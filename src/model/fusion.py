"""Fusion heads for pooled concatenation and SCOPE-style bilinear attention."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from src.model.common import activation_gain, init_batch_norm, init_linear, masked_mean


def _require_rank(tensor: torch.Tensor, rank: int, name: str) -> torch.Tensor:
    """Validate tensor rank with an actionable error message."""

    if tensor.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}. Received shape {tuple(tensor.shape)}.")
    return tensor


def _pooled_features(output: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Fetch pooled features or derive them from token-level features and masks."""

    pooled = output.get("pooled")
    if pooled is not None:
        return _require_rank(pooled, 2, f"{name} pooled")

    token_features = _require_rank(output["token_features"], 3, f"{name} token_features")
    mask = _require_rank(output["mask"], 2, f"{name} mask")
    return masked_mean(token_features, mask)


def _promote_pair_dtype(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Promote fusion inputs to a shared dtype before concatenation/attention."""
    common_dtype = torch.promote_types(left.dtype, right.dtype)
    if left.dtype != common_dtype:
        left = left.to(common_dtype)
    if right.dtype != common_dtype:
        right = right.to(common_dtype)
    return left, right


def _disabled_autocast_context(device_type: str) -> torch.amp.autocast_mode.autocast | nullcontext:
    """Disable AMP for numerically sensitive fusion submodules."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=False)
    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _make_weight_norm_linear(in_dim: int, out_dim: int, *, gain: float = 1.0, bias: bool = True) -> nn.Linear:
    """Create a weight-normalized linear layer with explicit initialization."""

    linear = nn.Linear(int(in_dim), int(out_dim), bias=bias)
    init_linear(linear, gain=gain)
    return weight_norm(linear, dim=None)


class FCNet(nn.Module):
    """Simple non-linear fully connected network from the BAN reference implementation."""

    def __init__(self, dims: list[int], act: str = "ReLU", dropout: float = 0.0) -> None:
        super().__init__()
        gain = activation_gain(act)
        layers: list[nn.Module] = []
        for index in range(len(dims) - 2):
            in_dim = int(dims[index])
            out_dim = int(dims[index + 1])
            layers.append(_make_weight_norm_linear(in_dim, out_dim, gain=gain))
            if act:
                layers.append(getattr(nn, act)())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(_make_weight_norm_linear(int(dims[-2]), int(dims[-1]), gain=gain))
        if act:
            layers.append(getattr(nn, act)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.main(inputs)


class BANLayer(nn.Module):
    """Bilinear attention layer adapted from SCOPE-DTI."""

    def __init__(
        self,
        v_dim: int,
        q_dim: int,
        h_dim: int,
        h_out: int,
        act: str = "ReLU",
        dropout: float = 0.2,
        k: int = 3,
    ) -> None:
        super().__init__()
        self.c = 32
        self.k = int(k)
        self.v_dim = int(v_dim)
        self.q_dim = int(q_dim)
        self.h_dim = int(h_dim)
        self.h_out = int(h_out)

        self.v_net = FCNet([self.v_dim, self.h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([self.q_dim, self.h_dim * self.k], act=act, dropout=dropout)
        if self.k > 1:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if self.h_out <= self.c:
            self.h_mat = nn.Parameter(torch.empty(1, self.h_out, 1, self.h_dim * self.k))
            self.h_bias = nn.Parameter(torch.empty(1, self.h_out, 1, 1))
        else:
            self.h_net = _make_weight_norm_linear(self.h_dim * self.k, self.h_out)

        self.bn = nn.BatchNorm1d(self.h_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if hasattr(self, "h_mat"):
            nn.init.xavier_uniform_(self.h_mat)
            nn.init.zeros_(self.h_bias)
        init_batch_norm(self.bn)

    def attention_pooling(self, v: torch.Tensor, q: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        fusion_logits = torch.einsum("bvk,bvq,bqk->bk", v, att_map, q)
        if self.k > 1:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v: torch.Tensor, q: torch.Tensor, softmax: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        v = _require_rank(v, 3, "drug BAN input")
        q = _require_rank(q, 3, "protein BAN input")
        if v.size(0) != q.size(0):
            raise ValueError(
                "Drug and protein BAN inputs must share the same batch size. "
                f"Received {v.size(0)} and {q.size(0)}."
            )

        v_num = v.size(1)
        q_num = q.size(1)
        v_proj = self.v_net(v)
        q_proj = self.q_net(q)

        if self.h_out <= self.c:
            att_maps = torch.einsum("xhyk,bvk,bqk->bhvq", self.h_mat, v_proj, q_proj) + self.h_bias
        else:
            bilinear = torch.matmul(v_proj.transpose(1, 2).unsqueeze(3), q_proj.transpose(1, 2).unsqueeze(2))
            att_maps = self.h_net(bilinear.transpose(1, 2).transpose(2, 3))
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)

        if softmax:
            probs = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), dim=2)
            att_maps = probs.view(-1, self.h_out, v_num, q_num)

        logits = self.attention_pooling(v_proj, q_proj, att_maps[:, 0, :, :])
        for index in range(1, self.h_out):
            logits = logits + self.attention_pooling(v_proj, q_proj, att_maps[:, index, :, :])
        logits = self.bn(logits)
        return logits, att_maps


class MLPDecoder(nn.Module):
    """Classifier head matching the stacked MLP used after BAN in SCOPE-DTI."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, binary: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        relu_gain = activation_gain("relu")
        init_linear(self.fc1, gain=relu_gain)
        init_batch_norm(self.bn1)
        init_linear(self.fc2, gain=relu_gain)
        init_batch_norm(self.bn2)
        init_linear(self.fc3, gain=relu_gain)
        init_batch_norm(self.bn3)
        init_linear(self.fc4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.bn1(torch.relu(self.fc1(inputs)))
        outputs = self.bn2(torch.relu(self.fc2(outputs)))
        outputs = self.bn3(torch.relu(self.fc3(outputs)))
        return self.fc4(outputs)


class ConcatFusion(nn.Module):
    """Baseline concatenation + MLP fusion head."""

    def __init__(self, drug_input_dim: int, protein_input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(int(drug_input_dim) + int(protein_input_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gelu_gain = activation_gain("gelu")
        init_linear(self.classifier[0], gain=gelu_gain)
        init_linear(self.classifier[3])

    def forward(self, drug_output: dict[str, torch.Tensor], protein_output: dict[str, torch.Tensor]) -> torch.Tensor:
        drug_pooled = _pooled_features(drug_output, "drug")
        protein_pooled = _pooled_features(protein_output, "protein")
        drug_pooled, protein_pooled = _promote_pair_dtype(drug_pooled, protein_pooled)
        if drug_pooled.size(0) != protein_pooled.size(0):
            raise ValueError(
                "Drug and protein pooled features must share the same batch size. "
                f"Received {drug_pooled.size(0)} and {protein_pooled.size(0)}."
            )

        fused = torch.cat([drug_pooled, protein_pooled], dim=-1)
        return self.classifier(fused).squeeze(-1)


class BANFusion(nn.Module):
    """SCOPE-style BAN over a pooled drug embedding and residue-level protein features."""

    def __init__(
        self,
        drug_input_dim: int,
        protein_input_dim: int,
        joint_dim: int,
        classifier_hidden_dim: int,
        glimpses: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.drug_input_dim = int(drug_input_dim)
        self.protein_input_dim = int(protein_input_dim)
        self.last_attention_maps: torch.Tensor | None = None

        ban = BANLayer(
            v_dim=self.drug_input_dim,
            q_dim=self.protein_input_dim,
            h_dim=int(joint_dim),
            h_out=int(glimpses),
            dropout=dropout,
        )
        self.ban = weight_norm(ban, name="h_mat", dim=None) if hasattr(ban, "h_mat") else ban
        self.classifier = MLPDecoder(
            in_dim=int(joint_dim),
            hidden_dim=int(classifier_hidden_dim),
            out_dim=int(joint_dim),
            binary=1,
        )

    def forward(self, drug_output: dict[str, torch.Tensor], protein_output: dict[str, torch.Tensor]) -> torch.Tensor:
        drug_pooled = _pooled_features(drug_output, "drug")
        protein_tokens = _require_rank(protein_output["token_features"], 3, "protein token_features")
        drug_pooled, protein_tokens = _promote_pair_dtype(drug_pooled, protein_tokens)

        if drug_pooled.size(0) != protein_tokens.size(0):
            raise ValueError(
                "Drug and protein features must share the same batch size. "
                f"Received {drug_pooled.size(0)} and {protein_tokens.size(0)}."
            )
        if drug_pooled.size(1) != self.drug_input_dim:
            raise ValueError(
                "Drug pooled feature width does not match the BAN input dimension. "
                f"Expected {self.drug_input_dim}, received {drug_pooled.size(1)}."
            )
        if protein_tokens.size(2) != self.protein_input_dim:
            raise ValueError(
                "Protein token feature width does not match the BAN input dimension. "
                f"Expected {self.protein_input_dim}, received {protein_tokens.size(2)}."
            )

        device_type = drug_pooled.device.type
        with _disabled_autocast_context(device_type):
            fused, attention_maps = self.ban(
                drug_pooled.to(torch.float32).unsqueeze(1),
                protein_tokens.to(torch.float32),
            )
        self.last_attention_maps = attention_maps.detach()
        return self.classifier(fused).squeeze(-1)
