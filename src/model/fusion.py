"""Fusion heads for pooled concatenation and SCOPE-style bilinear attention."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from src.model.common import activation_gain, init_batch_norm, init_layer_norm, init_linear, masked_mean


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


def _canonicalize_norm_kind(kind: str | None) -> str:
    """Normalize user-facing normalization aliases to a compact internal form."""

    if kind is None:
        return "none"
    normalized = str(kind).lower().replace("_", "").replace("-", "")
    if normalized in {"batchnorm", "bn"}:
        return "batchnorm"
    if normalized in {"layernorm", "ln"}:
        return "layernorm"
    if normalized in {"none", "identity"}:
        return "none"
    raise ValueError(f"Unsupported normalization kind: {kind!r}")


def _make_feature_norm(dim: int, kind: str) -> nn.Module:
    """Create a conservative feature normalization layer for rank-2 activations."""

    normalized_kind = _canonicalize_norm_kind(kind)
    if normalized_kind == "batchnorm":
        return nn.BatchNorm1d(int(dim))
    if normalized_kind == "layernorm":
        return nn.LayerNorm(int(dim))
    return nn.Identity()


def _reset_feature_norm(module: nn.Module) -> None:
    """Initialize the requested normalization layer as a no-op."""

    if isinstance(module, nn.BatchNorm1d):
        init_batch_norm(module)
    elif isinstance(module, nn.LayerNorm):
        init_layer_norm(module)


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
        norm: str = "layernorm",
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

        self.output_norm = _make_feature_norm(self.h_dim, norm)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if hasattr(self, "h_mat"):
            nn.init.xavier_uniform_(self.h_mat)
            nn.init.zeros_(self.h_bias)
        _reset_feature_norm(self.output_norm)

    def attention_pooling(self, v: torch.Tensor, q: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        fusion_logits = torch.einsum("bvk,bvq,bqk->bk", v, att_map, q)
        if self.k > 1:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(
        self,
        v: torch.Tensor,
        q: torch.Tensor,
        v_mask: torch.Tensor | None = None,
        q_mask: torch.Tensor | None = None,
        softmax: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = _require_rank(v, 3, "drug BAN input")
        q = _require_rank(q, 3, "protein BAN input")
        if v.size(0) != q.size(0):
            raise ValueError(
                "Drug and protein BAN inputs must share the same batch size. "
                f"Received {v.size(0)} and {q.size(0)}."
            )
        if v_mask is not None:
            v_mask = _require_rank(v_mask, 2, "drug BAN mask").to(dtype=torch.bool)
            if v_mask.shape != v.shape[:2]:
                raise ValueError(
                    "Drug BAN mask must match the first two dimensions of the BAN input. "
                    f"Received mask shape {tuple(v_mask.shape)} for features {tuple(v.shape)}."
                )
        if q_mask is not None:
            q_mask = _require_rank(q_mask, 2, "protein BAN mask").to(dtype=torch.bool)
            if q_mask.shape != q.shape[:2]:
                raise ValueError(
                    "Protein BAN mask must match the first two dimensions of the BAN input. "
                    f"Received mask shape {tuple(q_mask.shape)} for features {tuple(q.shape)}."
                )

        v_num = v.size(1)
        q_num = q.size(1)
        v_proj = self.v_net(v)
        q_proj = self.q_net(q)
        if v_mask is not None:
            v_proj = v_proj * v_mask.unsqueeze(-1).to(v_proj.dtype)
        if q_mask is not None:
            q_proj = q_proj * q_mask.unsqueeze(-1).to(q_proj.dtype)

        if self.h_out <= self.c:
            att_maps = torch.einsum("xhyk,bvk,bqk->bhvq", self.h_mat, v_proj, q_proj) + self.h_bias
        else:
            bilinear = torch.matmul(v_proj.transpose(1, 2).unsqueeze(3), q_proj.transpose(1, 2).unsqueeze(2))
            att_maps = self.h_net(bilinear.transpose(1, 2).transpose(2, 3))
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)

        valid_pairs: torch.Tensor | None = None
        if v_mask is not None or q_mask is not None:
            if v_mask is None:
                v_mask = torch.ones(v.shape[:2], dtype=torch.bool, device=v.device)
            if q_mask is None:
                q_mask = torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
            valid_pairs = v_mask.unsqueeze(1).unsqueeze(3) & q_mask.unsqueeze(1).unsqueeze(2)

        if softmax:
            flat_att_maps = att_maps.reshape(-1, self.h_out, v_num * q_num)
            if valid_pairs is not None:
                flat_valid_pairs = valid_pairs.reshape(-1, 1, v_num * q_num)
                fill_value = torch.finfo(flat_att_maps.dtype).min
                flat_att_maps = flat_att_maps.masked_fill(~flat_valid_pairs, fill_value)
            probs = nn.functional.softmax(flat_att_maps, dim=2)
            att_maps = probs.view(-1, self.h_out, v_num, q_num)
            if valid_pairs is not None:
                att_maps = att_maps * valid_pairs.to(att_maps.dtype)
                flat_probs = att_maps.reshape(-1, self.h_out, v_num * q_num)
                flat_probs = flat_probs / flat_probs.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
                att_maps = flat_probs.view(-1, self.h_out, v_num, q_num)
        elif valid_pairs is not None:
            att_maps = att_maps * valid_pairs.to(att_maps.dtype)

        logits = self.attention_pooling(v_proj, q_proj, att_maps[:, 0, :, :])
        for index in range(1, self.h_out):
            logits = logits + self.attention_pooling(v_proj, q_proj, att_maps[:, index, :, :])
        logits = self.output_norm(logits)
        return logits, att_maps


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block for classifier refinement."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float, norm: str) -> None:
        super().__init__()
        self.norm = _make_feature_norm(dim, norm)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        gelu_gain = activation_gain("gelu")
        _reset_feature_norm(self.norm)
        init_linear(self.fc1, gain=gelu_gain)
        init_linear(self.fc2, gain=gelu_gain)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.fc1(self.norm(inputs))
        outputs = torch.nn.functional.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        outputs = self.dropout(outputs)
        return inputs + outputs


class MLPDecoder(nn.Module):
    """Classifier head matching the stacked MLP used after BAN in SCOPE-DTI."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        binary: int = 1,
        dropout: float = 0.0,
        norm: str = "layernorm",
        num_blocks: int = 2,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_norm = _make_feature_norm(in_dim, norm)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    dim=hidden_dim,
                    hidden_dim=max(hidden_dim, int(round(hidden_dim * expansion))),
                    dropout=dropout,
                    norm=norm,
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.head_norm = _make_feature_norm(hidden_dim, norm)
        self.head_proj = nn.Linear(hidden_dim, out_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(out_dim, binary)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gelu_gain = activation_gain("gelu")
        _reset_feature_norm(self.input_norm)
        init_linear(self.input_proj, gain=gelu_gain)
        for block in self.blocks:
            block.reset_parameters()
        _reset_feature_norm(self.head_norm)
        init_linear(self.head_proj, gain=gelu_gain)
        init_linear(self.output)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.input_proj(self.input_norm(inputs))
        outputs = torch.nn.functional.gelu(outputs)
        outputs = self.input_dropout(outputs)
        for block in self.blocks:
            outputs = block(outputs)
        outputs = self.head_proj(self.head_norm(outputs))
        outputs = torch.nn.functional.gelu(outputs)
        outputs = self.head_dropout(outputs)
        return self.output(outputs)


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
    """Masked two-sided BAN over configurable drug/protein feature granularities."""

    def __init__(
        self,
        drug_input_dim: int,
        protein_input_dim: int,
        joint_dim: int,
        classifier_hidden_dim: int,
        glimpses: int,
        dropout: float,
        drug_feature_mode: str = "token",
        protein_feature_mode: str = "token",
        use_global_features: bool = True,
        attention_softmax: bool = True,
        norm: str = "layernorm",
        classifier_num_blocks: int = 2,
        classifier_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        self.drug_input_dim = int(drug_input_dim)
        self.protein_input_dim = int(protein_input_dim)
        self.drug_feature_mode = str(drug_feature_mode).lower()
        self.protein_feature_mode = str(protein_feature_mode).lower()
        if self.drug_feature_mode not in {"token", "pooled"}:
            raise ValueError(f"drug_feature_mode must be `token` or `pooled`. Received {drug_feature_mode!r}.")
        if self.protein_feature_mode not in {"token", "pooled"}:
            raise ValueError(f"protein_feature_mode must be `token` or `pooled`. Received {protein_feature_mode!r}.")
        self.use_global_features = bool(use_global_features)
        self.attention_softmax = bool(attention_softmax)
        self.joint_dim = int(joint_dim)
        self.last_attention_maps: torch.Tensor | None = None

        ban = BANLayer(
            v_dim=self.drug_input_dim,
            q_dim=self.protein_input_dim,
            h_dim=self.joint_dim,
            h_out=int(glimpses),
            dropout=dropout,
            norm=norm,
        )
        self.ban = weight_norm(ban, name="h_mat", dim=None) if hasattr(ban, "h_mat") else ban
        if self.use_global_features:
            self.global_residual = nn.Sequential(
                nn.Linear(self.drug_input_dim + self.protein_input_dim, self.joint_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            gelu_gain = activation_gain("gelu")
            init_linear(self.global_residual[0], gain=gelu_gain)
        else:
            self.global_residual = None
        self.classifier = MLPDecoder(
            in_dim=self.joint_dim,
            hidden_dim=int(classifier_hidden_dim),
            out_dim=self.joint_dim,
            binary=1,
            dropout=dropout,
            norm=norm,
            num_blocks=int(classifier_num_blocks),
            expansion=float(classifier_expansion),
        )

    @staticmethod
    def _select_sequence_features(
        output: dict[str, torch.Tensor],
        name: str,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "pooled":
            pooled = _pooled_features(output, name)
            pooled = _require_rank(pooled, 2, f"{name} pooled")
            mask = torch.ones(pooled.size(0), 1, dtype=torch.bool, device=pooled.device)
            return pooled.unsqueeze(1), mask

        token_features = _require_rank(output["token_features"], 3, f"{name} token_features")
        mask = _require_rank(output["mask"], 2, f"{name} mask").to(dtype=torch.bool)
        if token_features.shape[:2] != mask.shape:
            raise ValueError(
                f"{name} token features and mask must agree on batch/length. "
                f"Received {tuple(token_features.shape)} and {tuple(mask.shape)}."
            )
        return token_features, mask

    @staticmethod
    def _ensure_nonempty_sequence(
        features: torch.Tensor,
        mask: torch.Tensor,
        fallback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonempty = mask.any(dim=1)
        if bool(nonempty.all().item()):
            return features, mask

        repaired_features = features.clone()
        repaired_mask = mask.clone()
        empty_rows = ~nonempty
        repaired_features[empty_rows] = 0
        repaired_features[empty_rows, 0, :] = fallback[empty_rows]
        repaired_mask[empty_rows] = False
        repaired_mask[empty_rows, 0] = True
        return repaired_features, repaired_mask

    def forward(self, drug_output: dict[str, torch.Tensor], protein_output: dict[str, torch.Tensor]) -> torch.Tensor:
        drug_global = _pooled_features(drug_output, "drug")
        protein_global = _pooled_features(protein_output, "protein")
        drug_features, drug_mask = self._select_sequence_features(
            drug_output,
            name="drug",
            mode=self.drug_feature_mode,
        )
        protein_features, protein_mask = self._select_sequence_features(
            protein_output,
            name="protein",
            mode=self.protein_feature_mode,
        )

        drug_features, protein_features = _promote_pair_dtype(drug_features, protein_features)
        drug_global, protein_global = _promote_pair_dtype(drug_global, protein_global)

        if drug_features.size(0) != protein_features.size(0):
            raise ValueError(
                "Drug and protein features must share the same batch size. "
                f"Received {drug_features.size(0)} and {protein_features.size(0)}."
            )
        if drug_features.size(2) != self.drug_input_dim:
            raise ValueError(
                "Drug feature width does not match the BAN input dimension. "
                f"Expected {self.drug_input_dim}, received {drug_features.size(2)}."
            )
        if protein_features.size(2) != self.protein_input_dim:
            raise ValueError(
                "Protein feature width does not match the BAN input dimension. "
                f"Expected {self.protein_input_dim}, received {protein_features.size(2)}."
            )

        drug_features, drug_mask = self._ensure_nonempty_sequence(drug_features, drug_mask, drug_global)
        protein_features, protein_mask = self._ensure_nonempty_sequence(protein_features, protein_mask, protein_global)

        device_type = drug_features.device.type
        with _disabled_autocast_context(device_type):
            fused, attention_maps = self.ban(
                drug_features.to(torch.float32),
                protein_features.to(torch.float32),
                v_mask=drug_mask,
                q_mask=protein_mask,
                softmax=self.attention_softmax,
            )
            if self.global_residual is not None:
                global_features = torch.cat(
                    [drug_global.to(torch.float32), protein_global.to(torch.float32)],
                    dim=-1,
                )
                fused = fused + self.global_residual(global_features)
        self.last_attention_maps = attention_maps.detach()
        return self.classifier(fused).squeeze(-1)
