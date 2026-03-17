# AGENTS.md

## Project context

This repository is for a drug-target interaction / protein-generalization research project

Primary workflow:
- Code is written locally on Windows powershell
- The repository is synchronized to a remote CentOS 7 server via rsync
- Training and long-running experiments are executed remotely over SSH
- The server has A100 GPUs, but the software stack should remain conservative because CentOS 7 is old
- Use uv for dependency management

## Core engineering constraints

1. Do **not** assume a modern Ubuntu-like runtime on the remote server.
2. Prefer conservative dependencies and APIs.
3. Avoid introducing tools that are likely to fail on CentOS 7 unless explicitly requested.
4. Favor standard Python, PyTorch, argparse / yaml, and simple shell scripts.
5. All experiment code must be runnable non-interactively from CLI.
6. Every experiment must write machine-readable outputs, especially `metrics.json`.

## Dependency policy

Preferred:
- Python 3.10 or lower if required by cluster constraints
- PyTorch with standard AMP only
- PyYAML
- pandas / polars if already used in the repo
- numpy
- scikit-learn

Avoid by default:
- flash-attn
- xformers
- torch.compile
- DeepSpeed
- exotic CUDA-specific packages
- fragile distributed stacks unless clearly needed

## Coding policy

When generating code:
- keep modules small and explicit
- add docstrings for nontrivial functions
- prefer deterministic behavior when possible
- expose paths and hyperparameters through config files or CLI arguments
- never hard-code cluster-specific absolute paths without putting them behind environment variables
- log enough information for debugging on remote machines

## Experiment policy

Each experiment should:
- have a unique run directory
- save full config used for the run
- save `metrics.json`
- save a plain-text log or stdout redirection target
- support resume or safe rerun when feasible

Expected output layout:

```text
results/
  <run_name>/
    config.resolved.yaml
    metrics.json
    train.log
    checkpoints/
```

## Remote execution policy

Assume remote commands are launched through scripts and Makefile targets.
Code should support command lines like:

```bash
python -m src.train --config config/experiments/cp_easy.yaml --seed xxxx
```

and remote wrappers like:

```bash
bash scripts/remote_run.sh train config/experiments/cp_easy.yaml --seed xxxx
```

## What AI assistants should optimize for

1. Reliability over cleverness.
2. Minimal environment friction on CentOS 7.
3. Clear experiment orchestration.
4. Structured outputs for later aggregation.
5. Maintainability for a paper-oriented research codebase.

## What to avoid in generated code

- silent fallback behavior
- hidden global state
- implicit downloads during training
- code that only works in notebooks
- complicated shell one-liners when a script is clearer
- changing existing public interfaces unless necessary

## Preferred tasks for AI

Good tasks:
- scaffolding config-driven training scripts
- generating Makefile targets
- writing result aggregation scripts
- creating metrics loaders and summary tables
- writing smoke tests
- generating README / usage docs
- refactoring repeated experiment boilerplate

Higher-risk tasks requiring extra caution:
- modifying dataset split logic
- changing evaluation definitions
- altering negative sampling behavior
- adding distributed training
- adding new deep learning dependencies
