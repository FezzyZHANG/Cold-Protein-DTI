# Load optional local overrides.
-include .env.mk

SSH_HOST ?= gpu01
# SSH_PORT ?= 22
REMOTE_ROOT ?= /path/to/remote/project
LOCAL_ROOT ?= $(CURDIR)
REMOTE_PYTHON ?= python
REMOTE_ENV_NAME ?= bioml
REMOTE_ACTIVATE ?= source ~/.bashrc && conda activate $(REMOTE_ENV_NAME)
RSYNC_EXCLUDES ?= --exclude .git/ --exclude .venv/ --exclude __pycache__/ --exclude '*.pt' --exclude '*.ckpt' --exclude logs/ --exclude results/
DEFAULT_EXP ?= config/experiments.example.yaml
EXP ?= $(DEFAULT_EXP)
REMOTE_LOG ?= logs/remote_train.log

RSYNC := rsync -avz --progress -e "ssh -p $(SSH_PORT)"
# SSH := ssh -p $(SSH_PORT) $(SSH_HOST)
SSH := ssh $(SSH_HOST)

.PHONY: help check-env sync pull logs remote-smoke remote-train remote-eval remote-bash mkdirs bootstrap

help:
	@echo "Targets:"
	@echo "  make check-env         # print effective variables"
	@echo "  make mkdirs            # create local logs/results directories"
	@echo "  make sync              # rsync local repo to remote"
	@echo "  make pull              # pull remote results/logs back"
	@echo "  make bootstrap         # run remote environment sanity checks"
	@echo "  make remote-smoke      # run remote smoke test"
	@echo "  make remote-train EXP=...  # launch remote training"
	@echo "  make remote-eval EXP=...   # launch remote evaluation"
	@echo "  make logs              # tail remote log"
	@echo "  make remote-bash       # open a shell in remote project root"

check-env:
	@echo "SSH_HOST=$(SSH_HOST)"
	@echo "SSH_PORT=$(SSH_PORT)"
	@echo "REMOTE_ROOT=$(REMOTE_ROOT)"
	@echo "LOCAL_ROOT=$(LOCAL_ROOT)"
	@echo "REMOTE_PYTHON=$(REMOTE_PYTHON)"
	@echo "REMOTE_ENV_NAME=$(REMOTE_ENV_NAME)"
	@echo "DEFAULT_EXP=$(DEFAULT_EXP)"

mkdirs:
	mkdir -p logs results

sync:
	$(RSYNC) $(RSYNC_EXCLUDES) $(LOCAL_ROOT)/ $(SSH_HOST):$(REMOTE_ROOT)/

pull:
	$(RSYNC) $(SSH_HOST):$(REMOTE_ROOT)/results/ $(LOCAL_ROOT)/results/
	$(RSYNC) $(SSH_HOST):$(REMOTE_ROOT)/logs/ $(LOCAL_ROOT)/logs/

bootstrap:
	$(SSH) 'cd $(REMOTE_ROOT) && bash scripts/bootstrap_remote.sh'

remote-smoke:
	$(SSH) 'cd $(REMOTE_ROOT) && $(REMOTE_ACTIVATE) && $(REMOTE_PYTHON) -V && $(REMOTE_PYTHON) -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"'

remote-train:
	$(SSH) 'cd $(REMOTE_ROOT) && mkdir -p logs && nohup bash scripts/remote_run.sh train $(EXP) > $(REMOTE_LOG) 2>&1 & echo $$!'

remote-eval:
	$(SSH) 'cd $(REMOTE_ROOT) && mkdir -p logs && nohup bash scripts/remote_run.sh eval $(EXP) > logs/remote_eval.log 2>&1 & echo $$!'

logs:
	$(SSH) 'cd $(REMOTE_ROOT) && tail -n 200 -f $(REMOTE_LOG)'

remote-bash:
	$(SSH) 'cd $(REMOTE_ROOT) && bash -l'
