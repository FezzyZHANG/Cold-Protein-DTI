# Load optional local overrides.
-include .env.mk

SSH_HOST ?= hpc-zzy
REMOTE_ROOT ?= /share/home/grp-huangxd/zhangziyue/Cold-Protein-DTI
LOCAL_ROOT ?= $(CURDIR)
REMOTE_PYTHON ?= python
REMOTE_ACTIVATE ?= source .venv/bin/activate && 
RSYNC_EXCLUDES ?= --exclude .git/ --exclude .venv/ --exclude __pycache__/ --exclude '*.pt' --exclude '*.ckpt' --exclude logs/ --exclude results/
DEFAULT_EXP ?= config/experiments.example.yaml
EXP ?= $(DEFAULT_EXP)
REMOTE_LOG ?= logs/remote_train.log
RESULTS_DIR ?= results
RESULTS_REMOTE_DIR ?= $(REMOTE_ROOT)/$(RESULTS_DIR)
RESULTS_LOCAL_DIR ?= $(LOCAL_ROOT)/$(RESULTS_DIR)
RESULTS_RSYNC_FLAGS ?= -avz --progress

RSYNC := rsync $(RESULTS_RSYNC_FLAGS)
SSH := ssh $(SSH_HOST)

.PHONY: help check-env mkdirs sync pull pull-results push-results clean-results-list bootstrap remote-smoke remote-train remote-eval remote-kdbnet remote-scopedti logs remote-bash

help:
	@echo "Targets:"
	@echo "  make check-env              # print effective variables"
	@echo "  make mkdirs                 # create local logs/results directories"
	@echo "  make sync                   # rsync local repo to remote (excluding results)"
	@echo "  make pull                   # pull remote results/logs back"
	@echo "  make pull-results           # pull remote results/ back"
	@echo "  make push-results           # push local results/ to remote"
	@echo "  make clean-results-list     # list local result runs"
	@echo "  make bootstrap              # run remote environment sanity checks"
	@echo "  make remote-smoke           # run remote smoke test"
	@echo "  make remote-train EXP=...   # launch remote training"
	@echo "  make remote-eval EXP=...    # launch remote evaluation"
	@echo "  make remote-kdbnet EXP=...  # launch remote KDBNet benchmark"
	@echo "  make remote-scopedti EXP=... # launch remote Scope-DTI benchmark"
	@echo "  make logs                   # tail remote log"
	@echo "  make remote-bash            # open a shell in remote project root"

check-env:
	@echo "SSH_HOST=$(SSH_HOST)"
	@echo "REMOTE_ROOT=$(REMOTE_ROOT)"
	@echo "LOCAL_ROOT=$(LOCAL_ROOT)"
	@echo "REMOTE_PYTHON=$(REMOTE_PYTHON)"
	@echo "DEFAULT_EXP=$(DEFAULT_EXP)"
	@echo "RESULTS_DIR=$(RESULTS_DIR)"
	@echo "RESULTS_REMOTE_DIR=$(RESULTS_REMOTE_DIR)"
	@echo "RESULTS_LOCAL_DIR=$(RESULTS_LOCAL_DIR)"

mkdirs:
	mkdir -p logs $(RESULTS_DIR)

sync:
	$(RSYNC) $(RSYNC_EXCLUDES) $(LOCAL_ROOT)/ $(SSH_HOST):$(REMOTE_ROOT)/

pull: pull-results
	$(RSYNC) $(SSH_HOST):$(REMOTE_ROOT)/logs/ $(LOCAL_ROOT)/logs/

pull-results:
	$(SSH) 'mkdir -p $(RESULTS_REMOTE_DIR)'
	mkdir -p $(RESULTS_LOCAL_DIR)
	$(RSYNC) $(SSH_HOST):$(RESULTS_REMOTE_DIR)/ $(RESULTS_LOCAL_DIR)/

push-results:
	$(SSH) 'mkdir -p $(RESULTS_REMOTE_DIR)'
	mkdir -p $(RESULTS_LOCAL_DIR)
	$(RSYNC) $(RESULTS_LOCAL_DIR)/ $(SSH_HOST):$(RESULTS_REMOTE_DIR)/

clean-results-list:
	@if [ -d "$(RESULTS_LOCAL_DIR)" ]; then find "$(RESULTS_LOCAL_DIR)" -mindepth 1 -maxdepth 1 -type d | sort; fi

bootstrap:
	$(SSH) 'cd $(REMOTE_ROOT) && bash scripts/bootstrap_remote.sh'

remote-smoke:
	$(SSH) 'cd $(REMOTE_ROOT) && $(REMOTE_ACTIVATE) $(REMOTE_PYTHON) -V && $(REMOTE_PYTHON) -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"'

remote-train:
	$(SSH) 'cd $(REMOTE_ROOT) && mkdir -p logs && nohup bash scripts/remote_run.sh train $(EXP) > $(REMOTE_LOG) 2>&1 & echo $$!'

remote-eval:
	$(SSH) 'cd $(REMOTE_ROOT) && mkdir -p logs && nohup bash scripts/remote_run.sh eval $(EXP) > logs/remote_eval.log 2>&1 & echo $$!'

remote-kdbnet:
	$(SSH) 'cd $(REMOTE_ROOT) && mkdir -p logs runs/kdbnet artifacts/kdbnet && nohup bash scripts/remote_run.sh kdbnet $(EXP) > logs/remote_kdbnet.log 2>&1 & echo $$!'

remote-scopedti:
	$(SSH) 'cd $(REMOTE_ROOT) && mkdir -p logs runs/scopedti artifacts/ScopeDTI && nohup bash scripts/remote_run.sh scopedti $(EXP) > logs/remote_scopedti.log 2>&1 & echo $$!'

logs:
	$(SSH) 'cd $(REMOTE_ROOT) && tail -n 200 -f $(REMOTE_LOG)'

remote-bash:
	$(SSH) 'cd $(REMOTE_ROOT) && bash -l'
