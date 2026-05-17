# NOTE: venv is in /tmp because the project's parent directory contains a colon
# (/Users/harry/RL:ML Project/) which Python's venv module treats as a PATH
# separator. Override with: make VENV=/your/path venv

.PHONY: test data baselines ml analysis all venv

VENV ?= /tmp/ml-vol-momentum-venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip

venv:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	$(VENV)/bin/pytest tests/ -v

data:
	$(PYTHON) -m src.data.universe
	$(PYTHON) -m src.data.loaders

baselines:
	$(PYTHON) -m src.models.baselines

ml:
	$(PYTHON) -m src.models.gbm
	$(PYTHON) -m src.models.lstm_model

analysis:
	$(PYTHON) -m src.eval.comparison
	$(PYTHON) -m src.interp.shap_analysis
	$(PYTHON) -m src.interp.regime_analysis

all: data baselines ml analysis
