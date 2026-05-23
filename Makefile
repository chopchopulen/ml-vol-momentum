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
	# Run non-ML tests first, then ML model tests in separate invocations.
	# LightGBM (OpenMP) and PyTorch both use fork-unsafe C extensions that
	# cause SIGABRT when mixed in a single pytest process on macOS.
	$(VENV)/bin/pytest tests/ --ignore=tests/test_gbm.py --ignore=tests/test_lstm.py -v
	$(VENV)/bin/pytest tests/test_gbm.py -v
	$(VENV)/bin/pytest tests/test_lstm.py -v

data:
	$(PYTHON) scripts/build_data.py

baselines:
	$(PYTHON) scripts/run_baselines.py

ml:
	$(PYTHON) scripts/run_ml_models.py

analysis:
	$(PYTHON) scripts/compare_all_models.py
	$(PYTHON) scripts/run_phase4.py

all: data baselines ml analysis
