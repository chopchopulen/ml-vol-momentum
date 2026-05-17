.PHONY: test data baselines ml analysis all

PYTHON = /tmp/ml-vol-momentum-venv/bin/python

test:
	/tmp/ml-vol-momentum-venv/bin/pytest tests/ -v

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
