.PHONY: install simulate train evaluate serve test lint clean

install:
	pip install -r requirements.txt
	pip install -e .

simulate:
	python -m yieldguard.data.simulate --config configs/default.yaml

train:
	python -m yieldguard.pipelines.train --config configs/default.yaml

evaluate:
	python -m yieldguard.pipelines.evaluate --config configs/default.yaml

serve:
	uvicorn yieldguard.pipelines.api:app --reload --port 8000

test:
	pytest -q

lint:
	python -m pyflakes src/yieldguard

clean:
	rm -rf artifacts/models/* artifacts/reports/* data/processed/*
	find . -name "__pycache__" -type d -exec rm -rf {} +
