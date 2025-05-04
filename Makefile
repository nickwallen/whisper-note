VENV=venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
UVICORN=$(VENV)/bin/uvicorn

.DEFAULT_GOAL := default

default: install-dev format lint test

install:
	$(PIP) install -r requirements.txt -qq

install-dev:
	$(PIP) install -r requirements.txt -qq
	$(PIP) install -r requirements-dev.txt -qq

run:
	@mkdir -p logs; \
	( \
		set -m; \
		ollama serve > logs/ollama.log 2>&1 & \
		$(UVICORN) api:app --reload --port 8000 | tee logs/whisper-note.log \
	)

format:
	black .

lint:
	ruff check . --fix

test:
	$(VENV)/bin/pytest -x

test-unit:
	$(VENV)/bin/pytest -m "not integration"

test-integration:
	$(VENV)/bin/pytest -m integration

.PHONY: install run test format lint

