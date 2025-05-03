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
	$(UVICORN) api:app --reload --port 8000

format:
	black .

lint:
	ruff check .

test:
	$(VENV)/bin/pytest

test-unit:
	$(VENV)/bin/pytest -m "not integration"

test-integration:
	$(VENV)/bin/pytest -m integration

ollama:
	ollama serve

.PHONY: install run test ollama format

