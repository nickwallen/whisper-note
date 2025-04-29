VENV=venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
UVICORN=$(VENV)/bin/uvicorn

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

run: 
	$(UVICORN) api:app --reload --port 8000

format:
	black .

test:
	$(VENV)/bin/pytest

test-unit:
	$(VENV)/bin/pytest -m "not integration"

test-integration:
	$(VENV)/bin/pytest -m integration

ollama:
	ollama serve

.PHONY: install run test ollama format

