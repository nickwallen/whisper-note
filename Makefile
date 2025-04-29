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
	$(UVICORN) main:app --reload --port 8000

test:
	$(VENV)/bin/pytest

test-unit:
	$(VENV)/bin/pytest -m "not integration"

test-integration:
	$(VENV)/bin/pytest -m integration

ollama:
	ollama serve

.PHONY: install run test ollama