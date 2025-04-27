VENV=venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
UVICORN=$(VENV)/bin/uvicorn

install:
	$(PIP) install -r requirements.txt

run:
	$(UVICORN) main:app --reload --port 8000

.PHONY: install run