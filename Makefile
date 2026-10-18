PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: test

test:
	$(PYTHON) -m compileall -q app.py audit_cli.py src tests
	$(PYTHON) -m pytest
