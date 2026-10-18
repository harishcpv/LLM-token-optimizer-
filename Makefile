PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: test demo

test:
	$(PYTHON) -m compileall -q app.py audit_cli.py src tests
	$(PYTHON) -m pytest

demo:
	$(PYTHON) audit_cli.py \
		--code-file examples/agent.py \
		--prompt-file examples/prompt.json \
		--tools-file examples/tools.json \
		> demo-report.json
	@$(PYTHON) -c "import json; report=json.load(open('demo-report.json', encoding='utf-8')); print(f\"findings={len(report['findings'])} prompt_tokens={report['tokens']['prompt_total']}\")"
