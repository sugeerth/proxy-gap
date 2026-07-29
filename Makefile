PY := python3

.PHONY: all install test run site clean verify

all: install test run site
	@echo ""
	@echo "  PROXY GAP: full pipeline reproduced."
	@echo "  open docs/index.html"

install:
	$(PY) -m pip install -q -e ".[dev]"

test:
	$(PY) -m pytest -q

run:
	$(PY) -m proxygap.cli all --out docs/data

site: run
	@echo "  docs/data regenerated from source; site is static, nothing to build."

verify:
	$(PY) -m proxygap.cli verify

clean:
	rm -rf docs/data/*.json .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
