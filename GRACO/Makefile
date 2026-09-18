# GRACO developer shortcuts.  Run `make help` for the list.
.DEFAULT_GOAL := help
.PHONY: help install dev test lint format typecheck smoke clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package (core deps only)
	pip install -e .

dev:  ## Install with dev + all extras (tensorboard, wandb, ILP solver, ruff, pytest)
	pip install -e ".[all]"

test:  ## Run the test suite
	pytest -q

lint:  ## Lint with ruff
	ruff check graco tests examples benchmarks

format:  ## Auto-fix lint + import order
	ruff check --fix graco tests examples benchmarks

smoke:  ## Fast end-to-end training smoke test (CPU, seconds)
	graco train -c debug

clean:  ## Remove caches, build artifacts and run outputs
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache runs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
