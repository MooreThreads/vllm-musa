# vLLM MUSA Platform Plugin - Makefile
# =====================================

.PHONY: help install dev-install pre-commit test test-cov test-engine-plan build publish publish-test clean all

# Default target
help:
	@echo "vLLM MUSA Platform Plugin - Available targets:"
	@echo ""
	@echo "  Development:"
	@echo "    make dev-install  - Install package in development mode"
	@echo "    make install      - Install package"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make pre-commit   - Run pre-commit hooks on all files"
	@echo ""
	@echo "  Testing:"
	@echo "    make test         - Run all tests"
	@echo "    make test-cov     - Run tests with coverage report"
	@echo "    make test-engine-plan - Run focused RuntimePlan and plan-builder tests"
	@echo ""
	@echo "  Build & Publish:"
	@echo "    make build        - Build wheel and sdist"
	@echo "    make publish      - Build and publish to PyPI"
	@echo "    make publish-test - Build and publish to TestPyPI"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean        - Remove build artifacts"
	@echo ""
	@echo "  Combined:"
	@echo "    make all          - pre-commit, test, build"

# =============================================================================
# Development
# =============================================================================

dev-install:
	pip install -e ".[dev]" --no-build-isolation -v

install:
	pip install . --no-build-isolation -v

# =============================================================================
# Code Quality (via pre-commit)
# =============================================================================

pre-commit:
	pre-commit run --all-files

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=vllm_musa --cov-report=term-missing --cov-report=html

test-engine-plan:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
		python -m pytest -q -p no:cacheprovider \
		tests/test_engine_plan_artifact_io.py \
		tests/test_engine_plan_autotuner.py \
		tests/test_engine_plugin_ir_catalog.py \
		tests/test_engine_plan_builder_cli.py \
		tests/test_engine_plugins.py \
		tests/test_runtime_plan_core.py \
		tests/test_runtime_plan_declarative.py \
		tests/test_runtime_plan_engine_decisions.py \
		tests/test_builtin_engine_plan.py \
		tests/test_qwen_runtime_plan.py \
		tests/test_qwen_runtime_plan_source.py \
		tests/test_qwen3_qk_rope_kv_presplit.py \
		tests/test_deepseek_v4_runtime_plan.py \
		tests/test_deepseek_v4_runtime_plan_source.py

# =============================================================================
# Build & Publish
# =============================================================================

# Build wheel and source distribution
build: clean
	python -m build

# Publish to PyPI
publish: build
	python -m twine upload --repository pypi dist/*

# Publish to TestPyPI (for testing)
publish-test: build
	python -m twine upload --repository testpypi dist/*

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf vllm_musa.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# =============================================================================
# Combined Targets
# =============================================================================

all: pre-commit test build
