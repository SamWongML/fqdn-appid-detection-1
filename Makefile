# FQDN Orphan Detection - Makefile

.PHONY: help install dev test lint format clean train predict api docker-build docker-up

# Default target
help:
	@echo "FQDN Orphan Detection ML System"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Development:"
	@echo "  install      Install production dependencies"
	@echo "  dev          Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo ""
	@echo "Training & Inference:"
	@echo "  train        Train model with default settings"
	@echo "  train-cv     Train with cross-validation"
	@echo "  predict      Run batch prediction"
	@echo "  evaluate     Evaluate model on test set"
	@echo ""
	@echo "API & Docker:"
	@echo "  api          Start API server"
	@echo "  docker-build Build Docker images"
	@echo "  docker-up    Start Docker services"

# ============================================
# Development
# ============================================

install:
	uv sync

dev:
	uv sync --dev

test:
	uv run pytest tests/ -v --tb=short

test-cov:
	uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	uv run flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503,E203
	uv run mypy src/ --ignore-missing-imports

format:
	uv run black src/ tests/ scripts/ --line-length=120 --target-version py311
	uv run isort src/ tests/ scripts/

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	rm -rf models/trained/* reports/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# ... (omitted sections)

ci-test:
	uv run pytest tests/ -v --tb=short --junitxml=test-results.xml

ci-lint:
	uv run flake8 src/ --max-line-length=120 --ignore=E501,W503 --format=default > lint-results.txt || true
