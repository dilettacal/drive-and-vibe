.PHONY: help install install-dev clean lint format test setup data process process-sample download-notebooks run-all

# Default target
help:
	@echo "Drive-and-Vibe - ADAS AI Behavior Lab"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install dependencies with uv"
	@echo "  make install-dev  - Install dev dependencies"
	@echo "  make clean        - Clean generated files"
	@echo "  make lint         - Run linting with ruff"
	@echo "  make format       - Format code with black and ruff"
	@echo "  make test         - Run tests with pytest"
	@echo "  make setup        - Run pre-commit setup"
	@echo "  make data         - Download Geolife dataset"
	@echo "  make process      - Process full Geolife dataset (CPU intensive)"
	@echo "  make process-sample - Process a small sample (fast)"
	@echo "  make jupyter      - Launch Jupyter notebook"
	@echo "  make run-all      - Run all notebooks in sequence"

# Install dependencies using uv
install:
	@if [ ! -f uv.lock ]; then \
		echo "Creating uv.lock..."; \
		uv lock; \
	fi
	uv sync

install-dev: install
	uv sync --dev

# Setup pre-commit hooks
setup:
	uv run pre-commit install

# Linting and formatting
lint:
	uv run ruff check src/ tests/
	uv run black --check src/ tests/

format:
	uv run black src/ tests/
	uv run ruff check --fix src/ tests/

# Run tests
test:
	uv run pytest tests/ --cov=src --cov-report=term-missing

# Download Geolife dataset
data:
	@echo "Downloading Geolife dataset..."
	uv run python scripts/download_geolife.py

# Process Geolife data
process:
	@echo "Processing Geolife GPS trajectory data..."
	uv run python scripts/process_data.py

# Process sample data (faster for testing)
process-sample:
	@echo "Processing sample Geolife data (10 trajectories)..."
	uv run python scripts/process_data.py --sample 10

# Launch Jupyter
jupyter:
	uv run jupyter notebook notebooks/

# Run all notebooks in sequence
run-all:
	uv run jupyter nbconvert --to notebook --execute notebooks/*.ipynb

# Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

