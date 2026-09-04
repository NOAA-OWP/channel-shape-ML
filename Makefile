SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help env env-cuda docker-build run clean

help:
	@echo "=========================================================================="
	@echo "RIver ML Pipeline Execution Interface"
	@echo "=========================================================================="
	@echo "make env             : Create local Conda CPU environment (river_ml)"
	@echo "make env-cuda        : Create local Conda GPU environment (river_ml_cuda)"
	@echo "make docker-build    : Build multi-arch production Docker container"
	@echo "make run             : Run pipeline (requires explicit CLI parameter overrides)"
	@echo "make clean           : Purge local cache and temporary Python artifacts"

env:
	mamba env create -f environment.yml || conda env create -f environment.yml

env-cuda:
	mamba env create -f environment-cuda.yml || conda env create -f environment-cuda.yml

docker-build:
	docker build -t river_ml_pipeline:latest .

run:
	@if [ -z "$(FLOWLINES)" ] || [ -z "$(SLOPES)" ] || [ -z "$(OUTPUT)" ] || [ -z "$(MODELS)" ] || [ -z "$(DOMAIN)" ]; then \
		echo "ERROR: Missing required parameters. Usage:"; \
		echo "  make run FLOWLINES=<path> SLOPES=<path> OUTPUT=<path> MODELS=<path> DOMAIN=<name> [CHUNK=100000]"; \
		exit 1; \
	fi
	python run_inference.py \
		--flowlines_path "$(FLOWLINES)" \
		--slopes_path "$(SLOPES)" \
		--output_dir "$(OUTPUT)" \
		--save_model_path "$(MODELS)" \
		--process_domain "$(DOMAIN)" \
		--chunk_size $(or $(CHUNK),100000) \
		--lambda_reg $(or $(LAMBDA),10.0) \
		--gamma_reg $(or $(GAMMA),0.01)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete