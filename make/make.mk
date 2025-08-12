.PHONY: clean setup run-example

clean: ## Clean up the development dependencies
	@rm -f .venv
	@uv cache clean
	@rm -f uv.lock

setup: ## Install the development dependencies
	@rm -f uv.lock
	@echo "Installing python version: ${PYTHON_VERSION}"
	@uv venv --python ${PYTHON_VERSION} --allow-existing
	@uv sync --all-extras

run-example: ## Run an example. Filename must be passed with FILENAME
	@echo "running example ${FILENAME}"
	@uv run src/main.py -c examples/${FILENAME} -o /workspace/output
