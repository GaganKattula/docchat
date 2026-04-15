.PHONY: install dev run test lint format clean

install:
	uv pip install -r requirements.txt

dev:
	uv pip install -e ".[dev]"

run:
	streamlit run app.py

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=core --cov=llm_config --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ dist/ build/
