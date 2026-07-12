.PHONY: install run test lint eval docker-build docker-run

install:
	uv sync --all-extras --no-editable

run:
	uv run streamlit run app.py --server.port 8501

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

eval:
	uv run python src/evaluation/eval_runner.py

docker-build:
	docker build -t cpie:latest .

docker-run:
	docker-compose up
