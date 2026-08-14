.PHONY: install run test lint eval data ingest docker-build docker-up docker-down

install:
	uv sync --all-extras --no-editable

run:
	uv run streamlit run app.py --server.port 8501

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

eval:
	uv run python src/evaluation/retrieval_eval_runner.py
	uv run python src/evaluation/judge_runner.py

data:
	uv run python scripts/download_data.py

ingest:
	uv run python scripts/ingest.py
	uv run python scripts/build_indices.py

docker-build:
	docker build -t cpie:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down
