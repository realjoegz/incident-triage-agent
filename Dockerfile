# The build stage resolves the locked dependencies, installs the package and fetches the
# embedding model. The final stage holds the virtual environment, the model and the data, and
# no build tool.
FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# The dependencies first, so that a change to the source does not install them again.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# The model is put into the image so that a container embeds its first query without a download.
RUN .venv/bin/python -c "from pathlib import Path; from triage.embeddings import LocalEmbedder; LocalEmbedder(Path('/app/models')).embed(['warm'])"


FROM python:3.12-slim-bookworm

RUN useradd --system --uid 10001 --no-create-home triage
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/models /app/models
COPY data ./data
COPY eval ./eval

ENV PATH=/app/.venv/bin:$PATH \
    TRIAGE_EMBEDDING_CACHE_DIR=/app/models \
    HF_HUB_OFFLINE=1
USER triage
EXPOSE 8080

# Loading the corpus is idempotent: it replaces the 80 rows in one transaction.
CMD ["sh", "-c", "triage db load && exec triage serve --host 0.0.0.0 --port 8080"]
