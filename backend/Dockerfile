FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 1. Configuración clave para UV
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Copiar archivos de dependencias
COPY pyproject.toml uv.lock ./

# 2. Instalamos. Esto creará /app/.venv con certeza
RUN uv sync --frozen --no-install-project --all-extras

COPY . .

EXPOSE 8000

# Usamos la ruta absoluta del binario dentro del venv que creamos
CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]