FROM mcr.microsoft.com/vscode/devcontainers/base:bookworm
ARG USERNAME=vscode
ARG UV_VERSION

# install uv
USER ${USERNAME}
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/root/.local/bin:$PATH"
ENV PATH="/workspace/.venv/bin:$PATH"
ENV PYTHONPATH="/workspace/src:$PYTHONPATH"

