# HF Docker Space — THC LLM (CPU Basic)
FROM python:3.11-slim

# Prepara usuário não-root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HF_HOME="/home/user/.cache/huggingface"
WORKDIR /app

# Copia requirements.txt primeiro (cache layer)
COPY --chown=user requirements.txt .

# Instala llama-cpp-python via wheel pré-compilada (CPU x86_64)
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Instala resto das libs (transformers, fastapi, etc)
RUN pip install --no-cache-dir uv && \
    pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

# O vervoranteocum
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
