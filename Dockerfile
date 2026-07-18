# HF Docker Space — THC LLM
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HF_HOME="/home/user/.cache/huggingface"

WORKDIR /app

COPY --chown=user requirements.txt .

# Instala llama-cpp-python usando wheel pré-compilada pra CPU (evita build lento/travando)
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]