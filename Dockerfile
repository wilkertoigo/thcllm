# Utiliza imagem oficial leve do Python
FROM python:3.10-slim

# Instala dependências de compilação essenciais para o llama.cpp em C++
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configura o diretório de trabalho interno
WORKDIR /code

# Instala as dependências Python antecipadamente para aproveitar o cache do Docker
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Cria um usuário não-root (Exigência obrigatória de segurança do HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Define a pasta do app e copia os arquivos com a permissão correta
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# O Hugging Face Spaces exige que containers Docker escutem estritamente na porta 7860
EXPOSE 7860

# Comando para iniciar o servidor Flask
CMD ["python", "app.py"]