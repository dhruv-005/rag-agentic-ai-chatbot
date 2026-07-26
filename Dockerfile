FROM python:3.11-slim

# set working directory inside container
WORKDIR /app

# install system dependencies needed for pymupdf
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first so docker caches this layer
COPY requirements.txt .

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files into container
COPY . .

# create data directory inside container
RUN mkdir -p data/chroma_db

# expose the api port
EXPOSE 8000

# start the fastapi server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
