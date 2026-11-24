FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

WORKDIR /code
COPY requirements.txt requirements-extra.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Build with --build-arg INSTALL_EXTRAS=true to add sentence-transformers + anthropic (large image)
ARG INSTALL_EXTRAS=false
RUN if [ "$INSTALL_EXTRAS" = "true" ]; then pip install --no-cache-dir -r requirements-extra.txt; fi

COPY app ./app
RUN mkdir /data
VOLUME /data
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
