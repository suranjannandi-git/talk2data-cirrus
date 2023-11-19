FROM python:3.11.4

LABEL Author="Sudarshan Prasad"
LABEL version="0.0.1"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

# Copy application files
COPY . /app

RUN chmod -R 777 /app/src
RUN mkdir -p /.cache/huggingface/hub
RUN chmod -R 777 /.cache
RUN chmod -R 777 /.cache/huggingface/hub
RUN mkdir -p /tmp/sql_result
RUN chmod -R 777 /tmp
RUN chmod -R 777 /tmp/sql_result

EXPOSE 8080

CMD ["uvicorn", "src.opera_api:app", "--host", "0.0.0.0", "--port", "8080"]
