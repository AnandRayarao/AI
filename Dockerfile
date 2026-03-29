FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p docs

EXPOSE 5000

CMD ["flask", "--app", "app_flask.py", "run", "--host=0.0.0.0", "--port=5000"]