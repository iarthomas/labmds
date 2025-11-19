FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./app.py
ENV BACKEND_URL=http://backend:8000 \
GRADIO_SERVER_NAME=0.0.0.0 \
GRADIO_PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]