FROM python:3.11-slim-bookworm

# Set environment variables
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}


EXPOSE 8081
WORKDIR /app
COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

HEALTHCHECK CMD curl --fail http://localhost:8081/_stcore/health
ENTRYPOINT ["streamlit", "run", "LSTM_app.py", "--server.port=8081", "--server.address=0.0.0.0"]

