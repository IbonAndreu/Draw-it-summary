FROM python:3.8.12-buster
WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY canvas.py ./canvas.py
COPY .env .env
COPY logo.png ./logo.png
CMD streamlit run canvas.py --server.port=${PORT} --browser.serverAddress="0.0.0.0"
