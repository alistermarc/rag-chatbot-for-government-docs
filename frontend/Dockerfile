FROM python@sha256:9c85d1d49df54abca1c5db3b4016400e198e9e9bb699f32f1ef8e5c0c2149ccf

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip==25.1.1
RUN pip install --no-cache -r requirements.txt

COPY . .

CMD ["streamlit", "run", "streamlit.py", "--server.port=8501"]