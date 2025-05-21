FROM python@sha256:9c85d1d49df54abca1c5db3b4016400e198e9e9bb699f32f1ef8e5c0c2149ccf

WORKDIR /app

COPY . .

RUN pip install --upgrade pip==25.1.1
RUN pip install --no-cache -r requirements.txt
RUN pip install https://download.pytorch.org/whl/cpu/torch-2.1.1-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

CMD ["python3.11", "manage.py", "runserver", "0.0.0.0:8000"]