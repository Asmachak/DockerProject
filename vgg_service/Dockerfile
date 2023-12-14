FROM python:3.9-slim-buster

WORKDIR ./

ENV STATIC_URL /static
ENV STATIC_PATH /app/static

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY /app/saved_model.pb /app/saved_model.pb

COPY . .

EXPOSE 7000

CMD ["python", "/app/app.py"]
