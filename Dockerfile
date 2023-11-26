FROM continuumio/miniconda3

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

RUN pip install poetry
RUN mkdir app
WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install

COPY . /app

ENTRYPOINT ["poetry", "run", "uvicorn", "--host", "0.0.0.0", "app:app"]
