FROM python:3.8-slim

RUN pip install pipenv

#Set current directory as /app and cd into it
 WORKDIR /app

#Copy Pipfile and Pipfile.lock into ./
COPY ["Pipfile", "Pipfile.lock", "./"]

#We use this to install pipenv in the system, not in docker
RUN pipenv install --system --deploy

COPY ["flask_service.py", "proto.py", "./"]

#Expose port 9696
EXPOSE 9696

# We are actually runing this entrypoint=gunicorn --bind 0.0.0.0:9696 flask_service:app
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "flask_service:app"]
