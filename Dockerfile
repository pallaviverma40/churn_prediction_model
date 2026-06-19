
# DOCKER CODE INSTRUCTIONS

# This sets up the container with Python 3.10 installed.
FROM python:3.9-slim

# This ensures that the latest pip modules are installed
#RUN /usr/local/bin/python -m pip install --upgrade pip

# This sets the /app directory as the working directory for any RUN, CMD, ENTRYPOINT, or COPY instructions that follow.
WORKDIR /app

# This instruction copies the requirements file from the host machine to the container image
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# This instruction tells Docker that the container is listening on the specified network ports at runtime
EXPOSE 8501

# This copies everything in your current directory to the /app directory in the container.
COPY . .

# This creates an ENTRYPOINT to make the image executable
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]