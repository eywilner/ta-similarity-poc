FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
#FROM python:3.9-slim


# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Installing git which is required for flash-attn
RUN apt-get update -y && apt-get install git -y
# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install flash-attn package
RUN pip install flash-attn --no-build-isolation

# Make port 8000 available to the world outside this container
EXPOSE 4000

# Run the app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]
