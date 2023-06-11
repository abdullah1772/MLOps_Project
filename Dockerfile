# Use an official TensorFlow Docker image as the base image
FROM tensorflow/tensorflow:latest-gpu-py3

# Set the working directory in the container to /app
WORKDIR /app

# First copy only requirements.txt and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the rest of the application
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD streamlit run app.py
