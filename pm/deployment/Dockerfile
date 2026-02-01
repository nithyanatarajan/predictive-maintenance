# Use a minimal base image with Python 3.12 installed
FROM python:3.12-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip3 install -r requirements.txt

# Expose port (HuggingFace Spaces expects 7860)
EXPOSE 7860

# Create a non-root user with a home dir
RUN useradd -m app && chown -R app:app /app

# Switch to that user
USER app

# Point Streamlit config to the new user's home (writable)
ENV HOME=/home/app
ENV STREAMLIT_CONFIG_DIR=/home/app/.streamlit

# Define the command to run the Streamlit app on port 7860 and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
