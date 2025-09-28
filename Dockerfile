# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Copy the stock list file into the container
COPY /content/filtered_sp500_stocks.csv /app/filtered_sp500_stocks.csv

# Create directories for models if they don't exist
RUN mkdir -p trained_rf_models trained_lstm_models

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
# Use gunicorn or another production-ready server in a real deployment
# For this example, flask run is sufficient
CMD ["flask", "run", "--host", "0.0.0.0"]
