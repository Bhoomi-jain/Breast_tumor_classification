FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set the command to run the FastAPI app using uvicorn
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]



