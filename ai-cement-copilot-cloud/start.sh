#!/bin/bash
# start.sh - Simplified for Cloud Run IAM Authentication

echo "Starting Streamlit app. Using built-in Cloud Run IAM credentials for BigQuery/Vertex AI."

# Run Streamlit app
# Use 'exec' to replace the current shell process with Streamlit, which is a best practice for containers.
# We ensure it binds to all interfaces (0.0.0.0) on the required port (8080).
exec streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true