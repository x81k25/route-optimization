#!/bin/bash

# Launch script for the Streamlit Route Optimization Dashboard
# This script runs the app with dark theme and proper configuration

echo "🚀 Starting Route Optimization Dashboard..."
echo "The app will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Change to the project root directory
cd "$(dirname "$0")/.."

# Run the Streamlit app with dark theme
uv run streamlit run streamlit/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --theme.base dark \
    --theme.primaryColor "#4a9eff" \
    --theme.backgroundColor "#1a1a1a" \
    --theme.secondaryBackgroundColor "#2a2a2a" \
    --theme.textColor "#e0e0e0"