#!/usr/bin/env python3
"""
Simple runner script for the Streamlit Route Optimization Dashboard
"""

# standard library imports
import subprocess
import sys
from pathlib import Path

# 3rd-party imports
from loguru import logger

def main() -> None:
    """
    Run the Streamlit app.
    
    :return: None
    """
    app_path = Path(__file__).parent / "app.py"
    
    # run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
        "--theme.base", "dark"
    ]
    
    logger.info(f"starting Streamlit app at http://localhost:8501")
    logger.info(f"command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("shutting down Streamlit app...")
    except subprocess.CalledProcessError as e:
        logger.error(f"error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()