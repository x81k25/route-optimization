#!/usr/bin/env python3
"""
Test script to verify the Streamlit app works correctly
"""

# standard library imports
import json
import sys
from pathlib import Path

# 3rd-party imports
from loguru import logger
import pandas as pd
import polars as pl

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_data_loading() -> bool:
    """
    Test that required data files can be loaded.
    
    :return: True if data loading succeeds, False otherwise
    """
    logger.info("testing data loading...")
    
    try:
        # test itinerary loading
        itinerary_path = Path(__file__).parent.parent / "output" / "itinerary.jsonl"
        if not itinerary_path.exists():
            logger.warning(f"itinerary file not found: {itinerary_path}")
            return False
        
        # load a sample of data
        itinerary_data = []
        with open(itinerary_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 100:  # test first 100 lines
                    break
                line = line.strip()
                if line:
                    itinerary_data.append(json.loads(line))
        
        if not itinerary_data:
            logger.warning("no itinerary data found")
            return False
        
        itinerary_df = pl.DataFrame(itinerary_data)
        zones = itinerary_df['zone_id'].unique()
        logger.info(f"loaded {len(itinerary_data)} itinerary records")
        logger.info(f"found {len(zones)} zones: {sorted(zones)[:5]}{'...' if len(zones) > 5 else ''}")
        
        # test locations loading
        locations_path = Path(__file__).parent.parent / "data" / "locations.jsonl"
        locations = {}
        if locations_path.exists():
            with open(locations_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        location = json.loads(line)
                        locations[location['pos_id']] = location
            logger.info(f"loaded {len(locations)} location records")
        else:
            logger.warning("locations file not found (map will be limited)")
        
        return True
        
    except Exception as e:
        logger.error(f"data loading failed: {e}")
        return False

def test_imports() -> bool:
    """
    Test that all required packages can be imported.
    
    :return: True if all imports succeed, False otherwise
    """
    logger.info("testing package imports...")
    
    required_packages = [
        ('streamlit', 'st'),
        ('pandas', 'pd'),
        ('polars', 'pl'),
        ('plotly.graph_objects', 'go'),
        ('plotly.express', 'px'),
        ('yaml', None),
        ('json', None),
        ('pathlib', 'Path')
    ]
    
    failed_imports = []
    
    for package, alias in required_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            logger.info(f"imported {package}")
        except ImportError as e:
            logger.error(f"failed to import {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_config() -> bool:
    """
    Test configuration loading.
    
    :return: True if configuration loading succeeds, False otherwise
    """
    logger.info("testing configuration...")
    
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "model-params.yaml"
        
        if not config_path.exists():
            logger.warning("config file not found, using defaults")
            return True
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if 'model_params' in config:
            params = config['model_params']
            logger.info(f"config loaded with {len(params)} parameters")
            return True
        else:
            logger.warning("config file missing 'model_params' section")
            return True
            
    except Exception as e:
        logger.error(f"config loading failed: {e}")
        return False

def main() -> bool:
    """
    Run all tests.
    
    :return: True if all tests pass, False otherwise
    """
    logger.info("testing Streamlit route optimization dashboard")
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Configuration", test_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    logger.info("test results:")
    
    all_passed = True
    for test_name, result in results:
        status = "pass" if result else "fail"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    
    if all_passed:
        logger.success("all tests passed! the Streamlit app should work correctly.")
        logger.info("to run the app:")
        logger.info("  python streamlit/run.py")
        logger.info("or")
        logger.info("  uv run streamlit run streamlit/app.py --theme.base dark")
        return True
    else:
        logger.error("some tests failed. please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)