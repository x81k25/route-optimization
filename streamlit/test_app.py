#!/usr/bin/env python3
"""
Test script to verify the Streamlit app works correctly
"""

import sys
from pathlib import Path
import json
import pandas as pd
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_data_loading():
    """Test that required data files can be loaded."""
    print("Testing data loading...")
    
    try:
        # Test itinerary loading
        itinerary_path = Path(__file__).parent.parent / "output" / "itinerary.jsonl"
        if not itinerary_path.exists():
            print(f"✗ Itinerary file not found: {itinerary_path}")
            return False
        
        # Load a sample of data
        itinerary_data = []
        with open(itinerary_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 100:  # Test first 100 lines
                    break
                line = line.strip()
                if line:
                    itinerary_data.append(json.loads(line))
        
        if not itinerary_data:
            print("✗ No itinerary data found")
            return False
        
        itinerary_df = pl.DataFrame(itinerary_data)
        zones = itinerary_df['zone_id'].unique()
        print(f"✓ Loaded {len(itinerary_data)} itinerary records")
        print(f"  Found {len(zones)} zones: {sorted(zones)[:5]}{'...' if len(zones) > 5 else ''}")
        
        # Test locations loading
        locations_path = Path(__file__).parent.parent / "data" / "locations.jsonl"
        locations = {}
        if locations_path.exists():
            with open(locations_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        location = json.loads(line)
                        locations[location['pos_id']] = location
            print(f"✓ Loaded {len(locations)} location records")
        else:
            print("⚠ Locations file not found (map will be limited)")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
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
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "model-params.yaml"
        
        if not config_path.exists():
            print("⚠ Config file not found, using defaults")
            return True
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if 'model_params' in config:
            params = config['model_params']
            print(f"✓ Config loaded with {len(params)} parameters")
            return True
        else:
            print("⚠ Config file missing 'model_params' section")
            return True
            
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Streamlit Route Optimization Dashboard")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Configuration", test_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
        print()
    
    print("=" * 50)
    print("Test Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! The Streamlit app should work correctly.")
        print("\nTo run the app:")
        print("  python streamlit/run.py")
        print("or")
        print("  uv run streamlit run streamlit/app.py --theme.base dark")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)