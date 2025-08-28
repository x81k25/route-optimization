# Route Optimization

A Python project for optimizing delivery routes and solving vehicle routing problems (VRP).

## Installation

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv add -r requirements.txt
```

## Usage

```bash
# Run the main optimization algorithm
uv run main.py

# Run with specific parameters
uv run main.py --locations subway_locations.json --config config/model-params.yaml
```

## Configuration

Edit `config/model-params.yaml` to adjust optimization parameters:

- `days-per-week`: Working days per week
- `utilization`: Fleet utilization percentage
- `primary-hours-per-week`: Hours for primary locations
- `hours-per-non-primary`: Hours for secondary locations
- `locations-per-day-max`: Maximum locations per day
- `drive-inefficiency`: Routing inefficiency factor

## Development

```bash
# Install development dependencies
uv add --dev pytest black flake8 mypy

# Run tests
pytest

# Format code
black .

# Run linting
flake8

# Type checking
mypy .
```

## Project Structure

```
route-optimization/
├── config/
│   └── model-params.yaml    # Configuration parameters
├── data/
│   └── subway_locations.json # Location data
├── src/
│   ├── algorithms/          # Optimization algorithms
│   ├── models/             # Data models
│   └── utils/              # Utility functions
├── tests/                  # Test files
└── main.py                # Entry point
```