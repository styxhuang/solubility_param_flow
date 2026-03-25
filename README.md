# Solubility Parameter Calculation Workflow

A comprehensive workflow for calculating Hansen Solubility Parameters (HSP) and predicting solubility using molecular dynamics and machine learning.

## Features

- **HSP Calculation**: Calculate δD, δP, δH using MD simulations
- **Molecular Descriptors**: Compute chemical descriptors from SMILES/structures
- **Solubility Prediction**: ML models for solubility prediction
- **Visualization**: Plot HSP spheres and solubility maps

## Installation

```bash
pip install -e .
```

## Usage

```python
from solubility_param_flow import HSPCalculator

calc = HSPCalculator()
hsp = calc.calculate_from_smiles("CCO")  # Ethanol
print(f"δD={hsp.delta_d:.2f}, δP={hsp.delta_p:.2f}, δH={hsp.delta_h:.2f}")
```

## Project Structure

```
solubility_param_flow/
├── src/                    # Source code
│   ├── core/              # Core calculation modules
│   ├── descriptors/       # Molecular descriptor calculation
│   ├── models/            # ML models
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── examples/              # Example scripts
└── data/                  # Sample data
```

## License

MIT
