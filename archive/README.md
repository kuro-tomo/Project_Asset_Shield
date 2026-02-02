# TIR Quantitative Engine (Jörmungandr)

## Overview
TIR Quantitative Engine is an enterprise-grade algorithmic trading system designed for the Japanese equity market (J-Quants). It features a modular architecture with adaptive regime detection, financial fundamental analysis ("Financial Trinity"), and execution capabilities.

## Architecture

The system is organized into the following core components:

*   **shield**: Main package containing business logic.
    *   **adaptive_core**: Market regime detection and parameter adaptation.
    *   **brain**: AI/ML models for price prediction.
    *   **execution_core**: Order management and execution algorithms (VWAP, etc.).
    *   **jquants_client**: Interface to J-Quants API.
    *   **screener**: Fundamental analysis engine.

## Directory Structure

```
TIR_Quantitative_Engine/
├── src/
│   └── shield/       # Core source code
├── config/             # Configuration files
├── docs/               # Technical documentation and due diligence reports
├── infrastructure/     # Docker and Terraform definitions
├── scripts/            # Utility and maintenance scripts
├── tests/              # Unit and integration tests
├── data/               # Local data storage (ignored in version control)
├── logs/               # Application logs
└── manage.py           # CLI Entry Point
```

## Installation

1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## Usage

### Run Production Pipeline
To run the main analysis pipeline for a specific ticker:

```bash
python manage.py run-pipeline --ticker 7203 --dry-run
```

### Run Node Manager
To start the node in distributed mode (Master/Agent):

```bash
python manage.py node
```

### Run Tests
```bash
python manage.py test
```

## Documentation
See `docs/` for detailed strategic analysis and setup guides.

## License
Proprietary & Confidential. All rights reserved.
