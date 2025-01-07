# MarketSignals

## Directory Structure
```
├── notebooks/
│   ├── exploratory/    # Jupyter notebooks for initial exploration
│   └── finalized/      # Finalized notebooks ready for presentation
├── src/
│   ├── data/
│   │   ├── __init__.py            # Makes this folder a module
│   │   ├── base_data.py           # Defines the generic parent class (BaseData)
│   │   ├── order_book.py          # Defines the OrderBookData class
│   │   ├── ohlc_data.py           # Defines the OHLCData class
│   │   └── utils.py               # Utility functions specific to data (e.g., parsing, validation)
│   ├── preprocessing/  # Code for cleaning and transforming data
│   ├── models/         # Implementation of statistical/ML models
│   ├── utils/          # Utility functions used across the repository
│   └── visualizations/ # Scripts for data visualization
├── tests/
│   ├── unit/           # Unit tests for functions and classes
│   └── integration/    # Tests covering multiple components
├── docs/
│   └── figures/        # Images, plots, and other documentation assets
├── experiments/
│   ├── changepoint_detection/  # Individual experiments for research
│   └── meta_levels/            # Experiments for order book meta levels
└── scripts/
    ├── fetch_data.py   # Scripts for downloading data
    ├── run_model.py    # Script to train models
    └── visualize.py    # Generate visualizations
```

