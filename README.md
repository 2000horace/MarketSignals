# MarketSignals

## Directory Structure
```
├── notebooks/
│   ├── wip/            # Jupyter notebooks for initial exploration (work in progress notebooks)
│   └── finalized/      # Finalized notebooks ready for presentation
├── src/
│   ├── data/
│   │   ├── __init__.py            
│   │   ├── accessors                       # Folder to contain all external data layers               
│   │   │   ├── __init__.py        
│   │   │   ├── generic_accessor.py         # Abstract class for all external data classes
│   │   │   └── omi_arcticdb.py             # Implementation of GenericAccessor for OMI ArcticDB   
│   │   ├── internal                        # Folder to contain all internal data layers
│   │   │   ├── __init__.py
│   │   │   ├── base_data.py                # Abstract class for all internal data classes
│   │   │   ├── order_book.py               # Implementation of BaseData for limit order books
│   │   │   └── bar_data.py                 # Implementation of BaseData for OHLC bar data
│   │   └── utils.py                        # Utility functions specific to data (e.g., parsing, validation)
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

