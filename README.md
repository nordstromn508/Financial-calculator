# Financial Calculator

A Python-based financial calculator application that provides various financial computations and analysis tools.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [License](#license)

## Features
- Comprehensive financial calculations
- Configurable settings via settings.conf
- Easy-to-use command-line interface

## Installation
1. Clone or download this repository
2. Ensure you have Python installed on your system
3. Install any required dependencies (check main.py for imports)

## Usage
Run the application using:
```bash
python main.py
```

## Configuration
The application uses `settings.conf` for configuration. Modify this file to adjust calculator settings according to your needs.

## File Structure
- `main.py` - Main application code
- `settings.conf` - Configuration file
- `preprocess_csv.py` - Script to preprocess bank statement CSV files
- `add_merchant_column.py` - Script to add merchant identification to processed CSV files
- `README.md` - Project documentation

## CSV Preprocessing Script

This repository includes a preprocessing script to clean bank statement CSV files by removing sensitive columns.

### Usage:
```bash
python preprocess_csv.py input_file.csv [-o output_file.csv]
```

The script removes the following columns from bank statement CSVs:
- Bank RTN
- Account Number
- Check Number
- Account Running Balance

Output file will contain only: Date, Transaction Type, Description, Debit, and Credit columns.

## Merchant Identification

Additional functionality to identify merchants from transaction descriptions:

### Usage:
```bash
python add_merchant_column.py input_file.csv [output_file.csv]
```

- Analyzes the Description field to categorize transactions by merchant type
- Adds a new 'Merchant' column with categories like Grocery, Restaurant, Salary, etc.
- Output file will contain all original columns plus the new Merchant column

## License
This project is open source and available under the MIT License.