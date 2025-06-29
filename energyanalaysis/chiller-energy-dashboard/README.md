# Chiller Energy Dashboard

## Overview
The Chiller Energy Dashboard is a web application designed to analyze and visualize energy usage and performance metrics of chiller plants. Users can upload their chiller data in Excel or CSV format, preview the data, and perform various calculations to assess energy efficiency and performance.

## Features
- **Data Upload**: Users can upload Excel or CSV files containing chiller data.
- **Data Preview**: A preview of the uploaded data is displayed for user verification.
- **Column Mapping**: Automatically detects numeric columns and allows users to assign specific columns for time, chiller power, pumps, flowrate, cooling load, and efficiency.
- **Metrics Calculation**: Computes total power usage, kW/TR, COP, and average efficiency based on user-selected columns.
- **Visualizations**: Generates visualizations such as line charts for kW/TR, COP, and plant efficiency.

## Project Structure
```
chiller-energy-dashboard
├── src
│   ├── app.py
│   ├── components
│   │   ├── data_upload.py
│   │   ├── data_preview.py
│   │   ├── column_mapper.py
│   │   ├── metrics_calculator.py
│   │   └── visualizations.py
│   ├── utils
│   │   ├── data_processor.py
│   │   ├── energy_calculations.py
│   │   ├── efficiency_analyzer.py
│   │   └── file_handler.py
│   ├── models
│   │   ├── chiller_metrics.py
│   │   └── performance_indicators.py
│   └── config
│       └── settings.py
├── static
│   ├── css
│   │   └── styles.css
│   └── images
│       └── placeholder.txt
├── templates
│   ├── index.html
│   └── dashboard.html
├── data
│   ├── sample_data
│   │   └── sample_chiller_data.csv
│   └── uploads
│       └── placeholder.txt
├── tests
│   ├── test_data_processor.py
│   ├── test_calculations.py
│   └── test_file_handler.py
├── requirements.txt
└── .gitignore
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd chiller-energy-dashboard
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```
   python src/app.py
   ```
2. Open your web browser and go to `http://localhost:5000` to access the dashboard.
3. Upload your chiller data file and follow the prompts to analyze the data.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.