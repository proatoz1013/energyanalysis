# Configuration settings for the Chiller Energy Dashboard

# File upload settings
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit

# Application settings
DEBUG = True
SECRET_KEY = 'your_secret_key_here'  # Change this to a random secret key for production

# Visualization settings
DEFAULT_CHART_TYPE = 'line'
CHART_COLORS = {
    'power_usage': '#1f77b4',
    'efficiency': '#ff7f0e',
    'cop': '#2ca02c'
}

# Performance metrics settings
METRICS = {
    'total_power_usage': 'Total Power Usage (kW)',
    'kw_tr': 'kW/TR',
    'cop': 'Coefficient of Performance (COP)',
    'average_efficiency': 'Average Efficiency (%)'
}