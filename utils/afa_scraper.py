"""
TNB AFA Rate Scraper Module

This module scrapes the latest AFA (Automatic Fuel Adjustment) rates from TNB's official website
and provides them to the tariff comparison tool.
"""

import requests
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AFAScraper:
    """Scraper for TNB AFA rates from official website."""
    
    def __init__(self):
        self.base_url = "https://www.mytnb.com.my"
        self.afa_url = "https://www.mytnb.com.my/tariff/index.html?v=1.1.43#afa"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def get_latest_afa_rates(self) -> Dict[str, float]:
        """
        Scrape the latest AFA rates from TNB website.
        
        Returns:
            Dict containing the latest AFA rates with period as key and rate as value
        """
        try:
            # First, try to get the main page to extract any API endpoints or data
            response = self.session.get(self.afa_url, timeout=10)
            response.raise_for_status()
            
            html_content = response.text
            
            # Look for JSON data or API endpoints in the HTML
            afa_data = self._extract_afa_from_html(html_content)
            
            if afa_data:
                return afa_data
            
            # If direct HTML parsing fails, try to find API endpoints
            api_data = self._find_and_call_api(html_content)
            
            if api_data:
                return api_data
            
            # Fallback: return mock data with current known rates
            logger.warning("Could not scrape live data, using fallback rates")
            return self._get_fallback_rates()
            
        except requests.RequestException as e:
            logger.error(f"Network error while scraping AFA rates: {e}")
            return self._get_fallback_rates()
        except Exception as e:
            logger.error(f"Unexpected error while scraping AFA rates: {e}")
            return self._get_fallback_rates()
    
    def _extract_afa_from_html(self, html_content: str) -> Optional[Dict[str, float]]:
        """Extract AFA rates directly from HTML content."""
        try:
            # Look for patterns like "-1.45 sen / kWh" or similar
            afa_pattern = r'(-?\d+\.?\d*)\s*sen\s*/\s*kWh'
            date_pattern = r'(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(\w+)\s*(\d{4})'
            
            afa_matches = re.findall(afa_pattern, html_content, re.IGNORECASE)
            date_matches = re.findall(date_pattern, html_content)
            
            if afa_matches and date_matches:
                afa_data = {}
                
                # Try to pair dates with AFA rates
                for i, (rate) in enumerate(afa_matches):
                    if i < len(date_matches):
                        start_day, end_day, month, year = date_matches[i]
                        period_key = f"{start_day}-{end_day} {month} {year}"
                        afa_data[period_key] = float(rate)
                
                return afa_data if afa_data else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting AFA from HTML: {e}")
            return None
    
    def _find_and_call_api(self, html_content: str) -> Optional[Dict[str, float]]:
        """Find and call any API endpoints that might contain AFA data."""
        try:
            # Look for common API endpoint patterns
            api_patterns = [
                r'["\']([^"\']*api[^"\']*afa[^"\']*)["\']',
                r'["\']([^"\']*afa[^"\']*api[^"\']*)["\']',
                r'["\']([^"\']*tariff[^"\']*data[^"\']*)["\']',
                r'fetch\(["\']([^"\']*)["\']',
                r'ajax\(["\']([^"\']*)["\']',
            ]
            
            api_urls = []
            for pattern in api_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                api_urls.extend(matches)
            
            # Try each potential API URL
            for api_url in api_urls:
                try:
                    # Make URL absolute if relative
                    if api_url.startswith('/'):
                        api_url = self.base_url + api_url
                    elif not api_url.startswith('http'):
                        continue
                    
                    api_response = self.session.get(api_url, timeout=5)
                    if api_response.status_code == 200:
                        try:
                            json_data = api_response.json()
                            afa_data = self._parse_api_response(json_data)
                            if afa_data:
                                return afa_data
                        except json.JSONDecodeError:
                            continue
                            
                except requests.RequestException:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding API endpoints: {e}")
            return None
    
    def _parse_api_response(self, json_data: Dict) -> Optional[Dict[str, float]]:
        """Parse API response to extract AFA rates."""
        try:
            # This would need to be customized based on the actual API response structure
            # For now, we'll look for common patterns
            
            if isinstance(json_data, dict):
                # Look for AFA-related keys
                for key in ['afa', 'rates', 'adjustments', 'tariff']:
                    if key in json_data:
                        data = json_data[key]
                        if isinstance(data, dict):
                            return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
                        elif isinstance(data, list):
                            # Handle list of AFA entries
                            afa_dict = {}
                            for item in data:
                                if isinstance(item, dict) and 'period' in item and 'rate' in item:
                                    afa_dict[item['period']] = float(item['rate'])
                            return afa_dict if afa_dict else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
            return None
    
    def _get_fallback_rates(self) -> Dict[str, float]:
        """
        Return fallback AFA rates based on the screenshot provided.
        These should be updated periodically or when scraping fails.
        """
        # Based on the image you provided, here are the current rates
        current_date = datetime.now()
        
        fallback_rates = {
            "1-31 August 2025": -1.45,
            "1-30 September 2025": -1.10,
            "1-31 October 2025": -2.31,  # Forecast
            "1-30 November 2025": -1.06,  # Forecast
            "1-31 December 2025": -2.88,  # Forecast
        }
        
        return fallback_rates
    
    def get_current_afa_rate(self) -> float:
        """
        Get the AFA rate applicable for the current month.
        
        Returns:
            Current AFA rate in sen/kWh (can be negative)
        """
        afa_rates = self.get_latest_afa_rates()
        current_date = datetime.now()
        current_month = current_date.strftime("%B")
        current_year = str(current_date.year)
        
        # Look for current month rate
        for period, rate in afa_rates.items():
            if current_month in period and current_year in period:
                return rate
        
        # If current month not found, return the most recent rate
        if afa_rates:
            return list(afa_rates.values())[-1]
        
        # Ultimate fallback
        return -1.10  # Based on September 2025 rate from your screenshot
    
    def get_afa_rates_summary(self) -> List[Tuple[str, float, bool]]:
        """
        Get a summary of AFA rates with forecast indicators.
        
        Returns:
            List of tuples: (period, rate, is_forecast)
        """
        afa_rates = self.get_latest_afa_rates()
        current_date = datetime.now()
        
        summary = []
        for period, rate in afa_rates.items():
            # Determine if this is a forecast (future date)
            is_forecast = self._is_forecast_period(period, current_date)
            summary.append((period, rate, is_forecast))
        
        return summary
    
    def _is_forecast_period(self, period: str, current_date: datetime) -> bool:
        """Determine if a period is a forecast (future) period."""
        try:
            # Extract month and year from period string
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            
            for month in months:
                if month in period:
                    # Extract year
                    year_match = re.search(r'(\d{4})', period)
                    if year_match:
                        year = int(year_match.group(1))
                        month_num = months.index(month) + 1
                        
                        period_date = datetime(year, month_num, 1)
                        return period_date > current_date
            
            return False
            
        except Exception:
            return False


# Convenience function for easy import
def get_current_afa_rate() -> float:
    """Get the current AFA rate quickly."""
    scraper = AFAScraper()
    return scraper.get_current_afa_rate()


def get_all_afa_rates() -> Dict[str, float]:
    """Get all available AFA rates."""
    scraper = AFAScraper()
    return scraper.get_latest_afa_rates()


if __name__ == "__main__":
    # Test the scraper
    scraper = AFAScraper()
    
    print("Testing TNB AFA Rate Scraper...")
    print("="*50)
    
    try:
        # Get current rate
        current_rate = scraper.get_current_afa_rate()
        print(f"Current AFA Rate: {current_rate:.2f} sen/kWh")
        
        # Get all rates
        all_rates = scraper.get_latest_afa_rates()
        print(f"\nAll AFA Rates:")
        for period, rate in all_rates.items():
            print(f"  {period}: {rate:.2f} sen/kWh")
        
        # Get summary with forecast indicators
        summary = scraper.get_afa_rates_summary()
        print(f"\nAFA Rates Summary:")
        for period, rate, is_forecast in summary:
            forecast_indicator = " (Forecast)" if is_forecast else " (Current/Historical)"
            print(f"  {period}: {rate:.2f} sen/kWh{forecast_indicator}")
            
    except Exception as e:
        print(f"Error testing scraper: {e}")
