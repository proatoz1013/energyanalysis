"""
Malaysia Weather Panel for MD Shaving Solution V2
=================================================

Streamlit component that renders a Malaysia weather selector and 48h hourly forecast
using Open-Meteo API (no API key required) with timezone "Asia/Kuala_Lumpur".

Author: Energy Analysis Team
Dependencies: streamlit, requests, pandas
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
import json

# Malaysian states/FTs mapped to their capitals
MALAYSIA_STATES = {
    "Johor": {"capital": "Johor Bahru", "lat": 1.4927, "lon": 103.7414},
    "Kedah": {"capital": "Alor Setar", "lat": 6.1248, "lon": 100.3678},
    "Kelantan": {"capital": "Kota Bharu", "lat": 6.1254, "lon": 102.2386},
    "Kuala Lumpur": {"capital": "Kuala Lumpur", "lat": 3.1390, "lon": 101.6869},
    "Labuan": {"capital": "Victoria", "lat": 5.2831, "lon": 115.2308},
    "Malacca": {"capital": "Malacca City", "lat": 2.2089, "lon": 102.2378},
    "Negeri Sembilan": {"capital": "Seremban", "lat": 2.7297, "lon": 101.9381},
    "Pahang": {"capital": "Kuantan", "lat": 3.8077, "lon": 103.3260},
    "Penang": {"capital": "George Town", "lat": 5.4141, "lon": 100.3288},
    "Perak": {"capital": "Ipoh", "lat": 4.5975, "lon": 101.0901},
    "Perlis": {"capital": "Kangar", "lat": 6.4414, "lon": 100.1986},
    "Putrajaya": {"capital": "Putrajaya", "lat": 2.9264, "lon": 101.6964},
    "Sabah": {"capital": "Kota Kinabalu", "lat": 5.9749, "lon": 116.0724},
    "Sarawak": {"capital": "Kuching", "lat": 1.5533, "lon": 110.3592},
    "Selangor": {"capital": "Shah Alam", "lat": 3.0733, "lon": 101.5185},
    "Terengganu": {"capital": "Kuala Terengganu", "lat": 5.3302, "lon": 103.1408}
}


@st.cache_data(ttl=3600)  # Cache geocoding for 1 hour
def geocode_location(query: str) -> Optional[Dict[str, Any]]:
    """
    Geocode a location using Open-Meteo geocoding API.
    
    Args:
        query: Location query (place name or "lat,lon")
        
    Returns:
        Dict with name, lat, lon or None if not found
    """
    try:
        # Check if query is coordinates (lat,lon)
        if "," in query:
            parts = [p.strip() for p in query.split(",")]
            if len(parts) == 2:
                try:
                    lat, lon = float(parts[0]), float(parts[1])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return {
                            "name": f"Custom Location ({lat:.3f}, {lon:.3f})",
                            "latitude": lat,
                            "longitude": lon
                        }
                except ValueError:
                    pass
        
        # Use Open-Meteo geocoding API
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": query,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data.get("results") and len(data["results"]) > 0:
            result = data["results"][0]
            return {
                "name": result.get("name", "Unknown"),
                "latitude": result.get("latitude"),
                "longitude": result.get("longitude")
            }
            
    except Exception as e:
        st.warning(f"Geocoding error: {str(e)}")
    
    return None


@st.cache_data(ttl=900)  # Cache forecast for 15 minutes
def get_weather_forecast(lat: float, lon: float) -> Optional[pd.DataFrame]:
    """
    Get 48-hour hourly weather forecast using Open-Meteo API.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        DataFrame with 48 hours of weather data in Asia/Kuala_Lumpur timezone
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m", 
                "precipitation",
                "rain",
                "cloud_cover",
                "wind_speed_10m",
                "wind_gusts_10m"
            ],
            "timezone": "Asia/Kuala_Lumpur",
            "forecast_days": 2
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        hourly = data.get("hourly", {})
        
        # Create DataFrame
        df = pd.DataFrame({
            "time": pd.to_datetime(hourly.get("time", [])),
            "temperature_2m": hourly.get("temperature_2m", []),
            "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
            "precipitation": hourly.get("precipitation", []),
            "rain": hourly.get("rain", []),
            "cloud_cover": hourly.get("cloud_cover", []),
            "wind_speed_10m": hourly.get("wind_speed_10m", []),
            "wind_gusts_10m": hourly.get("wind_gusts_10m", [])
        })
        
        # Limit to 48 hours
        df = df.head(48)
        
        return df
        
    except Exception as e:
        st.error(f"Weather forecast error: {str(e)}")
        return None


def render_weather_charts(df: pd.DataFrame) -> None:
    """Render weather forecast charts."""
    if df.empty:
        return
    
    # Temperature chart
    st.markdown("##### 🌡️ Temperature (°C)")
    chart_data = df.set_index('time')[['temperature_2m']]
    st.line_chart(chart_data, height=300)
    
    # Precipitation & Rain chart
    st.markdown("##### 🌧️ Precipitation & Rain (mm)")
    chart_data = df.set_index('time')[['precipitation', 'rain']]
    st.line_chart(chart_data, height=300)
    
    # Wind & Gusts chart
    st.markdown("##### 💨 Wind Speed & Gusts (km/h)")
    chart_data = df.set_index('time')[['wind_speed_10m', 'wind_gusts_10m']]
    st.line_chart(chart_data, height=300)
    
    # Cloud cover chart
    st.markdown("##### ☁️ Cloud Cover (%)")
    chart_data = df.set_index('time')[['cloud_cover']]
    st.area_chart(chart_data, height=300)


def render_weather_panel(default_state: str = "Penang") -> Dict[str, Any]:
    """
    Renders the Malaysia weather panel and returns location and forecast data.
    
    Args:
        default_state: Default Malaysian state to select
        
    Returns:
        {
            "label": str,   # display name for selected location
            "lat": float,
            "lon": float, 
            "df48": pandas.DataFrame  # 48h hourly forecast in MYT
        }
    """
    st.markdown("#### ⛅ Malaysia Weather (48h Hourly Forecast)")
    
    # Location selection method
    selection_method = st.radio(
        "📍 **Location Selection:**",
        ["By state", "Type a place"],
        horizontal=True,
        key="weather_selection_method"
    )
    
    location_info = None
    
    if selection_method == "By state":
        # State selector
        selected_state = st.selectbox(
            "🏛️ **Select Malaysian State/Federal Territory:**",
            options=list(MALAYSIA_STATES.keys()),
            index=list(MALAYSIA_STATES.keys()).index(default_state) if default_state in MALAYSIA_STATES else 0,
            key="weather_state_selector"
        )
        
        state_data = MALAYSIA_STATES[selected_state]
        location_info = {
            "name": f"{state_data['capital']}, {selected_state}",
            "latitude": state_data["lat"],
            "longitude": state_data["lon"]
        }
        
    else:  # Type a place
        custom_location = st.text_input(
            "🌍 **Enter location (place name or lat,lon):**",
            placeholder="e.g., Bayan Lepas, Penang or 5.336,100.306",
            key="weather_custom_location"
        )
        
        if custom_location.strip():
            location_info = geocode_location(custom_location.strip())
            if not location_info:
                st.warning(f"❌ Could not find location: '{custom_location}'. Please try a different search term.")
        else:
            st.info("💡 Enter a location above to get weather forecast.")
    
    # Initialize return data
    result = {
        "label": "",
        "lat": 0.0,
        "lon": 0.0,
        "df48": pd.DataFrame()
    }
    
    if location_info:
        lat, lon = location_info["latitude"], location_info["longitude"]
        location_label = location_info["name"]
        
        # Update result
        result.update({
            "label": location_label,
            "lat": lat,
            "lon": lon
        })
        
        # Display resolved location
        st.success(f"📍 **Location**: {location_label} ({lat:.3f}, {lon:.3f})")
        
        # Get weather forecast
        with st.spinner("🌤️ Fetching weather forecast..."):
            df_weather = get_weather_forecast(lat, lon)
        
        if df_weather is not None and not df_weather.empty:
            result["df48"] = df_weather
            
            # Current weather metrics
            current = df_weather.iloc[0]
            st.markdown("##### 📊 Current Conditions")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🌡️ Temperature",
                    f"{current['temperature_2m']:.1f}°C",
                    help="Current temperature"
                )
            
            with col2:
                st.metric(
                    "💧 Humidity", 
                    f"{current['relative_humidity_2m']:.0f}%",
                    help="Relative humidity"
                )
            
            with col3:
                # 6-hour precipitation sum
                precip_6h = df_weather.head(6)['precipitation'].sum()
                st.metric(
                    "🌧️ 6h Precip",
                    f"{precip_6h:.1f}mm",
                    help="Precipitation sum over next 6 hours"
                )
            
            with col4:
                # Max 24h wind gust
                max_gust_24h = df_weather.head(24)['wind_gusts_10m'].max()
                st.metric(
                    "💨 Max 24h Gust",
                    f"{max_gust_24h:.1f}km/h", 
                    help="Maximum wind gust in next 24 hours"
                )
            
            # Weather charts
            st.markdown("##### 📈 48-Hour Forecast")
            render_weather_charts(df_weather)
            
            # Data table
            st.markdown("##### 📋 Hourly Data")
            
            # Format time for display
            df_display = df_weather.copy()
            df_display['time'] = df_display['time'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Format numeric columns
            numeric_cols = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 
                          'rain', 'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m']
            for col in numeric_cols:
                if col in df_display.columns:
                    df_display[col] = df_display[col].round(1)
            
            st.dataframe(df_display, use_container_width=True, height=400)
            
            # CSV download
            csv_data = df_weather.to_csv(index=False)
            st.download_button(
                label="📥 Download 48h Forecast CSV",
                data=csv_data,
                file_name=f"weather_forecast_{location_label.replace(' ', '_').replace(',', '')}.csv",
                mime="text/csv",
                help="Download the 48-hour hourly forecast data"
            )
            
        else:
            st.error("❌ Failed to retrieve weather forecast. Please try again later.")
    
    return result


def convert_weather_to_features(df48: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 48h weather forecast into feature columns suitable for forecasting.
    
    Args:
        df48: 48-hour weather forecast DataFrame
        
    Returns:
        DataFrame with weather features (TODO: implement full feature engineering)
    """
    # TODO: Implement proper feature engineering for energy forecasting
    # For now, just pass through the original data
    
    if df48.empty:
        return pd.DataFrame()
    
    # Simple feature engineering placeholder
    features = df48.copy()
    
    # Add some basic derived features
    features['temp_rolling_avg_6h'] = features['temperature_2m'].rolling(window=6, min_periods=1).mean()
    features['precip_cumsum_6h'] = features['precipitation'].rolling(window=6, min_periods=1).sum()
    features['wind_max_6h'] = features['wind_speed_10m'].rolling(window=6, min_periods=1).max()
    
    return features


if __name__ == "__main__":
    # Test the weather panel
    st.set_page_config(page_title="Weather Panel Test", layout="wide")
    
    st.title("🌤️ Weather Panel Test")
    
    result = render_weather_panel(default_state="Penang")
    
    if not result["df48"].empty:
        st.success(f"✅ Successfully loaded weather data for {result['label']}")
        st.write(f"📍 Coordinates: {result['lat']:.3f}, {result['lon']:.3f}")
        st.write(f"📊 Data points: {len(result['df48'])} hours")
