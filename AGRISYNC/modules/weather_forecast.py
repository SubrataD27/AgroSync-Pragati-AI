import requests
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

class WeatherForecastEngine:
    def __init__(self, api_key=None):
        """
        Initialize the weather forecast engine.
        
        Args:
            api_key (str, optional): API key for weather service
        """
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       'config', 'api_keys.json')
        self.api_key = api_key or self._load_api_key()
        
        # Base URLs for different weather APIs
        self.openweather_api_url = "https://api.openweathermap.org/data/2.5"
        self.weather_history_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                               'data', 'weather_history')
        os.makedirs(self.weather_history_path, exist_ok=True)
        
        # Load forecasting model if available
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'models', 'saved')
        self.model = self._load_model()
    
    def _load_api_key(self):
        """Load API key from configuration file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('weather_api_key', '')
            else:
                print("API configuration file not found.")
                return ''
        except Exception as e:
            print(f"Error loading API key: {e}")
            return ''
    
    def _load_model(self):
        """Load weather forecasting model if available."""
        model_path = os.path.join(self.model_dir, 'weather_forecast_model.pkl')
        try:
            if os.path.exists(model_path):
                import joblib
                return joblib.load(model_path)
            else:
                print("Weather forecasting model not found. Using API-based forecasts only.")
                return None
        except Exception as e:
            print(f"Error loading weather model: {e}")
            return None
    
    def get_current_weather(self, location):
        """
        Get current weather for a location.
        
        Args:
            location (dict): Location information with lat, lon or city, country
            
        Returns:
            dict: Current weather data
        """
        try:
            if not self.api_key:
                return self._get_mock_weather(location)
            
            # Construct query parameters
            params = {'appid': self.api_key, 'units': 'metric'}
            
            if 'lat' in location and 'lon' in location:
                params['lat'] = location['lat']
                params['lon'] = location['lon']
            elif 'city' in location:
                params['q'] = f"{location['city']}"
                if 'country' in location:
                    params['q'] += f",{location['country']}"
            else:
                return {"error": "Invalid location format"}
            
            # Make API request
            response = requests.get(f"{self.openweather_api_url}/weather", params=params)
            
            if response.status_code == 200:
                data = response.json()
                # Save to history for future model training
                self._save_weather_history(location, data)
                return self._format_weather_data(data)
            else:
                return {"error": f"API error: {response.status_code}", "message": response.text}
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_forecast(self, location, days=7):
        """
        Get weather forecast for a location.
        
        Args:
            location (dict): Location information with lat, lon or city, country
            days (int): Number of days to forecast
            
        Returns:
            dict: Weather forecast data
        """
        try:
            if not self.api_key:
                return self._get_mock_forecast(location, days)
            
            # Construct query parameters
            params = {'appid': self.api_key, 'units': 'metric'}
            
            if 'lat' in location and 'lon' in location:
                params['lat'] = location['lat']
                params['lon'] = location['lon']
            elif 'city' in location:
                params['q'] = f"{location['city']}"
                if 'country' in location:
                    params['q'] += f",{location['country']}"
            else:
                return {"error": "Invalid location format"}
            
            # Make API request
            response = requests.get(f"{self.openweather_api_url}/forecast", params=params)
            
            if response.status_code == 200:
                data = response.json()
                return self._format_forecast_data(data, days)
            else:
                return {"error": f"API error: {response.status_code}", "message": response.text}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _format_weather_data(self, data):
        """Format raw weather data into a more usable structure."""
        weather = {
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind']['deg'],
            'description': data['weather'][0]['description'],
            'icon': data['weather'][0]['icon'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add rainfall if available
        if 'rain' in data and '1h' in data['rain']:
            weather['rain_1h'] = data['rain']['1h']
        else:
            weather['rain_1h'] = 0
            
        return weather
    
    def _format_forecast_data(self, data, days):
        """Format raw forecast data into a more usable structure."""
        forecast = []
        
        # Group by day
        day_forecasts = {}
        for item in data['list']:
            dt = datetime.fromtimestamp(item['dt'])
            day = dt.strftime('%Y-%m-%d')
            
            if day not in day_forecasts:
                day_forecasts[day] = []
                
            day_forecasts[day].append({
                'time': dt.strftime('%H:%M'),
                'temperature': item['main']['temp'],
                'feels_like': item['main']['feels_like'],
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'wind_speed': item['wind']['speed'],
                'wind_direction': item['wind']['deg'],
                'description': item['weather'][0]['description'],
                'icon': item['weather'][0]['icon'],
                'rain_3h': item['rain']['3h'] if 'rain' in item and '3h' in item['rain'] else 0
            })
        
        # Limit to requested number of days
        days_list = list(day_forecasts.keys())[:days]
        
        # Calculate daily aggregates
        for day in days_list:
            day_data = day_forecasts[day]
            
            # Calculate averages and extract most common weather description
            temps = [entry['temperature'] for entry in day_data]
            humidity = [entry['humidity'] for entry in day_data]
            description_counts = {}
            for entry in day_data:
                desc = entry['description']
                description_counts[desc] = description_counts.get(desc, 0) + 1
            
            most_common_desc = max(description_counts.items(), key=lambda x: x[1])[0]
            most_common_icon = next(entry['icon'] for entry in day_data if entry['description'] == most_common_desc)
            
            daily_forecast = {
                'date': day,
                'avg_temp': sum(temps) / len(temps),
                'min_temp': min(temps),
                'max_temp': max(temps),
                'avg_humidity': sum(humidity) / len(humidity),
                'description': most_common_desc,
                'icon': most_common_icon,
                'hourly': day_data
            }
            
            forecast.append(daily_forecast)
        
        return forecast
    
    def _save_weather_history(self, location, data):
        """Save weather data to history for future model training."""
        try:
            location_id = f"{location.get('city', '')}{location.get('lat', '')}{location.get('lon', '')}"
            location_id = ''.join(c for c in location_id if c.isalnum())
            
            today = datetime.now().strftime('%Y-%m-%d')
            filename = os.path.join(self.weather_history_path, f"{location_id}_{today}.json")
            
            # Append to existing file or create new one
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    history = json.load(f)
            else:
                history = []
                
            # Add timestamp to data
            data['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            history.append(data)
            
            with open(filename, 'w') as f:
                json.dump(history, f)
                
        except Exception as e:
            print(f"Error saving weather history: {e}")
    
    def get_agricultural_weather_metrics(self, location):
        """
        Get weather metrics specifically relevant for agriculture.
        
        Args:
            location (dict): Location information
            
        Returns:
            dict: Agricultural weather metrics
        """
        current = self.get_current_weather(location)
        forecast = self.get_forecast(location)
        
        # Check for errors
        if 'error' in current or 'error' in forecast:
            return {"error": current.get('error') or forecast.get('error')}
        
        # Calculate growing degree days (base 10°C)
        gdd_base = 10
        avg_temps = [day['avg_temp'] for day in forecast]
        gdd = sum(max(0, temp - gdd_base) for temp in avg_temps)
        
        # Calculate precipitation forecast
        rain_forecast = []
        for day in forecast:
            daily_rain = sum(hour.get('rain_3h', 0) for hour in day['hourly'])
            rain_forecast.append({
                'date': day['date'],
                'amount': daily_rain
            })
        
        # Calculate potential evapotranspiration (simplified model)
        # Uses Hargreaves equation simplified
        et_forecast = []
        for day in forecast:
            # Simple estimation
            et = 0.0023 * (day['max_temp'] - day['min_temp'])**0.5 * (day['avg_temp'] + 17.8)
            et_forecast.append({
                'date': day['date'],
                'et': max(0, et)  # Ensure non-negative
            })
        
        return {
            'current': current,
            'growing_degree_days': gdd,
            'rainfall_forecast': rain_forecast,
            'evapotranspiration': et_forecast,
            'frost_risk': any(day['min_temp'] <= 0 for day in forecast),
            'heat_stress_risk': any(day['max_temp'] >= 35 for day in forecast)
        }
    
    def _get_mock_weather(self, location):
        """Provide mock weather data when API key is not available."""
        return {
            'temperature': 25.5,
            'feels_like': 26.0,
            'humidity': 65,
            'pressure': 1013,
            'wind_speed': 3.5,
            'wind_direction': 180,
            'description': 'partly cloudy',
            'icon': '03d',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rain_1h': 0,
            'note': 'This is mock data as no API key was provided'
        }
    
    def _get_mock_forecast(self, location, days=7):
        """Provide mock forecast data when API key is not available."""
        forecast = []
        base_date = datetime.now()
        
        for i in range(days):
            day_date = base_date + timedelta(days=i)
            forecast.append({
                'date': day_date.strftime('%Y-%m-%d'),
                'avg_temp': 25.0 + np.random.normal(0, 2),
                'min_temp': 20.0 + np.random.normal(0, 1),
                'max_temp': 30.0 + np.random.normal(0, 1),
                'avg_humidity': 65 + np.random.normal(0, 5),
                'description': np.random.choice(['clear sky', 'few clouds', 'scattered clouds', 'light rain']),
                'icon': np.random.choice(['01d', '02d', '03d', '10d']),
                'hourly': self._generate_mock_hourly(day_date),
                'note': 'This is mock data as no API key was provided'
            })
            
        return forecast
    
    def _generate_mock_hourly(self, date):
        """Generate mock hourly data for a given date."""
        hourly = []
        base_temp = 25.0 + np.random.normal(0, 2)
        
        for hour in range(0, 24, 3):  # Every 3 hours
            time = date.replace(hour=hour, minute=0)
            temp_variation = -2 * np.cos(hour * np.pi / 12)  # Cooler at night, warmer in day
            
            hourly.append({
                'time': time.strftime('%H:%M'),
                'temperature': base_temp + temp_variation,
                'feels_like': base_temp + temp_variation + np.random.normal(0, 0.5),
                'humidity': 65 + np.random.normal(0, 5),
                'pressure': 1013 + np.random.normal(0, 2),
                'wind_speed': 3.5 + np.random.normal(0, 1),
                'wind_direction': np.random.randint(0, 360),
                'description': np.random.choice(['clear sky', 'few clouds', 'scattered clouds', 'light rain']),
                'icon': np.random.choice(['01d', '02d', '03d', '10d']),
                'rain_3h': max(0, np.random.normal(0, 0.5))
            })
            
        return hourly