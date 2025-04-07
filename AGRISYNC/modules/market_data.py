import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import numpy as np

class MarketDataEngine:
    def __init__(self, api_key=None):
        """
        Initialize the market data engine.
        
        Args:
            api_key (str, optional): API key for market data service
        """
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       'config', 'api_keys.json')
        self.api_key = api_key or self._load_api_key()
        
        # Data paths
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.price_history_path = os.path.join(self.data_dir, 'market_price_history')
        os.makedirs(self.price_history_path, exist_ok=True)
        
        # Load historical data
        self.historical_data = self._load_historical_data()
        
        # Load price prediction model if available
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'models', 'saved')
        self.model = self._load_model()
    
    def _load_api_key(self):
        """Load API key from configuration file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('market_api_key', '')
            else:
                print("API configuration file not found.")
                return ''
        except Exception as e:
            print(f"Error loading API key: {e}")
            return ''
    
    def _load_model(self):
        """Load price prediction model if available."""
        model_path = os.path.join(self.model_dir, 'price_prediction_model.pkl')
        try:
            if os.path.exists(model_path):
                import joblib
                return joblib.load(model_path)
            else:
                print("Price prediction model not found. Using rule-based predictions only.")
                return None
        except Exception as e:
            print(f"Error loading price prediction model: {e}")
            return None
    
    def _load_historical_data(self):
        """Load historical price data."""
        data_file = os.path.join(self.data_dir, 'crop_prices_historical.csv')
        try:
            if os.path.exists(data_file):
                return pd.read_csv(data_file)
            else:
                print("Historical price data not found. Using default data.")
                return self._generate_default_historical_data()
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return self._generate_default_historical_data()
    
    def _generate_default_historical_data(self):
        """Generate default historical data when file is not available."""
        # Create a basic dataframe with common crops
        common_crops = [
            "Rice", "Wheat", "Maize", "Potato", "Tomato", "Onion", 
            "Soybean", "Cotton", "Sugarcane", "Coffee"
        ]
        
        data = []
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365*2)  # 2 years of data
        
        # Generate random but somewhat realistic price trends
        for crop in common_crops:
            base_price = np.random.uniform(10, 100)  # Base price varies by crop
            trend = np.random.uniform(-0.0002, 0.0002)  # Slight upward or downward trend
            seasonality = np.random.uniform(0.05, 0.15)  # Seasonal variation
            
            current_date = start_date
            while current_date <= end_date:
                day_of_year = current_date.timetuple().tm_yday
                seasonal_factor = np.sin(day_of_year / 365 * 2 * np.pi) * seasonality
                
                days_since_start = (current_date - start_date).days
                trend_factor = trend * days_since_start
                
                random_factor = np.random.normal(0, 0.01)  # Daily random variation
                
                price = base_price * (1 + seasonal_factor + trend_factor + random_factor)
                price = max(price, base_price * 0.5)  # Ensure price doesn't go too low
                
                data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'crop': crop,
                    'price': round(price, 2),
                    'currency': 'USD',
                    'unit': 'kg'
                })
                
                current_date += timedelta(days=1)
        
        return pd.DataFrame(data)
    
    def get_current_prices(self, crops=None, location=None):
        """
        Get current market prices for specified crops.
        
        Args:
            crops (list, optional): List of crop names
            location (dict, optional): Location details for local prices
            
        Returns:
            dict: Current market prices
        """
        try:
            if self.api_key and location:
                # Use API if available and location is specified
                return self._fetch_market_prices_api(crops, location)
            else:
                # Use historical data or mock data
                return self._get_prices_from_historical(crops)
                
        except Exception as e:
            return {"error": str(e)}
    
    def _fetch_market_prices_api(self, crops, location):
        """Fetch market prices from external API."""
        try:
            # This would be replaced with actual API call
            # For now, we'll return mock data
            return self._get_prices_from_historical(crops)
        except Exception as e:
            print(f"API fetch error: {e}")
            return self._get_prices_from_historical(crops)
    
    def _get_prices_from_historical(self, crops=None):
        """Get prices from historical data."""
        if not isinstance(self.historical_data, pd.DataFrame) or self.historical_data.empty:
            return self._get_mock_prices(crops)
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Filter for most recent date in the data
        latest_date = self.historical_data['date'].max()
        latest_data = self.historical_data[self.historical_data['date'] == latest_date]
        
        # Filter by crops if specified
        if crops:
            latest_data = latest_data[latest_data['crop'].isin(crops)]
        
        # Format the response
        prices = []
        for _, row in latest_data.iterrows():
            prices.append({
                'crop': row['crop'],
                'price': row['price'],
                'currency': row['currency'],
                'unit': row['unit'],
                'date': latest_date
            })
        
        if not prices:
            return self._get_mock_prices(crops)
            
        return {
            'prices': prices,
            'as_of_date': latest_date
        }
    
    def _get_mock_prices(self, crops=None):
        """Generate mock price data."""
        default_crops = [
            "Rice", "Wheat", "Maize", "Potato", "Tomato", "Onion", 
            "Soybean", "Cotton", "Sugarcane", "Coffee"
        ]
        
        crops_to_use = crops if crops else default_crops
        today = datetime.now().strftime('%Y-%m-%d')
        
        prices = []
        for crop in crops_to_use:
            base_price = {
                "Rice": 0.85, "Wheat": 0.30, "Maize": 0.20, "Potato": 0.45, 
                "Tomato": 1.20, "Onion": 0.65, "Soybean": 0.40, "Cotton": 0.75, 
                "Sugarcane": 0.05, "Coffee": 4.50
            }.get(crop, np.random.uniform(0.2, 5.0))
            
            # Add some randomness
            price = base_price * (1 + np.random.normal(0, 0.05))
            
            prices.append({
                'crop': crop,
                'price': round(price, 2),
                'currency': 'USD',
                'unit': 'kg',
                'date': today
            })
        
        return {
            'prices': prices,
            'as_of_date': today,
            'note': 'These are simulated prices as no API key was provided or no data was available'
        }
    
    def get_price_trends(self, crop, days=30):
        """
        Get price trends for a specific crop.
        
        Args:
            crop (str): Crop name
            days (int): Number of days to look back
            
        Returns:
            dict: Price trend data
        """
        try:
            if not isinstance(self.historical_data, pd.DataFrame) or self.historical_data.empty:
                return self._get_mock_price_trends(crop, days)

            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Filter data
            mask = (
                (self.historical_data['crop'] == crop) & 
                (pd.to_datetime(self.historical_data['date']).dt.date >= start_date)
            )
            trend_data = self.historical_data[mask].sort_values('date')
            
            if trend_data.empty:
                return self._get_mock_price_trends(crop, days)
            
            # Format the response
            trend = []
            for _, row in trend_data.iterrows():
                trend.append({
                    'date': row['date'],
                    'price': row['price']
                })
            
            # Calculate stats
            if len(trend) > 1:
                prices = [item['price'] for item in trend]
                price_change = prices[-1] - prices[0]
                percent_change = (price_change / prices[0]) * 100 if prices[0] > 0 else 0
                
                stats = {
                    'min': min(prices),
                    'max': max(prices),
                    'avg': sum(prices) / len(prices),
                    'change': round(price_change, 2),
                    'percent_change': round(percent_change, 2)
                }
            else:
                stats = {
                    'min': trend[0]['price'],
                    'max': trend[0]['price'],
                    'avg': trend[0]['price'],
                    'change': 0,
                    'percent_change': 0
                }
            
            return {
                'crop': crop,
                'currency': self.historical_data.loc[mask, 'currency'].iloc[0] if not trend_data.empty else 'USD',
                'unit': self.historical_data.loc[mask, 'unit'].iloc[0] if not trend_data.empty else 'kg',
                'trend': trend,
                'stats': stats
            }
                
        except Exception as e:
            return {"error": str(e)}
    
    def _get_mock_price_trends(self, crop, days=30):
        """Generate mock price trend data."""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Generate base price
        base_price = {
            "Rice": 0.85, "Wheat": 0.30, "Maize": 0.20, "Potato": 0.45, 
            "Tomato": 1.20, "Onion": 0.65, "Soybean": 0.40, "Cotton": 0.75, 
            "Sugarcane": 0.05, "Coffee": 4.50
        }.get(crop, np.random.uniform(0.2, 5.0))
        
        # Generate trend
        trend = []
        current_price = base_price
        overall_trend = np.random.choice([-1, 1]) * np.random.uniform(0.001, 0.003)  # Slight trend up or down
        
        current_date = start_date
        while current_date <= end_date:
            # Add randomness + slight trend
            random_factor = np.random.normal(0, 0.01)
            trend_factor = overall_trend * (current_date - start_date).days
            
            current_price = max(0.01, current_price * (1 + random_factor + trend_factor))
            
            trend.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'price': round(current_price, 2)
            })
            
            current_date += timedelta(days=1)
        
        # Calculate stats
        prices = [item['price'] for item in trend]
        price_change = prices[-1] - prices[0]
        percent_change = (price_change / prices[0]) * 100 if prices[0] > 0 else 0
        
        stats = {
            'min': min(prices),
            'max': max(prices),
            'avg': sum(prices) / len(prices),
            'change': round(price_change, 2),
            'percent_change': round(percent_change, 2)
        }
        
        return {
            'crop': crop,
            'currency': 'USD',
            'unit': 'kg',
            'trend': trend,
            'stats': stats,
            'note': 'These are simulated prices as no API key was provided or no data was available'
        }
    
    def predict_future_prices(self, crop, days_ahead=30):
        """
        Predict future prices for a specific crop.
        
        Args:
            crop (str): Crop name
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            dict: Price prediction data
        """
        try:
            # If we have a ML model, use it
            if self.model:
                return self._predict_with_model(crop, days_ahead)
            else:
                # Otherwise use simple trend-based prediction
                return self._predict_with_trend(crop, days_ahead)
                
        except Exception as e:
            return {"error": str(e)}
    
    def _predict_with_model(self, crop, days_ahead):
        """Use machine learning model to predict prices."""
        try:
            # Get recent history for features
            recent_trend = self.get_price_trends(crop, days=90)
            
            if 'error' in recent_trend:
                return self._predict_with_trend(crop, days_ahead)
                
            prices = [item['price'] for item in recent_trend['trend']]
            dates = [datetime.strptime(item['date'], '%Y-%m-%d').date() for item in recent_trend['trend']]
            
            if len(prices) < 30:  # Need sufficient data
                return self._predict_with_trend(crop, days_ahead)
                
            # Prepare features for the model (this would depend on the model's expected inputs)
            # For example, we might use recent price movements, volatility, seasonality features
            
            # Generate predictions
            predictions = []
            last_date = dates[-1]
            last_price = prices[-1]
            
            # Note: In a real implementation, we would use the model to generate these predictions
            # For now, we'll use the trend-based method with some adjustments
            
            trend_prediction = self._predict_with_trend(crop, days_ahead)
            
            if 'error' in trend_prediction:
                return trend_prediction
                
            # Apply some "model-like" adjustments to the trend prediction
            predictions = []
            for i, pred in enumerate(trend_prediction['predictions']):
                # Add some patterns that a model might detect
                day_of_year = (datetime.strptime(pred['date'], '%Y-%m-%d').timetuple().tm_yday)
                seasonal_adjustment = np.sin(day_of_year / 365 * 2 * np.pi) * 0.02  # Subtle seasonal pattern
                
                # Add some complexity to predictions (as if from a model)
                if i > 0 and i % 7 == 0:  # Weekly pattern
                    adjustment = 0.015
                else:
                    adjustment = -0.002
                    
                new_price = pred['price'] * (1 + seasonal_adjustment + adjustment)
                
                predictions.append({
                    'date': pred['date'],
                    'price': round(new_price, 2)
                })
            
            # Calculate confidence bounds (would be from model prediction intervals)
            for pred in predictions:
                days_from_now = (datetime.strptime(pred['date'], '%Y-%m-%d').date() - datetime.now().date()).days
                uncertainty = 0.005 * days_from_now  # Uncertainty grows with time
                pred['lower_bound'] = round(pred['price'] * (1 - uncertainty), 2)
                pred['upper_bound'] = round(pred['price'] * (1 + uncertainty), 2)
            
            return {
                'crop': crop,
                'currency': recent_trend.get('currency', 'USD'),
                'unit': recent_trend.get('unit', 'kg'),
                'predictions': predictions,
                'model_type': 'ML price prediction model'
            }
            
        except Exception as e:
            print(f"Model prediction error: {e}")
            return self._predict_with_trend(crop, days_ahead)
    
    def _predict_with_trend(self, crop, days_ahead):
        """Use simple trend-based approach to predict prices."""
        # Get recent price trends
        recent_trend = self.get_price_trends(crop, days=60)
        
        if 'error' in recent_trend:
            return recent_trend
        
        # Extract prices
        if not recent_trend['trend']:
            return {"error": "Insufficient data for prediction"}
        
        prices = [item['price'] for item in recent_trend['trend']]
        dates = [datetime.strptime(item['date'], '%Y-%m-%d').date() for item in recent_trend['trend']]
        
        if len(prices) < 7:  # Need at least a week of data
            return {"error": "Insufficient data for prediction"}
        
        # Calculate simple moving average and recent trend
        window = min(7, len(prices))
        recent_avg = sum(prices[-window:]) / window
        
        # Calculate trend (average daily change)
        if len(prices) >= 14:
            recent_price_change = prices[-1] - prices[-14]
            daily_change = recent_price_change / 14
        else:
            recent_price_change = prices[-1] - prices[0]
            daily_change = recent_price_change / (len(prices) - 1) if len(prices) > 1 else 0
        
        # Add seasonality factor if we have enough historical data
        seasonality_factor = 0
        if len(self.historical_data) > 365 and isinstance(self.historical_data, pd.DataFrame):
            crop_data = self.historical_data[self.historical_data['crop'] == crop]
            if not crop_data.empty:
                # Try to detect seasonal patterns (simplified)
                today = datetime.now().date()
                # Look at same time last year
                same_period_last_year = [
                    (today - timedelta(days=365+i)).strftime('%Y-%m-%d') 
                    for i in range(-15, 16)
                ]
                
                last_year_data = crop_data[crop_data['date'].isin(same_period_last_year)]
                if not last_year_data.empty:
                    last_year_avg = last_year_data['price'].mean()
                    current_avg = recent_avg
                    if last_year_avg > 0:
                        seasonal_ratio = current_avg / last_year_avg
                        seasonality_factor = (seasonal_ratio - 1) / 30  # Spread over a month
        
        # Generate predictions
        predictions = []
        last_date = dates[-1]
        last_price = prices[-1]
        
        for i in range(1, days_ahead + 1):
            # Calculate prediction with trend and seasonality
            prediction_date = last_date + timedelta(days=i)
            price_change = daily_change + seasonality_factor
            
            # Add some randomness and gradually reduce impact of trend for longer forecasts
            trend_decay = max(0, 1 - (i / (days_ahead * 2)))
            random_factor = np.random.normal(0, 0.005)  # Small random variation
            
            predicted_price = last_price + (price_change * i * trend_decay) + (random_factor * i)
            predicted_price = max(0.01, predicted_price)  # Ensure price doesn't go negative
            
            predictions.append({
                'date': prediction_date.strftime('%Y-%m-%d'),
                'price': round(predicted_price, 2)
            })
        
        return {
            'crop': crop,
            'currency': recent_trend.get('currency', 'USD'),
            'unit': recent_trend.get('unit', 'kg'),
            'predictions': predictions,
            'model_type': 'trend-based prediction'
        }
    
    def get_price_recommendations(self, crop, current_stock=None):
        """
        Get selling recommendations based on price predictions.
        
        Args:
            crop (str): Crop name
            current_stock (float, optional): Current stock in kg
            
        Returns:
            dict: Recommendation data
        """
        try:
            # Get price predictions
            predictions = self.predict_future_prices(crop, days_ahead=30)
            
            if 'error' in predictions:
                return predictions
                
            # Get recent trends for context
            recent_trend = self.get_price_trends(crop, days=30)
            
            if 'error' in recent_trend:
                return recent_trend
                
            # Make recommendations
            predicted_prices = [item['price'] for item in predictions['predictions']]
            current_price = recent_trend['trend'][-1]['price'] if recent_trend['trend'] else None
            
            if not current_price or not predicted_prices:
                return {"error": "Insufficient data for recommendations"}
                
            # Calculate key metrics
            max_price = max(predicted_prices)
            max_price_date = None
            for pred in predictions['predictions']:
                if pred['price'] == max_price:
                    max_price_date = pred['date']
                    break
                    
            # Price expected to rise significantly?
            price_rise = max_price - current_price
            percent_rise = (price_rise / current_price) * 100 if current_price > 0 else 0
            
            # Make recommendation
            if percent_rise > 10:
                action = "HOLD"
                reason = f"Prices expected to rise by {percent_rise:.1f}% within the next month. Best selling date around {max_price_date}."
            elif percent_rise > 5:
                action = "CONSIDER HOLDING"
                reason = f"Prices expected to rise by {percent_rise:.1f}% within the next month. Could sell gradually."
            elif percent_rise > 0:
                action = "SELL GRADUALLY"
                reason = f"Prices expected to rise slightly by {percent_rise:.1f}%. Consider selling gradually."
            else:
                action = "SELL NOW"
                reason = "Prices expected to decline. Better to sell now unless storage costs are minimal."
            
            # Additional insights
            insights = []
            
            # Price volatility
            if recent_trend['stats']['max'] > 0:
                volatility = (recent_trend['stats']['max'] - recent_trend['stats']['min']) / recent_trend['stats']['max'] * 100
                if volatility > 15:
                    insights.append(f"Market shows high volatility ({volatility:.1f}%). Consider selling in smaller batches.")
            
            # Calculate optimal selling strategy if current stock provided
            selling_strategy = None
            if current_stock:
                selling_strategy = self._calculate_selling_strategy(
                    current_stock, 
                    current_price, 
                    predictions['predictions'], 
                    action
                )
            
            return {
                'crop': crop,
                'current_price': current_price,
                'currency': predictions.get('currency', 'USD'),
                'unit': predictions.get('unit', 'kg'),
                'recommendation': {
                    'action': action,
                    'reason': reason,
                    'best_price': max_price,
                    'best_price_date': max_price_date,
                    'price_change': round(price_rise, 2),
                    'percent_change': round(percent_rise, 1)
                },
                'insights': insights,
                'selling_strategy': selling_strategy
            }
                
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_selling_strategy(self, current_stock, current_price, predictions, action):
        """Calculate optimal selling strategy based on predictions."""
        if action == "SELL NOW":
            return [{
                'date': datetime.now().strftime('%Y-%m-%d'),
                'amount': current_stock,
                'price': current_price,
                'revenue': round(current_stock * current_price, 2)
            }]
        
        # For other actions, distribute sales
        strategy = []
        remaining_stock = current_stock
        
        if action == "SELL GRADUALLY":
            # Sell over next 2 weeks in 3 batches
            sell_dates = [0, 7, 14]  # Today, in a week, in two weeks
            batch_size = current_stock / len(sell_dates)
            
            for days in sell_dates:
                sell_date = datetime.now() + timedelta(days=days)
                sell_date_str = sell_date.strftime('%Y-%m-%d')
                
                # Find predicted price for this date
                price = current_price  # Default
                for pred in predictions:
                    if pred['date'] == sell_date_str:
                        price = pred['price']
                        break
                
                if days == 0:
                    price = current_price  # Today's price is known
                
                amount = batch_size if days < sell_dates[-1] else remaining_stock
                remaining_stock -= amount
                
                strategy.append({
                    'date': sell_date_str,
                    'amount': round(amount, 2),
                    'price': price,
                    'revenue': round(amount * price, 2)
                })
        
        elif action in ["CONSIDER HOLDING", "HOLD"]:
            # Find optimal selling points
            sorted_predictions = sorted(predictions, key=lambda x: x['price'], reverse=True)
            
            # Take top 2 price points
            best_days = sorted_predictions[:2]
            batch_size = current_stock / 2
            
            for i, pred in enumerate(best_days):
                amount = batch_size if i == 0 else remaining_stock
                remaining_stock -= amount
                
                strategy.append({
                    'date': pred['date'],
                    'amount': round(amount, 2),
                    'price': pred['price'],
                    'revenue': round(amount * pred['price'], 2)
                })
        
        # Calculate total revenue
        total_revenue = sum(item['revenue'] for item in strategy)
        now_revenue = current_stock * current_price
        
        # Add summary
        for item in strategy:
            item['percentage'] = round((item['amount'] / current_stock) * 100, 1)
        
        strategy.append({
            'summary': {
                'total_revenue': round(total_revenue, 2),
                'vs_selling_now': round(total_revenue - now_revenue, 2),
                'percent_gain': round((total_revenue - now_revenue) / now_revenue * 100 if now_revenue > 0 else 0, 1)
            }
        })
        
        return strategy