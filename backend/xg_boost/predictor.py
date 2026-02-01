import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from django.utils import timezone

logger = logging.getLogger(__name__)

import os
from django.conf import settings

class XGBoostPredictor:
    def __init__(self, model_path=None):
        """
        Initialize the XGBoost predictor with the saved model package
        """
        if model_path is None:
            model_path = os.path.join(settings.BASE_DIR, 'xg_boost', 'complete_xgboost_package.pkl')
        
        self.model_path = model_path
        self.model_package = None
        self.model_package = None
        self.load_model()
        
    @property
    def loaded(self):
        """Check if model is successfully loaded"""
        return self.model_package is not None
        
    def load_model(self):
        """Load the pre-trained XGBoost model package"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_package = pickle.load(f)
            logger.info(f"Successfully loaded XGBoost model package")
            logger.info(f"Available targets: {list(self.model_package['all_models'].keys())}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.model_package = None
    
    def prepare_current_platform_data(self, historical_data):
        """
        Prepare current platform data for prediction
        historical_data: dict containing current platform metrics
        """
        # Get the latest date from existing data or use current date
        if historical_data and 'latest_date' in historical_data:
            latest_date = historical_data['latest_date']
        else:
            latest_date = timezone.now().date()
        
        # Create a base feature set based on current platform state
        features = {
            'timestamp': pd.Timestamp(latest_date).timestamp(),
            'user_id': 0,  # Placeholder for platform-level prediction
            'user_type': 0,  # 0 for platform, 1 for worker, 2 for employer
            'registration_date': pd.Timestamp(latest_date - timedelta(days=30)).timestamp(),
            'account_status': 1,  # Active
            'total_bookings': historical_data.get('total_bookings', 0),
            'completed_bookings': historical_data.get('completed_bookings', 0),
            'cancelled_bookings': historical_data.get('cancelled_bookings', 0),
            'total_spent': historical_data.get('total_spent', 0),
            'total_earned': historical_data.get('total_earned', 0),
            'platform_commission': historical_data.get('platform_commission', 0),
            'avg_rating': historical_data.get('avg_rating', 4.5),
            'total_reviews': historical_data.get('total_reviews', 0),
            'last_active': pd.Timestamp(latest_date).timestamp(),
            'days_since_registration': 30,
            'days_since_last_active': 0,
            'completion_rate': historical_data.get('completion_rate', 0.85),
            'cancellation_rate': historical_data.get('cancellation_rate', 0.05),
            'avg_earning_per_booking': historical_data.get('avg_earning_per_booking', 0)
        }
        
        return pd.DataFrame([features])
    
    def predict_next_month(self, historical_data):
        """
        Predict platform metrics for the next month
        Returns predictions for key metrics
        """
        if not self.model_package:
            return None
        
        try:
            # Prepare input data
            input_df = self.prepare_current_platform_data(historical_data)
            
            # Get feature names from the model (exclude target columns)
            target_columns = list(self.model_package['all_models'].keys())
            
            predictions = {}
            
            # Predict key metrics we care about for dashboard
            key_targets = [
                'platform_commission',  # Revenue prediction
                'total_bookings',  # Booking prediction
                'completed_bookings',  # Completed bookings
                'total_spent',  # Total amount spent on platform
                'avg_rating',  # Average rating
                'cancellation_rate'  # Churn indicator
            ]
            
            for target in key_targets:
                if target in self.model_package['all_models']:
                    model = self.model_package['all_models'][target]
                    logger.info(f"Predicting for {target}. Model type: {type(model)}")
                    if isinstance(model, dict):
                         logger.error(f"Model for {target} is a dict, keys: {model.keys()}")
                         # Try to extract actual model if it's nested
                         if 'model' in model:
                             model = model['model']
                    
                    
                    # Prepare features (exclude the target column)
                    feature_cols = [col for col in input_df.columns if col != target]
                    X_input = input_df[feature_cols]
                    
                    # Make prediction
                    if self.model_package['performance'][target].get('type') == 'regression':
                        pred = model.predict(X_input)[0]
                    else:
                        # For classification, get probability or class
                        pred = model.predict_proba(X_input)[0][1]  # Probability of positive class
                    
                    predictions[target] = float(pred)
            
            # Calculate derived predictions
            if 'platform_commission' in predictions:
                # Predict revenue growth percentage
                current_revenue = historical_data.get('platform_commission', 0)
                if current_revenue > 0:
                    revenue_growth = ((predictions['platform_commission'] - current_revenue) / current_revenue) * 100
                    predictions['revenue_growth_percent'] = revenue_growth
            
            if 'total_bookings' in predictions:
                current_bookings = historical_data.get('total_bookings', 0)
                if current_bookings > 0:
                    booking_growth = ((predictions['total_bookings'] - current_bookings) / current_bookings) * 100
                    predictions['booking_growth_percent'] = booking_growth
            
            # Get feature importance for visualization
            if 'platform_commission' in self.model_package['feature_importance']:
                # Sort feature importance for the revenue model
                feature_importance = self.model_package['feature_importance']['platform_commission']
                sorted_features = dict(sorted(feature_importance.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:10])  # Top 10 features
                predictions['top_features'] = sorted_features
            
            # --- DERIVED / HEURISTIC METRICS FOR DASHBOARD ---
            # These are needed for the charts but might not be in the model
            
            # 1. New Users Growth (Heuristic: correlated with bookings growth)
            booking_growth = predictions.get('booking_growth_percent', 5.0)
            current_new_users = historical_data.get('new_users_this_month', 50) # Fallback default
            predictions['new_users_next_month'] = int(current_new_users * (1 + (booking_growth/100)))
            
            # 2. Deleted Accounts (Heuristic: inverse to avg_rating)
            avg_rating = predictions.get('avg_rating', 4.5)
            # Lower rating -> higher deletions. Baseline 5 deletions.
            deletions = max(0, int(10 - avg_rating * 1.5)) 
            predictions['deleted_accounts_next_month'] = deletions
            
            # 3. Active Users
            current_active = historical_data.get('active_users', 100)
            predictions['active_users_next_month'] = int(current_active * (1 + (booking_growth/200))) # Grow at half rate of bookings
            
            # 4. Success Rate
            completed = predictions.get('completed_bookings', 0)
            total = predictions.get('total_bookings', 1)
            if total > 0:
                predictions['success_rate_next_month'] = (completed / total) * 100
            else:
                predictions['success_rate_next_month'] = 0
            
            # 5. Commission (ensure key matches view expectation)
            if 'platform_commission' in predictions:
                predictions['commission_next_month'] = predictions['platform_commission']
                predictions['revenue_next_month'] = predictions['platform_commission'] # Align keys
                
            # 6. Avg Rating
            if 'avg_rating' in predictions:
                predictions['avg_rating_next_month'] = predictions['avg_rating']
                
            # 7. Completed Bookings
            if 'completed_bookings' in predictions:
                predictions['completed_bookings_next_month'] = predictions['completed_bookings']
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

    def predict(self, historical_data):
        """Alias for predict_next_month to match view call"""
        return self.predict_next_month(historical_data)
    
    def generate_forecast_timeline(self, historical_data, periods=6):
        """
        Generate forecast for multiple future periods (months)
        """
        if not self.model_package:
            return None
        
        try:
            forecasts = []
            current_data = historical_data.copy()
            
            for i in range(periods):
                # Predict for next period
                prediction = self.predict_next_month(current_data)
                if prediction:
                    # Create forecast entry
                    forecast_date = timezone.now().date() + timedelta(days=30*(i+1))
                    forecast_entry = {
                        'period': f"Month {i+1}",
                        'date': forecast_date.strftime('%b %Y'),
                        'predicted_revenue': prediction.get('platform_commission', 0),
                        'predicted_bookings': prediction.get('total_bookings', 0),
                        'revenue_growth': prediction.get('revenue_growth_percent', 0),
                        'booking_growth': prediction.get('booking_growth_percent', 0)
                    }
                    forecasts.append(forecast_entry)
                    
                    # Update current data for next iteration (simulating growth)
                    current_data['platform_commission'] = prediction.get('platform_commission', current_data.get('platform_commission', 0))
                    current_data['total_bookings'] = prediction.get('total_bookings', current_data.get('total_bookings', 0))
                    current_data['completed_bookings'] = prediction.get('completed_bookings', current_data.get('completed_bookings', 0))
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Forecast generation error: {str(e)}")
            return None

    def get_available_predictions(self):
        """Get list of available prediction targets"""
        if not self.model_package or 'all_models' not in self.model_package:
            return []
        return list(self.model_package['all_models'].keys())

    def get_feature_importance(self, target='platform_commission'):
        """Get feature importance for a specific target"""
        if not self.model_package or 'feature_importance' not in self.model_package:
            return {}
        
        # If target in feature_importance, return it
        if target in self.model_package['feature_importance']:
            return self.model_package['feature_importance'][target]
            
        # If not, return the first one found or empty
        if self.model_package['feature_importance']:
            first_key = list(self.model_package['feature_importance'].keys())[0]
            return self.model_package['feature_importance'][first_key]
            
        return {}

# Create a singleton instance
predictor = XGBoostPredictor()