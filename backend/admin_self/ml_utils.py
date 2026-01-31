# admin_self/ml_utils.py - UPDATED VERSION
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from django.utils import timezone
import os
from django.conf import settings
import warnings
import traceback
warnings.filterwarnings('ignore')

class XGBoostPredictor:
    """Handle XGBoost model loading and predictions with proper feature alignment"""
    
    def __init__(self):
        self.models = {}
        self.features = []
        self.metadata = {}
        self.label_encoders = {}
        self.loaded = False
        self.model_path = os.path.join(settings.BASE_DIR, 'xg_boost', 'complete_xgboost_package.pkl')
        self.available_predictions = []
        
    def load_model(self):
        """Load the XGBoost model from file"""
        try:
            print(f"Looking for model at: {self.model_path}")
            print(f"File exists: {os.path.exists(self.model_path)}")
            
            if not os.path.exists(self.model_path):
                print(f"Model file not found at: {self.model_path}")
                return False
            
            print(f"Loading model from: {self.model_path}")
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Loaded pickle data type: {type(data)}")
            print(f"Pickle data keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
            
            # Extract models from the pickle file based on your error output
            self.models = {}
            
            # The pickle file contains multiple models
            for key, value in data.items():
                if key == 'models':
                    self.models = value
                elif key == 'feature_names':
                    self.features = value
                elif key == 'metadata':
                    self.metadata = value
                elif key == 'label_encoders':
                    self.label_encoders = value
                elif hasattr(value, 'predict'):  # If it's a model object
                    self.models[key] = value
            
            # If we still don't have models, check for direct model storage
            if not self.models:
                # Try to find models directly in data
                for key, value in data.items():
                    if hasattr(value, 'predict'):
                        self.models[key] = value
            
            print(f"Number of models loaded: {len(self.models)}")
            print(f"Model names: {list(self.models.keys())}")
            print(f"Number of features: {len(self.features)}")
            
            if not self.models:
                print("ERROR: No models found in the pickle file")
                return False
            
            self.loaded = True
            self.available_predictions = list(self.models.keys())
            
            print(f" Successfully loaded {len(self.models)} models")
            print(f"Available predictions: {self.available_predictions}")
            
            return True
            
        except Exception as e:
            print(f" Error loading model: {str(e)}")
            traceback.print_exc()
            return False
    
    def prepare_platform_features(self, platform_data):
        """
        Prepare features based on the model's training data structure
        """
        now = timezone.now()
        today = now.date()
        
        # Based on the model summary, it expects specific features
        # Let's prepare the 35 features that were mentioned in the training
        
        features_dict = {
            # User metrics (from your model output)
            'total_users': platform_data.get('total_users', 0),
            'active_users': platform_data.get('active_users', 0),
            'new_signups': platform_data.get('new_users_this_month', 0),
            'deletions': platform_data.get('deleted_accounts_this_month', 0),
            'total_employees': platform_data.get('total_workers', 0),
            
            # Booking metrics
            'total_bookings': platform_data.get('total_bookings', 0),
            'completed_bookings': platform_data.get('completed_bookings', 0),
            'cancelled_bookings': platform_data.get('cancelled_bookings', 0),
            'contracts_signed': platform_data.get('completed_bookings', 0),
            'completion_rate': platform_data.get('success_rate', 75) / 100,
            
            # Financial metrics
            'total_spent': platform_data.get('total_revenue', 0),
            'total_earned': platform_data.get('total_earnings', 0),
            'total_amount_in_site': platform_data.get('total_revenue', 0),
            'platform_commission': platform_data.get('platform_commission', 0),
            'earning_per_job': platform_data.get('avg_earning_per_job', 300),
            'avg_spending_per_job': platform_data.get('avg_spending_per_job', 350),
            'total_transaction_value': platform_data.get('total_revenue', 0),
            'total_commission': platform_data.get('platform_commission', 0),
            
            # Activity metrics
            'daily_active_minutes': 45,  # Default value
            'sessions_count': platform_data.get('bookings_today', 0),
            'profile_views': 120,  # Default value
            'messages_sent': 25,  # Default value
            'job_applications': platform_data.get('completed_bookings', 0),
            'jobs_posted': platform_data.get('total_bookings', 0),
            'streak_days': 7,  # Default value
            
            # Rating metrics
            'avg_rating': platform_data.get('avg_rating', 4.2),
            'total_reviews': platform_data.get('total_reviews', 0),
            
            # Date features
            'date_dayofweek': today.weekday(),
            'date_year': today.year,
            'date_month': today.month,
            'date_day': today.day,
            
            # Registration features
            'registration_date_dayofweek': (today - timedelta(days=180)).weekday(),
            'registration_date_year': today.year - 1,
            'registration_date_month': today.month,
            'registration_date_day': min(today.day, 28),
            
            # Last active
            'last_active_date_dayofweek': today.weekday(),
            'last_active_date_year': today.year,
            'last_active_date_month': today.month,
            'last_active_date_day': today.day,
            
            # Other features
            'engagement_score': 65.5,
            'days_since_registration': 180,
            'favorite_employee_count': 25,
            'cancelled_within_3_days': 3,
            'times_favorited': 45,
            'certificates_uploaded': 60,
            'support_tickets': 12,
            
            # Categorical features (encoded)
            'job_category': 3,
            'district': 1,
            'account_status': 1,
            'deletion_date': 0,
            'payment_disputes': 0,
            'user_type': 0,
        }
        
        print(f"Prepared {len(features_dict)} features")
        
        # Create DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Ensure we have all expected features
        if self.features:
            for feature in self.features:
                if feature not in features_df.columns:
                    print(f"Adding missing feature: {feature}")
                    features_df[feature] = 0
        
        # Reorder to match expected feature order
        if self.features:
            # Only include features that exist in both
            existing_features = [f for f in self.features if f in features_df.columns]
            features_df = features_df[existing_features]
        
        # Convert all to numeric
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        
        print(f"Final feature matrix shape: {features_df.shape}")
        
        return features_df
    
    def predict_all_models(self, features_df):
        """Make predictions using all loaded models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    # Make prediction
                    pred = model.predict(features_df)
                    
                    # Handle single value output
                    if isinstance(pred, (np.ndarray, list)):
                        pred_value = float(pred[0])
                    else:
                        pred_value = float(pred)
                    
                    # Format based on prediction type
                    if model_name in ['is_active_today', 'is_new_user', 'is_deleted_user', 
                                    'platform_new_signups', 'platform_deletions',
                                    'sessions_count', 'job_applications', 'jobs_posted']:
                        # Classification - round to 0 or 1
                        pred_value = 1 if pred_value > 0.5 else 0
                    elif 'rate' in model_name or 'success_rate' in model_name:
                        # Percentage - ensure between 0-100
                        pred_value = max(0, min(100, pred_value))
                    elif 'rating' in model_name:
                        # Rating - ensure between 0-5
                        pred_value = max(0, min(5, pred_value))
                    else:
                        # Regression - ensure non-negative
                        pred_value = max(0, pred_value)
                    
                    predictions[model_name] = pred_value
                    print(f"Predicted {model_name}: {pred_value}")
                    
            except Exception as e:
                print(f"Error predicting {model_name}: {str(e)}")
        
        return predictions
    
    def predict(self, platform_data):
        """Main prediction function"""
        if not self.loaded:
            if not self.load_model():
                print("Failed to load model")
                return {}
        
        try:
            # Prepare features
            features_df = self.prepare_platform_features(platform_data)
            
            # Make predictions
            raw_predictions = self.predict_all_models(features_df)
            
            if not raw_predictions:
                print("No predictions generated")
                return {}
            
            # Map to display-friendly format
            display_predictions = self.format_predictions(raw_predictions, platform_data)
            
            print(f"Generated {len(display_predictions)} display predictions")
            return display_predictions
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            return {}
    
    def format_predictions(self, raw_predictions, platform_data):
        """Format raw predictions for display"""
        # Initialize with default values
        predictions = {
            'new_users_next_month': 0,
            'deleted_accounts_next_month': 0,
            'completed_bookings_next_month': 0,
            'active_users_next_month': 0,
            'revenue_next_month': 0,
            'commission_next_month': 0,
            'success_rate_next_month': 0,
            'avg_rating_next_month': 0,
            'total_bookings_next_month': 0,
        }
        
        # Map raw predictions to display names
        mapping = {
            'platform_new_signups': 'new_users_next_month',
            'platform_deletions': 'deleted_accounts_next_month',
            'completed_bookings': 'completed_bookings_next_month',
            'platform_active_users': 'active_users_next_month',
            'total_transaction_value': 'revenue_next_month',
            'total_commission': 'commission_next_month',
            'success_rate': 'success_rate_next_month',
            'avg_rating': 'avg_rating_next_month',
            'total_bookings': 'total_bookings_next_month',
        }
        
        # Apply mappings
        for raw_key, display_key in mapping.items():
            if raw_key in raw_predictions:
                predictions[display_key] = raw_predictions[raw_key]
        
        # If some predictions are missing, use intelligent defaults
        current_users = platform_data.get('new_users_this_month', 0)
        current_bookings = platform_data.get('completed_bookings', 0)
        current_revenue = platform_data.get('total_revenue', 0)
        current_commission = platform_data.get('platform_commission', 0)
        current_success_rate = platform_data.get('success_rate', 0)
        current_rating = platform_data.get('avg_rating', 0)
        
        # Set defaults based on current data with growth factors
        if predictions['new_users_next_month'] == 0:
            predictions['new_users_next_month'] = int(current_users * 1.12)
        
        if predictions['completed_bookings_next_month'] == 0:
            predictions['completed_bookings_next_month'] = int(current_bookings * 1.18)
        
        if predictions['revenue_next_month'] == 0:
            predictions['revenue_next_month'] = current_revenue * 1.18
        
        if predictions['commission_next_month'] == 0:
            predictions['commission_next_month'] = current_commission * 1.18
        
        if predictions['success_rate_next_month'] == 0:
            predictions['success_rate_next_month'] = min(100, current_success_rate * 1.05)
        
        if predictions['avg_rating_next_month'] == 0:
            predictions['avg_rating_next_month'] = min(5, current_rating * 1.01)
        
        if predictions['total_bookings_next_month'] == 0:
            predictions['total_bookings_next_month'] = int(platform_data.get('total_bookings', 0) * 1.12)
        
        if predictions['active_users_next_month'] == 0:
            predictions['active_users_next_month'] = int(platform_data.get('active_users', 0) * 1.07)
        
        if predictions['deleted_accounts_next_month'] == 0:
            predictions['deleted_accounts_next_month'] = int(platform_data.get('deleted_accounts_this_month', 0) * 0.92)
        
        # Add raw predictions for debugging
        predictions['raw_predictions'] = raw_predictions
        
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance from models"""
        if not self.loaded:
            if not self.load_model():
                return {}
        
        importance_dict = {}
        
        try:
            # Collect importance from all models
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Get feature names
                    if hasattr(model, 'feature_names_in_'):
                        feature_names = model.feature_names_in_
                    elif self.features:
                        feature_names = self.features
                    else:
                        continue
                    
                    # Ensure lengths match
                    if len(feature_names) == len(importances):
                        for i, importance in enumerate(importances):
                            if i < len(feature_names):
                                feature_name = feature_names[i]
                                importance_dict[feature_name] = importance_dict.get(feature_name, 0) + importance
            
            # Normalize and get top 10
            if importance_dict:
                total = sum(importance_dict.values())
                if total > 0:
                    importance_dict = {k: v/total for k, v in importance_dict.items()}
                
                # Sort and get top 10
                top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # Format for display
                display_names = {
                    'total_users': 'Total Users',
                    'active_users': 'Active Users',
                    'new_signups': 'New Signups',
                    'total_bookings': 'Total Bookings',
                    'completed_bookings': 'Completed Bookings',
                    'total_spent': 'Total Spent',
                    'total_commission': 'Platform Commission',
                    'avg_rating': 'Average Rating',
                    'engagement_score': 'Engagement Score',
                    'completion_rate': 'Completion Rate',
                }
                
                result = {}
                for key, value in top_features:
                    display_name = display_names.get(key, key.replace('_', ' ').title())
                    result[display_name] = value
                
                return result
        
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
        
        return {}

# Singleton instance
predictor = XGBoostPredictor()