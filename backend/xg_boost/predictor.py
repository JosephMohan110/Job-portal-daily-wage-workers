# xg_boost/predictor.py - UPDATED VERSION
import pickle
import pandas as pd
import numpy as np
import os
from django.conf import settings
from datetime import datetime, timedelta
from django.utils import timezone
from decimal import Decimal
import traceback

class XGBoostPredictor:
    def __init__(self):
        self.model_path = os.path.join(settings.BASE_DIR, 'xg_boost', 'complete_xgboost_package.pkl')
        self.models = {}
        self.feature_names = []
        self.loaded = False
        self.load_model()

    def load_model(self):
        """Load the XGBoost model from pickle file"""
        try:
            if not os.path.exists(self.model_path):
                print(f" Model file not found at {self.model_path}")
                return False
            
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f" Loaded pickle data. Keys: {list(data.keys())}")
            
            # Extract models from the complex structure
            if 'all_models' in data:
                self.models = data['all_models']
                print(f" Found {len(self.models)} models in 'all_models'")
            
            # Get feature names
            if 'data_info' in data and 'original_features' in data['data_info']:
                self.feature_names = data['data_info']['original_features']
            elif 'metadata' in data and 'feature_names' in data['metadata']:
                self.feature_names = data['metadata']['feature_names']
            
            print(f" Feature names: {self.feature_names}")
            print(f" Available predictions: {list(self.models.keys())}")
            
            self.loaded = True
            return True
            
        except Exception as e:
            print(f" Error loading model: {str(e)}")
            traceback.print_exc()
            return False

    def prepare_platform_features(self, platform_data):
        """
        Prepare EXACTLY the 19 features the model expects:
        ['timestamp', 'user_id', 'user_type', 'registration_date', 'account_status', 
         'total_bookings', 'completed_bookings', 'cancelled_bookings', 'total_spent', 
         'total_earned', 'platform_commission', 'avg_rating', 'total_reviews', 
         'last_active', 'days_since_registration', 'days_since_last_active', 
         'completion_rate', 'cancellation_rate', 'avg_earning_per_booking']
        """
        try:
            now = timezone.now()
            today = now.date()
            
            # Create next month timestamp
            next_month = today.replace(day=28) + timedelta(days=4)  # Next month
            next_month_timestamp = datetime.combine(next_month, datetime.min.time()).timestamp()
            
            # Get current platform metrics
            total_bookings = platform_data.get('total_bookings', 0)
            completed_bookings = platform_data.get('completed_bookings', 0)
            cancelled_bookings = platform_data.get('cancelled_bookings', 0)
            total_revenue = platform_data.get('total_revenue', 0)
            total_earnings = platform_data.get('total_earnings', 0)
            platform_commission = platform_data.get('platform_commission', 0)
            avg_rating = platform_data.get('avg_rating', 0)
            total_reviews = platform_data.get('total_reviews', 0)
            
            # Calculate rates
            completion_rate = (completed_bookings / total_bookings * 100) if total_bookings > 0 else 0
            cancellation_rate = (cancelled_bookings / total_bookings * 100) if total_bookings > 0 else 0
            avg_earning_per_booking = (total_earnings / completed_bookings) if completed_bookings > 0 else 0
            
            # Prepare features EXACTLY as model expects
            features = {
                'timestamp': next_month_timestamp,
                'user_id': 0,  # Platform aggregate
                'user_type': 0,  # Platform type
                'registration_date': now.timestamp() - (180 * 24 * 3600),  # 180 days ago
                'account_status': 1,  # Active
                'total_bookings': total_bookings,
                'completed_bookings': completed_bookings,
                'cancelled_bookings': cancelled_bookings,
                'total_spent': total_revenue,
                'total_earned': total_earnings,
                'platform_commission': platform_commission,
                'avg_rating': avg_rating,
                'total_reviews': total_reviews,
                'last_active': now.timestamp(),
                'days_since_registration': 180,  
                'days_since_last_active': 1, 
                'completion_rate': completion_rate,
                'cancellation_rate': cancellation_rate,
                'avg_earning_per_booking': avg_earning_per_booking
            }
            
            print(f" Prepared {len(features)} features for prediction")
            return pd.DataFrame([features])
            
        except Exception as e:
            print(f" Error preparing features: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()

    def predict(self, platform_data):
        """Make predictions using the loaded models"""
        if not self.loaded:
            print(" Model not loaded")
            return {}
        
        try:
            # Prepare features
            features_df = self.prepare_platform_features(platform_data)
            
            if features_df.empty:
                print(" Features DataFrame is empty")
                return {}
            
            # Initialize predictions dict
            raw_predictions = {}
            
            # Make predictions for each target
            for target_name, model_info in self.models.items():
                try:
                    # Extract model from the complex structure
                    if isinstance(model_info, dict) and 'model' in model_info:
                        model = model_info['model']
                    else:
                        model = model_info
                    
                    if hasattr(model, 'predict'):
                        # For each model, we need to remove the target feature from input
                        input_features = features_df.copy()
                        
                        # Remove the target feature if it exists in input
                        if target_name in input_features.columns:
                            input_features = input_features.drop(columns=[target_name])
                        
                        # Ensure we have all required features
                        if hasattr(model, 'feature_names_in_'):
                            model_features = model.feature_names_in_
                            # Reorder columns to match model expectations
                            input_features = input_features[model_features]
                        
                        # Make prediction
                        prediction = model.predict(input_features)
                        pred_value = float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction)
                        
                        # Apply constraints based on target type
                        if target_name in ['completed_bookings', 'total_bookings', 'cancelled_bookings', 'total_reviews']:
                            pred_value = max(0, int(pred_value))  # Non-negative integers
                        elif target_name in ['completion_rate', 'cancellation_rate']:
                            pred_value = max(0, min(100, pred_value))  # Percentage 0-100
                        elif target_name == 'avg_rating':
                            pred_value = max(0, min(5, pred_value))  # Rating 0-5
                        elif target_name in ['total_spent', 'total_earned', 'platform_commission', 'avg_earning_per_booking']:
                            pred_value = max(0, pred_value)  # Non-negative currency
                        
                        raw_predictions[target_name] = pred_value
                        print(f" Predicted {target_name}: {pred_value}")
                        
                except Exception as e:
                    print(f" Error predicting {target_name}: {str(e)}")
                    continue
            
            # Format predictions for dashboard display
            formatted_predictions = self.format_predictions(raw_predictions, platform_data)
            
            print(f" Generated {len(formatted_predictions)} formatted predictions")
            return formatted_predictions
            
        except Exception as e:
            print(f" Prediction error: {str(e)}")
            traceback.print_exc()
            return {}

    def format_predictions(self, raw_predictions, platform_data):
        """Format raw model predictions for dashboard display"""
        predictions = {
            # Map model predictions to dashboard metrics
            'revenue_next_month': raw_predictions.get('total_spent', 0),
            'commission_next_month': raw_predictions.get('platform_commission', 0),
            'completed_bookings_next_month': int(raw_predictions.get('completed_bookings', 0)),
            'total_bookings_next_month': int(raw_predictions.get('total_bookings', 0)),
            'success_rate_next_month': min(100, max(0, raw_predictions.get('completion_rate', 0))),
            'avg_rating_next_month': min(5, max(0, raw_predictions.get('avg_rating', 0))),
            
            # Calculate derived predictions
            'new_users_next_month': int(platform_data.get('new_users_this_month', 0) * 1.1),
            'active_users_next_month': int(platform_data.get('active_users', 0) * 1.05),
            'deleted_accounts_next_month': int(platform_data.get('deleted_accounts_this_month', 0) * 0.95),
            
            # Store raw predictions for debugging
            'raw_predictions': raw_predictions,
            'using_real_ml': True,
            'ml_model_loaded': True,
        }
        
        return predictions

    def get_available_predictions(self):
        """Get list of available prediction targets"""
        return list(self.models.keys()) if self.loaded else []

    def get_feature_importance(self):
        """Get feature importance for dashboard display"""
        try:
            # We want to show what drives Revenue (total_spent)
            target_model_info = self.models.get('total_spent')
            if not target_model_info and self.models:
                # Fallback to first available model
                target_model_info = next(iter(self.models.values()))
            
            if not target_model_info:
                return {}

            # Extract model object
            if isinstance(target_model_info, dict) and 'model' in target_model_info:
                model = target_model_info['model']
            else:
                model = target_model_info
            
            # Get importance scores
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                
                # Use loaded feature names or fallback to fixed list
                names = self.feature_names
                if not names or len(names) != len(importance_scores):
                    # Fallback to the known list if data mismatch
                    names = ['timestamp', 'user_id', 'user_type', 'registration_date', 'account_status', 
                             'total_bookings', 'completed_bookings', 'cancelled_bookings', 'total_spent', 
                             'total_earned', 'platform_commission', 'avg_rating', 'total_reviews', 
                             'last_active', 'days_since_registration', 'days_since_last_active', 
                             'completion_rate', 'cancellation_rate', 'avg_earning_per_booking']
                
                # Zip names and scores
                # Ensure we don't index out of bounds
                limit = min(len(names), len(importance_scores))
                feature_map = {names[i]: float(importance_scores[i]) for i in range(limit)}
                
                # Sort and top 10
                top_features = sorted(
                    feature_map.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
                
                # Format for display
                display_names = {
                    'total_spent': 'Total Revenue',
                    'completed_bookings': 'Completed Bookings',
                    'total_bookings': 'Total Bookings',
                    'platform_commission': 'Platform Commission',
                    'avg_rating': 'Average Rating',
                    'completion_rate': 'Completion Rate',
                    'total_earned': 'Worker Earnings',
                    'cancelled_bookings': 'Cancelled Bookings',
                    'total_reviews': 'Total Reviews',
                    'last_active': 'Last Active Days',
                    'days_since_registration': 'Account Age',
                    'avg_earning_per_booking': 'Avg Earning/Booking'
                }
                
                result = {}
                for key, value in top_features:
                    display_name = display_names.get(key, key.replace('_', ' ').title())
                    result[display_name] = value
                
                return result
            
            return {}
            
        except Exception as e:
            print(f" Error getting feature importance: {str(e)}")
            return {}

# Singleton instance
predictor = XGBoostPredictor()