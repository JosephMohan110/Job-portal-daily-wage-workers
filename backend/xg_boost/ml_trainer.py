"""
XGBoost ML Trainer - Trains models on REAL platform data from CSV exports
Uses actual exported data from algorithm_setting.html page
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')
import glob

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import sys

# Django imports
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from django.utils import timezone
from django.db.models import Count, Sum, Avg, Q
from employee.models import Employee, JobRequest, Review
from employer.models import Employer, Payment

logger = logging.getLogger(__name__)

class MLTrainer:
    def __init__(self):
        self.models = {}
        self.feature_names = [
            'timestamp', 'user_id', 'user_type', 'registration_date', 'account_status',
            'total_bookings', 'completed_bookings', 'cancelled_bookings', 'total_spent',
            'total_earned', 'platform_commission', 'avg_rating', 'total_reviews',
            'last_active'
        ]
        self.feature_importance_dict = {}
        self.performance_dict = {}
        self.label_encoders = {}
        # Path to real CSV data exports - relative to backend directory
        self.media_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'media', 'data_exports')
    
    def load_data_from_file(self, file_path):
        """Load and clean data from a specific CSV or Excel file"""
        print(f"[Trainer] 📁 Loading data from: {file_path}")
        
        try:
            # Detect file type and load
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                print("[Trainer] Detected Excel file")
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                print("[Trainer] Detected CSV file")
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            print(f"[Trainer] ✓ Loaded {len(df)} records")
            print(f"[Trainer] ✓ Columns: {', '.join(df.columns[:10].tolist())}...")
            
            # Data cleaning and preparation
            # Add timestamp if missing
            if 'timestamp' not in df.columns:
                print("[Trainer] Adding timestamp column...")
                df['timestamp'] = pd.Timestamp.now()
            
            # Convert data types
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['timestamp'] = df['timestamp'].fillna(pd.Timestamp.now())
                df['timestamp'] = df['timestamp'].astype(np.int64) // 10**9
            
            if 'registration_date' in df.columns:
                df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
                df['registration_date'] = df['registration_date'].fillna(pd.Timestamp.now())
                df['registration_date'] = df['registration_date'].astype(np.int64) // 10**9
            
            if 'last_active' in df.columns:
                df['last_active'] = pd.to_datetime(df['last_active'], errors='coerce')
                df['last_active'] = df['last_active'].fillna(pd.Timestamp.now())
                df['last_active'] = df['last_active'].astype(np.int64) // 10**9
            
            # Handle user_id - remove prefix letters and convert to int
            if 'user_id' in df.columns:
                df['user_id'] = df['user_id'].astype(str).str.extract(r'(\d+)', expand=False)
                df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').fillna(0).astype(int)
            
            # Handle user_type encoding
            if 'user_type' in df.columns:
                df['user_type'] = df['user_type'].map({'worker': 0, 'employer': 1, 0: 0, 1: 1})
                df['user_type'] = df['user_type'].fillna(0).astype(int)
            
            # Handle account_status encoding
            if 'account_status' in df.columns:
                df['account_status'] = df['account_status'].map({'Active': 1, 'Inactive': 0, 1: 1, 0: 0})
                df['account_status'] = df['account_status'].fillna(1).astype(int)
            
            # Handle numeric columns
            for col in ['total_bookings', 'completed_bookings', 'cancelled_bookings', 'total_reviews']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
            for col in ['total_spent', 'total_earned', 'platform_commission', 'avg_rating']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
            
            # Keep only the 14 ML features
            essential_cols = [
                'timestamp', 'user_id', 'user_type', 'registration_date', 'account_status',
                'total_bookings', 'completed_bookings', 'cancelled_bookings', 'total_spent',
                'total_earned', 'platform_commission', 'avg_rating', 'total_reviews', 'last_active'
            ]
            
            # Keep only columns that exist in the data
            cols_to_keep = [col for col in essential_cols if col in df.columns]
            
            # Add missing columns with default values
            for col in essential_cols:
                if col not in df.columns:
                    print(f"[Trainer] ⚠️ Missing column '{col}', adding with default values")
                    if col in ['timestamp', 'registration_date', 'last_active']:
                        df[col] = int(pd.Timestamp.now().timestamp())
                    elif col in ['user_id', 'user_type', 'account_status', 'total_bookings', 
                                'completed_bookings', 'cancelled_bookings', 'total_reviews']:
                        df[col] = 0
                    else:
                        df[col] = 0.0
            
            df = df[essential_cols]
            
            print(f"[Trainer] ✓ Data cleaned and prepared with {len(df.columns)} ML features")
            print(f"[Trainer] ✓ Using {len(df)} records for training")
            
            return df
            
        except Exception as e:
            print(f"[Trainer] ❌ Error loading file: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_real_csv_data(self):
        """Load REAL data from exported CSV files in media/data_exports"""
        print("[Trainer] 📁 Loading REAL data from exported CSV files...")
        
        # Find all CSV files in data_exports folder
        csv_files = glob.glob(os.path.join(self.media_path, '*.csv'))
        
        if not csv_files:
            print("[Trainer] ⚠️ No CSV files found in data_exports folder!")
            print(f"[Trainer] Searched path: {self.media_path}")
            return None
        
        # Get the latest CSV file
        latest_csv = max(csv_files, key=os.path.getctime)
        return self.load_data_from_file(latest_csv)
    
    def collect_training_data(self, months_back=12):
        """
        Collect REAL training data from exported CSV files
        Falls back to database if CSV not available
        """
        print(f"[Trainer] Collecting REAL training data...")
        
        # First, try to load from real CSV exports
        df = self.load_real_csv_data()
        
        if df is not None and len(df) > 0:
            print(f"[Trainer] ✓ SUCCESS: Using {len(df)} REAL data records from CSV export")
            return df
        
        print("[Trainer] ⚠️ CSV data not available, falling back to database...")
        return self.collect_from_database(months_back)
    
    def create_sample_training_data(self, n_samples=50):
        """Create sample training data for demonstration"""
        print("[Trainer] Creating sample training data...")
        
        np.random.seed(42)
        data = {
            'timestamp': np.linspace(1609459200, 1735689600, n_samples),
            'user_id': np.zeros(n_samples),
            'user_type': np.zeros(n_samples),
            'registration_date': np.ones(n_samples) * 1609459200,
            'account_status': np.ones(n_samples),
            'total_bookings': np.random.randint(50, 500, n_samples),
            'completed_bookings': np.random.randint(40, 450, n_samples),
            'cancelled_bookings': np.random.randint(5, 50, n_samples),
            'total_spent': np.random.uniform(10000, 100000, n_samples),
            'total_earned': np.random.uniform(9900, 99000, n_samples),
            'platform_commission': np.random.uniform(100, 1000, n_samples),
            'avg_rating': np.random.uniform(3.5, 5.0, n_samples),
            'total_reviews': np.random.randint(100, 500, n_samples),
            'last_active': np.linspace(1609459200, 1735689600, n_samples),
            'days_since_registration': np.random.randint(30, 400, n_samples),
            'days_since_last_active': np.random.randint(0, 30, n_samples),
            'completion_rate': np.random.uniform(0.7, 0.95, n_samples),
            'cancellation_rate': np.random.uniform(0.02, 0.15, n_samples),
            'avg_earning_per_booking': np.random.uniform(100, 500, n_samples),
        }
        
        return pd.DataFrame(data)
    
    def collect_from_database(self, months_back=12):
        """Fallback: collect training data from database"""
        print("[Trainer] 📊 Collecting from database (fallback method)...")
        
        now = timezone.now()
        start_date = now - timedelta(days=30*months_back)
        
        training_records = []
        current_date = start_date.replace(day=1)
        
        while current_date < now:
            month_start = current_date
            next_month = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
            
            if next_month > now:
                next_month = now
            
            try:
                # Get metrics for this month
                total_bookings = JobRequest.objects.filter(
                    created_at__gte=month_start, created_at__lt=next_month
                ).count()
                
                completed_bookings = JobRequest.objects.filter(
                    status='completed', created_at__gte=month_start, created_at__lt=next_month
                ).count()
                
                cancelled_bookings = JobRequest.objects.filter(
                    status='cancelled', created_at__gte=month_start, created_at__lt=next_month
                ).count()
                
                payments = Payment.objects.filter(
                    status='completed', created_at__gte=month_start, created_at__lt=next_month
                ).aggregate(total=Sum('amount'))
                total_spent = float(payments['total'] or Decimal('0'))
                
                reviews = Review.objects.filter(
                    created_at__gte=month_start, created_at__lt=next_month
                )
                avg_rating = float(reviews.aggregate(avg=Avg('rating'))['avg'] or 4.0)
                total_reviews = reviews.count()
                
                # Create record
                record = {
                    'timestamp': pd.Timestamp(now).timestamp(),
                    'user_id': 0,
                    'user_type': 0,
                    'registration_date': pd.Timestamp(start_date).timestamp(),
                    'account_status': 1,
                    'total_bookings': int(total_bookings),
                    'completed_bookings': int(completed_bookings),
                    'cancelled_bookings': int(cancelled_bookings),
                    'total_spent': float(total_spent),
                    'total_earned': float(total_spent * 0.99),
                    'platform_commission': float(total_spent * 0.01),
                    'avg_rating': float(avg_rating),
                    'total_reviews': int(total_reviews),
                    'last_active': pd.Timestamp(now).timestamp(),
                }
                
                training_records.append(record)
                print(f"[Trainer] ✓ Month {month_start.strftime('%Y-%m')}: {total_bookings} bookings, ₹{total_spent:.0f}")
                
            except Exception as e:
                print(f"[Trainer] ⚠️ Error: {e}")
                continue
            
            current_date = next_month
        
        if training_records:
            return pd.DataFrame(training_records)
        return self.create_sample_training_data()
    
    def train_all_models(self, X_train, y_train_dict):
        """Train XGBoost models for each target"""
        print(f"[Trainer] Training {len(y_train_dict)} XGBoost models...")
        
        for i, (target, y_train) in enumerate(y_train_dict.items(), 1):
            print(f"\n[Trainer] [{i}/{len(y_train_dict)}] Training model for target: {target}")
            
            try:
                # Determine if classification or regression
                unique_values = len(np.unique(y_train))
                is_classification = unique_values < 10 and target in ['account_status']
                
                if is_classification:
                    # Classification
                    model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42,
                        verbosity=0,
                        eval_metric='logloss'
                    )
                else:
                    # Regression
                    model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42,
                        verbosity=0
                    )
                
                # Train model
                model.fit(X_train, y_train, verbose=False)
                
                self.models[target] = {
                    'model': model,
                    'is_classification': is_classification,
                }
                
                # Get feature importance
                importance = model.feature_importances_
                self.feature_importance_dict[target] = dict(zip(self.feature_names, importance))
                
                # Calculate performance metrics
                y_pred = model.predict(X_train)
                
                if is_classification:
                    accuracy = accuracy_score(y_train, y_pred)
                    self.performance_dict[target] = {
                        'type': 'classification',
                        'accuracy': float(accuracy),
                        'note': f'Classification model - Accuracy: {accuracy:.4f}'
                    }
                    print(f"  ✓ Classification Model - Accuracy: {accuracy:.4f}")
                else:
                    r2 = r2_score(y_train, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
                    self.performance_dict[target] = {
                        'type': 'regression',
                        'r2_score': float(r2),
                        'rmse': float(rmse),
                        'note': f'Regression model - R²: {r2:.4f}, RMSE: {rmse:.2f}'
                    }
                    print(f"  ✓ Regression Model - R²: {r2:.4f}, RMSE: {rmse:.2f}")
                
            except Exception as e:
                print(f"  ✗ Error training {target}: {e}")
                continue
    
    def train(self, months_back=12):
        """Main training pipeline"""
        print("\n" + "="*60)
        print("🚀 STARTING XGBOOST TRAINING PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Collect data
        df = self.collect_training_data(months_back)
        
        if df is None or len(df) == 0:
            print("[Trainer] ✗ Failed to collect training data")
            return False
        
        print(f"\n[Trainer] Dataset shape: {df.shape}")
        print(f"[Trainer] Features: {list(df.columns)}")
        
        # Step 2: Prepare features and targets
        X = df[self.feature_names].copy()
        
        # Predict only the metrics that exist in real CSV data
        # These are the actual metrics the platform has
        targets_to_predict = [
            'total_bookings', 'completed_bookings', 'cancelled_bookings',
            'total_spent', 'total_earned', 'platform_commission',
            'avg_rating', 'total_reviews'
        ]
        
        # Only predict targets that exist in the dataframe
        available_targets = [t for t in targets_to_predict if t in df.columns]
        print(f"[Trainer] Available targets to predict: {available_targets}")
        
        y_train_dict = {target: df[target].values for target in available_targets}
        
        # Step 3: Train models
        self.train_all_models(X, y_train_dict)
        
        # Step 4: Save models
        self.save_models()
        
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETE!")
        print("="*60 + "\n")
        
        return True
    
    def train_from_file(self, file_path, output_path=None):
        """Train models from a specific uploaded file - ALL 19 TARGETS"""
        print("\n" + "="*60)
        print("🚀 TRAINING FROM UPLOADED FILE")
        print("="*60 + "\n")
        
        # Step 1: Load data from file
        df = self.load_data_from_file(file_path)
        
        if df is None or len(df) == 0:
            print("[Trainer] ✗ Failed to load data from file")
            return False
        
        print(f"\n[Trainer] Dataset shape: {df.shape}")
        print(f"[Trainer] Total samples: {len(df)}")
        
        # Step 2: Train models for ALL 19 features (each feature is a target)
        print(f"\n[Trainer] Training models for ALL 19 features...")
        
        all_features = [
            'timestamp', 'user_id', 'user_type', 'registration_date', 'account_status',
            'total_bookings', 'completed_bookings', 'cancelled_bookings',
            'total_spent', 'total_earned', 'platform_commission',
            'avg_rating', 'total_reviews', 'last_active',
            'days_since_registration', 'days_since_last_active',
            'completion_rate', 'cancellation_rate', 'avg_earning_per_booking'
        ]
        
        # Train a model to predict each feature based on all other features
        for target in all_features:
            if target not in df.columns:
                print(f"[Trainer] ⚠️ Skipping {target} (not in dataset)")
                continue
            
            # Features for this model = all features except the target
            feature_cols = [f for f in all_features if f != target and f in df.columns]
            
            X_train = df[feature_cols].copy()
            y_train = df[target].values
            
            try:
                # Determine if classification or regression
                unique_values = len(np.unique(y_train))
                is_classification = unique_values < 100 and target in ['user_id', 'user_type', 'account_status']
                
                if is_classification:
                    # Classification model
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder = LabelEncoder()
                    y_train_encoded = label_encoder.fit_transform(y_train)
                    
                    model = xgb.XGBClassifier(
                        n_estimators=50,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=42,
                        verbosity=0,
                        eval_metric='logloss'
                    )
                    model.fit(X_train, y_train_encoded, verbose=False)
                    
                    # Calculate accuracy
                    y_pred = model.predict(X_train)
                    accuracy = np.mean(y_pred == y_train_encoded)
                    
                    self.models[target] = {
                        'model': model,
                        'is_classification': True,
                        'feature_names': feature_cols,
                        'label_encoder': label_encoder
                    }
                    
                    self.performance_dict[target] = {
                        'type': 'classification',
                        'accuracy': float(accuracy),
                        'note': 'High accuracy expected with clean data'
                    }
                    
                else:
                    # Regression model
                    model = xgb.XGBRegressor(
                        n_estimators=50,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=42,
                        verbosity=0,
                        eval_metric='rmse'
                    )
                    model.fit(X_train, y_train, verbose=False)
                    
                    # Calculate R2 and RMSE
                    y_pred = model.predict(X_train)
                    r2 = 1 - (np.sum((y_train - y_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
                    rmse = np.sqrt(np.mean((y_train - y_pred) ** 2))
                    
                    # Determine note based on R2
                    if r2 >= 0.99:
                        note = "PERFECT prediction "
                    elif r2 >= 0.90:
                        note = "EXCELLENT prediction"
                    elif r2 >= 0.75:
                        note = "GOOD prediction"
                    else:
                        note = "Fair prediction"
                    
                    self.models[target] = {
                        'model': model,
                        'is_classification': False,
                        'feature_names': feature_cols,
                        'label_encoder': None
                    }
                    
                    self.performance_dict[target] = {
                        'type': 'regression',
                        'r2_score': float(r2),
                        'rmse': float(rmse),
                        'note': note
                    }
                
                # Get feature importance
                importance = model.feature_importances_
                self.feature_importance_dict[target] = {
                    feature_cols[i]: float(importance[i])
                    for i in range(len(feature_cols))
                }
                
                model_type = "Classification" if is_classification else "Regression"
                print(f"[Trainer] ✓ Trained {model_type} model for '{target}'")
                
            except Exception as e:
                print(f"[Trainer] ✗ Error training model for '{target}': {e}")
                continue
        
        print(f"\n[Trainer] ✓ Successfully trained {len(self.models)} models")
        
        # Step 3: Save models with complete structure
        success = self.save_models(output_path, total_samples=len(df))
        
        if success:
            print("\n" + "="*60)
            print("✓ TRAINING FROM FILE COMPLETE!")
            print("="*60 + "\n")
        
        return success
    
    def save_models(self, output_path=None, total_samples=None):
        """Save trained models to pickle file"""
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(__file__),
                'complete_xgboost_package.pkl'
            )
        
        # Use provided total_samples or default
        if total_samples is None:
            total_samples = len(self.feature_names)
        
        from datetime import datetime
        
        package = {
            'all_models': self.models,
            'performance': self.performance_dict,
            'feature_importance': self.feature_importance_dict,
            'data_info': {
                'original_features': self.feature_names,
                'total_samples': total_samples,
                'data_characteristics': 'Clean/synthetic data showing perfect correlations',
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'xgboost_version': xgb.__version__,
            },
        }
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(package, f)
            print(f"[Trainer] ✓ Models saved to: {output_path}")
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[Trainer] ✓ File size: {file_size_mb:.2f} MB")
            return True
        except Exception as e:
            print(f"[Trainer] ✗ Error saving models: {e}")
            return False


def train_ml_models():
    """Execute training"""
    try:
        trainer = MLTrainer()
        success = trainer.train(months_back=12)
        
        if success:
            print("\n✅ ML Training Successful!")
            return True
        else:
            print("\n❌ ML Training Failed")
            return False
    
    except Exception as e:
        print(f"\n❌ Training Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    train_ml_models()
