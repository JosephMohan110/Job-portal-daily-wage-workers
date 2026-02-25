"""
XGBoost Training Service
Handles data preprocessing, model training, and package creation
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import logging
from io import BytesIO
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, r2_score
)

logger = logging.getLogger(__name__)


class XGBoostTrainingService:
    """Service for training XGBoost models from uploaded datasets"""
    
    def __init__(self):
        """Initialize the training service"""
        self.data = None
        self.models = {}
        self.feature_columns = []
        self.label_encoders = {}
        
    def train(self, file_obj, filename):
        """
        Train XGBoost models from uploaded file
        
        Args:
            file_obj: File object (file-like object or BytesIO)
            filename: Original filename
            
        Returns:
            dict: Training result with keys:
                - success: bool indicating if training was successful
                - package: dict containing trained models
                - analysis: dict with data analysis
                - summary: dict with training summary
        """
        try:
            logger.info(f"[Training Service] Starting training with file: {filename}")
            
            # Load data
            self.data = self._load_data(file_obj, filename)
            if self.data is None:
                return {
                    'success': False,
                    'package': None,
                    'analysis': {},
                    'summary': {'success': 0, 'total': 0, 'failed': 1}
                }
            
            # Analyze data
            analysis = self._analyze_data()
            
            # Clean data
            self._clean_data()
            
            # Train models
            summary = self._train_models()
            
            # Create package
            package = {
                'all_models': self.models,
                'feature_columns': self.feature_columns,
                'label_encoders': self.label_encoders,
                'training_date': datetime.now().isoformat(),
                'data_shape': self.data.shape,
            }
            
            logger.info(f"[Training Service] Training completed successfully")
            
            return {
                'success': True,
                'package': package,
                'analysis': analysis,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"[Training Service] Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'package': None,
                'analysis': {},
                'summary': {'success': 0, 'total': 0, 'failed': 1, 'error': str(e)}
            }
    
    def _load_data(self, file_obj, filename):
        """Load data from CSV or Excel file"""
        try:
            logger.info(f"[Training Service] Loading data from {filename}")
            
            # Determine file type
            if filename.endswith('.csv'):
                data = pd.read_csv(file_obj)
            elif filename.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file_obj)
            else:
                logger.error(f"[Training Service] Unsupported file type: {filename}")
                return None
            
            logger.info(f"[Training Service] Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
            
        except Exception as e:
            logger.error(f"[Training Service] Error loading data: {str(e)}")
            return None
    
    def _analyze_data(self):
        """Analyze the loaded data"""
        analysis = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': self.data.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': self.data.select_dtypes(include=['object']).columns.tolist(),
        }
        
        logger.info(f"[Training Service] Data analysis completed")
        return analysis
    
    def _clean_data(self):
        """Clean and preprocess the data"""
        try:
            logger.info("[Training Service] Starting data cleaning")
            
            # Remove rows with missing target values
            numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) > 0:
                # Use the first numeric column as target
                target_col = numeric_cols[-1]
                self.data = self.data.dropna(subset=[target_col])
                logger.info(f"[Training Service] Removed rows with missing {target_col}")
            
            # Handle missing values in numeric columns
            for col in numeric_cols:
                if self.data[col].isnull().any():
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                    logger.info(f"[Training Service] Filled missing values in {col}")
            
            # Handle missing values in categorical columns
            categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                if self.data[col].isnull().any():
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                    logger.info(f"[Training Service] Filled missing values in {col}")
            
            # Encode categorical features
            for col in categorical_cols:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"[Training Service] Encoded categorical column: {col}")
            
            logger.info("[Training Service] Data cleaning completed")
            
        except Exception as e:
            logger.error(f"[Training Service] Error during data cleaning: {str(e)}")
    
    def _train_models(self):
        """Train XGBoost models on the data"""
        try:
            logger.info("[Training Service] Starting model training")
            
            # Get numeric columns for training
            numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) < 2:
                logger.error("[Training Service] Insufficient numeric columns for training")
                return {'success': 0, 'total': 0, 'failed': 1}
            
            # Split features and targets
            target_col = numeric_cols[-1]
            self.feature_columns = numeric_cols[:-1]
            
            X = self.data[self.feature_columns].values
            y = self.data[target_col].values
            
            logger.info(f"[Training Service] Training features: {self.feature_columns}")
            logger.info(f"[Training Service] Target column: {target_col}")
            
            # Train XGBoost Regressor (for numeric targets)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train model
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store model
                self.models[target_col] = {
                    'model': model,
                    'type': 'regression',
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'accuracy_score': float(r2 * 100) if r2 >= 0 else 0.0,  # Convert R² to percentage
                }
                
                logger.info(
                    f"[Training Service] Trained {target_col} - "
                    f"RMSE: {rmse:.2f}, R²: {r2:.4f}"
                )
                
                success_count = 1
                
            except Exception as e:
                logger.error(f"[Training Service] Error training {target_col}: {str(e)}")
                success_count = 0
            
            # Summary
            summary = {
                'success': success_count,
                'total': 1,
                'failed': 1 - success_count,
                'training_date': datetime.now().isoformat(),
                'models_trained': list(self.models.keys()),
            }
            
            logger.info(f"[Training Service] Model training completed: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"[Training Service] Error during model training: {str(e)}")
            return {'success': 0, 'total': 0, 'failed': 1}
