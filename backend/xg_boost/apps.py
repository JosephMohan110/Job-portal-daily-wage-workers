from django.apps import AppConfig

class XgBoostConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'xg_boost'
    
    def ready(self):
        # Initialize ML predictor when app is ready
        try:
            from .predictor import predictor
            print("XGBoost app initialized successfully")
        except Exception as e:
            print(f"Error initializing XGBoost app: {e}")
