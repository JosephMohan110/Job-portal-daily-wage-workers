"""
Django management command to train XGBoost models
Usage: python manage.py train_ml_models
"""

from django.core.management.base import BaseCommand
from xg_boost.ml_trainer import train_ml_models

class Command(BaseCommand):
    help = 'Train XGBoost models on real platform data'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--months',
            type=int,
            default=12,
            help='Number of months of historical data to use for training (default: 12)'
        )
    
    def handle(self, *args, **options):
        months = options.get('months', 12)
        
        self.stdout.write(
            self.style.SUCCESS(
                f'\n🚀 Starting XGBoost Model Training (using {months} months of data)...\n'
            )
        )
        
        try:
            success = train_ml_models()
            
            if success:
                self.stdout.write(
                    self.style.SUCCESS(
                        '\n✅ ML Models trained successfully!\n'
                        'Models saved to: backend/xg_boost/complete_xgboost_package.pkl\n'
                        'You can now use real predictions in analytics_prediction.html\n'
                    )
                )
            else:
                self.stdout.write(
                    self.style.ERROR('\n❌ ML Training failed. Check logs above.\n')
                )
        
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'\n❌ Error: {str(e)}\n')
            )
