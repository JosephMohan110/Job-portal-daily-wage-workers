# admin_self/models.py
from django.db import models
from django.conf import settings
from django.utils import timezone
from decimal import Decimal
from django.db.models import Sum
from django.core.validators import FileExtensionValidator
import os

class Commission(models.Model):
    """Model for tracking platform commissions from payments"""
    COMMISSION_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('calculated', 'Calculated'),
        ('paid', 'Paid'),
        ('failed', 'Failed'),
    ]
    
    commission_id = models.AutoField(primary_key=True)
    payment = models.ForeignKey('employer.Payment', on_delete=models.CASCADE, related_name='commissions')
    employer = models.ForeignKey('employer.Employer', on_delete=models.CASCADE, related_name='commissions')
    employee = models.ForeignKey('employee.Employee', on_delete=models.CASCADE, related_name='commissions')
    
    # Commission details
    transaction_amount = models.DecimalField(max_digits=10, decimal_places=2)
    commission_rate = models.DecimalField(max_digits=5, decimal_places=4, default=0.0010)
    commission_amount = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    description = models.TextField(blank=True)
    
    # Status and timestamps
    status = models.CharField(max_length=20, choices=COMMISSION_STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    calculated_at = models.DateTimeField(null=True, blank=True)
    paid_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'commission_table'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Commission #{self.commission_id} - ₹{self.commission_amount}"
    
    def save(self, *args, **kwargs):
        # Auto-calculate commission amount if not set (0.10% = 0.0010)
        if not self.commission_amount and self.transaction_amount:
            self.commission_amount = self.transaction_amount * Decimal('0.0010')
        super().save(*args, **kwargs)


class Payout(models.Model):
    """Model for tracking worker payouts"""
    PAYOUT_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    PAYOUT_METHOD_CHOICES = [
        ('bank_transfer', 'Bank Transfer'),
        ('upi', 'UPI'),
        ('paypal', 'PayPal'),
        ('razorpay', 'Razorpay'),
    ]
    
    payout_id = models.AutoField(primary_key=True)
    employee = models.ForeignKey('employee.Employee', on_delete=models.CASCADE, related_name='payouts', null=True, blank=True)
    
    # Payout details
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    payout_method = models.CharField(max_length=20, choices=PAYOUT_METHOD_CHOICES, default='bank_transfer')
    reference_number = models.CharField(max_length=100, null=True, blank=True)
    description = models.TextField(blank=True)
    
    # Status and timestamps
    status = models.CharField(max_length=20, choices=PAYOUT_STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Payment method details
    bank_account_number = models.CharField(max_length=50, null=True, blank=True)
    bank_ifsc = models.CharField(max_length=20, null=True, blank=True)
    bank_name = models.CharField(max_length=100, null=True, blank=True)
    upi_id = models.CharField(max_length=100, null=True, blank=True)
    
    # Razorpay specific fields
    razorpay_payment_id = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    razorpay_order_id = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    razorpay_contact_id = models.CharField(max_length=100, null=True, blank=True)
    razorpay_fund_account_id = models.CharField(max_length=100, null=True, blank=True)
    razorpay_account_type = models.CharField(max_length=20, null=True, blank=True)  # 'bank_account' or 'upi'
    
    class Meta:
        db_table = 'payout_table'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Payout #{self.payout_id} - ₹{self.amount}"


class PlatformRevenue(models.Model):
    """Model for tracking platform revenue periods"""
    PERIOD_CHOICES = [
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
    ]
    
    revenue_id = models.AutoField(primary_key=True)
    period_type = models.CharField(max_length=20, choices=PERIOD_CHOICES, default='monthly')
    period_start = models.DateField()
    period_end = models.DateField()
    
    # Revenue calculations
    total_transactions = models.IntegerField(default=0)
    total_transaction_amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_commission = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_payouts = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    platform_balance = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    
    # Status
    is_finalized = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    finalized_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'platform_revenue_table'
        ordering = ['-period_start']
    
    def __str__(self):
        return f"Revenue {self.period_type.capitalize()} {self.period_start} to {self.period_end}"
    
    def calculate_revenue(self):
        """Calculate revenue for this period"""
        # Import here to avoid circular imports
        from employer.models import Payment
        
        # Get payments in this period
        payments = Payment.objects.filter(
            payment_date__date__gte=self.period_start,
            payment_date__date__lte=self.period_end,
            status='completed'
        )
        
        self.total_transactions = payments.count()
        self.total_transaction_amount = payments.aggregate(total=Sum('amount'))['total'] or Decimal('0')
        
        # Calculate commission (0.10% = 0.0010)
        self.total_commission = self.total_transaction_amount * Decimal('0.0010')
        
        # Get payouts in this period
        payouts = Payout.objects.filter(
            created_at__date__gte=self.period_start,
            created_at__date__lte=self.period_end,
            status='completed'
        )
        self.total_payouts = payouts.aggregate(total=Sum('amount'))['total'] or Decimal('0')
        
        # Calculate platform balance
        self.platform_balance = self.total_commission - self.total_payouts
        self.save()


class MLModel(models.Model):
    """Model for storing uploaded ML models (XGBoost only)"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending Review'),
        ('testing', 'Testing'),
        ('deployed', 'Deployed'),
        ('failed', 'Failed'),
        ('archived', 'Archived'),
    ]
    
    MODEL_TYPE_CHOICES = [
        ('churn', 'User Churn Prediction'),
        ('booking', 'Booking Success Prediction'),
        ('revenue', 'Revenue Forecast'),
        ('recommendation', 'Service Recommendation'),
        ('matching', 'Worker-Employer Matching'),
    ]
    
    model_id = models.AutoField(primary_key=True)
    model_name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=50, choices=MODEL_TYPE_CHOICES)
    version = models.CharField(max_length=20, default='1.0.0')
    description = models.TextField(blank=True)
    
    # Algorithm (always XGBoost)
    algorithm = models.CharField(max_length=50, default='XGBoost')
    
    # File storage
    model_file = models.FileField(
        upload_to='ml_models/xgboost/',
        validators=[FileExtensionValidator(allowed_extensions=['pkl', 'joblib', 'model'])]
    )
    file_size = models.IntegerField(default=0)  # in bytes
    features_file = models.FileField(
        upload_to='ml_models/features/',
        null=True,
        blank=True,
        validators=[FileExtensionValidator(allowed_extensions=['json', 'csv'])]
    )
    
    # Training info
    training_date = models.DateField(default=timezone.now)
    training_data_size = models.IntegerField(default=0)  # number of samples
    accuracy_score = models.FloatField(null=True, blank=True)
    precision_score = models.FloatField(null=True, blank=True)
    recall_score = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    is_active = models.BooleanField(default=False)
    
    # Upload info
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    deployed_at = models.DateTimeField(null=True, blank=True)
    
    # Model parameters
    n_estimators = models.IntegerField(default=100)
    max_depth = models.IntegerField(default=6)
    learning_rate = models.FloatField(default=0.3)
    min_child_weight = models.IntegerField(default=1)
    subsample = models.FloatField(default=1.0)
    colsample_bytree = models.FloatField(default=1.0)
    
    class Meta:
        db_table = 'ml_model_table'
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.model_name} v{self.version} ({self.model_type})"
    
    def save(self, *args, **kwargs):
        # Set algorithm to XGBoost
        self.algorithm = 'XGBoost'
        
        # Calculate file size if file exists
        if self.model_file:
            try:
                self.file_size = self.model_file.size
            except:
                pass
        
        super().save(*args, **kwargs)
    
    @property
    def file_size_mb(self):
        """Return file size in MB"""
        return round(self.file_size / (1024 * 1024), 2) if self.file_size else 0
    
    @property
    def status_display(self):
        """Get status display with appropriate class"""
        status_classes = {
            'pending': 'status-pending',
            'testing': 'status-pending',
            'deployed': 'status-deployed',
            'failed': 'status-failed',
            'archived': 'status-failed',
        }
        return {
            'text': self.get_status_display(),
            'class': status_classes.get(self.status, 'status-pending')
        }
    
    @property
    def filename(self):
        """Get filename from model_file"""
        if self.model_file:
            return os.path.basename(self.model_file.name)
        return None


class ModelTrainingData(models.Model):
    """Model for tracking training data used for ML models"""
    
    DATA_SOURCE_CHOICES = [
        ('user_data', 'User Data'),
        ('booking_data', 'Booking Data'),
        ('revenue_data', 'Revenue Data'),
        ('review_data', 'Review Data'),
        ('combined', 'Combined Data'),
    ]
    
    data_id = models.AutoField(primary_key=True)
    ml_model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='training_data')
    data_source = models.CharField(max_length=50, choices=DATA_SOURCE_CHOICES)
    data_file = models.FileField(upload_to='training_data/', null=True, blank=True)
    total_samples = models.IntegerField(default=0)
    total_features = models.IntegerField(default=0)
    
    # Data collection period
    period_start = models.DateField()
    period_end = models.DateField()
    
    # Preprocessing info
    preprocessing_steps = models.JSONField(default=list, blank=True)
    feature_columns = models.JSONField(default=list, blank=True)
    target_column = models.CharField(max_length=100, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'model_training_data_table'
    
    def __str__(self):
        return f"Training Data for {self.ml_model.model_name}"


class ModelPerformance(models.Model):
    """Model for tracking model performance metrics"""
    
    METRIC_CHOICES = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1 Score'),
        ('auc_roc', 'AUC-ROC'),
        ('mse', 'Mean Squared Error'),
        ('mae', 'Mean Absolute Error'),
        ('rmse', 'Root Mean Squared Error'),
    ]
    
    SPLIT_CHOICES = [
        ('train', 'Training Set'),
        ('test', 'Testing Set'),
        ('validation', 'Validation Set'),
        ('cross_val', 'Cross Validation'),
    ]
    
    performance_id = models.AutoField(primary_key=True)
    ml_model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='performance_metrics')
    metric_name = models.CharField(max_length=50, choices=METRIC_CHOICES)
    split_type = models.CharField(max_length=50, choices=SPLIT_CHOICES)
    metric_value = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'model_performance_table'
    
    def __str__(self):
        return f"{self.metric_name}: {self.metric_value} for {self.ml_model.model_name}"


class DataCollectionLog(models.Model):
    """Model for logging data collection activities"""
    
    COLLECTION_TYPE_CHOICES = [
        ('automatic', 'Automatic Collection'),
        ('manual', 'Manual Export'),
        ('scheduled', 'Scheduled Export'),
    ]
    
    log_id = models.AutoField(primary_key=True)
    collection_type = models.CharField(max_length=50, choices=COLLECTION_TYPE_CHOICES)
    data_type = models.CharField(max_length=100)
    
    # Statistics
    records_collected = models.IntegerField(default=0)
    file_size = models.IntegerField(default=0)  # in bytes
    file_format = models.CharField(max_length=20, default='csv')
    
    # Status
    status = models.CharField(max_length=20, choices=[
        ('success', 'Success'),
        ('failed', 'Failed'),
        ('partial', 'Partial Success'),
    ], default='success')
    
    error_message = models.TextField(blank=True)
    
    # Use settings.AUTH_USER_MODEL
    collected_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    
    # Export file
    export_file = models.FileField(upload_to='data_exports/', null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'data_collection_log_table'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Data Collection: {self.data_type} - {self.status}"
    
    @property
    def duration_seconds(self):
        """Calculate duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0
    
    @property
    def file_size_mb(self):
        """Return file size in MB"""
        return round(self.file_size / (1024 * 1024), 2) if self.file_size else 0




