# employer/models.py

from django.db import models
from django.contrib.auth.hashers import make_password, check_password
from django.utils import timezone
from decimal import Decimal
from django.conf import settings  



class Employer(models.Model):
    STATUS_CHOICES = [
        ('Active', 'Active'),
        ('Inactive', 'Inactive'),
        ('Suspended', 'Suspended'),
    ]
    
    VISIBILITY_CHOICES = [
        ('full', 'Show full address to workers'),
        ('partial', 'Show only city and state to workers'),
        ('hidden', 'Hide location from workers'),
    ]
    
    LANGUAGE_CHOICES = [
        ('english', 'English'),
        ('hindi', 'Hindi'),
        ('spanish', 'Spanish'),
        ('french', 'French'),
        ('german', 'German'),
    ]
    
    CURRENCY_CHOICES = [
        ('inr', 'Indian Rupee (₹)'),
        ('usd', 'US Dollar ($)'),
        ('eur', 'Euro (€)'),
        ('gbp', 'British Pound (£)'),
    ]
    
    TIMEZONE_CHOICES = [
        ('ist', 'India Standard Time (IST)'),
        ('est', 'Eastern Standard Time (EST)'),
        ('pst', 'Pacific Standard Time (PST)'),
        ('gmt', 'Greenwich Mean Time (GMT)'),
    ]
    
    DATE_FORMAT_CHOICES = [
        ('dd-mm-yyyy', 'DD-MM-YYYY'),
        ('mm-dd-yyyy', 'MM-DD-YYYY'),
        ('yyyy-mm-dd', 'YYYY-MM-DD'),
    ]

    # Basic Information
    employer_id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    email = models.EmailField(max_length=100, unique=True)
    phone = models.CharField(max_length=15, unique=True)
    profile_image = models.ImageField(upload_to='employer_profiles/', null=True, blank=True)
    business_document = models.FileField(upload_to='employer_documents/business/', null=True, blank=True)
    business_verified = models.BooleanField(default=False)
    aadhar_document = models.FileField(upload_to='employer_documents/aadhar/', null=True, blank=True)
    aadhar_verified = models.BooleanField(default=False)
    aadhar_upload_date = models.DateTimeField(null=True, blank=True)
    total_spent = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Additional Profile Fields (from settings page)
    company_name = models.CharField(max_length=200, null=True, blank=True)
    bio = models.TextField(null=True, blank=True)
    
    # Location Information
    address = models.TextField(null=True, blank=True)
    city = models.CharField(max_length=100, null=True, blank=True)
    state = models.CharField(max_length=100, null=True, blank=True)
    zip_code = models.CharField(max_length=20, null=True, blank=True)
    country = models.CharField(max_length=100, default='India')
    location_visibility = models.CharField(
        max_length=10, 
        choices=VISIBILITY_CHOICES, 
        default='partial'
    )
    
    # Preferences
    language = models.CharField(
        max_length=20, 
        choices=LANGUAGE_CHOICES, 
        default='english'
    )
    currency = models.CharField(
        max_length=10, 
        choices=CURRENCY_CHOICES, 
        default='inr'
    )
    timezone = models.CharField(
        max_length=10, 
        choices=TIMEZONE_CHOICES, 
        default='ist'
    )
    date_format = models.CharField(
        max_length=20, 
        choices=DATE_FORMAT_CHOICES, 
        default='dd-mm-yyyy'
    )
    
    # Privacy & Security Settings
    two_factor_auth = models.BooleanField(default=True)
    show_profile_to_workers = models.BooleanField(default=True)
    data_sharing_analytics = models.BooleanField(default=False)
    
    # Notification Preferences
    email_notifications = models.BooleanField(default=True)
    sms_notifications = models.BooleanField(default=False)
    push_notifications = models.BooleanField(default=True)
    marketing_communications = models.BooleanField(default=False)

    # Status and Timestamps
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Active')
    last_password_change = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)  

    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.company_name or 'No Company'})"
    
    class Meta:
        db_table = 'employer_detail_table'
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_active(self):
        return self.status == 'Active'


class EmployerLogin(models.Model):
    STATUS_CHOICES = [
        ('Active', 'Active'),
        ('Inactive', 'Inactive'),
    ]
    
    login_id = models.AutoField(primary_key=True)
    employer = models.ForeignKey(Employer, on_delete=models.CASCADE, related_name='logins')
    email = models.EmailField(max_length=100, unique=True)
    password = models.CharField(max_length=255)
    last_login = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Active')

    def __str__(self):
        return f"Employer Login - {self.email}"
    
    class Meta:
        db_table = 'employer_login_table'


class EmployerFavorite(models.Model):
    """Model for storing employer's favorite employees"""
    id = models.AutoField(primary_key=True)
    employer = models.ForeignKey(Employer, on_delete=models.CASCADE, related_name='favorites')
    employee = models.ForeignKey('employee.Employee', on_delete=models.CASCADE, related_name='favorited_by')
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Optional: Add notes or tags for the favorite
    notes = models.TextField(blank=True, null=True)
    
    class Meta:
        db_table = 'employer_favorite_table'
        unique_together = ['employer', 'employee']
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.employer.full_name} → {self.employee.full_name}"
    





class Payment(models.Model):
    """Model for tracking payments to employees"""
    PAYMENT_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('refunded', 'Refunded'),
    ]
    
    PAYMENT_METHOD_CHOICES = [
        ('razorpay', 'Razorpay'),
        ('cash', 'Cash'),
        ('bank_transfer', 'Bank Transfer'),
        ('upi', 'UPI'),
        ('card', 'Card'),
    ]
    
    payment_id = models.AutoField(primary_key=True)
    employer = models.ForeignKey('Employer', on_delete=models.CASCADE, related_name='payments')
    employee = models.ForeignKey('employee.Employee', on_delete=models.CASCADE, related_name='payments')
    job = models.ForeignKey('employee.JobRequest', on_delete=models.SET_NULL, null=True, blank=True, related_name='payments')
    
    # Payment details
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=10, default='INR')
    description = models.TextField(blank=True)
    payment_method = models.CharField(max_length=50, choices=PAYMENT_METHOD_CHOICES, default='razorpay')
    
    # Razorpay fields
    razorpay_order_id = models.CharField(max_length=255, null=True, blank=True)
    razorpay_payment_id = models.CharField(max_length=255, null=True, blank=True)
    razorpay_signature = models.CharField(max_length=255, null=True, blank=True)
    
    # Status and timestamps
    status = models.CharField(max_length=20, choices=PAYMENT_STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    payment_date = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'payment_table'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Payment #{self.payment_id} - ₹{self.amount} - {self.status}"
    
    @property
    def is_successful(self):
        return self.status == 'completed'
    
    @property
    def formatted_amount(self):
        return f"₹{self.amount}"


class PaymentInvoice(models.Model):
    """Model for storing payment invoices"""
    invoice_id = models.AutoField(primary_key=True)
    payment = models.ForeignKey(Payment, on_delete=models.CASCADE, related_name='invoices')
    invoice_number = models.CharField(max_length=100, unique=True)
    invoice_date = models.DateTimeField(auto_now_add=True)
    due_date = models.DateTimeField(null=True, blank=True)
    
    # File storage
    invoice_file = models.FileField(upload_to='payment_invoices/', null=True, blank=True)
    
    # Tax details
    subtotal = models.DecimalField(max_digits=10, decimal_places=2)
    tax_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'payment_invoice_table'
        ordering = ['-invoice_date']
    
    def __str__(self):
        return f"Invoice {self.invoice_number}"
    
    def generate_invoice_number(self):
        """Generate invoice number"""
        if not self.invoice_number:
            date_str = timezone.now().strftime('%Y%m%d')
            last_invoice = PaymentInvoice.objects.order_by('-invoice_id').first()
            last_num = last_invoice.invoice_id if last_invoice else 0
            self.invoice_number = f"INV-{date_str}-{last_num + 1:05d}"
        return self.invoice_number
    
    def save(self, *args, **kwargs):
        if not self.invoice_number:
            self.generate_invoice_number()
        super().save(*args, **kwargs)





class SiteReview(models.Model):
    """Model for site/platform reviews"""
    REVIEW_TYPE_CHOICES = [
        ('platform', 'Platform Experience'),
        ('payment', 'Payment Process'),
        ('support', 'Customer Support'),
        ('feature', 'Feature Request'),
        ('bug', 'Bug Report'),
    ]
    
    AREA_CHOICES = [
        ('ui_ux', 'Website/App Design & Usability'),
        ('payment', 'Payment System'),
        ('matching', 'Worker Matching System'),
        ('support', 'Customer Support'),
        ('safety', 'Safety & Security'),
    ]
    
    id = models.AutoField(primary_key=True)
    employer = models.ForeignKey(Employer, on_delete=models.CASCADE, related_name='site_reviews', null=True, blank=True)
    employee = models.ForeignKey('employee.Employee', on_delete=models.SET_NULL, null=True, blank=True)
    
    review_type = models.CharField(max_length=50, choices=REVIEW_TYPE_CHOICES)
    rating = models.IntegerField()  # 1-5
    title = models.CharField(max_length=200)
    review_text = models.TextField()
    areas = models.JSONField(default=list)  # List of selected areas
    recommendation = models.CharField(max_length=20, choices=[('yes', 'Yes'), ('maybe', 'Maybe'), ('no', 'No')])
    
    sentiment_score = models.FloatField(default=0.0)
    is_published = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'site_review_table'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Site Review by {self.employer.full_name} - {self.review_type}"


class Report(models.Model):
    """Model for reports"""
    REPORT_TYPE_CHOICES = [
        ('employee_issue', 'Employee Issue'),
        ('platform_issue', 'Platform Issue'),
        ('safety_concern', 'Safety Concern'),
        ('payment_dispute', 'Payment Dispute'),
        ('misconduct', 'Misconduct'),
        ('other', 'Other'),
    ]
    
    SEVERITY_CHOICES = [
        ('low', 'Low - Minor issue'),
        ('medium', 'Medium - Needs attention'),
        ('high', 'High - Urgent attention needed'),
        ('critical', 'Critical - Immediate action required'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('resolved', 'Resolved'),
        ('dismissed', 'Dismissed'),
    ]
    
    id = models.AutoField(primary_key=True)
    employer = models.ForeignKey(Employer, on_delete=models.CASCADE, related_name='reports')
    employee = models.ForeignKey('employee.Employee', on_delete=models.SET_NULL, null=True, blank=True)
    job = models.ForeignKey('employee.JobRequest', on_delete=models.SET_NULL, null=True, blank=True)
    
    report_type = models.CharField(max_length=50, choices=REPORT_TYPE_CHOICES)
    title = models.CharField(max_length=200)
    description = models.TextField()
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    resolution_preference = models.TextField(blank=True)
    contact_methods = models.JSONField(default=list)  # List of preferred contact methods
    share_with_employee = models.BooleanField(default=False)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    # FIX: Use settings.AUTH_USER_MODEL instead of 'auth.User'
    assigned_to = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    resolution_notes = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'report_table'
        ordering = ['-created_at']
    
    
    def __str__(self):
        return f"Report #{self.id} - {self.title}"


class EmployerNotification(models.Model):
    """Model for general employer notifications (System, Job Updates, etc.)"""
    NOTIFICATION_TYPES = [
        ('system', 'System Message'),
        ('profile', 'Profile Update'),
        ('security', 'Security Alert'),
        ('payment', 'Payment Update'),
        ('job', 'Job Update'),
    ]
    
    notification_id = models.AutoField(primary_key=True)
    employer = models.ForeignKey(Employer, on_delete=models.CASCADE, related_name='notifications')
    title = models.CharField(max_length=200)
    message = models.TextField()
    notification_type = models.CharField(max_length=20, choices=NOTIFICATION_TYPES, default='system')
    is_read = models.BooleanField(default=False)
    link = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'employer_notification_table'
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.title} - {self.employer.full_name}"
