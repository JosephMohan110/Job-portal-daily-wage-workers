# employee/models.py

from django.db import models
from django.contrib.auth.hashers import make_password, check_password
from django.utils import timezone
from django.db.models import Avg, Count
import os

class Employee(models.Model):
    STATUS_CHOICES = [
        ('Active', 'Active'),
        ('Inactive', 'Inactive'),
        ('Suspended', 'Suspended'),
    ]
    
    PRIVACY_CHOICES = [
        ('full', 'Show full profile to employers'),
        ('partial', 'Show only basic information'),
        ('hidden', 'Hide profile from employers'),
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
    
    AVAILABILITY_CHOICES = [
        ('available', 'Available for Jobs'),
        ('busy', 'Busy - Limited Availability'),
        ('unavailable', 'Unavailable - On Break'),
    ]
    
    SERVICE_RADIUS_CHOICES = [
        (5, '5 km'),
        (10, '10 km'),
        (15, '15 km'),
        (20, '20 km'),
        (30, '30 km'),
    ]
    
    DATE_FORMAT_CHOICES = [
        ('dd-mm-yyyy', 'DD-MM-YYYY'),
        ('mm-dd-yyyy', 'MM-DD-YYYY'),
        ('yyyy-mm-dd', 'YYYY-MM-DD'),
    ]

    # Basic Information
    employee_id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    email = models.EmailField(max_length=100, unique=True)
    phone = models.CharField(max_length=15, unique=True, null=True, blank=True)
    profile_image = models.ImageField(upload_to='employee_profiles/', null=True, blank=True)
    cover_image = models.ImageField(upload_to='employee_covers/', null=True, blank=True)
    
    # Professional Information
    job_title = models.CharField(max_length=200, null=True, blank=True, verbose_name="Professional Title")
    location = models.CharField(max_length=255, null=True, blank=True, verbose_name="City/Area")
    bio = models.TextField(null=True, blank=True, help_text="Tell employers about your skills and experience")
    
    # Experience and Skills
    years_experience = models.IntegerField(default=0, verbose_name="Years of Experience")
    work_experience = models.TextField(null=True, blank=True)
    skills = models.TextField(null=True, blank=True, help_text="Comma separated skills")
    
    # Performance Metrics
    rating = models.FloatField(default=0)
    total_earnings = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_jobs_done = models.IntegerField(default=0)
    success_rate = models.IntegerField(default=0, verbose_name="Success Rate (%)")
    
    # Availability Information - ADDED FIELDS
    availability_status = models.CharField(max_length=100, default='Currently Available')
    response_time = models.CharField(max_length=50, default='Within 1 hour')
    working_hours = models.CharField(max_length=100, default='9:00 AM - 7:00 PM')
    service_area = models.CharField(max_length=100, default='Within 10 km radius')
    
    # Location Information
    address = models.TextField(null=True, blank=True)
    city = models.CharField(max_length=100, null=True, blank=True)
    state = models.CharField(max_length=100, null=True, blank=True)
    zip_code = models.CharField(max_length=20, null=True, blank=True)
    country = models.CharField(max_length=100, default='India')
    service_radius = models.IntegerField(choices=SERVICE_RADIUS_CHOICES, default=10)
    
    # Privacy & Security Settings
    two_factor_auth = models.BooleanField(default=True)
    show_profile_to_employer = models.BooleanField(default=True)
    share_work_history = models.BooleanField(default=True)
    privacy_level = models.CharField(
        max_length=10, 
        choices=PRIVACY_CHOICES, 
        default='partial'
    )
    data_sharing_analytics = models.BooleanField(default=False)
    
    # Notification Preferences
    job_alerts = models.BooleanField(default=True)
    message_notifications = models.BooleanField(default=True)
    payment_alerts = models.BooleanField(default=True)
    platform_updates = models.BooleanField(default=False)
    
    # Account Preferences
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
    availability = models.CharField(
        max_length=20, 
        choices=AVAILABILITY_CHOICES, 
        default='available'
    )
    
    # Security
    last_password_change = models.DateTimeField(null=True, blank=True)
    last_login_time = models.DateTimeField(null=True, blank=True)
    login_count = models.IntegerField(default=0)
    
    # Status and Timestamps
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Active')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deactivation_date = models.DateTimeField(null=True, blank=True)
    deactivation_reason = models.TextField(null=True, blank=True)
    
    # Documents
    aadhar_document = models.FileField(upload_to='employee_documents/aadhar/', null=True, blank=True)
    aadhar_verified = models.BooleanField(default=False)
    aadhar_upload_date = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name}"
    
    class Meta:
        db_table = 'employee_detail_table'
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_active(self):
        return self.status == 'Active'
    
    @property
    def rating_stars(self):
        """Convert rating to star representation"""
        full_stars = int(self.rating)
        half_star = self.rating - full_stars >= 0.5
        return full_stars, half_star
    
    @property
    def profile_stats(self):
        """Get profile statistics for display"""
        return {
            'jobs_completed': self.total_jobs_done,
            'rating': self.rating,
            'success_rate': self.success_rate,
            'years_experience': self.years_experience,
        }
    
    @property
    def skills_list(self):
        """Convert skills string to list"""
        if self.skills:
            return [skill.strip() for skill in self.skills.split(',') if skill.strip()]
        return []
    
    def save(self, *args, **kwargs):
        # Set default values if not provided
        if not self.availability_status:
            self.availability_status = 'Currently Available'
        if not self.response_time:
            self.response_time = 'Within 1 hour'
        if not self.working_hours:
            self.working_hours = '9:00 AM - 7:00 PM'
        if not self.service_area:
            self.service_area = f'Within {self.service_radius} km radius'
        super().save(*args, **kwargs)


class EmployeeExperience(models.Model):
    """Model for storing employee work experience"""
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='experiences')
    job_title = models.CharField(max_length=200)
    company = models.CharField(max_length=200)
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    currently_working = models.BooleanField(default=False)
    description = models.TextField(blank=True)
    
    class Meta:
        db_table = 'employee_experience_table'
        ordering = ['-start_date']
    
    def __str__(self):
        return f"{self.job_title} at {self.company}"
    
    @property
    def duration(self):
        if self.currently_working:
            return f"{self.start_date.strftime('%b %Y')} - Present"
        elif self.end_date:
            return f"{self.start_date.strftime('%b %Y')} - {self.end_date.strftime('%b %Y')}"
        return self.start_date.strftime('%b %Y')


class EmployeeCertificate(models.Model):
    """Model for storing employee certificates and licenses"""
    id = models.AutoField(primary_key=True)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='certificates')
    name = models.CharField(max_length=200)
    issuer = models.CharField(max_length=200)
    issue_date = models.DateField()
    expiry_date = models.DateField(null=True, blank=True)
    certificate_file = models.FileField(upload_to='employee_certificates/', null=True, blank=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'employee_certificate_table'
        ordering = ['-issue_date']
    
    def __str__(self):
        return f"{self.name} - {self.issuer}"
    
    @property
    def is_expired(self):
        if self.expiry_date:
            return self.expiry_date < timezone.now().date()
        return False
    
    @property
    def file_name(self):
        if self.certificate_file:
            return os.path.basename(self.certificate_file.name)
        return None


class EmployeePortfolio(models.Model):
    """Model for storing employee portfolio images"""
    id = models.AutoField(primary_key=True)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='portfolio_items')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='employee_portfolio/')
    upload_date = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'employee_portfolio_table'
        ordering = ['-upload_date']
    
    def __str__(self):
        return self.title
    
    @property
    def image_url(self):
        if self.image:
            return self.image.url
        return None


class EmployeeSkill(models.Model):
    """Model for storing employee skills"""
    id = models.AutoField(primary_key=True)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='employee_skills')
    skill_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'employee_skill_table'
        unique_together = ['employee', 'skill_name']
    
    def __str__(self):
        return self.skill_name


class EmployeeLogin(models.Model):
    STATUS_CHOICES = [
        ('Active', 'Active'),
        ('Inactive', 'Inactive'),
    ]
    
    login_id = models.AutoField(primary_key=True)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='login')
    email = models.EmailField(max_length=100, unique=True)
    password = models.CharField(max_length=255)
    last_login = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Active')
    failed_login_attempts = models.IntegerField(default=0)
    account_locked_until = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Employee Login - {self.email}"
    
    class Meta:
        db_table = 'employee_login_table'




class Review(models.Model):
    """
    Model for reviews posted by employers about employees.
    Now supports multiple reviews for different jobs.
    """
    employee = models.ForeignKey('Employee', on_delete=models.CASCADE, related_name='reviews')
    employer = models.ForeignKey('employer.Employer', on_delete=models.CASCADE, related_name='posted_reviews', null=True, blank=True)
    job = models.ForeignKey('JobRequest', on_delete=models.SET_NULL, null=True, blank=True, related_name='reviews')  # Link to specific job
    text = models.TextField()
    rating = models.FloatField(null=True, blank=True, help_text="Numerical rating (1-5 stars, optional)")
    sentiment_score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'employee_review_table'
        ordering = ['-created_at']
        # Allow multiple reviews for same employer-employee but different jobs
        unique_together = ['employer', 'employee', 'job']

    def __str__(self):
        return f"Review for {self.employee.full_name} - Job #{self.job.job_id if self.job else 'N/A'}"
    
    

class JobRequest(models.Model):
    """Model for job requests from employers to employees"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('accepted', 'Accepted'),
        ('rejected', 'Rejected'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]
    
    PRIORITY_CHOICES = [
        ('normal', 'Normal'),
        ('urgent', 'Urgent'),
        ('asap', 'ASAP'),
    ]
    
    job_id = models.AutoField(primary_key=True)
    employer = models.ForeignKey('employer.Employer', on_delete=models.CASCADE, related_name='job_requests')
    employee = models.ForeignKey('Employee', on_delete=models.CASCADE, related_name='job_requests')
    
    title = models.CharField(max_length=200)
    description = models.TextField()
    category = models.CharField(max_length=100, null=True, blank=True)
    
    # Job details
    proposed_date = models.DateField()
    proposed_time = models.TimeField(null=True, blank=True)
    estimated_duration = models.CharField(max_length=100, null=True, blank=True)
    budget = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    location = models.TextField()
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default='normal')
    
    # Communication
    employer_notes = models.TextField(null=True, blank=True)
    employee_notes = models.TextField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    accepted_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Location details
    address = models.TextField(null=True, blank=True)
    city = models.CharField(max_length=100, null=True, blank=True)
    state = models.CharField(max_length=100, null=True, blank=True)

    rejection_reason = models.TextField(null=True, blank=True)
    rejection_message_sent = models.BooleanField(default=False)

    
    class Meta:
        db_table = 'job_request_table'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Job #{self.job_id}: {self.title} - {self.get_status_display()}"
    
    @property
    def is_urgent(self):
        return self.priority in ['urgent', 'asap']
    
    @property
    def formatted_budget(self):
        if self.budget:
            return f"₹{self.budget}"
        return "Negotiable"
    
    @property
    def time_ago(self):
        from django.utils.timesince import timesince
        return timesince(self.created_at)
    
    @property
    def id(self):
        """Alias for job_id to match template expectations"""
        return self.job_id
    
    @property
    def job_title(self):
        """Alias for title field"""
        return self.title
    
    @property
    def job_description(self):
        """Alias for description field"""
        return self.description
    
    @property
    def job_category(self):
        """Alias for category field"""
        return self.category
    
    @property
    def duration(self):
        """Alias for estimated_duration field"""
        return self.estimated_duration
    
    @property
    def posted_date(self):
        """Alias for created_at field"""
        return self.created_at
    
    @property
    def experience_level(self):
        """Get experience level from employer data or default"""
        return getattr(self.employer, 'experience_level', 'Any Level')
    
    @property
    def proposal_count(self):
        """Get the count of proposals/applicants for this job"""
        return self.actions.filter(action_type='viewed').count() or 1
    
    @property
    def employer_name(self):
        """Get formatted employer name"""
        if self.employer:
            name = f"{self.employer.first_name} {self.employer.last_name}".strip()
            return name if name else self.employer.company_name or 'Unknown Employer'
        return 'Unknown Employer'
    
    @property
    def employer_avatar(self):
        """Get employer profile image URL"""
        if self.employer and self.employer.profile_image:
            return self.employer.profile_image.url
        return None
    
    @property
    def employer_rating(self):
        """Get employer rating from related data or default"""
        return getattr(self.employer, 'rating', 4.5)
    
    @property
    def employer_reviews(self):
        """Get employer review count or default"""
        return getattr(self.employer, 'total_reviews', 12)
    
    @property
    def employer_completed_jobs(self):
        """Get employer completed jobs count"""
        if self.employer:
            return self.employer.job_requests.filter(status='completed').count()
        return 0
    
    @property
    def paid_amount(self):
        """Calculate total paid amount from Payment objects"""
        from employer.models import Payment
        payments = Payment.objects.filter(job=self, status='completed')
        total = sum(p.amount for p in payments) if payments else 0
        return f"{total:.2f}"
    
    @property
    def remaining_amount(self):
        """Calculate remaining amount (budget - paid amount)"""
        if not self.budget:
            return "0.00"
        from employer.models import Payment
        payments = Payment.objects.filter(job=self, status='completed')
        paid = sum(p.amount for p in payments) if payments else 0
        remaining = float(self.budget) - float(paid)
        return f"{max(0, remaining):.2f}"


class JobAction(models.Model):
    """Model for tracking job request actions"""
    ACTION_CHOICES = [
        ('created', 'Created'),
        ('viewed', 'Viewed'),
        ('accepted', 'Accepted'),
        ('rejected', 'Rejected'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('rescheduled', 'Rescheduled'),
        ('message', 'Message Sent'),
    ]
    
    id = models.AutoField(primary_key=True)
    job = models.ForeignKey(JobRequest, on_delete=models.CASCADE, related_name='actions')
    employee = models.ForeignKey('Employee', on_delete=models.CASCADE, null=True, blank=True)
    employer = models.ForeignKey('employer.Employer', on_delete=models.CASCADE, null=True, blank=True)
    
    action_type = models.CharField(max_length=20, choices=ACTION_CHOICES)
    notes = models.TextField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'job_action_table'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.get_action_type_display()} - Job #{self.job.job_id}"
    



# # Add this to employee/models.py


class EmployeeAvailability(models.Model):
   """Model to track employee manual availability (unavailable days)"""
   availability_id = models.AutoField(primary_key=True)
   employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='manual_availability')
   date = models.DateField()
   is_available = models.BooleanField(default=True) # True=Available, False=Unavailable/Busy
   reason = models.CharField(max_length=255, null=True, blank=True)
   
   created_at = models.DateTimeField(auto_now_add=True)
   updated_at = models.DateTimeField(auto_now=True)
   
   class Meta:
       db_table = 'employee_availability_table'
       unique_together = ['employee', 'date']
       ordering = ['date']
       
   def __str__(self):
       return f"{self.employee.full_name} - {self.date} - {'Available' if self.is_available else 'Unavailable'}"


# class Refund(models.Model):
#     """Model for tracking refunds for cancelled bookings"""
#     REFUND_STATUS_CHOICES = [
#         ('pending', 'Pending'),
#         ('processing', 'Processing'),
#         ('completed', 'Completed'),
#         ('failed', 'Failed'),
#     ]
    
#     refund_id = models.AutoField(primary_key=True)
#     booking = models.ForeignKey(JobRequest, on_delete=models.CASCADE, related_name='refunds')
#     employer = models.ForeignKey('employer.Employer', on_delete=models.CASCADE)
#     employee = models.ForeignKey('Employee', on_delete=models.CASCADE)
    
#     # Refund details
#     amount = models.DecimalField(max_digits=10, decimal_places=2)
#     platform_fee = models.DecimalField(max_digits=10, decimal_places=2, default=0)
#     worker_earnings = models.DecimalField(max_digits=10, decimal_places=2, default=0)
#     reason = models.TextField()
    
#     # Status and timestamps
#     status = models.CharField(max_length=20, choices=REFUND_STATUS_CHOICES, default='pending')
#     created_at = models.DateTimeField(auto_now_add=True)
#     processed_at = models.DateTimeField(null=True, blank=True)
#     completed_at = models.DateTimeField(null=True, blank=True)
    
#     class Meta:
#         db_table = 'refund_table'
#         ordering = ['-created_at']
    
#     def __str__(self):
#         return f"Refund #{self.refund_id} - ₹{self.amount}"


class EmployeeNotification(models.Model):
    """Model for general employee notifications (System, Profile, etc.)"""
    NOTIFICATION_TYPES = [
        ('system', 'System Message'),
        ('profile', 'Profile Update'),
        ('security', 'Security Alert'),
        ('payment', 'Payment Update'),
        ('job', 'Job Update'),
    ]
    
    notification_id = models.AutoField(primary_key=True)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='notifications')
    title = models.CharField(max_length=200)
    message = models.TextField()
    notification_type = models.CharField(max_length=20, choices=NOTIFICATION_TYPES, default='system')
    is_read = models.BooleanField(default=False)
    link = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'employee_notification_table'
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.title} - {self.employee.full_name}"
