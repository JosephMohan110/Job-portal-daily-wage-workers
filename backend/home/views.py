
# #working of backend or logic  make here 

# # Create your views here.
# def index(request):       # name any can be give as home
#     return render(request,'index.html')   # function home is give, and html page want to run for that we give this line

# def register(request):   #using this function we will load te reg_client.html
#     return render(request,'register.html')

from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.hashers import make_password, check_password
import re

# Replace with:
from employee.models import Employee, EmployeeLogin
from employer.models import Employer, EmployerLogin  # Import from employer app


#***********************************************************

def contact_us(request):
    return render(request, 'home_page_html/contact_us.html')

#*******************************************************

def about_us(request):
    # Calculate real-time statistics
    from django.db.models import Count, Avg, Q
    from employee.models import Employee, JobRequest
    from employer.models import Employer, SiteReview

    # Skilled Workers: Count of active employees
    skilled_workers = Employee.objects.filter(status='Active').count()

    # Jobs Completed: Count of completed job requests
    jobs_completed = JobRequest.objects.filter(status='completed').count()

    # Cities Served: Count of distinct cities from employees and employers
    employee_cities = Employee.objects.filter(status='Active').exclude(city__isnull=True).exclude(city='').values('city').distinct().count()
    employer_cities = Employer.objects.filter(status='Active').exclude(city__isnull=True).exclude(city='').values('city').distinct().count()
    cities_served = max(employee_cities, employer_cities)  # Use the higher count

    # Average Rating: Average rating from site reviews
    avg_rating = SiteReview.objects.filter(is_published=True).aggregate(avg_rating=Avg('rating'))['avg_rating']
    if avg_rating:
        avg_rating = round(avg_rating, 1)
    else:
        avg_rating = 4.8  # Default fallback

    context = {
        'skilled_workers': skilled_workers,
        'jobs_completed': jobs_completed,
        'cities_served': cities_served,
        'avg_rating': avg_rating,
    }

    return render(request, 'home_page_html/about_us.html', context)

#*******************************************************

def chat_bot_icon(request):
    return render(request, 'chat_bot/chatbot.html')

#*******************************************************

from employer.models import SiteReview

def index(request):
    # Fetch published site reviews
    published_reviews = SiteReview.objects.filter(is_published=True).order_by('-created_at')[:5]
    
    context = {
        'reviews': published_reviews,
        'voice_assistant_enabled': True,
    }
    return render(request, 'home_page_html/index.html', context)

#***********************************************************

def privacy_terms(request):
    return render(request, 'home_page_html/privacy_terms.html')

#*****************************************************************


def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    # Accepts 10-digit Indian numbers or numbers with country code
    pattern = r'^(\+91[\-\s]?)?[0]?(91)?[6789]\d{9}$'
    return re.match(pattern, phone) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    
    if not re.search(r'[A-Z]', password):
        return "Password must contain at least one uppercase letter."
    
    if not re.search(r'[a-z]', password):
        return "Password must contain at least one lowercase letter."
    
    if not re.search(r'[0-9]', password):
        return "Password must contain at least one number."
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return "Password must contain at least one special character."
    
    return None

def register(request):
    if request.method == 'POST':
        # Get form data
        user_type = request.POST.get('user_type', 'employee')
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        email = request.POST.get('email', '').strip().lower()
        phone = request.POST.get('phone', '').strip()
        password1 = request.POST.get('password1', '')
        password2 = request.POST.get('password2', '')
        skills = request.POST.get('skills', '').strip()
        company_name = request.POST.get('company_name', '').strip()
        location = request.POST.get('location', '').strip()
        profile_photo = request.FILES.get('profile_photo')
        terms = request.POST.get('terms')
        
        # Store form data in session for persistence
        request.session['register_form_data'] = {
            'user_type': user_type,
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'phone': phone,
            'skills': skills,
            'company_name': company_name,
            'location': location,
        }
        
        # Validation
        errors = {}
        
        # Required fields validation
        if not first_name:
            errors['first_name'] = "First name is required."
        elif len(first_name) < 2:
            errors['first_name'] = "First name must be at least 2 characters long."
        
        if not last_name:
            errors['last_name'] = "Last name is required."
        elif len(last_name) < 2:
            errors['last_name'] = "Last name must be at least 2 characters long."
        
        # Email validation
        if not email:
            errors['email'] = "Email is required."
        elif not validate_email(email):
            errors['email'] = "Please enter a valid email address."
        
        # Phone validation
        if not phone:
            errors['phone'] = "Phone number is required."
        elif not validate_phone(phone):
            errors['phone'] = "Please enter a valid 10-digit Indian phone number."
        
        # Password validation
        if not password1:
            errors['password1'] = "Password is required."
        else:
            password_error = validate_password(password1)
            if password_error:
                errors['password1'] = password_error
        
        if not password2:
            errors['password2'] = "Please confirm your password."
        elif password1 != password2:
            errors['password2'] = "Passwords do not match."
        
        # Location validation
        if not location:
            errors['location'] = "Location is required."
        
        # User type specific validations
        if user_type == 'employee':
            if not skills:
                errors['skills'] = "Skills are required for employees."
        else:  # employer
            if not company_name:
                errors['company_name'] = "Company name is required for employers."
        
        # Terms and conditions
        if not terms:
            errors['terms'] = "You must agree to the Terms & Conditions and Privacy Policy."
        
        # Check if email already exists
        if not errors.get('email'):
            if Employer.objects.filter(email=email).exists() or Employee.objects.filter(email=email).exists():
                errors['email'] = "Email already exists. Please use a different email."
        
        # Check if phone already exists
        if not errors.get('phone'):
            if Employer.objects.filter(phone=phone).exists() or Employee.objects.filter(phone=phone).exists():
                errors['phone'] = "Phone number already exists. Please use a different phone number."
        
        # Profile photo validation
        if profile_photo:
            # Validate file size (max 5MB)
            if profile_photo.size > 5 * 1024 * 1024:
                errors['profile_photo'] = "Image size should be less than 5MB."
            
            # Validate file type
            allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif']
            if profile_photo.content_type not in allowed_types:
                errors['profile_photo'] = "Only JPEG, PNG, JPG, and GIF images are allowed."
        
        if errors:
            # Store errors in session
            request.session['register_errors'] = errors
            
            # Set error class for fields with errors
            error_fields = {key: 'error-field' for key in errors.keys()}
            request.session['error_fields'] = error_fields
            
            return redirect('register')
        
        try:
            if user_type == 'employer':
                # Create Employer
                employer = Employer(
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    phone=phone,
                    company_name=company_name,
                    address=location,
                    city=location.split(',')[0] if location else '',
                )
                
                if profile_photo:
                    employer.profile_image = profile_photo
                
                employer.save()
                
                # Create Employer Login
                employer_login = EmployerLogin(
                    employer=employer,
                    email=email,
                    password=make_password(password1),
                    status='Active'
                )
                employer_login.save()
                
                messages.success(request, 'Employer account created successfully! Please login.')
                
            else:  # employee
                # Create Employee
                employee = Employee(
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    phone=phone,
                    skills=skills
                )
                
                if profile_photo:
                    employee.profile_image = profile_photo
                
                employee.save()
                
                # Create Employee Login
                employee_login = EmployeeLogin(
                    employee=employee,
                    email=email,
                    password=make_password(password1),
                    status='Active'
                )
                employee_login.save()
                
                messages.success(request, 'Employee account created successfully! Please login.')
            
            # Clear stored form data and errors
            if 'register_form_data' in request.session:
                del request.session['register_form_data']
            if 'register_errors' in request.session:
                del request.session['register_errors']
            if 'error_fields' in request.session:
                del request.session['error_fields']
            
            return redirect('index')
            
        except Exception as e:
            messages.error(request, f'Error creating account: {str(e)}')
            return redirect('register')
    
    # GET request - check for stored form data
    form_data = request.session.get('register_form_data', {})
    errors = request.session.get('register_errors', {})
    error_fields = request.session.get('error_fields', {})
    
    # Clear stored errors after displaying
    if 'register_errors' in request.session:
        del request.session['register_errors']
    if 'error_fields' in request.session:
        del request.session['error_fields']
    
    context = {
        'form_data': form_data,
        'errors': errors,
        'error_fields': error_fields,
    }
    
    return render(request, 'register.html', context)


#************************************************************


def employee_login(request):
    if request.method == 'POST':
        email = request.POST.get('email', '').strip().lower()
        password = request.POST.get('password', '')
        
        # Validation
        if not email or not password:
            messages.error(request, "Please enter both email and password.")
            return render(request, 'home_page_html/index.html')
        
        try:
            # Check if employee exists
            employee_login = EmployeeLogin.objects.get(email=email)
            
            # Verify password
            if check_password(password, employee_login.password):
                # Login successful - store in session
                request.session['employee_id'] = employee_login.employee.employee_id
                request.session['employee_name'] = f"{employee_login.employee.first_name} {employee_login.employee.last_name}"
                request.session['employee_email'] = employee_login.email
                messages.success(request, f"Welcome back, {employee_login.employee.first_name}!")
                return redirect('employee_dashboard')
            else:
                messages.error(request, "Invalid password.")
        except EmployeeLogin.DoesNotExist:
            messages.error(request, "No employee account found with this email.")
        
        return render(request, 'home_page_html/index.html')
    
    return redirect('index')


#*****************************************************************

def employer_login(request):
    if request.method == 'POST':
        email = request.POST.get('email', '').strip().lower()
        password = request.POST.get('password', '')
        
        # Validation
        if not email or not password:
            messages.error(request, "Please enter both email and password.")
            return render(request, 'home_page_html/index.html')
        
        try:
            # Check if employer exists
            employer_login = EmployerLogin.objects.get(email=email)
            
            # Check if account is active
            if employer_login.status != 'Active':
                messages.error(request, "Your account is deactivated. Please contact support to reactivate.")
                return render(request, 'home_page_html/index.html')
            
            # Verify password
            if check_password(password, employer_login.password):
                # Also check if the associated employer is active
                if employer_login.employer.status != 'Active':
                    messages.error(request, "Your account is deactivated. Please contact support to reactivate.")
                    return render(request, 'home_page_html/index.html')
                
                # Login successful - store in session
                request.session['employer_id'] = employer_login.employer.employer_id
                request.session['employer_name'] = f"{employer_login.employer.first_name} {employer_login.employer.last_name}"
                request.session['employer_email'] = employer_login.email
                messages.success(request, f"Welcome back, {employer_login.employer.first_name}!")
                return redirect('employer_dashboard')
            else:
                messages.error(request, "Invalid password.")
        except EmployerLogin.DoesNotExist:
            messages.error(request, "No employer account found with this email.")
        
        return render(request, 'home_page_html/index.html')
    
    return redirect('index')


#***************************************************


def logout(request):
    # Clear session
    request.session.flush()
    messages.success(request, "Logged out successfully.")
    return redirect('index')



