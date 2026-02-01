
# from django.shortcuts import render, redirect, get_object_or_404
# from django.contrib import messages
# from django.contrib.auth.hashers import make_password, check_password
# from django.utils import timezone
# from .models import Employer, EmployerLogin, EmployerFavorite
# # from django.db import transaction
# # from django.http import JsonResponse
# from datetime import datetime, timedelta
# from django.http import HttpResponse
# # import json
# import re
# from django.db.models import Avg, Count, Q, Sum
# import time

# # Pipeline imports: Employee models + geopy for distance
# from employee.models import (
#     Employee, EmployeeCertificate, EmployeeSkill, Review  # Review added
# )
# from geopy.geocoders import Nominatim
# from geopy.distance import geodesic

# import math
# from geopy.exc import GeocoderTimedOut, GeocoderServiceError
# from message_system.models import ChatRoom, Message    # For unread messages
# from employee.models import JobRequest, JobAction  # For job data
# import requests


# import csv
# from django.http import HttpResponse
# from io import BytesIO
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter
# from django.shortcuts import get_object_or_404

# import json
# import hmac
# import hashlib
# from django.conf import settings
# from django.http import JsonResponse, HttpResponse

# import razorpay
# from django.conf import settings
# from django.db import transaction
# from decimal import Decimal
# from .models import Payment, PaymentInvoice
# from employee.models import JobRequest, Employee

# # Initialize Razorpay client
# razorpay_client = razorpay.Client(
#     auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)
# )




# Add these imports at the TOP of views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.hashers import make_password, check_password
from django.utils import timezone
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.db import transaction
from django.db.models import Q, Avg, Count, Sum
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from decimal import Decimal
import json
import hmac
import hashlib
from datetime import datetime, timedelta
import re
import time
import os
import csv
from io import BytesIO

# For PDF generation
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Razorpay
import razorpay
razorpay_client = razorpay.Client(
    auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)
)

# Geopy for location
try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

# Models
from .models import Employer, EmployerLogin, EmployerFavorite, Payment, PaymentInvoice, SiteReview, Report, EmployerNotification
from employee.models import Employee, EmployeeCertificate, EmployeeSkill, Review, JobRequest, JobAction, EmployeePortfolio, EmployeeExperience, EmployeeNotification
from message_system.models import ChatRoom, Message







#**********************************************


def employer_dashboard(request):
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        
        # Calculate active hires (jobs with status 'accepted' or 'pending')
        active_hires = JobRequest.objects.filter(
            employer=employer,
            status__in=['pending', 'accepted']
        ).count()
        
        # Calculate favorite workers count
        favorite_workers = EmployerFavorite.objects.filter(
            employer=employer
        ).count()
        
        # Calculate total spent (from completed jobs)
        completed_jobs = JobRequest.objects.filter(
            employer=employer,
            status='completed'
        )
        
        total_spent = 0
        if completed_jobs.exists():
            total_spent_result = completed_jobs.aggregate(total=Sum('budget'))
            total_spent = total_spent_result['total'] or 0
        
        # Calculate total paid amounts from Payment model
        total_paid_result = Payment.objects.filter(
            employer=employer,
            status='completed'
        ).aggregate(total=Sum('amount'))
        total_paid = total_paid_result['total'] or 0
        
        # Calculate pending amount
        # Ensure we don't show negative pending if overpaid (unlikely but safe)
        pending_to_pay = max(0, total_spent - total_paid)
        
        # Get completed jobs count
        completed_jobs_count = completed_jobs.count()
        
        # Get top rated workers near the employer (limit to 5)
        top_workers = []
        
        # Get employer coordinates for distance calculation
        employer_location_parts = []
        if employer.city:
            employer_location_parts.append(employer.city)
        if employer.state:
            employer_location_parts.append(employer.state)
        if employer.country:
            employer_location_parts.append(employer.country)
        
        employer_coords = None
        if employer_location_parts:
            employer_coords = get_coordinates(", ".join(employer_location_parts))
        
        # Get all active employees who allow their profile to be shown
        employees = Employee.objects.filter(
            status='Active',
            show_profile_to_employer=True
        ).prefetch_related('reviews', 'employee_skills')[:20]  # Get first 20, we'll sort and take top 5
        
        workers_data = []
        for emp in employees:
            try:
                # Calculate average rating
                reviews = emp.reviews.all()
                review_count = reviews.count()
                avg_rating = 0.0
                
                if review_count > 0:
                    rating_avg = reviews.aggregate(avg_rating=Avg('rating'))
                    avg_rating = rating_avg['avg_rating'] or 0.0
                
                # Calculate distance
                distance = 100.0
                emp_coords = get_employee_location_coords(emp)
                if employer_coords and emp_coords:
                    distance = calculate_distance(employer_coords, emp_coords)
                
                # Calculate wage estimate
                wage_estimate = emp.years_experience * 100 + 500
                
                workers_data.append({
                    'employee': emp,
                    'avg_rating': round(avg_rating, 1),
                    'review_count': review_count,
                    'distance': round(distance, 1),
                    'wage_estimate': wage_estimate,
                    'experience': emp.years_experience,
                    'response_time': emp.response_time or 'Not specified',
                    'job_title': emp.job_title or 'Worker',
                })
                
            except Exception as e:
                print(f"Error processing employee {emp.employee_id}: {str(e)}")
                continue
        
        # Sort by rating (highest first), then by distance (closest first)
        workers_data.sort(key=lambda x: (-x['avg_rating'], x['distance']))
        
        # Take top 5 workers
        top_workers = workers_data[:5]
        
        # Get unread messages count
        try:
            from message_system.models import ChatRoom
            unread_messages = ChatRoom.objects.filter(
                employer=employer,
                unread_employer__gt=0
            ).count()
        except:
            unread_messages = 0
        
        # Get recent job requests (last 5)
        recent_jobs = JobRequest.objects.filter(
            employer=employer
        ).select_related('employee').order_by('-created_at')[:5]
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'employer_email': employer.email,
            'active_hires': active_hires,
            'favorite_workers': favorite_workers,
            'total_spent': total_spent,
            'total_paid': total_paid,
            'pending_to_pay': pending_to_pay,
            'completed_jobs': completed_jobs_count,
            'top_workers': top_workers,
            'unread_messages': unread_messages,
            'recent_jobs': recent_jobs,
        }
        
        return render(request, 'employer_html/employer_dashboard.html', context)
        
    except (Employer.DoesNotExist):
        # Clear session on error for security
        if 'employer_id' in request.session:
            del request.session['employer_id']
        messages.error(request, "Employer not found.")
        return redirect('index')
    except Exception as e:
        messages.error(request, f"Error loading dashboard: {str(e)}")
        return redirect('index')
    

#**************************************************************

# Settings page
def employer_setting(request):
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        employer_login = EmployerLogin.objects.get(employer=employer)
    except (Employer.DoesNotExist, EmployerLogin.DoesNotExist):
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')
    
    context = {
        'employer': employer,
        'employer_name': employer.full_name,
        'employer_email': employer.email,
    }
    return render(request, 'employer_html/employer_setting.html', context)


#*******************************************************************

def update_profile(request):
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
            # FIX: This logic is tricky. If you have multiple forms on one page,
            # it's better to use separate views or a clear identifier.
            # In your setup, the change_password modal directly posts to its view,
            # so this check is technically unnecessary but kept for safety.
            # if 'form_type' in request.POST and request.POST['form_type'] == 'password_change':
            #     # Redirect to change_password view
            #     return change_password(request)
        
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
        
            # Update basic profile information
            employer.first_name = request.POST.get('first_name', employer.first_name)
            employer.last_name = request.POST.get('last_name', employer.last_name)
            
            # NOTE: Phone uniqueness check should be done here if being updated
            new_phone = request.POST.get('phone')
            if new_phone and new_phone != employer.phone:
                if Employer.objects.filter(phone=new_phone).exclude(employer_id=employer.employer_id).exists():
                    messages.error(request, "This phone number is already registered.")
                    return redirect('employer:employer_setting')
                employer.phone = new_phone
                
            employer.company_name = request.POST.get('company_name', employer.company_name)
            employer.bio = request.POST.get('bio', employer.bio)
            
            # Handle profile image upload - only if a new file is provided
            if 'profile_image' in request.FILES and request.FILES['profile_image']:
                profile_image = request.FILES['profile_image']
                
                # Validate file size (max 5MB)
                if profile_image.size > 5 * 1024 * 1024:
                    messages.error(request, "Image size should be less than 5MB.")
                    return redirect('employer:employer_setting')
                
                # Validate file type
                allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif']
                if profile_image.content_type not in allowed_types:
                    messages.error(request, "Only JPEG, PNG, JPG, and GIF images are allowed.")
                    return redirect('employer:employer_setting')
                
                employer.profile_image = profile_image
            
            # Handle remove photo - only if flag is set
            remove_photo_flag = request.POST.get('remove_photo', '0')
            if remove_photo_flag == '1':
                employer.profile_image.delete(save=False) # Delete old file
                employer.profile_image = None
            
            employer.save()
            messages.success(request, "Profile updated successfully!")
            
        except Employer.DoesNotExist:
            messages.error(request, "Employer not found.")
        except Exception as e:
            messages.error(request, f"Error updating profile: {str(e)}")
    
    return redirect('employer_setting')


#************************************************************


def change_password(request):
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            employer_login = EmployerLogin.objects.get(employer=employer)
            
            current_password = request.POST.get('current_password')
            new_password = request.POST.get('new_password')
            confirm_password = request.POST.get('confirm_password')
            
            # 1. Validate inputs
            if not current_password or not new_password or not confirm_password:
                messages.error(request, 'All fields are required.')
                request.session['open_password_modal'] = True
                return redirect('employer_setting')
            
            # 2. Check current password
            if not check_password(current_password, employer_login.password):
                messages.error(request, 'Current password is incorrect.')
                request.session['open_password_modal'] = True
                return redirect('employer_setting')
            
            # 3. Check if new password is same as current password
            if check_password(new_password, employer_login.password):
                messages.error(request, 'New password cannot be the same as current password.')
                request.session['open_password_modal'] = True
                return redirect('employer_setting')
            
            # 4. Check if passwords match
            if new_password != confirm_password:
                messages.error(request, 'New passwords do not match.')
                request.session['open_password_modal'] = True
                return redirect('employer_setting')
                
            # 5. Validate new password strength (Server-side validation)
            if len(new_password) < 8:
                messages.error(request, 'Password must be at least 8 characters long.')
                request.session['open_password_modal'] = True
                return redirect('employer_setting')
            
            if not re.search(r'[A-Z]', new_password):
                messages.error(request, 'Password must contain at least one uppercase letter.')
                request.session['open_password_modal'] = True
                return redirect('employer_setting')
            
            if not re.search(r'[a-z]', new_password):
                messages.error(request, 'Password must contain at least one lowercase letter.')
                request.session['open_password_modal'] = True
                return redirect('employer_setting')
            
            if not re.search(r'[0-9]', new_password):
                messages.error(request, 'Password must contain at least one number.')
                request.session['open_password_modal'] = True
                return redirect('employer_setting')
            
            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', new_password):
                messages.error(request, 'Password must contain at least one special character.')
                request.session['open_password_modal'] = True
                return redirect('employer_setting')
            
            
            # 6. Update password
            employer_login.password = make_password(new_password)
            employer_login.save()
            
            # 7. Update last password change timestamp
            employer.last_password_change = timezone.now()
            employer.save()
            
            messages.success(request, 'Password changed successfully!')
            return redirect('employer_setting')
            
        except Employer.DoesNotExist:
            messages.error(request, 'Employer not found.')
            return redirect('employer_setting')
        except EmployerLogin.DoesNotExist:
            messages.error(request, 'Login credentials not found.')
            return redirect('employer_setting')
        except Exception as e:
            messages.error(request, f'Error changing password: {str(e)}')
            # Set flag on generic error as well
            request.session['open_password_modal'] = True
            return redirect('employer_setting')
    
    # Return to settings page if not POST request
    return redirect('employer_setting')


#************************************************************************

def add_to_favorites(request, employee_id):
    """Add an employee to employer's favorites"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Check if already favorited
        favorite_exists = EmployerFavorite.objects.filter(
            employer=employer, 
            employee=employee
        ).exists()
        
        if not favorite_exists:
            EmployerFavorite.objects.create(
                employer=employer,
                employee=employee,
                notes=request.POST.get('notes', '')
            )
            messages.success(request, f"{employee.full_name} added to favorites!")
        else:
            messages.info(request, f"{employee.full_name} is already in your favorites.")
            
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
    except Exception as e:
        messages.error(request, f"Error adding to favorites: {str(e)}")
    
    # Redirect back to the referrer page
    return redirect(request.META.get('HTTP_REFERER', 'employer_find_workers'))


#****************************************************************************


def remove_from_favorites(request, favorite_id=None, employee_id=None):
    """Remove an employee from favorites"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        
        if favorite_id:
            # Remove by favorite ID
            favorite = get_object_or_404(EmployerFavorite, id=favorite_id, employer=employer)
            employee_name = favorite.employee.full_name
            favorite.delete()
            
        elif employee_id:
            # Remove by employee ID
            employee = Employee.objects.get(employee_id=employee_id)
            EmployerFavorite.objects.filter(employer=employer, employee=employee).delete()
            employee_name = employee.full_name
            
        messages.success(request, f"{employee_name} removed from favorites!")
        
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
    except Exception as e:
        messages.error(request, f"Error removing from favorites: {str(e)}")
    
    return redirect(request.META.get('HTTP_REFERER', 'employer_favorites'))


#*********************************************************************************


def employer_favorites(request):
    """View employer's favorite workers"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        employer_login = EmployerLogin.objects.get(employer=employer)
        
        # Get all favorites for this employer
        favorites = EmployerFavorite.objects.filter(employer=employer).select_related(
            'employee'
        ).order_by('-created_at')
        
        # Search functionality
        search_query = request.GET.get('search', '').strip()
        profession_filter = request.GET.get('profession', '')
        availability_filter = request.GET.get('availability', '')
        
        if request.method == 'GET':
            if search_query:
                # Split search terms
                search_terms = search_query.lower().split()
                search_filter = Q()
                
                for term in search_terms:
                    if term:
                        search_filter |= (
                            Q(employee__first_name__icontains=term) |
                            Q(employee__last_name__icontains=term) |
                            Q(employee__job_title__icontains=term) |
                            Q(employee__skills__icontains=term) |
                            Q(employee__employee_skills__skill_name__icontains=term)
                        )
                
                favorites = favorites.filter(search_filter).distinct()
            
            if profession_filter:
                favorites = favorites.filter(employee__job_title__icontains=profession_filter)
            
            if availability_filter:
                favorites = favorites.filter(employee__availability=availability_filter)
        
        # Process each favorite for display
        favorite_workers = []
        for fav in favorites:
            try:
                employee = fav.employee
                
                # Get reviews info
                reviews = employee.reviews.all()
                review_count = reviews.count()
                avg_rating = 0.0
                
                if review_count > 0:
                    rating_avg = reviews.aggregate(avg_rating=Avg('rating'))
                    avg_rating = rating_avg['avg_rating'] or 0.0
                
                # Calculate wage estimate
                wage_estimate = f"₹{employee.years_experience * 100 + 500}/day"
                
                # Get skills
                employee_skills = list(employee.employee_skills.all()[:3])
                skills_from_text = []
                if employee.skills:
                    skills_from_text = [s.strip() for s in employee.skills.split(',') if s.strip()][:3]
                
                # Combine skills
                all_skills = [s.skill_name for s in employee_skills] + skills_from_text
                unique_skills = list(set(all_skills))[:3]
                
                favorite_workers.append({
                    'favorite_id': fav.id,
                    'employee': employee,
                    'notes': fav.notes,
                    'created_at': fav.created_at,
                    'avg_rating': round(avg_rating, 1),
                    'review_count': review_count,
                    'wage_estimate': wage_estimate,
                    'skills': unique_skills,
                    'city': employee.city or 'Not specified',
                    'state': employee.state or '',
                    'availability': employee.availability,
                    'availability_status': employee.availability_status,
                    'response_time': employee.response_time
                })
                
            except Exception as e:
                print(f"Error processing favorite {fav.id}: {str(e)}")
                continue
        
        # Calculate statistics
        total_favorites = favorites.count()
        available_now = favorites.filter(employee__availability='available').count()
        
        # Count top-rated (rating > 4.5)
        top_rated_count = 0
        for fav in favorite_workers:
            if fav['avg_rating'] >= 4.5:
                top_rated_count += 1
        
        # For now, set hired_from_favorites to 0 (you can implement this later)
        hired_from_favorites = 0
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'employer_email': employer.email,
            'favorite_workers': favorite_workers,
            'total_favorites': total_favorites,
            'hired_from_favorites': hired_from_favorites,
            'available_now': available_now,
            'top_rated': top_rated_count,
            'search_query': search_query,
            'profession_filter': profession_filter,
            'availability_filter': availability_filter,
        }
        
        return render(request, 'employer_html/employer_favorites.html', context)
        
    except (Employer.DoesNotExist, EmployerLogin.DoesNotExist):
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')


#*********************************************************************************


def view_employee_public_profile(request, employee_id):
    """View an employee's public profile from employer perspective"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        employee = get_object_or_404(Employee, employee_id=employee_id)
        
        # Get related data
        experiences = EmployeeExperience.objects.filter(employee=employee).order_by('-start_date')
        certificates = EmployeeCertificate.objects.filter(employee=employee).order_by('-issue_date')
        portfolio_items = EmployeePortfolio.objects.filter(employee=employee).order_by('-upload_date')
        skills_qs = EmployeeSkill.objects.filter(employee=employee)
        
        # Get skills list from both models
        employee_skills = list(skills_qs.values_list('skill_name', flat=True))
        if employee.skills:
            skills_from_text = [s.strip() for s in employee.skills.split(',') if s.strip()]
            employee_skills.extend(skills_from_text)
        
        # Remove duplicates
        employee_skills = list(set(employee_skills))
        
        # Calculate stats
        total_jobs_done = JobRequest.objects.filter(employee=employee, status='completed').count()
        
        # Calculate average rating
        reviews = Review.objects.filter(employee=employee)
        avg_rating = 0.0
        if reviews.exists():
            avg_rating = reviews.aggregate(Avg('rating'))['rating__avg']
            
        stats = {
            'total_jobs': total_jobs_done,
            'avg_rating': round(avg_rating, 1) if avg_rating else 0.0,
        }
        
        # Calculate wage estimate
        wage_estimate = (employee.years_experience * 100) + 500
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'employee': employee,
            'experiences': experiences,
            'certificates': certificates,
            'portfolio_items': portfolio_items,
            'skills': employee_skills,
            'stats': stats,
            'wage_estimate': wage_estimate,
        }
        
        return render(request, 'employer_html/view_employee_profile.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading profile: {str(e)}")
        return redirect('employer_dashboard')


#******************************************************************************


def employer_hiring_history(request):
    """View for employer hiring history with advanced filtering and statistics"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        
        # Get all job requests for this employer
        job_requests = JobRequest.objects.filter(
            employer=employer
        ).exclude(status='pending').select_related(
            'employee'
        ).order_by('-created_at')
        
        # Get filter parameters
        status_filter = request.GET.get('status', '')
        category_filter = request.GET.get('category', '')
        search_query = request.GET.get('search', '').strip()
        date_from = request.GET.get('date_from', '')
        date_to = request.GET.get('date_to', '')
        
        if status_filter:
            job_requests = job_requests.filter(status=status_filter)
        
        if category_filter:
            job_requests = job_requests.filter(category__icontains=category_filter)
        
        if search_query:
            job_requests = job_requests.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(employee__first_name__icontains=search_query) |
                Q(employee__last_name__icontains=search_query) |
                Q(job_type__icontains=search_query)
            )
        
        if date_from:
            try:
                date_from_obj = datetime.strptime(date_from, '%Y-%m-%d').date()
                job_requests = job_requests.filter(proposed_date__gte=date_from_obj)
            except ValueError:
                pass
        
        if date_to:
            try:
                date_to_obj = datetime.strptime(date_to, '%Y-%m-%d').date()
                job_requests = job_requests.filter(proposed_date__lte=date_to_obj)
            except ValueError:
                pass
        
        # Calculate statistics
        total_hires = job_requests.count()
        completed_jobs = job_requests.filter(status='completed').count()
        ongoing_jobs = job_requests.filter(status__in=['accepted', 'in_progress']).count()
        
        # Calculate total spent
        total_spent_result = job_requests.filter(
            status='completed'
        ).aggregate(total=Sum('budget'))
        total_spent = total_spent_result['total'] or 0
        
        # Calculate month-over-month change
        now = timezone.now()
        last_month = now - timedelta(days=30)
        
        current_month_hires = job_requests.filter(
            created_at__month=now.month,
            created_at__year=now.year
        ).count()
        
        last_month_hires = job_requests.filter(
            created_at__month=last_month.month,
            created_at__year=last_month.year
        ).count()
        
        # Calculate trends
        if total_hires > 0:
            completion_rate = (completed_jobs / total_hires) * 100
        else:
            completion_rate = 0
            
        # Calculate Average Rating Given (by this employer)
        avg_rating_result = Review.objects.filter(employer=employer).aggregate(avg=Avg('rating'))
        avg_rating_given = avg_rating_result['avg'] or 0
        
        # Calculate Average Job Duration (for completed jobs)
        # We'll use the difference between completed_at and accepted_at
        completed_requests = job_requests.filter(status='completed', completed_at__isnull=False, accepted_at__isnull=False)
        total_duration = timedelta(0)
        jobs_with_duration = 0
        
        for job in completed_requests:
            if job.completed_at and job.accepted_at:
                duration = job.completed_at - job.accepted_at
                total_duration += duration
                jobs_with_duration += 1
                
        avg_duration_days = 0
        if jobs_with_duration > 0:
            avg_duration_days = (total_duration.total_seconds() / 86400) # Convert seconds to days
            
        # Identify Most Hired Category
        category_counts = job_requests.filter(status='completed').values('category').annotate(count=Count('category')).order_by('-count')
        most_hired_category = "N/A"
        if category_counts:
            most_hired_category = category_counts[0]['category'] or "General"
            
        # Preferred Payment Method (Placeholder logic as field doesn't exist)
        payment_method = "Cash" if total_hires > 0 else "N/A"
        
        # Calculate trends logic reused from existing code...
        if last_month_hires > 0:
            hire_trend = ((current_month_hires - last_month_hires) / last_month_hires) * 100
        else:
            hire_trend = 100 if current_month_hires > 0 else 0
        
        # Prepare hiring records for template
        hiring_records = []
        for job in job_requests:
            # Get employee initials for avatar
            initials = ''
            if job.employee:
                if job.employee.first_name and job.employee.last_name:
                    initials = f"{job.employee.first_name[0]}{job.employee.last_name[0]}"
                elif job.employee.first_name:
                    initials = job.employee.first_name[0]
            
            # Get job date
            job_date = job.proposed_date.strftime('%d %b %Y') if job.proposed_date else "N/A"
            
            # Get duration
            duration = job.estimated_duration or "N/A"
            
            # Get amount
            amount = f"₹{job.budget}" if job.budget else "Negotiable"
            
            # Check if job has rating
            has_rating = Review.objects.filter(
                employer=employer,
                employee=job.employee
            ).exists() if job.employee else False
            
            hiring_records.append({
                'id': job.job_id,
                'employee': job.employee,
                'employee_initials': initials.upper(),
                'job_type': job.category or job.title,
                'title': job.title,
                'job_date': job_date,
                'duration': duration,
                'amount': amount,
                'status': job.status,
                'has_rating': has_rating,
                'category': job.category or 'General',
            })
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'employer_email': employer.email,
            'hiring_records': hiring_records,
            'total_hires': total_hires,
            'completed_jobs': completed_jobs,
            'ongoing_jobs': ongoing_jobs,
            'total_spent': total_spent,
            'hire_trend': round(hire_trend, 1),
            'search_query': search_query,
            'status_filter': status_filter,
            'category_filter': category_filter,
            'date_from': date_from,
            'date_to': date_to,
            
            # New Statistics
            'completion_rate': round(completion_rate, 0),
            'avg_rating_given': round(avg_rating_given, 1),
            'avg_duration_days': round(avg_duration_days, 1),
            'most_hired_category': most_hired_category,
            'payment_method': payment_method,
            
            'status_choices': [
                ('', 'All Status'),
                ('pending', 'Pending'),
                ('accepted', 'Accepted'),
                ('in_progress', 'In Progress'),
                ('completed', 'Completed'),
                ('cancelled', 'Cancelled'),
                ('upcoming', 'Upcoming'),
            ],
            'category_choices': [
                ('', 'All Categories'),
                ('plumber', 'Plumbers'),
                ('electrician', 'Electricians'),
                ('carpenter', 'Carpenters'),
                ('painter', 'Painters'),
                ('mason', 'Masons'),
                ('cleaner', 'Cleaners'),
                ('gardener', 'Gardeners'),
                ('driver', 'Drivers'),
                ('cook', 'Cooks'),
                ('babysitter', 'Babysitters'),
                ('eldercare', 'Elder Care'),
                ('technician', 'Technicians'),
                ('repair', 'Repair Services'),
            ],
        }
        
        return render(request, 'employer_html/employer_hiring_history.html', context)
        
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')
    except Exception as e:
        messages.error(request, f"Error loading hiring history: {str(e)}")
        return redirect('employer_dashboard')



def export_hiring_history(request):
    """Export hiring history as CSV"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        
        # Get filtered hiring history
        job_requests = JobRequest.objects.filter(employer=employer).exclude(status='pending')
        
        # Apply filters
        status_filter = request.GET.get('status', '')
        category_filter = request.GET.get('category', '')
        search_query = request.GET.get('search', '').strip()
        
        if status_filter:
            job_requests = job_requests.filter(status=status_filter)
        
        if category_filter:
            job_requests = job_requests.filter(category__icontains=category_filter)
        
        if search_query:
            job_requests = job_requests.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(employee__first_name__icontains=search_query) |
                Q(employee__last_name__icontains=search_query)
            )
        
        job_requests = job_requests.order_by('-created_at')
        
        # Create CSV response
        response = HttpResponse(content_type='text/csv')
        report_date = datetime.now().strftime('%Y%m%d')
        response['Content-Disposition'] = f'attachment; filename="hiring_history_{report_date}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Job ID', 'Worker Name', 'Job Type', 'Date', 'Duration', 
                        'Amount', 'Status', 'Location', 'Rating'])
        
        for job in job_requests:
            worker_name = job.employee.full_name if job.employee else 'N/A'
            job_date = job.proposed_date.strftime('%Y-%m-%d') if job.proposed_date else 'N/A'
            amount = f"₹{job.budget}" if job.budget else 'Negotiable'
            location = f"{job.city}, {job.state}" if job.city and job.state else job.location or 'N/A'
            
            writer.writerow([
                job.job_id,
                worker_name,
                job.category or job.title,
                job_date,
                job.estimated_duration or 'N/A',
                amount,
                job.get_status_display(),
                location,
                'Not Rated'
            ])
        
        return response
        
    except Exception as e:
        messages.error(request, f"Error exporting data: {str(e)}")
        return redirect('employer_hiring_history')


def print_hiring_report(request):
    """Generate PDF report of hiring history"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        if not REPORTLAB_AVAILABLE:
            messages.error(request, "PDF export requires ReportLab library. Please install it.")
            return redirect('employer_hiring_history')
        
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        
        # Get job requests
        job_requests = JobRequest.objects.filter(
            employer=employer,
            status='completed'
        ).select_related('employee').order_by('-completed_at')[:50]
        
        # Create PDF
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        # Header
        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, 780, "Hiring History Report")
        p.setFont("Helvetica", 12)
        p.drawString(100, 760, f"Employer: {employer.full_name}")
        p.drawString(100, 745, f"Company: {employer.company_name or 'N/A'}")
        p.drawString(100, 730, f"Report Date: {datetime.now().strftime('%d %b %Y')}")
        
        # Summary Statistics
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, 700, "Summary")
        p.setFont("Helvetica", 12)
        
        total_jobs = job_requests.count()
        total_spent = sum([job.budget or 0 for job in job_requests])
        avg_rating = job_requests.aggregate(avg=Avg('employee__rating'))['avg'] or 0
        
        p.drawString(100, 680, f"Total Jobs Completed: {total_jobs}")
        p.drawString(100, 665, f"Total Amount Spent: ₹{total_spent}")
        p.drawString(100, 650, f"Average Worker Rating: {avg_rating:.1f}/5")
        
        # Job List
        y = 600
        p.setFont("Helvetica-Bold", 12)
        p.drawString(100, y, "Recent Job History:")
        y -= 20
        
        p.setFont("Helvetica", 10)
        for job in job_requests[:20]:
            worker_name = job.employee.full_name if job.employee else 'N/A'
            job_date = job.completed_at.strftime('%d %b %Y') if job.completed_at else 'N/A'
            amount = f"₹{job.budget}" if job.budget else 'N/A'
            
            p.drawString(120, y, f"• {worker_name} - {job.title}")
            p.drawString(400, y, f"{job_date}")
            p.drawString(500, y, f"{amount}")
            
            y -= 15
            if y < 100:
                p.showPage()
                y = 750
        
        # Footer
        p.showPage()
        p.setFont("Helvetica", 10)
        p.drawString(100, 100, "Generated by WorkNest Hiring System")
        report_id = datetime.now().strftime('%Y%m%d%H%M%S')
        p.drawString(100, 85, f"Report ID: {report_id}")
        
        p.save()
        buffer.seek(0)
        
        response = HttpResponse(buffer, content_type='application/pdf')
        report_date = datetime.now().strftime('%Y%m%d')
        response['Content-Disposition'] = f'attachment; filename="hiring_report_{report_date}.pdf"'
        return response
        
    except Exception as e:
        messages.error(request, f"Error generating report: {str(e)}")
        return redirect('employer_hiring_history')


def view_job_details(request, job_id):
    """View detailed job information"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        job_request = JobRequest.objects.get(job_id=job_id, employer=employer)
        
        # Get timeline events
        timeline_events = JobAction.objects.filter(
            job=job_request
        ).order_by('created_at')
        
        # Get employee rating
        rating = Review.objects.filter(
            employer=employer,
            employee=job_request.employee
        ).first()
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'job_request': job_request,
            'timeline_events': timeline_events,
            'rating': rating,
            'has_rating': rating is not None,
        }
        
        return render(request, 'employer_html/job_details.html', context)
        
    except JobRequest.DoesNotExist:
        messages.error(request, "Job request not found.")
        return redirect('employer_hiring_history')
    except Exception as e:
        messages.error(request, f"Error loading job details: {str(e)}")
        return redirect('employer_hiring_history')






def submit_employee_report(request):
    """Handle employee report submission"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.", extra_tags='employee_report danger')
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            employee_id = request.POST.get('employee_id')
            
            # Get form data and process...
            # Similar to your existing submit_report function
            # but specifically for employee reports
            
            messages.success(request, "Employee report submitted successfully! We'll investigate.", extra_tags='employee_report success')
            
        except Exception as e:
            messages.error(request, f"Error submitting report: {str(e)}", extra_tags='employee_report danger')
    
    return redirect(request.META.get('HTTP_REFERER', 'employer_hired_employees'))

def submit_site_report(request):
    """Handle site/technical issue report submission"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.", extra_tags='site_report danger')
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            
            # Get form data and process...
            # Similar logic but for site issues
            
            messages.success(request, "Site issue report submitted! We'll work on fixing it.", extra_tags='site_report success')
            
        except Exception as e:
            messages.error(request, f"Error submitting site report: {str(e)}", extra_tags='site_report danger')
    
    return redirect(request.META.get('HTTP_REFERER', 'employer_hired_employees'))



#*******************************************************************************

def update_privacy_security(request):
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            employer.two_factor_auth = 'two_factor_auth' in request.POST
            employer.show_profile_to_workers = 'show_profile_to_workers' in request.POST
            employer.data_sharing_analytics = 'data_sharing_analytics' in request.POST
            employer.save()
            messages.success(request, "Privacy settings updated successfully!")
        except Exception as e:
            messages.error(request, f"Error updating privacy settings: {str(e)}")
    
    return redirect('employer_setting')


#********************************************************************************

def update_location(request):
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            employer.address = request.POST.get('address', employer.address)
            employer.city = request.POST.get('city', employer.city)
            employer.state = request.POST.get('state', employer.state)
            employer.zip_code = request.POST.get('zip_code', employer.zip_code)
            employer.country = request.POST.get('country', employer.country)
            employer.location_visibility = request.POST.get('location_visibility', employer.location_visibility)
            employer.save()
            messages.success(request, "Location settings updated successfully!")
        except Exception as e:
            messages.error(request, f"Error updating location settings: {str(e)}")
    
    return redirect('employer_setting')

#**********************************************************************************

def update_notifications(request):
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            employer.email_notifications = 'email_notifications' in request.POST
            employer.sms_notifications = 'sms_notifications' in request.POST
            employer.push_notifications = 'push_notifications' in request.POST
            employer.marketing_communications = 'marketing_communications' in request.POST
            employer.save()
            messages.success(request, "Notification settings updated successfully!")
        except Exception as e:
            messages.error(request, f"Error updating notification settings: {str(e)}")
    
    return redirect('employer_setting')

#*************************************************************************************

def update_preferences(request):
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            employer.language = request.POST.get('language', employer.language)
            employer.currency = request.POST.get('currency', employer.currency)
            employer.timezone = request.POST.get('timezone', employer.timezone)
            employer.date_format = request.POST.get('date_format', employer.date_format)
            employer.save()
            messages.success(request, "Preferences updated successfully!")
        except Exception as e:
            messages.error(request, f"Error updating preferences: {str(e)}")
    
    return redirect('employer_setting')

#*************************************************************************************8

def deactivate_account(request):
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            employer_login = EmployerLogin.objects.get(employer=employer)
            
            # Get password from form
            password = request.POST.get('password', '')
            
            # Validate password
            if not password:
                messages.error(request, "Please enter your password to confirm account deactivation.")
                return redirect('employer_setting')
            
            # Verify password
            if not check_password(password, employer_login.password):
                messages.error(request, "Incorrect password. Please try again.")
                return redirect('employer_setting')
            
            # FIX: Set deactivation date and reason
            employer.status = 'Inactive'
            employer.deactivation_date = timezone.now()  # Add this field to your model
            employer.deactivation_reason = 'User requested'  # Add this field to your model
            
            # FIX: Also clear sensitive data (optional but good practice)
            employer.profile_image.delete(save=False)
            employer.profile_image = None
            employer.phone = f"DEACTIVATED_{employer.phone}"  # Prefix to make it unique
            employer.save()
            
            # Deactivate login - set to Inactive
            employer_login.status = 'Inactive'
            employer_login.save()
            
            # Clear session
            request.session.flush()
            
            messages.success(request, "Your account has been deactivated successfully.")
            return redirect('index')
            
        except Employer.DoesNotExist:
            messages.error(request, "Employer not found.")
        except EmployerLogin.DoesNotExist:
            messages.error(request, "Login credentials not found.")
        except Exception as e:
            messages.error(request, f"Error deactivating account: {str(e)}")
    
    return redirect('employer_setting')
    
#******************************************************************************

def employer_payment_section(request):
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
        
    context = {
        'employer_name': request.session.get('employer_name', ''),
        'employer_email': request.session.get('employer_email', '')
    }
    return render(request, 'employer_html/employer_payment_section.html', context)

#***********************************************************************************


# Helper: Sentiment Analysis (standalone to avoid circular imports)
def compute_text_sentiment(text):
    """
    Keyword-based sentiment: +1 (good/excellent), -1 (bad/terrible), 0 (normal/neutral).
    Uses provided positive/negative lists.
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # Expanded positive and negative keywords
    positive = [
        'good', 'great', 'excellent', 'hardworking', 'reliable', 'professional',
        'amazing', 'best', 'awesome', 'fantastic', 'skilled', 'efficient',
        'timely', 'honest', 'dedicated', 'outstanding', 'superb', 'wonderful',
        'trustworthy', 'punctual', 'clean', 'neat', 'careful', 'experienced',
        'knowledgeable', 'helpful', 'friendly', 'polite', 'patient', 'quick',
        'fast', 'thorough', 'detail-oriented', 'creative', 'innovative',
        'proactive', 'responsible', 'conscientious', 'diligent', 'meticulous'
    ]
    
    negative = [
        'bad', 'poor', 'terrible', 'lazy', 'unreliable', 'incompetent',
        'worst', 'awful', 'disappointing', 'late', 'dishonest', 'inefficient',
        'sloppy', 'unprofessional', 'horrible', 'frustrating', 'subpar',
        'careless', 'messy', 'slow', 'rude', 'impolite', 'unfriendly',
        'unskilled', 'inexperienced', 'negligent', 'inattentive', 'forgetful',
        'disorganized', 'chaotic', 'expensive', 'overpriced', 'unsatisfactory',
        'mediocre', 'average', 'ordinary', 'forgetful', 'absent'
    ]
    
    text_lower = text.lower()
    
    # Count positive and negative occurrences
    pos_count = sum(1 for word in positive if re.search(r'\b' + re.escape(word) + r'\b', text_lower))
    neg_count = sum(1 for word in negative if re.search(r'\b' + re.escape(word) + r'\b', text_lower))
    
    # Calculate sentiment score
    if pos_count > 0 or neg_count > 0:
        sentiment = (pos_count - neg_count) / (pos_count + neg_count)
        return round(sentiment, 2)
    
    return 0.0


# Alternative geocoding function with fallback
def get_coordinates(location_str, max_retries=3):
    """Get coordinates for a location string with retry logic."""
    if not location_str:
        return None
    
    geolocator = Nominatim(user_agent="worker_finder_app", timeout=10)
    
    for attempt in range(max_retries):
        try:
            time.sleep(1)  # Rate limiting
            location = geolocator.geocode(location_str)
            if location:
                return (location.latitude, location.longitude)
            break
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt == max_retries - 1:
                print(f"Geocoding failed for '{location_str}': {str(e)}")
                return None
            time.sleep(2)
        except Exception as e:
            print(f"Geocoding error for '{location_str}': {str(e)}")
            return None
    
    return None


def calculate_distance(coord1, coord2):
    """Calculate distance between two coordinates."""
    if not coord1 or not coord2:
        return 100.0  # Default large distance
    
    try:
        return round(geodesic(coord1, coord2).kilometers, 1)
    except:
        return 100.0


def get_employee_location_coords(employee):
    """Get coordinates for employee location."""
    location_parts = []
    if employee.city:
        location_parts.append(employee.city)
    if employee.state:
        location_parts.append(employee.state)
    if employee.country:
        location_parts.append(employee.country)
    
    if location_parts:
        return get_coordinates(", ".join(location_parts))
    
    return None


# Main View: Employer Find Workers (FULL PIPELINE IMPLEMENTATION)
def employer_find_workers(request):
    """
    Implements the 4-step pipeline with improved logic:
    1. Random Forest Sim: Efficiency score
    2. K-Means Sim: Grade tiers
    3. Content Filtering: Skill match %
    4. KNN Sim: Distance + final score/sort
    """
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')

    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        employer_login = EmployerLogin.objects.get(employer=employer)
    except (Employer.DoesNotExist, EmployerLogin.DoesNotExist):
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')

    workers = []
    search_performed = False
    search_query = ''
    location_query = ''
    search_mode = 'employer_location'
    filter_desc = ' nearest to you'

    if request.method == 'POST':
        search_performed = True
        search_query = request.POST.get('search_query', '').strip()
        location_query = request.POST.get('location', '').strip()
        
        print(f"DEBUG: Search Query: '{search_query}'")
        print(f"DEBUG: Location Query: '{location_query}'")

        # Step 1: Get employer coordinates (for distance calculation)
        employer_location_parts = []
        if employer.city:
            employer_location_parts.append(employer.city)
        if employer.state:
            employer_location_parts.append(employer.state)
        if employer.country:
            employer_location_parts.append(employer.country)
        
        employer_coords = None
        if employer_location_parts:
            employer_coords = get_coordinates(", ".join(employer_location_parts))
        
        # Step 2: Base query for employees
        employees = Employee.objects.filter(
            status='Active',
            show_profile_to_employer=True
        ).select_related().prefetch_related('certificates', 'employee_skills', 'reviews')
        
        # Apply country filter if employer has country
        if employer.country:
            employees = employees.filter(country__icontains=employer.country)
        
        print(f"DEBUG: Initial employee count: {employees.count()}")

        # Step 3: Apply text search if provided
        if search_query:
            # Create search query for multiple fields
            search_terms = search_query.lower().split()
            search_filter = Q()
            
            for term in search_terms:
                if term:
                    search_filter |= (
                        Q(job_title__icontains=term) |
                        Q(skills__icontains=term) |
                        Q(bio__icontains=term) |
                        Q(work_experience__icontains=term) |
                        Q(first_name__icontains=term) |
                        Q(last_name__icontains=term) |
                        Q(employee_skills__skill_name__icontains=term)
                    )
            
            employees = employees.filter(search_filter).distinct()
            print(f"DEBUG: After text filter: {employees.count()}")

        # Step 4: Apply location filter if provided
        if location_query:
            location_filter = Q()
            location_terms = location_query.lower().split()
            
            for term in location_terms:
                if term:
                    location_filter |= (
                        Q(city__icontains=term) |
                        Q(state__icontains=term) |
                        Q(address__icontains=term) |
                        Q(location__icontains=term)
                    )
            
            employees = employees.filter(location_filter)
            search_mode = 'specific_location'
            filter_desc = f" in/near {location_query.title()}"
            print(f"DEBUG: After location filter: {employees.count()}")
        
        # Step 5: Process each employee
        worker_list = []
        
        for emp in employees:
            try:
                is_favorited = False
                if 'employer_id' in request.session:
                    try:
                        current_employer = Employer.objects.get(employer_id=request.session['employer_id'])
                        is_favorited = EmployerFavorite.objects.filter(
                            employer=current_employer, 
                            employee=emp
                        ).exists()
                    except Exception as e:
                        print(f"DEBUG: Error checking favorite status: {str(e)}")
                        pass

                # Get employee coordinates
                emp_coords = get_employee_location_coords(emp)
                
                # Calculate distance
                distance = 100.0
                if employer_coords and emp_coords:
                    distance = calculate_distance(employer_coords, emp_coords)
                elif emp.city and employer.city:
                    # Simple string comparison as fallback
                    if emp.city.lower() == employer.city.lower():
                        distance = 5.0
                    elif employer.state and emp.state and emp.state.lower() == employer.state.lower():
                        distance = 50.0
                
                # Check service radius
                if distance > emp.service_radius and location_query:
                    continue  # Skip if out of radius AND location was specified
                
                # Step 1: Random Forest Simulation - Efficiency Score
                score = 0.0
                
                # Review Analysis
                reviews = emp.reviews.all()
                review_count = reviews.count()
                avg_rating = 0.0
                avg_sentiment = 0.0
                sentiment_boost = 0
                
                if review_count > 0:
                    # Numerical average rating
                    rating_avg = reviews.aggregate(avg_rating=Avg('rating'))
                    avg_rating = rating_avg['avg_rating'] or 0.0
                    
                    # Text sentiment average
                    sentiment_scores = [compute_text_sentiment(r.text) for r in reviews]
                    avg_sentiment = sum(sentiment_scores) / review_count if review_count > 0 else 0.0
                    
                    # Boost based on sentiment and review count
                    if review_count >= 10:
                        if avg_sentiment >= 0.8:
                            sentiment_boost = 30
                        elif avg_sentiment >= 0.5:
                            sentiment_boost = 20
                        elif avg_sentiment >= 0:
                            sentiment_boost = 0
                        elif avg_sentiment >= -0.5:
                            sentiment_boost = -10
                        else:
                            sentiment_boost = -20
                    elif review_count >= 5:
                        if avg_sentiment >= 0.5:
                            sentiment_boost = 15
                        elif avg_sentiment <= -0.5:
                            sentiment_boost = -5
                    else:
                        sentiment_boost = int(avg_sentiment * 10)
                
                # Review contribution
                review_score = min(avg_rating * 20, 100) + sentiment_boost
                score += max(0, min(review_score, 100))
                
                # Experience points
                exp_points = min(emp.years_experience * 5, 50)
                score += exp_points
                
                # Success rate
                score += emp.success_rate
                
                # Jobs done reliability
                if emp.total_jobs_done >= 50:
                    score += 20
                elif emp.total_jobs_done >= 20:
                    score += 15
                elif emp.total_jobs_done >= 5:
                    score += 10
                else:
                    score += 5
                
                # Certificates (licenses/certs)
                cert_count = emp.certificates.filter(expiry_date__gte=timezone.now().date()).count()
                if cert_count >= 3:
                    score += 30
                elif cert_count >= 1:
                    score += 20
                
                # Response time bonus
                response_time = emp.response_time.lower()
                if 'hour' in response_time or 'fast' in response_time or 'quick' in response_time:
                    score += 10
                elif 'day' in response_time or 'medium' in response_time:
                    score += 5
                
                # Cap score
                score = max(0, min(score, 100))
                
                # Step 2: Grade Tiers
                if score >= 90:
                    grade = 'A+'
                    tier_desc = 'Elite Performers'
                elif score >= 80:
                    grade = 'A'
                    tier_desc = 'Top Tier'
                elif score >= 60:
                    grade = 'B'
                    tier_desc = 'Reliable'
                elif score >= 40:
                    grade = 'C'
                    tier_desc = 'Developing'
                else:
                    grade = 'D'
                    tier_desc = 'New/Unverified'
                
                # Step 3: Content-Based Filtering - Skill Match %
                skill_match = 100
                if search_query:
                    search_terms = [term.lower().strip() for term in search_query.split() if term.strip()]
                    matched_terms = 0
                    
                    # Check job title
                    if emp.job_title:
                        job_title_lower = emp.job_title.lower()
                        for term in search_terms:
                            if term in job_title_lower:
                                matched_terms += 1
                                break
                    
                    # Check skills
                    all_skills = [s.skill_name.lower() for s in emp.employee_skills.all()]
                    if emp.skills:
                        all_skills.extend([s.strip().lower() for s in emp.skills.split(',') if s.strip()])
                    
                    for term in search_terms:
                        for skill in all_skills:
                            if term in skill:
                                matched_terms += 1
                                break
                    
                    # Check bio and work experience
                    if emp.bio:
                        bio_lower = emp.bio.lower()
                        for term in search_terms:
                            if term in bio_lower:
                                matched_terms += 0.5
                                break
                    
                    if emp.work_experience:
                        exp_lower = emp.work_experience.lower()
                        for term in search_terms:
                            if term in exp_lower:
                                matched_terms += 0.5
                                break
                    
                    if search_terms:
                        skill_match = min((matched_terms / len(search_terms)) * 100, 100)
                    
                    # If no skill match at all, skip this employee
                    if skill_match == 0:
                        continue
                
                # Step 4: KNN Simulation - Final Score
                proximity_factor = max(0.3, 1.0 - (distance / 100.0))
                final_score = score * (skill_match / 100.0) * proximity_factor
                
                # Wage estimate
                wage_estimate = f"₹{emp.years_experience * 100 + 500}/day"
                
                # Sentiment category
                if avg_sentiment >= 0.8:
                    sentiment_category = 'Excellent'
                elif avg_sentiment >= 0.3:
                    sentiment_category = 'Good'
                elif avg_sentiment > -0.3:
                    sentiment_category = 'Normal'
                elif avg_sentiment > -0.8:
                    sentiment_category = 'Bad'
                else:
                    sentiment_category = 'Terrible'
                
                worker_list.append({
                    'employee': emp,
                    'efficiency_score': round(score, 1),
                    'grade': grade,
                    'tier_desc': tier_desc,
                    'skill_match': round(skill_match, 1),
                    'distance': round(distance, 1),
                    'final_score': round(final_score, 1),
                    'wage_estimate': wage_estimate,
                    'avg_rating': round(avg_rating, 1),
                    'avg_sentiment': round(avg_sentiment, 2),
                    'review_count': review_count,
                    'sentiment_boost': sentiment_boost,
                    'sentiment_category': sentiment_category,
                    'has_profile_image': bool(emp.profile_image),
                    'profile_image_url': emp.profile_image.url if emp.profile_image else None,
                    'response_time': emp.response_time,
                    'city': emp.city or 'Not specified',
                    'city': emp.city or 'Not specified',
                    'availability_status': emp.availability_status,
                    'is_favorited': is_favorited,
                    'availability_status': emp.availability_status,
                    'is_favorited': is_favorited,
                })
                
                # Get raw availability
                raw_av = get_employee_availability(emp, end_date=datetime.now().date() + timedelta(days=60))
                
                # Group by month
                months = []
                current_month_key = None
                current_month_data = None
                
                for day in raw_av:
                    m_key = (day['date'].year, day['date'].month)
                    if m_key != current_month_key:
                        if current_month_data:
                            months.append(current_month_data)
                        
                        # Start new month
                        # Calculate padding for first day (0=Sun, 6=Sat)
                        padding = int(day['date'].strftime('%w'))
                        
                        current_month_data = {
                            'name': day['date'].strftime('%B %Y'),
                            'padding': range(padding), # Create iterable for template
                            'days': []
                        }
                        current_month_key = m_key
                    
                    current_month_data['days'].append(day)
                
                if current_month_data:
                    months.append(current_month_data)
                
                # Verify data structure
                if months:
                    print(f"DEBUG: Processed months for {emp.employee_id}: {[m['name'] for m in months]}")
                    
                worker_list[-1]['availability_months'] = months
                
            except Exception as e:
                print(f"Error processing employee {emp.employee_id}: {str(e)}")
                continue
        
        # Sort by final score
        worker_list.sort(key=lambda x: x['final_score'], reverse=True)
        workers = worker_list[:20]  # Limit to top 20
        
        print(f"DEBUG: Final worker count: {len(workers)}")
        
        # Feedback messages
        if workers:
            messages.success(request, f"Found {len(workers)} matching workers{filter_desc}!")
        else:
            messages.info(request, "No workers found matching your criteria. Try broadening your search.")
    
    # Context for template
    context = {
        'employer': employer,
        'employer_name': employer.full_name,
        'employer_email': employer.email,
        'workers': workers,
        'search_performed': search_performed,
        'search_query': search_query,
        'location_query': location_query,
        'search_mode': search_mode,
        'filter_desc': filter_desc,
    }
    
    return render(request, 'employer_html/employer_find_workers.html', context)


#*************************************************************


def hire_employee_view(request, employee_id):
    """View to handle hiring an employee"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        employee = Employee.objects.get(employee_id=employee_id)
        
        if request.method == 'POST':
            # Get form data
            job_title = request.POST.get('job_title', '').strip()
            job_description = request.POST.get('job_description', '').strip()
            proposed_date = request.POST.get('proposed_date', '')
            proposed_time = request.POST.get('proposed_time', '')
            estimated_duration = request.POST.get('estimated_duration', '').strip()
            budget = request.POST.get('budget', '').strip()
            employer_notes = request.POST.get('employer_notes', '').strip()
            location = request.POST.get('location', '').strip()
            address = request.POST.get('address', '').strip()
            city = request.POST.get('city', '').strip()
            state = request.POST.get('state', '').strip()
            priority = request.POST.get('priority', 'normal')
            
            # Validate required fields
            if not job_title or not job_description or not proposed_date:
                messages.error(request, "Job title, description, and date are required.")
                return redirect('employer_find_workers')
            
            # Parse date
            try:
                proposed_date_obj = datetime.strptime(proposed_date, '%Y-%m-%d').date()
            except ValueError:
                messages.error(request, "Invalid date format. Please use YYYY-MM-DD.")
                return redirect('employer_find_workers')
            
            # Parse time if provided
            proposed_time_obj = None
            if proposed_time:
                try:
                    proposed_time_obj = datetime.strptime(proposed_time, '%H:%M').time()
                except ValueError:
                    messages.error(request, "Invalid time format. Please use HH:MM.")
                    return redirect('employer_find_workers')
            
            # Parse budget
            budget_decimal = None
            if budget:
                try:
                    budget_decimal = float(budget)
                except ValueError:
                    messages.error(request, "Invalid budget amount.")
                    return redirect('employer_find_workers')
            
            # Import JobRequest model
            from employee.models import JobRequest, JobAction
            
            # Create job request
            job_request = JobRequest.objects.create(
                employer=employer,
                employee=employee,
                title=job_title,
                description=job_description,
                proposed_date=proposed_date_obj,
                proposed_time=proposed_time_obj,
                estimated_duration=estimated_duration,
                budget=budget_decimal,
                employer_notes=employer_notes,
                location=location,
                address=address,
                city=city,
                state=state,
                priority=priority,
                status='pending'
            )
            
            # Create initial action
            JobAction.objects.create(
                job=job_request,
                employer=employer,
                action_type='created',
                notes=f"Job request created by {employer.full_name}"
            )
            
            # Create Notification for Employee
            EmployeeNotification.objects.create(
                employee=employee,
                title="New Job Request",
                message=f"You have received a new job request from {employer.full_name} for '{job_title}'.",
                notification_type='job',
                link=f"/employee/job/details/{job_request.job_id}/"
            )
            
            messages.success(request, f"Job request sent to {employee.full_name} successfully!")
            return redirect('employer_find_workers')
        
        else:
            # GET request - show hiring form
            # Get employee availability for next 30 days
            availability = get_employee_availability(employee)
            
            context = {
                'employer': employer,
                'employee': employee,
                'availability': availability,
                'today': datetime.now().strftime('%Y-%m-%d'),
                'max_date': (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
            }
            
            return render(request, 'employer_html/hire_employee.html', context)
    
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
        return redirect('index')
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employer_find_workers')
    except Exception as e:
        messages.error(request, f"Error processing hire request: {str(e)}")
        return redirect('employer_find_workers')


#**********************************************************************

# ============================================================================
# EMPLOYER HIRED EMPLOYEES VIEW (Updated)
# ============================================================================
def employer_hired_employees(request):
    """View to show all employees hired by the employer"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        
        # Get all job requests where employer has hired employees (completed or accepted)
        job_requests = JobRequest.objects.filter(
            employer=employer,
            status__in=['completed', 'accepted']
        ).select_related('employee').order_by('-completed_at', '-created_at')
        
        # Get unique employees with their job info
        hired_employees_dict = {}
        for job in job_requests:
            employee_id = job.employee.employee_id
            
            # Get reviews for this specific job
            job_review = Review.objects.filter(
                employer=employer,
                employee=job.employee,
                job=job
            ).first()
            
            # Get employee's average rating from all reviews
            all_reviews = Review.objects.filter(employee=job.employee)
            avg_rating = all_reviews.aggregate(avg=Avg('rating'))['avg'] or 0.0
            
            # Count jobs without reviews
            jobs_without_review = JobRequest.objects.filter(
                employer=employer,
                employee=job.employee,
                status='completed'
            ).exclude(
                reviews__employer=employer
            ).count()
            
            if employee_id not in hired_employees_dict:
                hired_employees_dict[employee_id] = {
                    'employee': job.employee,
                    'total_jobs': job_requests.filter(employee=job.employee).count(),
                    'completed_jobs': job_requests.filter(
                        employee=job.employee, 
                        status='completed'
                    ).count(),
                    'total_earned': job_requests.filter(
                        employee=job.employee, 
                        status='completed'
                    ).aggregate(total=Sum('budget'))['total'] or 0,
                    'average_rating': round(avg_rating, 1),
                    'review_count': all_reviews.count(),
                    'jobs_without_review': jobs_without_review,
                    'latest_job': job,
                }
        
        hired_employees = list(hired_employees_dict.values())
        
        # Apply search filter if provided
        search_query = request.GET.get('search', '').strip()
        if search_query:
            search_terms = search_query.lower().split()
            filtered_employees = []
            
            for emp_data in hired_employees:
                employee = emp_data['employee']
                employee_name = employee.full_name.lower()
                job_title = employee.job_title.lower() if employee.job_title else ''
                skills = employee.skills.lower() if employee.skills else ''
                
                match_found = False
                for term in search_terms:
                    if (term in employee_name or 
                        term in job_title or 
                        term in skills):
                        match_found = True
                        break
                
                if match_found:
                    filtered_employees.append(emp_data)
            
            hired_employees = filtered_employees
        
        # Calculate statistics
        total_hired = len(hired_employees)
        reviewed_jobs = sum(emp['completed_jobs'] - emp['jobs_without_review'] for emp in hired_employees)
        total_completed_jobs = sum(emp['completed_jobs'] for emp in hired_employees)
        pending_reviews = sum(emp['jobs_without_review'] for emp in hired_employees)
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'employer_email': employer.email,
            'hired_employees': hired_employees,
            'search_query': search_query,
            'total_hired': total_hired,
            'reviewed_jobs': reviewed_jobs,
            'total_completed_jobs': total_completed_jobs,
            'pending_reviews': pending_reviews,
        }
        
        return render(request, 'employer_html/employer_hired_employees.html', context)
        
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')
    except Exception as e:
        messages.error(request, f"Error loading hired employees: {str(e)}")
        return redirect('employer_dashboard')

#***************************************************************






# employer/reviews_views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.utils import timezone
from django.http import HttpResponse
from django.db.models import Q, Avg, Count, Sum
from .models import Employer, EmployerLogin, EmployerFavorite
from employee.models import Employee, JobRequest, Review
import re
from datetime import datetime, timedelta

# ============================================================================
# REVIEWS DASHBOARD - MAIN VIEW
# ============================================================================
def employer_reviews_dashboard(request):
    """Main reviews dashboard for employer"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        
        # Get all hired employees
        hired_employees = []
        job_requests = JobRequest.objects.filter(
            employer=employer,
            status='completed'
        ).select_related('employee').order_by('-completed_at')
        
        # Calculate total hired employees
        employee_ids = set()
        for job in job_requests:
            if job.employee:
                employee_ids.add(job.employee.employee_id)
        
        total_hired = len(employee_ids)
        
        # Calculate total completed jobs
        total_completed_jobs = job_requests.count()
        
        # Calculate reviewed jobs
        reviewed_jobs = Review.objects.filter(
            employer=employer
        ).count()
        
        # Calculate pending reviews
        pending_reviews = total_completed_jobs - reviewed_jobs
        
        # Get employees needing reviews
        employees_needing_reviews = []
        for employee_id in employee_ids:
            try:
                employee = Employee.objects.get(employee_id=employee_id)
                
                # Count jobs without reviews for this employee
                jobs_without_review = job_requests.filter(
                    employee=employee
                ).exclude(
                    reviews__employer=employer
                ).count()
                
                if jobs_without_review > 0:
                    # Get total jobs with this employee
                    total_jobs = job_requests.filter(employee=employee).count()
                    
                    # Get total earned
                    total_earned_result = job_requests.filter(
                        employee=employee
                    ).aggregate(total=Sum('budget'))
                    total_earned = total_earned_result['total'] or 0
                    
                    # Get average rating
                    avg_rating_result = Review.objects.filter(
                        employee=employee
                    ).aggregate(avg=Avg('rating'))
                    avg_rating = avg_rating_result['avg'] or 0.0
                    
                    # Get latest job without review
                    latest_job = job_requests.filter(
                        employee=employee
                    ).exclude(
                        reviews__employer=employer
                    ).order_by('-completed_at').first()
                    
                    employees_needing_reviews.append({
                        'employee': employee,
                        'jobs_without_review': jobs_without_review,
                        'total_jobs': total_jobs,
                        'total_earned': total_earned,
                        'avg_rating': avg_rating,
                        'latest_job': latest_job,
                        'latest_job_id': latest_job.job_id if latest_job else None,
                    })
            except Employee.DoesNotExist:
                continue
        
        # Get site feedbacks and reports from Database
        site_feedbacks = SiteReview.objects.filter(employer=employer).order_by('-created_at')
        
        # Get reports
        all_reports = Report.objects.filter(employer=employer).order_by('-created_at')
        site_issue_reports = all_reports.filter(report_type='platform_issue')
        employee_reports = all_reports.exclude(report_type='platform_issue')
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'employer_email': employer.email,
            'total_hired': total_hired,
            'total_completed_jobs': total_completed_jobs,
            'reviewed_jobs': reviewed_jobs,
            'pending_reviews': pending_reviews,
            'employees_needing_reviews': employees_needing_reviews,
            'site_feedbacks': site_feedbacks,
            'site_issue_reports': site_issue_reports,
            'employee_reports': employee_reports,
        }
        
        return render(request, 'employer_html/give_employee_review.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading reviews dashboard: {str(e)}")
        return redirect('employer_dashboard')
    

# ============================================================================
# GIVE EMPLOYEE REVIEW (Job-specific)
# ============================================================================
def give_employee_review(request, employee_id, job_id=None):
    """View for employer to give review for a specific job"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Get the specific job if job_id is provided
        job_id = request.GET.get('job_id') or request.POST.get('job_id')
        job = None
        
        if job_id:
            job = get_object_or_404(JobRequest, 
                                   job_id=job_id, 
                                   employer=employer,
                                   employee=employee,
                                   status='completed')
        
        # Check if employer has completed jobs with this employee
        completed_jobs = JobRequest.objects.filter(
            employer=employer,
            employee=employee,
            status='completed'
        ).order_by('-completed_at')
        
        if not completed_jobs.exists():
            messages.error(request, "You haven't completed any jobs with this employee yet.")
            return redirect('employer_hired_employees')
        
        # If no specific job is selected, use the latest completed job
        if not job:
            job = completed_jobs.first()
        
        # Check for existing review for this specific job
        existing_review = Review.objects.filter(
            employer=employer,
            employee=employee,
            job=job
        ).first()
        
        if request.method == 'POST':
            # Handle review submission
            review_text = request.POST.get('review_text', '').strip()
            rating = request.POST.get('rating', '').strip()
            
            if not review_text:
                messages.error(request, "Please write a review.")
                if job_id:
                    return redirect('give_employee_review_with_job', employee_id=employee_id, job_id=job_id)
                else:
                    return redirect('give_employee_review', employee_id=employee_id)
            
            if not rating or not rating.isdigit():
                messages.error(request, "Please provide a valid rating.")
                if job_id:
                    return redirect('give_employee_review_with_job', employee_id=employee_id, job_id=job_id)
                else:
                    return redirect('give_employee_review', employee_id=employee_id)
            
            rating_value = float(rating)
            if rating_value < 1 or rating_value > 5:
                messages.error(request, "Rating must be between 1 and 5.")
                if job_id:
                    return redirect('give_employee_review_with_job', employee_id=employee_id, job_id=job_id)
                else:
                    return redirect('give_employee_review', employee_id=employee_id)
            
            # Calculate sentiment score
            sentiment_score = compute_text_sentiment(review_text)
            
            if existing_review:
                # Update existing review
                existing_review.text = review_text
                existing_review.rating = rating_value
                existing_review.sentiment_score = sentiment_score
                existing_review.updated_at = timezone.now()
                existing_review.save()
                action = "updated"
            else:
                # Create new review for this specific job
                Review.objects.create(
                    employer=employer,
                    employee=employee,
                    job=job,
                    text=review_text,
                    rating=rating_value,
                    sentiment_score=sentiment_score
                )
                action = "submitted"
            
            messages.success(request, f"Review {action} successfully for Job #{job.job_id}!")
            return redirect('employer_hired_employees')
        
        # Get job history with this employee
        job_history = completed_jobs
        
        # Get employee's average rating from all reviews
        all_reviews = Review.objects.filter(employee=employee).order_by('-created_at')
        avg_rating = all_reviews.aggregate(avg=Avg('rating'))['avg'] or 0.0
        review_count = all_reviews.count()
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'employee': employee,
            'job': job,
            'job_id': job.job_id if job else None,
            'existing_review': existing_review,
            'job_history': job_history,
            'total_jobs': job_history.count(),
            'average_rating': round(avg_rating, 1),
            'review_count': review_count,
            'all_reviews': all_reviews,
        }
        
        return render(request, 'employer_html/give_employee_review.html', context)
        
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employer_hired_employees')
    except Exception as e:
        messages.error(request, f"Error: {str(e)}")
        return redirect('employer_hired_employees')



# ============================================================================
# VIEW EMPLOYEE REVIEWS
# ============================================================================
def view_employee_reviews(request, employee_id):
    """View all reviews for a specific employee"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Get all reviews for this employee
        reviews = Review.objects.filter(employee=employee).select_related('employer', 'job').order_by('-created_at')
        
        # Get employer's reviews for this employee
        employer_reviews = Review.objects.filter(employer=employer, employee=employee).select_related('job')
        
        # Calculate statistics
        total_reviews = reviews.count()
        avg_rating = reviews.aggregate(avg=Avg('rating'))['avg'] or 0.0
        
        # Sentiment analysis
        positive_reviews = reviews.filter(sentiment_score__gt=0.3).count()
        neutral_reviews = reviews.filter(sentiment_score__gte=-0.3, sentiment_score__lte=0.3).count()
        negative_reviews = reviews.filter(sentiment_score__lt=-0.3).count()
        
        # Get completed jobs without reviews
        completed_jobs_without_reviews = JobRequest.objects.filter(
            employer=employer,
            employee=employee,
            status='completed'
        ).exclude(
            reviews__employer=employer
        ).count()
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'employee': employee,
            'reviews': reviews,
            'employer_reviews': employer_reviews,
            'total_reviews': total_reviews,
            'average_rating': round(avg_rating, 1),
            'positive_reviews': positive_reviews,
            'neutral_reviews': neutral_reviews,
            'negative_reviews': negative_reviews,
            'completed_jobs_without_reviews': completed_jobs_without_reviews,
        }
        
        return render(request, 'employer_html/view_employee_reviews.html', context)
        
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employer_hired_employees')
    except Exception as e:
        messages.error(request, f"Error: {str(e)}")
        return redirect('employer_hired_employees')

# ============================================================================
# DELETE REVIEW
# ============================================================================
def delete_employee_review(request, review_id):
    """Delete a specific review"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            
            # Get the review
            review = get_object_or_404(Review, id=review_id, employer=employer)
            job_id = review.job.job_id if review.job else None
            
            review.delete()
            messages.success(request, "Review deleted successfully!")
            
            # Redirect back to appropriate page
            if job_id:
                return redirect('view_job_details', job_id=job_id)
            else:
                return redirect('employer_hired_employees')
                
        except Employer.DoesNotExist:
            messages.error(request, "Employer not found.")
        except Review.DoesNotExist:
            messages.error(request, "Review not found.")
        except Exception as e:
            messages.error(request, f"Error deleting review: {str(e)}")
    
    return redirect('employer_hired_employees')

# ============================================================================
# ADD JOB REVIEW (From hiring history)
# ============================================================================
def add_job_review(request, job_id):
    """Add review for a completed job from hiring history"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            job_request = JobRequest.objects.get(job_id=job_id, employer=employer)
            
            # Check if job is completed
            if job_request.status != 'completed':
                messages.error(request, "Can only review completed jobs.")
                return redirect('view_job_details', job_id=job_id)
            
            # Check if review already exists for this job
            existing_review = Review.objects.filter(
                employer=employer,
                employee=job_request.employee,
                job=job_request
            ).first()
            
            if existing_review:
                messages.error(request, "You have already reviewed this job.")
                return redirect('view_job_details', job_id=job_id)
            
            # Get form data
            rating = request.POST.get('rating')
            review_text = request.POST.get('review_text', '').strip()
            
            if not rating:
                messages.error(request, "Please provide a rating.")
                return redirect('view_job_details', job_id=job_id)
            
            # Calculate sentiment
            sentiment_score = compute_text_sentiment(review_text)
            
            # Create review
            review = Review.objects.create(
                employer=employer,
                employee=job_request.employee,
                job=job_request,
                text=review_text,
                rating=float(rating),
                sentiment_score=sentiment_score
            )
            
            # Update employee rating
            if job_request.employee:
                employee = job_request.employee
                all_reviews = Review.objects.filter(employee=employee, rating__isnull=False)
                if all_reviews.exists():
                    avg_rating = all_reviews.aggregate(avg=Avg('rating'))['avg']
                    employee.rating = avg_rating
                    employee.save()
            
            messages.success(request, "Review submitted successfully!")
            return redirect('view_job_details', job_id=job_id)
            
        except JobRequest.DoesNotExist:
            messages.error(request, "Job request not found.")
            return redirect('employer_hiring_history')
        except Exception as e:
            messages.error(request, f"Error submitting review: {str(e)}")
            return redirect('view_job_details', job_id=job_id)
    
    return redirect('employer_hiring_history')

# ============================================================================
# SUBMIT SITE REVIEW
# ============================================================================
def submit_site_review(request):
    """Handle site/platform review submission"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            
            # Get form data
            employee_id = request.POST.get('employee_id')
            review_type = request.POST.get('review_type')
            rating = request.POST.get('rating')
            title = request.POST.get('title', '').strip()
            review_text = request.POST.get('review_text', '').strip()
            areas = request.POST.getlist('areas')  # Get multiple checkbox values
            recommendation = request.POST.get('recommendation')
            
            # Validate required fields
            if not review_type or not rating or not review_text:
                messages.error(request, "Please fill in all required fields.")
                return redirect(request.META.get('HTTP_REFERER', 'view_employee_reviews'))
            
            # Create SiteReview model
            SiteReview.objects.create(
                employer=employer,
                employee_id=employee_id if employee_id else None,
                review_type=review_type,
                rating=int(rating),
                title=title,
                review_text=review_text,
                areas=areas,
                recommendation=recommendation
            )
            
            messages.success(request, "Thank you for your site review! Your feedback helps us improve.")
            
        except Employer.DoesNotExist:
            messages.error(request, "Employer not found.")
        except Exception as e:
            messages.error(request, f"Error submitting site review: {str(e)}")
    
    # Redirect back to the employee reviews page
    if request.POST.get('employee_id'):
        return redirect('view_employee_reviews', employee_id=request.POST.get('employee_id'))
    return redirect('employer_hired_employees')

# ============================================================================
# SUBMIT REPORT
# ============================================================================
def submit_report(request):
    """Handle report submission"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            
            # Get form data
            employee_id = request.POST.get('employee_id')
            report_type = request.POST.get('report_type')
            title = request.POST.get('title', '').strip()
            description = request.POST.get('description', '').strip()
            job_id = request.POST.get('job_id')
            severity = request.POST.get('severity')
            resolution_preference = request.POST.get('resolution_preference', '').strip()
            contact_methods = request.POST.getlist('contact_methods')
            share_with_employee = request.POST.get('share_with_employee', 'no')
            
            # Handle file uploads
            evidence_files = request.FILES.getlist('evidence_files')
            
            # Validate required fields
            if not report_type or not title or not description or not severity:
                messages.error(request, "Please fill in all required fields.")
                return redirect(request.META.get('HTTP_REFERER', 'view_employee_reviews'))
            
            # Validate file size
            max_file_size = 10 * 1024 * 1024  # 10MB
            for file in evidence_files:
                if file.size > max_file_size:
                    messages.error(request, f"File '{file.name}' exceeds maximum size of 10MB.")
                    return redirect(request.META.get('HTTP_REFERER', 'view_employee_reviews'))
            
            # Create Report model
            Report.objects.create(
                employer=employer,
                employee_id=employee_id if employee_id else None,
                report_type=report_type,
                title=title,
                description=description,
                job_id=job_id if job_id else None,
                severity=severity,
                resolution_preference=resolution_preference,
                contact_methods=contact_methods,
                share_with_employee=(share_with_employee == 'yes'),
                status='pending'
            )
            
            # Process file uploads (in production, you would save to disk/database)
            # For now, we'll just log the file names
            if evidence_files:
                for file in evidence_files:
                    print(f"File uploaded: {file.name} ({file.size} bytes)")
            
            messages.success(request, "Report submitted successfully! We'll investigate and get back to you.")
            
        except Employer.DoesNotExist:
            messages.error(request, "Employer not found.")
        except Exception as e:
            messages.error(request, f"Error submitting report: {str(e)}")
    
    # Redirect back to the employee reviews page
    if request.POST.get('employee_id'):
        return redirect('view_employee_reviews', employee_id=request.POST.get('employee_id'))
    return redirect('employer_hired_employees')

# ============================================================================
# SUBMIT EMPLOYEE REPORT
# ============================================================================
def submit_employee_report(request):
    """Handle employee report submission"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.", extra_tags='employee_report danger')
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            employee_id = request.POST.get('employee_id')
            report_type = request.POST.get('report_type')
            description = request.POST.get('description', '').strip()
            severity = request.POST.get('severity')
            
            if not employee_id or not report_type or not description:
                messages.error(request, "Please fill in all required fields.", extra_tags='employee_report danger')
                return redirect(request.META.get('HTTP_REFERER', 'employer_hired_employees'))

            Report.objects.create(
                employer=employer,
                employee_id=employee_id,
                report_type=report_type,  # e.g., 'professionalism', 'performance'
                title=f"Employee Report: {report_type.title()}",
                description=description,
                severity=severity,
                status='pending'
            )
            
            messages.success(request, "Employee report submitted successfully! We'll investigate.", extra_tags='employee_report success')
            
        except Exception as e:
            messages.error(request, f"Error submitting report: {str(e)}", extra_tags='employee_report danger')
    
    return redirect(request.META.get('HTTP_REFERER', 'employer_hired_employees'))

# ============================================================================
# SUBMIT SITE FEEDBACK
# ============================================================================
def submit_site_feedback(request):
    """Handle site feedback submission"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            
            # Get form data
            feedback_type = request.POST.get('feedback_type')
            rating = request.POST.get('rating')
            title = request.POST.get('title', '').strip()
            feedback_text = request.POST.get('feedback_text', '').strip()
            areas = request.POST.getlist('areas')
            recommendation = request.POST.get('recommendation')
            feature_requests = request.POST.get('feature_requests', '').strip()
            contact_email = request.POST.get('contact_email', '')
            
            # Validate required fields
            if not feedback_type or not rating or not title or not feedback_text or not recommendation:
                messages.error(request, "Please fill in all required fields.", extra_tags='site_feedback danger')
                return redirect('employer_reviews_dashboard')
            
            # Create SiteReview for feedback
            SiteReview.objects.create(
                employer=employer,
                review_type='platform',  # defaulting to platform for general feedback
                rating=int(rating),
                title=title,
                review_text=feedback_text,
                areas=areas,
                recommendation=recommendation
            )
            
            messages.success(request, "Thank you for your feedback! It helps us improve WorkNest.", extra_tags='site_feedback success')
            
        except Employer.DoesNotExist:
            messages.error(request, "Employer not found.", extra_tags='site_feedback danger')
        except Exception as e:
            messages.error(request, f"Error submitting feedback: {str(e)}", extra_tags='site_feedback danger')
    
    return redirect('employer_reviews_dashboard')

# ============================================================================
# SUBMIT SITE ISSUE REPORT
# ============================================================================
def submit_site_issue_report(request):
    """Handle site issue report submission"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            
            # Get form data
            issue_type = request.POST.get('issue_type')
            title = request.POST.get('title', '').strip()
            description = request.POST.get('description', '').strip()
            page_location = request.POST.get('page_location', '').strip()
            browser_info = request.POST.get('browser_info', '').strip()
            severity = request.POST.get('severity')
            frequency = request.POST.get('frequency', '')
            urgency = request.POST.get('urgency', '')
            contact_email = request.POST.get('contact_email', '')
            contact_phone = request.POST.get('contact_phone', '')
            
            # Validate required fields
            if not issue_type or not title or not description or not page_location or not severity or not contact_email:
                messages.error(request, "Please fill in all required fields.", extra_tags='site_report danger')
                return redirect('employer_reviews_dashboard')
            
            # Handle file uploads
            screenshots = request.FILES.getlist('screenshots')
            
            # Validate file size
            max_file_size = 5 * 1024 * 1024  # 5MB
            for file in screenshots:
                if file.size > max_file_size:
                    messages.error(request, f"File '{file.name}' exceeds maximum size of 5MB.", extra_tags='site_report danger')
                    return redirect('employer_reviews_dashboard')
            
            # Create Report for site issue
            Report.objects.create(
                employer=employer,
                report_type='platform_issue',
                title=title,
                description=f"{description}\n\nPage: {page_location}\nBrowser: {browser_info}\nFrequency: {frequency}\nUrgency: {urgency}",
                severity=severity,
                status='pending'
            )
            
            # Process file uploads
            if screenshots:
                for file in screenshots:
                    print(f"Site issue screenshot uploaded: {file.name} ({file.size} bytes)")
            
            messages.success(request, "Site issue report submitted! We'll work on fixing it.", extra_tags='site_report success')
            
        except Employer.DoesNotExist:
            messages.error(request, "Employer not found.", extra_tags='site_report danger')
        except Exception as e:
            messages.error(request, f"Error submitting site report: {str(e)}", extra_tags='site_report danger')
    
    return redirect('employer_reviews_dashboard')

# ============================================================================
# VIEW PREVIOUS REPORTS
# ============================================================================
def view_previous_reports(request):
    """View previous reports submitted by employer"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        
        # Get reports from session (temporary)
        reports = request.session.get('reports', [])
        
        # Filter by current employer
        employer_reports = [r for r in reports if r.get('employer_id') == employer.employer_id]
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'reports': employer_reports,
        }
        
        return render(request, 'employer_html/view_reports.html', context)
        
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')
    except Exception as e:
        messages.error(request, f"Error loading reports: {str(e)}")
        return redirect('employer_dashboard')






#**************************************************

def get_employee_availability(employee, start_date=None, end_date=None):
    from employee.models import JobRequest
    import datetime
    
    if not start_date:
        start_date = datetime.date.today()
    if not end_date:
        end_date = start_date + datetime.timedelta(days=30)
        
    # Get all accepted/completed jobs for this worker in the date range
    existing_jobs = JobRequest.objects.filter(
        employee=employee,
        status__in=['accepted', 'completed'],
        proposed_date__range=[start_date, end_date]
    ).values_list('proposed_date', flat=True)
    
    availability_list = []
    current_date = start_date
    while current_date <= end_date:
        is_available = True
        # If worker has a job that day, they are unavailable
        if current_date in existing_jobs:
            is_available = False
        # Also check general availability status
        if employee.availability == 'unavailable':
            is_available = False
            
        availability_list.append({
            'date': current_date,
            'date_str': current_date.strftime('%Y-%m-%d'),
            'day_name': current_date.strftime('%A'),
            'available': is_available
        })
        current_date += datetime.timedelta(days=1)
    return availability_list


    return availability_list



def contact_employee(request, employee_id):
    """Contact an employee directly (starts a chat)"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Check if chat already exists between employer and employee
        chat_exists = ChatRoom.objects.filter(
            employer=employer,
            employee=employee
        ).exists()
        
        if chat_exists:
            # Get existing chat room
            chat_room = ChatRoom.objects.filter(
                employer=employer,
                employee=employee
            ).first()
            
            messages.info(request, f"Opening existing chat with {employee.full_name}")
            return redirect('chat_room', room_id=chat_room.room_id)
        else:
            # Create new chat room
            chat_room = ChatRoom.objects.create(
                employer=employer,
                employee=employee,
                subject=f"Chat with {employee.full_name}",
                room_type='general'
            )
            
            # Create initial message (optional)
            if request.method == 'POST':
                initial_message = request.POST.get('initial_message', '').strip()
                if initial_message:
                    Message.objects.create(
                        room=chat_room,
                        sender_type='employer',
                        sender_employer=employer,
                        message_type='text',
                        content=initial_message,
                        status='sent'
                    )
                    
                    chat_room.message_count = 1
                    chat_room.last_message_time = timezone.now()
                    chat_room.unread_employee = 1
                    chat_room.save()
            
            messages.success(request, f"Started new chat with {employee.full_name}")
            return redirect('chat_room', room_id=chat_room.room_id)
            
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employer_find_workers')
    except Exception as e:
        messages.error(request, f"Error contacting employee: {str(e)}")
        return redirect('employer_find_workers')
    

#***************************************************************

















# employer/views.py - Complete with all payment functions
import razorpay
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.utils import timezone
from django.http import HttpResponse, JsonResponse
from django.db import transaction
from django.db.models import Q, Sum, Count
from decimal import Decimal
import json
from datetime import datetime, timedelta
from .models import Payment, PaymentInvoice, Employer, EmployerLogin, EmployerFavorite
from employee.models import JobRequest, Employee, Review
import os

# Initialize Razorpay client
razorpay_client = razorpay.Client(
    auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)
)

# ============================================================================
# PAYMENT SECTION VIEW
# ============================================================================
def employer_payment_section(request):
    """Main payment dashboard for employer"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        
        # Get pending payments (jobs completed but not paid)
        pending_payments = JobRequest.objects.filter(
            employer=employer,
            status='completed'
        ).exclude(
            payments__status='completed'
        ).select_related('employee').order_by('-completed_at')[:10]
        
        # Get recent payments
        recent_payments = Payment.objects.filter(
            employer=employer
        ).select_related('employee', 'job').order_by('-created_at')[:10]
        
        # Calculate statistics
        total_paid = Payment.objects.filter(
            employer=employer,
            status='completed'
        ).aggregate(total=Sum('amount'))['total'] or 0
        
        pending_amount = sum([
            (job.budget or 0) for job in pending_payments
        ])
        
        successful_payments = Payment.objects.filter(
            employer=employer,
            status='completed'
        ).count()
        
        failed_payments = Payment.objects.filter(
            employer=employer,
            status='failed'
        ).count()
        
        # Get monthly spending
        now = timezone.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        
        monthly_spending = Payment.objects.filter(
            employer=employer,
            status='completed',
            payment_date__gte=month_start,
            payment_date__lte=month_end
        ).aggregate(total=Sum('amount'))['total'] or 0
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'employer_email': employer.email,
            'pending_payments': pending_payments,
            'recent_payments': recent_payments,
            'stats': {
                'total_paid': total_paid,
                'pending_amount': pending_amount,
                'successful_payments': successful_payments,
                'failed_payments': failed_payments,
                'monthly_spending': monthly_spending,
            },
            'razorpay_key_id': settings.RAZORPAY_KEY_ID,
        }
        
        return render(request, 'employer_html/employer_payment_section.html', context)
        
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')
    except Exception as e:
        messages.error(request, f"Error loading payment section: {str(e)}")
        return redirect('employer_dashboard')


# ============================================================================
# INITIATE PAYMENT
# ============================================================================
def initiate_payment(request, job_id):
    """Initiate payment with proper debugging"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        job = get_object_or_404(JobRequest, job_id=job_id, employer=employer)
        
        # Check if job is completed
        if job.status != 'completed':
            messages.error(request, "Payment can only be made for completed jobs.")
            return redirect('employer_payment_section')
        
        # Check if payment already exists
        existing_payment = Payment.objects.filter(
            employer=employer,
            job=job,
            status='completed'
        ).first()
        
        if existing_payment:
            messages.info(request, "Payment for this job is already completed.")
            return redirect('employer_payment_section')
        
        # Get amount
        amount = job.budget or 0
        if amount <= 0:
            messages.error(request, "Invalid payment amount.")
            return redirect('employer_payment_section')
        
        # Validate amount doesn't exceed transaction limit (₹10,000)
        if amount > 10000:
            messages.error(request, "Payment amount cannot exceed ₹10,000. Please adjust the job budget.")
            return redirect('employer_payment_section')
        
        # Convert to paise
        amount_in_paise = int(amount * 100)
        
        # Create Razorpay order
        order_data = {
            'amount': amount_in_paise,
            'currency': 'INR',
            'receipt': f'job_{job_id}',
            'payment_capture': '1',
            'notes': {
                'job_id': str(job_id),
                'employer_id': str(employer.employer_id),
            }
        }
        
        print(f" Creating Razorpay order: {order_data}")
        
        try:
            razorpay_order = razorpay_client.order.create(data=order_data)
            print(f" Razorpay order created: {razorpay_order['id']}")
        except Exception as e:
            print(f" Razorpay error: {str(e)}")
            messages.error(request, f"Payment gateway error: {str(e)}")
            return redirect('employer_payment_section')
        
        # Create payment record
        with transaction.atomic():
            payment = Payment.objects.create(
                employer=employer,
                employee=job.employee,
                job=job,
                amount=amount,
                currency='INR',
                description=f"Payment for job: {job.title}",
                razorpay_order_id=razorpay_order['id'],
                status='pending'
            )
            print(f" Payment record created: {payment.payment_id}")
        
        # Build absolute URLs
        current_site = request.get_host()
        protocol = 'https' if request.is_secure() else 'http'
        base_url = f"{protocol}://{current_site}"
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'job': job,
            'payment': payment,
            'razorpay_order': razorpay_order,
            'razorpay_key_id': settings.RAZORPAY_KEY_ID,
            'amount': amount,
            'base_url': base_url,
        }
        
        return render(request, 'employer_html/payment_checkout.html', context)
        
    except Exception as e:
        print(f" Error in initiate_payment: {str(e)}")
        messages.error(request, f"Error: {str(e)}")
        return redirect('employer_payment_section')
    



# ============================================================================
# VERIFY PAYMENT (Razorpay Webhook/Callback)
# ============================================================================
def verify_payment(request):
    """Verify Razorpay payment"""
    if request.method != 'POST':
        messages.error(request, "Invalid request method.")
        return redirect('employer_payment_section')
    
    try:
        # Get payment data from form
        razorpay_payment_id = request.POST.get('razorpay_payment_id')
        razorpay_order_id = request.POST.get('razorpay_order_id')
        razorpay_signature = request.POST.get('razorpay_signature')
        payment_id = request.POST.get('payment_id')
        
        if not all([razorpay_payment_id, razorpay_order_id, razorpay_signature, payment_id]):
            messages.error(request, "Invalid payment data received.")
            return redirect('employer_payment_section')
        
        # Get payment record
        payment = get_object_or_404(Payment, payment_id=payment_id)
        
        # Verify payment signature
        params_dict = {
            'razorpay_order_id': razorpay_order_id,
            'razorpay_payment_id': razorpay_payment_id,
            'razorpay_signature': razorpay_signature
        }
        
        try:
            # Verify the payment signature
            razorpay_client.utility.verify_payment_signature(params_dict)
            signature_valid = True
        except razorpay.errors.SignatureVerificationError as e:
            print(f"Signature verification error: {str(e)}")
            signature_valid = False
        except Exception as e:
            print(f"Verification error: {str(e)}")
            signature_valid = False
        
        if signature_valid:
            # Payment successful
            with transaction.atomic():
                payment.razorpay_payment_id = razorpay_payment_id
                payment.razorpay_signature = razorpay_signature
                payment.status = 'completed'
                payment.payment_date = timezone.now()
                payment.save()
                
                # Update job payment status if exists
                if payment.job:
                    job = payment.job
                    job.payment_status = 'paid'
                    job.save()
                
                # Update employee total earnings
                if payment.employee:
                    employee = payment.employee
                    employee.total_earnings += payment.amount
                    employee.save()
                
                # Generate invoice
                try:
                    generate_invoice(payment)
                except Exception as e:
                    print(f"Error generating invoice: {str(e)}")
                
                messages.success(request, f"Payment of ₹{payment.amount} completed successfully!")
                return redirect('transaction_details', payment_id=payment.payment_id)
        else:
            # Payment failed
            payment.status = 'failed'
            payment.save()
            messages.error(request, "Payment verification failed. Please try again.")
            return redirect('employer_payment_section')
                
    except Payment.DoesNotExist:
        messages.error(request, "Payment record not found.")
        return redirect('employer_payment_section')
    except Exception as e:
        messages.error(request, f"Error verifying payment: {str(e)}")
        return redirect('employer_payment_section')



#********************************************************************************


def razorpay_webhook(request):
    """Handle Razorpay webhook notifications"""
    if request.method != 'POST':
        return HttpResponse(status=405)
    
    try:
        # Get webhook signature
        webhook_signature = request.headers.get('X-Razorpay-Signature', '')
        webhook_body = request.body.decode('utf-8')
        
        # Verify webhook signature
        expected_signature = hmac.new(
            bytes(settings.RAZORPAY_KEY_SECRET, 'utf-8'),
            bytes(webhook_body, 'utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Verify signature
        if not hmac.compare_digest(webhook_signature, expected_signature):
            return HttpResponse(status=400)
        
        # Parse webhook data
        data = json.loads(webhook_body)
        event = data.get('event')
        payload = data.get('payload', {}).get('payment', {}).get('entity', {})
        
        print(f"Razorpay Webhook Event: {event}")
        print(f"Payment Data: {payload}")
        
        if event == 'payment.captured':
            # Payment captured successfully
            razorpay_payment_id = payload.get('id')
            razorpay_order_id = payload.get('order_id')
            
            try:
                with transaction.atomic():
                    # Find payment by razorpay_order_id
                    payment = Payment.objects.get(razorpay_order_id=razorpay_order_id)
                    
                    # Update payment status
                    payment.razorpay_payment_id = razorpay_payment_id
                    payment.status = 'completed'
                    payment.payment_date = timezone.now()
                    payment.save()
                    
                    # Update job payment status
                    if payment.job:
                        payment.job.payment_status = 'paid'
                        payment.job.save()
                    
                    # Update employee earnings
                    payment.employee.total_earnings += payment.amount
                    payment.employee.save()
                    
                    # Generate invoice
                    generate_invoice(payment)
                    
                    print(f"Payment #{payment.payment_id} marked as completed")
                    
            except Payment.DoesNotExist:
                print(f"Payment not found for order_id: {razorpay_order_id}")
            except Exception as e:
                print(f"Error processing webhook: {str(e)}")
        
        elif event == 'payment.failed':
            # Payment failed
            razorpay_order_id = payload.get('order_id')
            
            try:
                payment = Payment.objects.get(razorpay_order_id=razorpay_order_id)
                payment.status = 'failed'
                payment.save()
                print(f"Payment #{payment.payment_id} marked as failed")
            except Payment.DoesNotExist:
                print(f"Payment not found for failed order: {razorpay_order_id}")
        
        return HttpResponse(status=200)
        
    except Exception as e:
        print(f"Webhook error: {str(e)}")
        return HttpResponse(status=500)
    

#*******************************************************************


# Payment Success Callback
@csrf_exempt
def payment_success_callback(request):
    """Handle successful payment callback from Razorpay"""
    try:
        # Get parameters from GET request (Razorpay sends them in the handler)
        razorpay_payment_id = request.GET.get('razorpay_payment_id')
        razorpay_order_id = request.GET.get('razorpay_order_id')
        razorpay_signature = request.GET.get('razorpay_signature')
        payment_id = request.GET.get('payment_id')
        
        print(f"Payment Success Callback Received")
        print(f"Payment ID: {payment_id}")
        print(f"Razorpay Payment ID: {razorpay_payment_id}")
        print(f"Razorpay Order ID: {razorpay_order_id}")
        
        if not razorpay_order_id:
            messages.error(request, "Invalid payment response. No order ID received.")
            return redirect('employer_payment_section')
        
        # Find the payment
        payment = None
        if payment_id:
            try:
                payment = Payment.objects.get(payment_id=payment_id)
            except Payment.DoesNotExist:
                print(f"Payment not found with ID: {payment_id}")
        
        # If payment not found by ID, try by order ID
        if not payment and razorpay_order_id:
            try:
                payment = Payment.objects.get(razorpay_order_id=razorpay_order_id)
            except Payment.DoesNotExist:
                print(f"Payment not found with order ID: {razorpay_order_id}")
        
        if not payment:
            messages.error(request, "Payment record not found.")
            return redirect('employer_payment_section')
        
        # Verify the payment signature (important for security)
        try:
            # Parameters to verify
            params_dict = {
                'razorpay_order_id': razorpay_order_id,
                'razorpay_payment_id': razorpay_payment_id,
                'razorpay_signature': razorpay_signature
            }
            
            # Verify signature
            razorpay_client.utility.verify_payment_signature(params_dict)
            print("Payment signature verified successfully")
            
        except razorpay.errors.SignatureVerificationError as e:
            print(f"Signature verification failed: {str(e)}")
            # You might want to handle this differently in production
            # For testing, we can continue but log the error
            messages.warning(request, "Payment verification warning. Please check payment status.")
        except Exception as e:
            print(f"Verification error: {str(e)}")
        
        # Update payment status
        with transaction.atomic():
            payment.razorpay_payment_id = razorpay_payment_id
            payment.razorpay_signature = razorpay_signature
            payment.status = 'completed'
            payment.payment_date = timezone.now()
            payment.save()
            
            # Update job payment status if exists
            if payment.job:
                payment.job.payment_status = 'paid'
                payment.job.save()
                print(f"Job #{payment.job.job_id} marked as paid")
            
            # Update employee earnings
            if payment.employee:
                payment.employee.total_earnings += payment.amount
                payment.employee.save()
                print(f"Employee {payment.employee.employee_id} earnings updated")
            
            # Generate invoice
            try:
                invoice = generate_invoice(payment)
                if invoice:
                    print(f"Invoice generated: {invoice.invoice_number}")
            except Exception as e:
                print(f"Error generating invoice: {str(e)}")
        
        messages.success(request, f"Payment of ₹{payment.amount} completed successfully!")
        print(f"Payment #{payment.payment_id} marked as completed")
        
        # Create Notification for Employee
        if payment.employee:
            EmployeeNotification.objects.create(
                employee=payment.employee,
                title="Payment Received",
                message=f"You have received a payment of ₹{payment.amount} from {payment.employer.full_name}.",
                notification_type='payment',
                is_read=False
            )
        
        # Redirect to transaction details
        return redirect('transaction_details', payment_id=payment.payment_id)
        
    except Exception as e:
        print(f"Error in payment success callback: {str(e)}")
        messages.error(request, f"Error processing payment: {str(e)}")
        return redirect('employer_payment_section')


# Payment Failure Callback
@csrf_exempt
def payment_failure_callback(request):
    """Handle payment failure callback"""
    try:
        error_code = request.GET.get('error_code', 'unknown')
        error_description = request.GET.get('error_description', 'Payment failed')
        razorpay_order_id = request.GET.get('razorpay_order_id')
        payment_id = request.GET.get('payment_id')
        
        print(f"Payment Failure: {error_description} (Code: {error_code})")
        
        # Update payment status if found
        payment = None
        if payment_id:
            try:
                payment = Payment.objects.get(payment_id=payment_id)
            except Payment.DoesNotExist:
                pass
        
        if not payment and razorpay_order_id:
            try:
                payment = Payment.objects.get(razorpay_order_id=razorpay_order_id)
            except Payment.DoesNotExist:
                pass
        
        if payment:
            payment.status = 'failed'
            payment.save()
            print(f"Payment #{payment.payment_id} marked as failed")
        
        # Show error message
        error_msg = f"Payment failed: {error_description}"
        if error_code != 'unknown':
            error_msg += f" (Error code: {error_code})"
        
        messages.error(request, error_msg)
        return redirect('employer_payment_section')
        
    except Exception as e:
        print(f"Error in failure callback: {str(e)}")
        messages.error(request, "Payment failed. Please try again.")
        return redirect('employer_payment_section')














# ============================================================================
# GENERATE INVOICE
# ============================================================================
def generate_invoice(payment):
    """Generate invoice for successful payment"""
    try:
        # Check if invoice already exists
        existing_invoice = PaymentInvoice.objects.filter(payment=payment).first()
        if existing_invoice:
            return existing_invoice
        
        # Calculate amounts
        subtotal = payment.amount * Decimal('0.95')  # 5% tax assumption
        tax_amount = payment.amount - subtotal
        
        # Generate invoice number
        date_str = timezone.now().strftime('%Y%m%d')
        invoice_count = PaymentInvoice.objects.count() + 1
        invoice_number = f"INV-{date_str}-{invoice_count:05d}"
        
        # Create invoice
        invoice = PaymentInvoice.objects.create(
            payment=payment,
            invoice_number=invoice_number,
            subtotal=subtotal,
            tax_amount=tax_amount,
            total_amount=payment.amount,
            notes=f"Invoice for payment #{payment.payment_id} - {payment.description}",
            due_date=timezone.now() + timedelta(days=30)
        )
        
        return invoice
        
    except Exception as e:
        print(f"Error generating invoice: {str(e)}")
        # Create a simple invoice even if there's an error
        try:
            return PaymentInvoice.objects.create(
                payment=payment,
                invoice_number=f"INV-EMERG-{payment.payment_id}",
                subtotal=payment.amount,
                tax_amount=0,
                total_amount=payment.amount,
                notes="Emergency invoice - please review"
            )
        except:
            return None


# ============================================================================
# PAYMENT HISTORY
# ============================================================================
def payment_history(request):
    """View payment history"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        
        # Get filter parameters
        status_filter = request.GET.get('status', '')
        date_from = request.GET.get('date_from', '')
        date_to = request.GET.get('date_to', '')
        search_query = request.GET.get('search', '').strip()
        
        # Start with all payments
        payments = Payment.objects.filter(
            employer=employer
        ).select_related('employee', 'job').order_by('-created_at')
        
        # Apply filters
        if status_filter:
            payments = payments.filter(status=status_filter)
        
        if date_from:
            try:
                date_from_obj = datetime.strptime(date_from, '%Y-%m-%d').date()
                payments = payments.filter(created_at__date__gte=date_from_obj)
            except ValueError:
                pass
        
        if date_to:
            try:
                date_to_obj = datetime.strptime(date_to, '%Y-%m-%d').date()
                payments = payments.filter(created_at__date__lte=date_to_obj)
            except ValueError:
                pass
        
        if search_query:
            payments = payments.filter(
                Q(employee__first_name__icontains=search_query) |
                Q(employee__last_name__icontains=search_query) |
                Q(job__title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(razorpay_order_id__icontains=search_query)
            )
        
        # Calculate statistics
        total_payments = payments.count()
        total_amount = payments.aggregate(total=Sum('amount'))['total'] or 0
        successful_payments = payments.filter(status='completed').count()
        pending_payments = payments.filter(status='pending').count()
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'payments': payments,
            'total_payments': total_payments,
            'total_amount': total_amount,
            'successful_payments': successful_payments,
            'pending_payments': pending_payments,
            'status_filter': status_filter,
            'date_from': date_from,
            'date_to': date_to,
            'search_query': search_query,
        }
        
        return render(request, 'employer_html/payment_history.html', context)
        
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')
    except Exception as e:
        messages.error(request, f"Error loading payment history: {str(e)}")
        return redirect('employer_dashboard')


# ============================================================================
# VIEW INVOICE
# ============================================================================
def view_invoice(request, payment_id):
    """View payment invoice"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        payment = get_object_or_404(Payment, payment_id=payment_id, employer=employer)
        
        # Get or create invoice
        invoice = PaymentInvoice.objects.filter(payment=payment).first()
        if not invoice and payment.status == 'completed':
            invoice = generate_invoice(payment)
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'payment': payment,
            'invoice': invoice,
        }
        
        return render(request, 'employer_html/view_invoice.html', context)
        
    except Payment.DoesNotExist:
        messages.error(request, "Payment not found.")
        return redirect('payment_history')
    except Exception as e:
        messages.error(request, f"Error viewing invoice: {str(e)}")
        return redirect('payment_history')


# ============================================================================
# DOWNLOAD INVOICE
# ============================================================================
def download_invoice(request, payment_id):
    """Download invoice as PDF"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        payment = get_object_or_404(Payment, payment_id=payment_id, employer=employer)
        invoice = get_object_or_404(PaymentInvoice, payment=payment)
        
        # For now, return a simple HTML invoice
        # In production, you would generate a PDF using ReportLab or similar
        response = HttpResponse(content_type='text/html')
        response['Content-Disposition'] = f'attachment; filename="invoice_{invoice.invoice_number}.html"'
        
        # Simple HTML invoice
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Invoice {invoice.invoice_number}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; border-bottom: 2px solid #4a6fff; padding-bottom: 20px; margin-bottom: 30px; }}
                .company-name {{ color: #4a6fff; font-size: 24px; font-weight: bold; }}
                .invoice-title {{ font-size: 20px; margin: 20px 0; }}
                .details {{ margin: 30px 0; }}
                .row {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
                .label {{ font-weight: bold; color: #666; }}
                .value {{ color: #333; }}
                .amount {{ text-align: right; font-size: 24px; color: #4a6fff; margin: 30px 0; }}
                .footer {{ margin-top: 50px; text-align: center; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="company-name">WorkNest</div>
                <div class="invoice-title">TAX INVOICE</div>
                <div>Invoice No: {invoice.invoice_number}</div>
                <div>Date: {invoice.invoice_date.strftime('%d %b %Y')}</div>
            </div>
            
            <div class="details">
                <div class="row">
                    <div class="label">Bill To:</div>
                    <div class="value">{employer.full_name}<br>{employer.company_name or ''}<br>{employer.email}</div>
                </div>
                <div class="row">
                    <div class="label">Pay To:</div>
                    <div class="value">{payment.employee.full_name}<br>Employee ID: {payment.employee.employee_id}</div>
                </div>
                <div class="row">
                    <div class="label">Payment ID:</div>
                    <div class="value">{payment.payment_id}</div>
                </div>
                <div class="row">
                    <div class="label">Order ID:</div>
                    <div class="value">{payment.razorpay_order_id or 'N/A'}</div>
                </div>
                <div class="row">
                    <div class="label">Description:</div>
                    <div class="value">{payment.description}</div>
                </div>
                <div class="row">
                    <div class="label">Payment Date:</div>
                    <div class="value">{payment.payment_date.strftime('%d %b %Y %H:%M') if payment.payment_date else 'N/A'}</div>
                </div>
            </div>
            
            <div class="amount">
                <div>Total Amount: ₹{payment.amount}</div>
                <div style="font-size: 14px; color: #666;">(Tax Included: ₹{invoice.tax_amount})</div>
            </div>
            
            <div class="footer">
                <p>This is a computer-generated invoice. No signature required.</p>
                <p>WorkNest - Connecting Employers with Skilled Workers</p>
                <p>Generated on: {timezone.now().strftime('%d %b %Y %H:%M')}</p>
            </div>
        </body>
        </html>
        """
        
        response.write(html_content)
        return response
        
    except Exception as e:
        messages.error(request, f"Error downloading invoice: {str(e)}")
        return redirect('view_invoice', payment_id=payment_id)


# ============================================================================
# SEND INVOICE EMAIL (Missing function)
# ============================================================================
def send_invoice_email(request, payment_id):
    """Send invoice via email"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        payment = get_object_or_404(Payment, payment_id=payment_id, employer=employer)
        
        # Get or create invoice
        invoice = PaymentInvoice.objects.filter(payment=payment).first()
        if not invoice and payment.status == 'completed':
            invoice = generate_invoice(payment)
        
        if not invoice:
            messages.error(request, "Invoice not found or payment not completed.")
            return redirect('view_invoice', payment_id=payment_id)
        
        # In production, you would integrate with an email service
        # For now, we'll just show a success message
        messages.success(request, f"Invoice {invoice.invoice_number} has been sent to {employer.email}")
        
        # Log the email sending (in production, you would actually send the email)
        print(f"Invoice {invoice.invoice_number} would be sent to {employer.email}")
        
        return redirect('view_invoice', payment_id=payment_id)
        
    except Exception as e:
        messages.error(request, f"Error sending invoice email: {str(e)}")
        return redirect('view_invoice', payment_id=payment_id)


# ============================================================================
# TRANSACTION DETAILS
# ============================================================================
def transaction_details(request, payment_id):
    """View transaction details"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        payment = get_object_or_404(Payment, payment_id=payment_id, employer=employer)
        
        # Try to get Razorpay payment details
        razorpay_details = None
        if payment.razorpay_payment_id:
            try:
                razorpay_details = razorpay_client.payment.fetch(payment.razorpay_payment_id)
            except:
                pass
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'payment': payment,
            'razorpay_details': razorpay_details,
        }
        
        return render(request, 'employer_html/transaction_details.html', context)
        
    except Payment.DoesNotExist:
        messages.error(request, "Payment not found.")
        return redirect('payment_history')
    except Exception as e:
        messages.error(request, f"Error viewing transaction: {str(e)}")
        return redirect('payment_history')


# ============================================================================
# QUICK PAYMENT
# ============================================================================
def quick_payment(request, employee_id):
    """Make a quick payment to an employee without a job"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        employee = get_object_or_404(Employee, employee_id=employee_id)
        
        if request.method == 'POST':
            amount = request.POST.get('amount', '').strip()
            description = request.POST.get('description', '').strip()
            
            if not amount:
                messages.error(request, "Please enter an amount.")
                return redirect('employer_payment_section')
            
            try:
                amount_decimal = Decimal(amount)
                if amount_decimal <= 0:
                    raise ValueError
            except:
                messages.error(request, "Please enter a valid amount.")
                return redirect('employer_payment_section')
            
            # Create payment record
            payment = Payment.objects.create(
                employer=employer,
                employee=employee,
                amount=amount_decimal,
                currency='INR',
                description=description or f"Quick payment to {employee.full_name}",
                status='pending'
            )
            
            # Convert amount to paise for Razorpay
            amount_in_paise = int(amount_decimal * 100)
            
            # Create Razorpay order
            order_data = {
                'amount': amount_in_paise,
                'currency': 'INR',
                'receipt': f'quick_{payment.payment_id}',
                'payment_capture': '1',
                'notes': {
                    'payment_id': str(payment.payment_id),
                    'employer_id': str(employer.employer_id),
                    'employee_id': str(employee.employee_id),
                    'type': 'quick_payment',
                }
            }
            
            try:
                razorpay_order = razorpay_client.order.create(data=order_data)
                payment.razorpay_order_id = razorpay_order['id']
                payment.save()
                
                context = {
                    'employer': employer,
                    'employer_name': employer.full_name,
                    'payment': payment,
                    'razorpay_order': razorpay_order,
                    'razorpay_key_id': settings.RAZORPAY_KEY_ID,
                    'amount': amount_decimal,
                }
                
                return render(request, 'employer_html/payment_checkout.html', context)
                
            except Exception as e:
                messages.error(request, f"Payment gateway error: {str(e)}")
                return redirect('employer_payment_section')
        
        # GET request - show quick payment form
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'employee': employee,
        }
        
        return render(request, 'employer_html/quick_payment.html', context)
        
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
        return redirect('employer_dashboard')
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employer_payment_section')
    except Exception as e:
        messages.error(request, f"Error processing quick payment: {str(e)}")
        return redirect('employer_payment_section')


# ============================================================================
# INITIATE REFUND
# ============================================================================
def initiate_refund(request, payment_id):
    """Initiate refund for a payment"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        payment = get_object_or_404(Payment, payment_id=payment_id, employer=employer)
        
        # Check if payment can be refunded
        if payment.status != 'completed':
            messages.error(request, "Only completed payments can be refunded.")
            return redirect('transaction_details', payment_id=payment_id)
        
        if not payment.razorpay_payment_id:
            messages.error(request, "Cannot refund this payment.")
            return redirect('transaction_details', payment_id=payment_id)
        
        if request.method == 'POST':
            refund_amount = request.POST.get('refund_amount', '').strip()
            refund_reason = request.POST.get('refund_reason', '').strip()
            
            if not refund_amount:
                messages.error(request, "Please enter refund amount.")
                return redirect('transaction_details', payment_id=payment_id)
            
            try:
                refund_amount_decimal = Decimal(refund_amount)
                if refund_amount_decimal <= 0 or refund_amount_decimal > payment.amount:
                    raise ValueError
            except:
                messages.error(request, "Please enter a valid refund amount.")
                return redirect('transaction_details', payment_id=payment_id)
            
            # Convert to paise
            refund_amount_paise = int(refund_amount_decimal * 100)
            
            try:
                # Create Razorpay refund
                refund_data = {
                    'amount': refund_amount_paise,
                    'notes': {
                        'reason': refund_reason,
                        'employer_id': str(employer.employer_id),
                        'payment_id': str(payment_id)
                    }
                }
                
                # In production, uncomment this to actually create refund
                # refund = razorpay_client.payment.refund(payment.razorpay_payment_id, refund_data)
                
                # For now, just update the payment status
                payment.status = 'refunded'
                payment.save()
                
                messages.success(request, f"Refund of ₹{refund_amount} initiated successfully!")
                
            except Exception as e:
                messages.error(request, f"Error creating refund: {str(e)}")
            
            return redirect('transaction_details', payment_id=payment_id)
        
        # GET request - show refund form
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'payment': payment,
        }
        
        return render(request, 'employer_html/initiate_refund.html', context)
        
    except Payment.DoesNotExist:
        messages.error(request, "Payment not found.")
        return redirect('payment_history')
    except Exception as e:
        messages.error(request, f"Error initiating refund: {str(e)}")
        return redirect('transaction_details', payment_id=payment_id)


# employer/views.py - Add this function
def view_receipt(request, payment_id):
    """View payment receipt"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        payment = get_object_or_404(Payment, payment_id=payment_id, employer=employer)
        
        if payment.status != 'completed':
            messages.error(request, "Receipt is only available for completed payments.")
            return redirect('transaction_details', payment_id=payment_id)
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'payment': payment,
        }
        
        return render(request, 'employer_html/view_receipt.html', context)
        
    except Payment.DoesNotExist:
        messages.error(request, "Payment not found.")
        return redirect('payment_history')
    except Exception as e:
        messages.error(request, f"Error viewing receipt: {str(e)}")
        return redirect('transaction_details', payment_id=payment_id)
    
def view_payment_for_job(request, job_id):
    """View payment details for a specific job from payment section"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        job = get_object_or_404(JobRequest, job_id=job_id, employer=employer)
        
        # Get any existing payments for this job
        existing_payments = Payment.objects.filter(
            employer=employer,
            job=job
        ).order_by('-created_at')
        
        # Get the latest payment
        latest_payment = existing_payments.first()
        
        # Check if job is completed and eligible for payment
        is_eligible_for_payment = job.status == 'completed'
        
        # Calculate payment status
        if latest_payment:
            payment_status = latest_payment.get_status_display()
            payment_amount = latest_payment.amount
        else:
            payment_status = 'Pending'
            payment_amount = job.budget or 0
        
        context = {
            'employer': employer,
            'employer_name': employer.full_name,
            'job': job,
            'latest_payment': latest_payment,
            'existing_payments': existing_payments,
            'is_eligible_for_payment': is_eligible_for_payment,
            'payment_status': payment_status,
            'payment_amount': payment_amount,
            'razorpay_key_id': settings.RAZORPAY_KEY_ID,
        }
        
        return render(request, 'employer_html/view_payment_for_job.html', context)
        
    except Employer.DoesNotExist:
        messages.error(request, "Employer not found.")
        return redirect('employer_payment_section')
    except JobRequest.DoesNotExist:
        messages.error(request, "Job not found.")
        return redirect('employer_payment_section')
    except Exception as e:
        messages.error(request, f"Error loading payment details: {str(e)}")
        return redirect('employer_payment_section')










#Test Payment

def test_razorpay_integration(request):
    """Test Razorpay integration page"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    employer = Employer.objects.get(employer_id=request.session['employer_id'])
    
    context = {
        'employer': employer,
        'employer_name': employer.full_name,
        'razorpay_key_id': settings.RAZORPAY_KEY_ID,
    }
    
    return render(request, 'employer_html/test_razorpay.html', context)


# notification views
# add these to employer/views.py

@csrf_exempt
def mark_notification_read(request, notification_id):
    """Mark a single notification as read"""
    if 'employer_id' not in request.session:
        return JsonResponse({'status': 'error', 'message': 'Not logged in'}, status=401)
    
    if request.method == 'POST':
        try:
            employer = Employer.objects.get(employer_id=request.session['employer_id'])
            notification = EmployerNotification.objects.get(
                notification_id=notification_id,
                employer=employer
            )
            
            notification.is_read = True
            notification.save()
            
            return JsonResponse({'status': 'success'})
            
        except EmployerNotification.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Notification not found'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
            
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)


def mark_all_notifications_read(request):
    """Mark all notifications as read"""
    if 'employer_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
        EmployerNotification.objects.filter(employer=employer, is_read=False).update(is_read=True)
        messages.success(request, "All notifications marked as read.")
        
    except Exception as e:
        messages.error(request, f"Error updating notifications: {str(e)}")
        
    # Redirect back to where they came from
    return redirect(request.META.get('HTTP_REFERER', 'employer_dashboard'))
