
# Create your views here.

# employee/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth.hashers import make_password, check_password
from django.utils import timezone
from django.db import transaction
from django.http import JsonResponse, HttpResponse
# from django.core.files.storage import FileSystemStorage
from home.models import Location 
from django.db.models import Q, Sum, Count, Avg
from datetime import datetime, timedelta
import calendar
from .models import JobRequest, JobAction, Review
import re

from django.core.paginator import Paginator
from employer.models import Employer, SiteReview


# Import models properly
from .models import (
    Employee, 
    EmployeeLogin, 
    EmployeeExperience, 
    EmployeeCertificate, 
    EmployeePortfolio, 
    EmployeeSkill,
    EmployeeNotification,
    EmployeeAvailability
)






#**********************************************************

def employee_dashboard(request):
    # Check if employee is logged in
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Get current date and time
        now = timezone.now()
        current_date = now.strftime("%A, %B %d, %Y")
        current_time = now.strftime("%I:%M %p")
        
        # Determine greeting based on time of day
        hour = now.hour
        if hour < 12:
            greeting_message = "Good morning! Ready for a productive day?"
        elif hour < 17:
            greeting_message = "Good afternoon! Hope you're having a great day."
        else:
            greeting_message = "Good evening! How was your day?"
        
        # Calculate statistics
        total_jobs = JobRequest.objects.filter(employee=employee).count()
        
        # Calculate jobs from last week
        week_ago = now - timedelta(days=7)
        jobs_last_week = JobRequest.objects.filter(
            employee=employee,
            created_at__gte=week_ago
        ).count()
        
        job_change = total_jobs - jobs_last_week
        
        # Get review count from employee reviews
        review_count = employee.reviews.count()
        
        # Calculate last month earnings
        month_ago = now - timedelta(days=30)
        last_month_earnings = JobRequest.objects.filter(
            employee=employee,
            status='completed',
            completed_at__gte=month_ago
        ).aggregate(total=Sum('budget'))['total'] or 0
        
        # Calculate profile completion with detailed tracking
        profile_items = {
            'profile_photo': {
                'name': 'Profile Photo',
                'completed': bool(employee.profile_image),
                'count': 1 if employee.profile_image else 0,
                'total': 1
            },
            'bio': {
                'name': 'About Me',
                'completed': bool(employee.bio and len(employee.bio.strip()) > 0),
                'count': 1 if employee.bio and len(employee.bio.strip()) > 0 else 0,
                'total': 1
            },
            'skills': {
                'name': 'Skills',
                'completed': bool(employee.skills and len(employee.skills.strip()) > 0),
                'count': EmployeeSkill.objects.filter(employee=employee).count(),
                'total': 1
            },
            'experience': {
                'name': 'Work Experience',
                'completed': EmployeeExperience.objects.filter(employee=employee).exists(),
                'count': EmployeeExperience.objects.filter(employee=employee).count(),
                'total': 1
            },
            'certificates': {
                'name': 'Certificates',
                'completed': EmployeeCertificate.objects.filter(employee=employee).exists(),
                'count': EmployeeCertificate.objects.filter(employee=employee).count(),
                'total': 1
            },
            'portfolio': {
                'name': 'Portfolio',
                'completed': EmployeePortfolio.objects.filter(employee=employee).exists(),
                'count': EmployeePortfolio.objects.filter(employee=employee).count(),
                'total': 1
            }
        }
        
        # Calculate completion percentages
        completed_items = sum(1 for item in profile_items.values() if item['completed'])
        total_items = len(profile_items)
        profile_completion = int((completed_items / total_items) * 100) if total_items > 0 else 0
        
        # Calculate circle progress offset (314 is circumference for r=50)
        profile_circle_offset = 314 - (profile_completion / 100 * 314)
        
        # Get profile status message
        if profile_completion >= 90:
            profile_status_message = "Excellent! Your profile is almost complete."
        elif profile_completion >= 70:
            profile_status_message = "Good job! Your profile looks great."
        elif profile_completion >= 50:
            profile_status_message = "Halfway there! Complete more sections."
        else:
            profile_status_message = "Start completing your profile to get more job opportunities."
        
        # Get recent activities (job actions)
        recent_activities = []
        recent_actions = JobAction.objects.filter(
            Q(employee=employee) | Q(job__employee=employee)
        ).select_related('job').order_by('-created_at')[:5]
        
        for action in recent_actions:
            activity = {
                'title': f"Job #{action.job.job_id} - {action.get_action_type_display()}",
                'description': action.notes or f"Job '{action.job.title}' was {action.get_action_type_display().lower()}",
                'time': action.created_at.strftime("%I:%M %p"),
                'icon': get_activity_icon(action.action_type)
            }
            recent_activities.append(activity)
        
        # Add profile completion as activity if low
        if profile_completion < 70:
            recent_activities.insert(0, {
                'title': "Complete Your Profile",
                'description': f"Your profile is {profile_completion}% complete. Complete it to get more job opportunities.",
                'time': "Just now",
                'icon': 'user-check'
            })
        
        # Get recent job requests (last 3)
        recent_jobs = JobRequest.objects.filter(
            employee=employee
        ).select_related('employer').order_by('-created_at')[:3]
        
        # Prepare stats for template
        stats = {
            'total_jobs': total_jobs,
            'job_change': job_change,
            'success_rate': employee.success_rate,
            'review_count': review_count,
            'last_month_earnings': last_month_earnings
        }
        
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'employee_email': employee.email,
            'greeting_message': greeting_message,
            'current_date': current_date,
            'current_time': current_time,
            'stats': stats,
            'profile_completion': profile_completion,
            'profile_circle_offset': profile_circle_offset,
            'profile_completed_items': completed_items,
            'profile_pending_items': total_items - completed_items,
            'profile_items': profile_items,
            'profile_status_message': profile_status_message,
            'recent_activities': recent_activities,
            'recent_jobs': recent_jobs,
            'recent_jobs': recent_jobs,
            # Notifications are now handled by context processor
        }
        
        return render(request, 'employee_html/employee_dashboard.html', context)
        
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('index')
    except Exception as e:
        messages.error(request, f"Error loading dashboard: {str(e)}")
        return redirect('index')


#**********************************************************************************

def get_activity_icon(action_type):
    """Map action type to FontAwesome icon"""
    icon_map = {
        'created': 'plus-circle',
        'viewed': 'eye',
        'accepted': 'check-circle',
        'rejected': 'times-circle',
        'completed': 'check-double',
        'cancelled': 'ban',
        'rescheduled': 'calendar-alt',
        'message': 'comment-alt'
    }
    return icon_map.get(action_type, 'bell')

#******************************************************


def employee_earnings(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Get filter parameters
        payment_method_filter = request.GET.get('payment_method', 'all')
        month_filter = request.GET.get('month', 'all')
        date_filter = request.GET.get('date_filter', 'all')
        search_query = request.GET.get('search', '').strip()
        chart_period = request.GET.get('chart_period', 'monthly')  # Default to monthly
        
        # Get completed jobs for this employee
        completed_jobs = JobRequest.objects.filter(
            employee=employee,
            status='completed'
        ).select_related('employer').order_by('-completed_at')
        
        # Apply search filter
        if search_query:
            completed_jobs = completed_jobs.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(employer__first_name__icontains=search_query) |
                Q(employer__last_name__icontains=search_query) |
                Q(employer__company_name__icontains=search_query)
            )
        
        # Apply date filter
        now = timezone.now()
        if date_filter == 'today':
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            completed_jobs = completed_jobs.filter(completed_at__gte=today_start)
        elif date_filter == 'week':
            week_ago = now - timedelta(days=7)
            completed_jobs = completed_jobs.filter(completed_at__gte=week_ago)
        elif date_filter == 'month':
            month_ago = now - timedelta(days=30)
            completed_jobs = completed_jobs.filter(completed_at__gte=month_ago)
        
        # Calculate statistics from actual data
        total_earnings = completed_jobs.aggregate(total=Sum('budget'))['total'] or 0
        total_jobs_completed = completed_jobs.count()
        
        # Calculate chart data based on selected period
        chart_labels = []
        chart_data = []
        
        if chart_period == 'daily':
            # Get last 30 days of data
            for i in range(29, -1, -1):
                date = now - timedelta(days=i)
                day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = date.replace(hour=23, minute=59, second=59, microsecond=999999)
                
                day_earnings = JobRequest.objects.filter(
                    employee=employee,
                    status='completed',
                    completed_at__gte=day_start,
                    completed_at__lte=day_end
                ).aggregate(total=Sum('budget'))['total'] or 0
                
                chart_data.append(float(day_earnings))
                chart_labels.append(date.strftime('%d %b'))
        
        elif chart_period == 'weekly':
            # Get last 12 weeks of data
            for i in range(11, -1, -1):
                week_end = now - timedelta(days=i*7)
                week_start = week_end - timedelta(days=6)
                week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
                week_end = week_end.replace(hour=23, minute=59, second=59, microsecond=999999)
                
                week_earnings = JobRequest.objects.filter(
                    employee=employee,
                    status='completed',
                    completed_at__gte=week_start,
                    completed_at__lte=week_end
                ).aggregate(total=Sum('budget'))['total'] or 0
                
                chart_data.append(float(week_earnings))
                chart_labels.append(f"Week {week_start.strftime('%d/%m')}")
        
        else:  # monthly (default)
            # Get last 6 months for chart
            for i in range(5, -1, -1):
                month_date = now - timedelta(days=30*i)
                month_start = month_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                
                month_earnings = JobRequest.objects.filter(
                    employee=employee,
                    status='completed',
                    completed_at__gte=month_start,
                    completed_at__lte=month_end
                ).aggregate(total=Sum('budget'))['total'] or 0
                
                chart_data.append(float(month_earnings))
                chart_labels.append(month_start.strftime('%b'))
        
        # Get payment history from completed jobs
        earnings_history = []
        for job in completed_jobs:
            # Determine payment method (you can enhance this with actual Payment model)
            payment_method = get_payment_method(job)
            
            # Format date
            payment_date = job.completed_at or job.updated_at
            date_str = payment_date.strftime('%d %b %Y')
            
            earnings_history.append({
                'date': date_str,
                'job_title': job.title,
                'employer': job.employer.company_name or f"{job.employer.first_name} {job.employer.last_name}",
                'amount': float(job.budget) if job.budget else 0,
                'payment_method': payment_method,
                'payment_date': payment_date
            })
        
        # Sort by date
        earnings_history.sort(key=lambda x: x['payment_date'], reverse=True)
        
        # Pagination
        paginator = Paginator(earnings_history, 10)
        page_number = request.GET.get('page', 1)
        earnings_page = paginator.get_page(page_number)
        
        # Calculate recent stats
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_earnings = JobRequest.objects.filter(
            employee=employee,
            status='completed',
            completed_at__gte=today_start
        ).aggregate(total=Sum('budget'))['total'] or 0
        
        week_ago = now - timedelta(days=7)
        weekly_earnings = JobRequest.objects.filter(
            employee=employee,
            status='completed',
            completed_at__gte=week_ago
        ).aggregate(total=Sum('budget'))['total'] or 0
        
        month_ago = now - timedelta(days=30)
        monthly_earnings_total = JobRequest.objects.filter(
            employee=employee,
            status='completed',
            completed_at__gte=month_ago
        ).aggregate(total=Sum('budget'))['total'] or 0
        
        # Get average earnings per job
        avg_per_job = total_earnings / total_jobs_completed if total_jobs_completed > 0 else 0
        
        # Prepare context
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'employee_email': employee.email,
            'earnings_history': earnings_page,
            'total_earnings': total_earnings,
            'total_jobs_completed': total_jobs_completed,
            'today_earnings': today_earnings,
            'weekly_earnings': weekly_earnings,
            'monthly_earnings_total': monthly_earnings_total,
            'avg_per_job': avg_per_job,
            'chart_data': chart_data,
            'chart_labels': chart_labels,
            'chart_period': chart_period,
            'search_query': search_query,
            'payment_method_filter': payment_method_filter,
            'month_filter': month_filter,
            'date_filter': date_filter,
        }
        
        return render(request, 'employee_html/employee_earnings.html', context)
        
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employee_dashboard')
    except Exception as e:
        messages.error(request, f"Error loading earnings: {str(e)}")
        return redirect('employee_dashboard')


#**********************************************************


def get_payment_method(job):
    """Determine payment method for a job"""
    
    if job.employer_notes:
        notes_lower = job.employer_notes.lower()
        if 'cash' in notes_lower:
            return 'cash'
        elif 'upi' in notes_lower:
            return 'upi'
        elif 'bank' in notes_lower or 'transfer' in notes_lower:
            return 'bank'
    
    # Default based on amount
    if job.budget and job.budget >= 2000:
        return 'bank'
    elif job.budget and job.budget >= 500:
        return 'upi'
    else:
        return 'cash'

        

#*************************************************************

# Employee Job History View
def employee_job_history(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Check if we're viewing payment details
        view_payment = request.GET.get('view_payment')
        job_id = request.GET.get('job_id')
        
        if view_payment == 'true' and job_id:
            return job_history_payment(request, job_id)
        
        # Check if we're viewing review details
        view_review = request.GET.get('view_review')
        if view_review == 'true' and job_id:
            return job_history_review(request, job_id)
        
        # Get filter parameters
        status_filter = request.GET.get('status', 'all')
        sort_filter = request.GET.get('sort', 'newest')
        time_filter = request.GET.get('time', 'all')
        search_query = request.GET.get('search', '').strip()
        current_view = request.GET.get('view', 'all')
        
        # Start with all job requests for this employee (excluding pending)
        job_history = JobRequest.objects.filter(
            employee=employee
        ).exclude(status='pending').select_related('employer')
        
        # Apply status filter
        if status_filter != 'all':
            job_history = job_history.filter(status=status_filter)
        
        # Apply time filter
        now = timezone.now()
        if time_filter == 'week':
            week_ago = now - timedelta(days=7)
            job_history = job_history.filter(updated_at__gte=week_ago)
        elif time_filter == 'month':
            month_ago = now - timedelta(days=30)
            job_history = job_history.filter(updated_at__gte=month_ago)
        elif time_filter == 'quarter':
            quarter_ago = now - timedelta(days=90)
            job_history = job_history.filter(updated_at__gte=quarter_ago)
        elif time_filter == 'year':
            year_ago = now - timedelta(days=365)
            job_history = job_history.filter(updated_at__gte=year_ago)
        
        # Apply search filter
        if search_query:
            job_history = job_history.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(employer__first_name__icontains=search_query) |
                Q(employer__last_name__icontains=search_query) |
                Q(employer__company_name__icontains=search_query) |
                Q(city__icontains=search_query) |
                Q(state__icontains=search_query)
            )
        
        # Apply sorting
        if sort_filter == 'newest':
            job_history = job_history.order_by('-updated_at')
        elif sort_filter == 'oldest':
            job_history = job_history.order_by('updated_at')
        elif sort_filter == 'rating-high':
            # This would require joining with reviews table
            # For now, sort by job completion date
            job_history = job_history.order_by('-completed_at')
        elif sort_filter == 'rating-low':
            job_history = job_history.order_by('completed_at')
        elif sort_filter == 'earning-high':
            job_history = job_history.order_by('-budget')
        
        # Calculate statistics
        total_jobs = JobRequest.objects.filter(
            employee=employee,
            status='completed'
        ).count()
        
        completed_jobs = total_jobs
        
        # Calculate total earnings from completed jobs
        total_earnings = JobRequest.objects.filter(
            employee=employee,
            status='completed'
        ).aggregate(total=Sum('budget'))['total'] or 0
        
        # Pagination
        paginator = Paginator(job_history, 10)  # 10 jobs per page
        page_number = request.GET.get('page', 1)
        job_history_page = paginator.get_page(page_number)
        
        # Prepare stats for template
        stats = {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'total_earnings': total_earnings,
        }
        
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'employee_email': employee.email,
            'job_history': job_history_page,
            'stats': stats,
            'current_filters': {
                'status': status_filter,
                'sort': sort_filter,
                'time': time_filter,
            },
            'current_view': current_view,
        }
        
        return render(request, 'employee_html/employee_job_history.html', context)
        
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employee_dashboard')
    except Exception as e:
        messages.error(request, f"Error loading job history: {str(e)}")
        return redirect('employee_dashboard')


#**********************************************************

# Job History Payment Details View
def job_history_payment(request, job_id):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Get job details
        job = get_object_or_404(JobRequest, job_id=job_id, employee=employee)
        
        # Determine payment details based on job status
        if job.status == 'completed':
            paid = True
            payment_date = job.completed_at or job.updated_at
            payment_method = 'Online Payment'  # This would come from a Payment model
            transaction_id = f'TX{job.job_id:08d}'
            amount = job.budget or 0
            cancellation_reason = None
        elif job.status == 'cancelled':
            paid = False
            payment_date = job.updated_at
            payment_method = None
            transaction_id = None
            amount = 0
            # Get cancellation reason from job actions
            from .models import JobAction
            cancellation_action = JobAction.objects.filter(
                job=job,
                action_type='cancelled'
            ).first()
            cancellation_reason = cancellation_action.notes if cancellation_action else "Job was cancelled"
        else:
            paid = False
            payment_date = job.updated_at
            payment_method = None
            transaction_id = None
            amount = job.budget or 0
            cancellation_reason = None
        
        payment_details = {
            'job': job,
            'employer_name': job.employer.company_name or job.employer.full_name,
            'paid': paid,
            'payment_date': payment_date,
            'payment_method': payment_method,
            'transaction_id': transaction_id,
            'amount': amount,
            'cancellation_reason': cancellation_reason,
        }
        
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'payment_details': payment_details,
        }
        
        return render(request, 'employee_html/employee_job_history.html', context)
        
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employee_dashboard')
    except Exception as e:
        messages.error(request, f"Error loading payment details: {str(e)}")
        return redirect('employee_job_history')


#**********************************************************

# Job History Review Details View
def job_history_review(request, job_id):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Get job details
        job = get_object_or_404(JobRequest, job_id=job_id, employee=employee)
        
        # Get reviews for this job
        reviews = Review.objects.filter(
            employee=employee,
            employer=job.employer
        ).order_by('-created_at')
        
        # Get employer reviews
        employer_reviews = Review.objects.filter(
            employer=job.employer
        ).order_by('-created_at')
        
        review_details = {
            'job': job,
            'reviews': reviews,
            'employer_reviews': employer_reviews,
        }
        
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'review_details': review_details,
        }
        
        return render(request, 'employee_html/employee_job_history.html', context)
        
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employee_dashboard')
    except Exception as e:
        messages.error(request, f"Error loading review details: {str(e)}")
        return redirect('employee_job_history')
    

#**************************************************************


def employee_review_list(request):
    """View for employee to see all reviews from employers"""
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Get all reviews for this employee
        reviews = Review.objects.filter(employee=employee).select_related('employer', 'job').order_by('-created_at')
        
        # Calculate statistics
        total_reviews = reviews.count()
        
        # Calculate average rating
        avg_rating = 0.0
        if total_reviews > 0:
            rating_avg = reviews.aggregate(avg_rating=Avg('rating'))
            avg_rating = rating_avg['avg_rating'] or 0.0
        
        # Calculate rating distribution
        rating_distribution = {
            '5': reviews.filter(rating=5).count(),
            '4': reviews.filter(rating=4).count(),
            '3': reviews.filter(rating=3).count(),
            '2': reviews.filter(rating=2).count(),
            '1': reviews.filter(rating=1).count(),
        }
        
        # Calculate positive reviews (4-5 stars)
        positive_reviews = reviews.filter(rating__gte=4).count()
        positive_percentage = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0
        
        # Get filter parameters
        rating_filter = request.GET.get('rating', '')
        date_filter = request.GET.get('date_filter', '')
        search_query = request.GET.get('search', '').strip()
        
        # Apply filters
        if rating_filter and rating_filter != 'all':
            reviews = reviews.filter(rating=int(rating_filter))
        
        if date_filter:
            now = timezone.now()
            if date_filter == 'month':
                month_ago = now - timedelta(days=30)
                reviews = reviews.filter(created_at__gte=month_ago)
            elif date_filter == '3months':
                three_months_ago = now - timedelta(days=90)
                reviews = reviews.filter(created_at__gte=three_months_ago)
            elif date_filter == 'year':
                year_ago = now - timedelta(days=365)
                reviews = reviews.filter(created_at__gte=year_ago)
        
        if search_query:
            reviews = reviews.filter(
                Q(employer__first_name__icontains=search_query) |
                Q(employer__last_name__icontains=search_query) |
                Q(employer__company_name__icontains=search_query) |
                Q(text__icontains=search_query) |
                Q(job__title__icontains=search_query)
            )
        
        # Get sentiment for each review
        for review in reviews:
            review.sentiment = compute_text_sentiment(review.text)
            review.sentiment_category = get_sentiment_category(review.sentiment)
        
        
        # Prepare context
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'employee_email': employee.email,
            'reviews': reviews,
            'total_reviews': total_reviews,
            'avg_rating': round(avg_rating, 1),
            'positive_percentage': round(positive_percentage, 1),
            'rating_distribution': rating_distribution,
            'rating_filter': rating_filter,
            'date_filter': date_filter,
            'search_query': search_query,
            # context processor handles notifications
        }
        
        return render(request, 'employee_html/employee_review_list.html', context)
        
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employee_dashboard')
    except Exception as e:

        raise e

#**********************************************************

def get_sentiment_category(sentiment_score):
    """Categorize sentiment score"""
    if sentiment_score >= 0.5:
        return 'positive'
    elif sentiment_score >= -0.3:
        return 'neutral'
    else:
        return 'negative'
    
#****************************************************

def compute_text_sentiment(text):
    """Simple sentiment analysis for reviews"""
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
        'proactive', 'responsible', 'conscientious', 'diligent', 'meticulous',
        'perfect', 'excellent', 'superior', 'exceptional', 'commendable'
    ]
    
    negative = [
        'bad', 'poor', 'terrible', 'lazy', 'unreliable', 'incompetent',
        'worst', 'awful', 'disappointing', 'late', 'dishonest', 'inefficient',
        'sloppy', 'unprofessional', 'horrible', 'frustrating', 'subpar',
        'careless', 'messy', 'slow', 'rude', 'impolite', 'unfriendly',
        'unskilled', 'inexperienced', 'negligent', 'inattentive', 'forgetful',
        'disorganized', 'chaotic', 'expensive', 'overpriced', 'unsatisfactory',
        'mediocre', 'average', 'ordinary'
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


#**************************************************************


# Employee Profile View
def employee_profile(request):
    # Check if employee is logged in
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Get related data
        experiences = EmployeeExperience.objects.filter(employee=employee).order_by('-start_date')
        certificates = EmployeeCertificate.objects.filter(employee=employee).order_by('-issue_date')
        portfolio_items = EmployeePortfolio.objects.filter(employee=employee).order_by('-upload_date')
        skills = EmployeeSkill.objects.filter(employee=employee)
        
        # Get skills list from both models
        employee_skills = list(skills.values_list('skill_name', flat=True))
        if employee.skills:
            skills_from_text = [s.strip() for s in employee.skills.split(',') if s.strip()]
            employee_skills.extend(skills_from_text)
        
        # Remove duplicates
        employee_skills = list(set(employee_skills))
        
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'employee_email': employee.email,
            'experiences': experiences,
            'certificates': certificates,
            'portfolio_items': portfolio_items,
            'skills': employee_skills,
            'profile_stats': employee.profile_stats,
        }
        
        return render(request, 'employee_html/employee_profile.html', context)
        
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employee_dashboard')


#**********************************************************************


def employee_job_request(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Check if we're viewing details of a specific job
        view_details = request.GET.get('view_details', 'false')
        job_id = request.GET.get('job_id')
        
        if view_details == 'true' and job_id:
            try:
                # Get job details
                job = JobRequest.objects.get(job_id=job_id, employee=employee)
                actions = JobAction.objects.filter(job=job).order_by('-created_at')
                
                context = {
                    'employee': employee,
                    'employee_name': employee.full_name,
                    'job_details': job,
                    'actions': actions,
                }
                return render(request, 'employee_html/employee_job_request.html', context)
                
            except JobRequest.DoesNotExist:
                messages.error(request, "Job request not found.")
                return redirect('employee_job_request')
        
        # Normal list view with tabs
        # Get current tab from request
        current_tab = request.GET.get('tab', 'new-jobs')
        
        # Get filter parameters
        status_filter = request.GET.get('status', 'all')
        time_filter = request.GET.get('time', 'all')
        search_query = request.GET.get('search', '').strip()
        
        # Start with all job requests for this employee
        job_requests = JobRequest.objects.filter(employee=employee)
        
        # Apply status filter
        if status_filter != 'all':
            job_requests = job_requests.filter(status=status_filter)
        
        # Apply time filter
        now = timezone.now()
        if time_filter == 'today':
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            job_requests = job_requests.filter(created_at__gte=today_start)
        elif time_filter == 'week':
            week_ago = now - timedelta(days=7)
            job_requests = job_requests.filter(created_at__gte=week_ago)
        elif time_filter == 'month':
            month_ago = now - timedelta(days=30)
            job_requests = job_requests.filter(created_at__gte=month_ago)
        
        # Apply search filter
        if search_query:
            job_requests = job_requests.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(category__icontains=search_query) |
                Q(location__icontains=search_query)
            )
        
        # Split into categories
        # 1. New Jobs (Pending)
        new_jobs = job_requests.filter(status='pending').order_by('-created_at')
        
        # 2. Active Jobs (Accepted)
        active_jobs = job_requests.filter(status='accepted').order_by('-updated_at')
        
        # 3. Previous Jobs (Completed, Cancelled)
        previous_jobs = job_requests.filter(status__in=['completed', 'cancelled']).order_by('-updated_at')
        
        # 4. Rejected Jobs
        rejected_jobs = job_requests.filter(status='rejected').order_by('-updated_at')
        
        # Count completed jobs
        completed_jobs_count = job_requests.filter(status='completed').count()
        
        # Get statistics
        total_requests = JobRequest.objects.filter(employee=employee).count()
        accepted_requests = JobRequest.objects.filter(employee=employee, status='accepted').count()
        pending_requests = JobRequest.objects.filter(employee=employee, status='pending').count()
        rejected_requests = JobRequest.objects.filter(employee=employee, status='rejected').count()
        completed_requests = JobRequest.objects.filter(employee=employee, status='completed').count()
        
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'new_jobs': new_jobs,
            'active_jobs': active_jobs,
            'previous_jobs': previous_jobs,
            'rejected_jobs': rejected_jobs,
            
            'new_jobs_count': new_jobs.count(),
            'active_jobs_count': active_jobs.count(),
            'previous_jobs_count': previous_jobs.count(),
            'rejected_jobs_count': rejected_jobs.count(),
            
            'completed_jobs_count': completed_jobs_count,
            'stats': {
                'total': total_requests,
                'accepted': accepted_requests,
                'pending': pending_requests,
                'rejected': rejected_requests,
                'completed': completed_requests,
            },
            'current_filters': {
                'status': status_filter,
                'time': time_filter,
            },
            'current_tab': current_tab,
        }
        return render(request, 'employee_html/employee_job_request.html', context)
        
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employee_dashboard')


#**************************************************************

def job_request_details(request, job_id):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Get job details
        job = JobRequest.objects.get(job_id=job_id, employee=employee)
        actions = JobAction.objects.filter(job=job).order_by('-created_at')
        
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'job_details': job,
            'actions': actions,
        }
        return render(request, 'employee_html/employee_job_request.html', context)
        
    except JobRequest.DoesNotExist:
        messages.error(request, "Job request not found.")
        return redirect('employee_job_request')
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('employee_dashboard')
    except Exception as e:
        messages.error(request, f"Error loading job details: {str(e)}")
        return redirect('employee_job_request')


#**************************************************************

# Accept Job Request
def accept_job_request(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            job_id = request.POST.get('job_id')
            
            if not job_id:
                messages.error(request, "Job ID is required.")
                return redirect('employee_job_request')
            
            job = JobRequest.objects.get(job_id=job_id, employee=employee)
            
            # Validate that job is pending
            if job.status != 'pending':
                messages.error(request, f"Job is already {job.get_status_display().lower()}.")
                return redirect('employee_job_request')
            
            # Update job status
            job.status = 'accepted'
            job.accepted_at = timezone.now()
            job.updated_at = timezone.now()
            job.save()
            
            # Create action record
            JobAction.objects.create(
                job=job,
                employee=employee,
                action_type='accepted',
                notes=f"Job accepted by {employee.full_name}"
            )
            
            messages.success(request, f"Job '{job.title}' accepted successfully!")
            
            # Create Notification
            EmployeeNotification.objects.create(
                employee=employee,
                title="Job Accepted",
                message=f"You have accepted the job request '{job.title}'.",
                notification_type='job',
                is_read=True,
                link=f"/employee/job/details/{job.job_id}/"
            )

            # Create Employer Notification
            from employer.models import EmployerNotification
            EmployerNotification.objects.create(
                employer=job.employer,
                title="Job Accepted",
                message=f"{employee.full_name} has accepted your job request '{job.title}'.",
                notification_type='job',
                is_read=False,
                link=f"/employer/job/details/{job.job_id}/" # Assuming this URL exists or similar
            )
            
        except JobRequest.DoesNotExist:
            messages.error(request, "Job request not found.")
        except Exception as e:
            messages.error(request, f"Error accepting job: {str(e)}")
    
    # Redirect back to the same page with details view if we were on details
    referer = request.META.get('HTTP_REFERER', '')
    if 'job/details' in referer or 'view_details=true' in referer:
        job_id = request.POST.get('job_id')
        if job_id:
            return redirect('job_request_details', job_id=job_id)
    
    return redirect('employee_job_request')


#**************************************************************

# Reject Job Request
def reject_job_request(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            job_id = request.POST.get('job_id')
            rejection_reason = request.POST.get('rejection_reason', '')
            
            if not job_id:
                messages.error(request, "Job ID is required.")
                return redirect('employee_job_request')
            
            job = JobRequest.objects.get(job_id=job_id, employee=employee)
            
            # Validate that job is pending
            if job.status != 'pending':
                messages.error(request, f"Job is already {job.get_status_display().lower()}.")
                return redirect('employee_job_request')
            
            # Update job status
            job.status = 'rejected'
            job.updated_at = timezone.now()
            job.save()
            
            # Create action record
            JobAction.objects.create(
                job=job,
                employee=employee,
                action_type='rejected',
                notes=f"Job rejected by {employee.full_name}. Reason: {rejection_reason}"
            )
            
            messages.success(request, f"Job '{job.title}' rejected successfully.")
            
            # Create Notification
            EmployeeNotification.objects.create(
                employee=employee,
                title="Job Rejected",
                message=f"You have rejected the job request '{job.title}'.",
                notification_type='job',
                is_read=True
            )

            # Create Employer Notification
            from employer.models import EmployerNotification
            EmployerNotification.objects.create(
                employer=job.employer,
                title="Job Rejected",
                message=f"{employee.full_name} has rejected your job request '{job.title}'.",
                notification_type='job',
                is_read=False
            )
            
        except JobRequest.DoesNotExist:
            messages.error(request, "Job request not found.")
        except Exception as e:
            messages.error(request, f"Error rejecting job: {str(e)}")
    
    # Redirect back to the same page with details view if we were on details
    referer = request.META.get('HTTP_REFERER', '')
    if 'job/details' in referer or 'view_details=true' in referer:
        job_id = request.POST.get('job_id')
        if job_id:
            return redirect('job_request_details', job_id=job_id)
    
    return redirect('employee_job_request')


#**************************************************************

def update_job_status(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            job_id = request.POST.get('job_id')
            new_status = request.POST.get('status')
            notes = request.POST.get('notes', '')
            
            if not job_id or not new_status:
                messages.error(request, "Job ID and status are required.")
                return redirect('employee_job_request')
            
            job = JobRequest.objects.get(job_id=job_id, employee=employee)
            
            # Validate status transition
            valid_transitions = {
                'pending': ['accepted', 'rejected'],
                'accepted': ['completed', 'cancelled'],
                'rejected': [],
                'completed': [],
                'cancelled': []
            }
            
            if new_status not in valid_transitions.get(job.status, []):
                messages.error(request, f"Cannot change status from {job.status} to {new_status}.")
                return redirect('employee_job_request')
            
            # Update job status
            old_status = job.status
            job.status = new_status
            job.updated_at = timezone.now()
            
            # Set completion time if marking as completed
            if new_status == 'completed':
                job.completed_at = timezone.now()
            
            job.save()
            
            # Create action record
            action_notes = f"Status changed from {old_status} to {new_status}."
            if notes:
                action_notes += f" Notes: {notes}"
            
            JobAction.objects.create(
                job=job,
                employee=employee,
                action_type=new_status,
                notes=action_notes
            )
            
            # Update employee stats if job is completed
            if new_status == 'completed':
                employee.total_jobs_done += 1
                employee.save()
            
            # Create Notification
            EmployeeNotification.objects.create(
                employee=employee,
                title=f"Job Status Updated",
                message=f"You updated the status of Job #{job.job_id} ({job.title}) to {new_status}.",
                notification_type='job',
                link=f"/employee/job/details/{job.job_id}/"
            )

            # Create Employer Notification
            from employer.models import EmployerNotification
            EmployerNotification.objects.create(
                employer=job.employer,
                title=f"Job Status Updated",
                message=f"{employee.full_name} updated the status of Job '{job.title}' to {new_status}.",
                notification_type='job',
                is_read=False
            )
            
            messages.success(request, f"Job status updated to {new_status}.")
            
        except JobRequest.DoesNotExist:
            messages.error(request, "Job request not found.")
        except Exception as e:
            messages.error(request, f"Error updating job status: {str(e)}")
    
    # Redirect back to the same page with details view if we were on details
    referer = request.META.get('HTTP_REFERER', '')
    if 'job/details' in referer or 'view_details=true' in referer:
        job_id = request.POST.get('job_id')
        if job_id:
            return redirect('job_request_details', job_id=job_id)
    
    return redirect('employee_job_request')


#**************************************************************

# Filter Job Requests
def filter_job_requests(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'GET':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            # Get filter parameters
            status = request.GET.get('status', 'all')
            priority = request.GET.get('priority', 'all')
            search = request.GET.get('search', '')
            
            # Start with base queryset
            job_requests = JobRequest.objects.filter(employee=employee)
            
            # Apply filters
            if status != 'all':
                job_requests = job_requests.filter(status=status)
            
            if priority != 'all':
                job_requests = job_requests.filter(priority=priority)
            
            if search:
                from django.db.models import Q
                job_requests = job_requests.filter(
                    Q(title__icontains=search) |
                    Q(description__icontains=search) |
                    Q(category__icontains=search) |
                    Q(location__icontains=search)
                )
            
            # Get statistics
            total_requests = JobRequest.objects.filter(employee=employee).count()
            accepted_requests = JobRequest.objects.filter(employee=employee, status='accepted').count()
            pending_requests = JobRequest.objects.filter(employee=employee, status='pending').count()
            rejected_requests = JobRequest.objects.filter(employee=employee, status='rejected').count()
            
            context = {
                'employee': employee,
                'employee_name': employee.full_name,
                'job_requests': job_requests,
                'stats': {
                    'total': total_requests,
                    'accepted': accepted_requests,
                    'pending': pending_requests,
                    'rejected': rejected_requests,
                },
                'filters': {
                    'status': status,
                    'priority': priority,
                    'search': search
                }
            }
            return render(request, 'employee_html/employee_job_request.html', context)
            
        except Exception as e:
            messages.error(request, f"Error filtering jobs: {str(e)}")
            return redirect('employee_job_request')
        

#*********************************************************************


def employee_schedule(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee = Employee.objects.get(employee_id=request.session['employee_id'])
        
        # Import models
        from .models import EmployeeAvailability, JobRequest
        
        import calendar
        from datetime import date, datetime, timedelta
        
        # Handle POST requests (Toggle Availability)
        if request.method == 'POST':
            action = request.POST.get('action')
            date_str = request.POST.get('date')
            
            if action == 'toggle_availability' and date_str:
                try:
                    target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    
                    # Check if there are any booked jobs on this date
                    has_booked_jobs = JobRequest.objects.filter(
                        employee=employee, 
                        proposed_date=target_date, 
                        status__in=['accepted', 'pending']
                    ).exists()
                    
                    if has_booked_jobs:
                        messages.error(request, f"Cannot change availability for {target_date}, you have booked job(s).")
                    else:
                        # Toggle availability - check if record exists
                        availability_record, created = EmployeeAvailability.objects.get_or_create(
                            employee=employee, 
                            date=target_date,
                            defaults={'is_available': False}  # Default to unavailable when manually toggled
                        )
                        
                        if not created:
                            # If record exists, toggle it
                            availability_record.is_available = not availability_record.is_available
                            availability_record.reason = "Manually changed by employee"
                            availability_record.save()
                        
                        status_msg = "available" if availability_record.is_available else "unavailable"
                        messages.success(request, f"You are now {status_msg} on {target_date}")
                        
                except ValueError:
                    messages.error(request, "Invalid date format.")
                except Exception as e:
                    messages.error(request, f"Error updating availability: {str(e)}")
                    
            return redirect('employee_schedule')
        
        # GET request - Show calendar
        # Get year and month from URL parameters
        today = date.today()
        try:
            year = int(request.GET.get('year', today.year))
            month = int(request.GET.get('month', today.month))
        except (ValueError, TypeError):
            year, month = today.year, today.month
        
        # Create calendar
        cal = calendar.Calendar(firstweekday=6)  # Sunday first
        month_days = cal.monthdatescalendar(year, month)
        
        # Calculate date range for database queries
        if month_days:
            start_date = month_days[0][0]
            end_date = month_days[-1][-1]
            
            # Get all jobs for this employee in the date range
            jobs_in_range = JobRequest.objects.filter(
                employee=employee,
                proposed_date__range=(start_date, end_date)
            ).select_related('employer')
            
            # Get all manual availability records for this employee in the date range
            availability_records = EmployeeAvailability.objects.filter(
                employee=employee,
                date__range=(start_date, end_date)
            )
            
            # Create a dictionary for quick lookup
            availability_dict = {record.date: record.is_available for record in availability_records}
            jobs_dict = {}
            
            for job in jobs_in_range:
                if job.proposed_date not in jobs_dict:
                    jobs_dict[job.proposed_date] = []
                jobs_dict[job.proposed_date].append(job)
        
        # Prepare calendar data for template
        calendar_weeks = []
        for week in month_days:
            week_data = []
            for d in week:
                # Determine status for this day
                status = 'available'  # Default
                css_class = 'day-available'
                events = []
                
                # Check for jobs on this date
                if d in jobs_dict:
                    jobs = jobs_dict[d]
                    status = 'booked'
                    css_class = 'day-booked'
                    
                    for job in jobs:
                        events.append({
                            'title': f"{job.title} ({job.employer.company_name or job.employer.full_name})",
                            'color': 'var(--danger)',
                            'job_id': job.job_id,
                            'status': job.status
                        })
                
                # Check for manual unavailability
                elif d in availability_dict and not availability_dict[d]:
                    status = 'unavailable'
                    css_class = 'day-unavailable'
                    events.append({
                        'title': 'Manually Unavailable',
                        'color': 'var(--warning)'
                    })
                elif d < today:
                    # Past dates
                    status = 'past'
                    css_class = 'day-past'
                elif d == today:
                    # Today
                    css_class = 'day-today'
                
                week_data.append({
                    'date': d,
                    'day_number': d.day,
                    'in_month': d.month == month,
                    'is_today': d == today,
                    'status': status,
                    'events': events,
                    'css_class': css_class,
                    'date_str': d.strftime('%Y-%m-%d')
                })
            calendar_weeks.append(week_data)
        
        # Navigation URLs
        first = date(year, month, 1)
        prev_month_date = (first - timedelta(days=1))
        next_month_date = (first + timedelta(days=32)).replace(day=1)
        
        prev_month_url = f"?month={prev_month_date.month}&year={prev_month_date.year}"
        next_month_url = f"?month={next_month_date.month}&year={next_month_date.year}"
        today_url = f"?month={today.month}&year={today.year}"
        
        # Get upcoming jobs for display
        upcoming_jobs = JobRequest.objects.filter(
            employee=employee,
            status__in=['accepted', 'pending'],
            proposed_date__gte=today
        ).select_related('employer').order_by('proposed_date')[:5]
        
        # Count stats
        jobs_this_week = JobRequest.objects.filter(
            employee=employee,
            proposed_date__range=[today, today + timedelta(days=7)],
            status__in=['accepted', 'pending']
        ).count()
        
        jobs_in_progress = JobRequest.objects.filter(
            employee=employee,
            status='accepted',
            proposed_date__lte=today
        ).count()
        
        upcoming_jobs_count = JobRequest.objects.filter(
            employee=employee,
            status='pending',
            proposed_date__gte=today
        ).count()
        
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'calendar_weeks': calendar_weeks,
            'current_month_name': first.strftime('%B %Y'),
            'prev_month_url': prev_month_url,
            'next_month_url': next_month_url,
            'today_url': today_url,
            'upcoming_jobs': upcoming_jobs,
            'stats': {
                'jobs_this_week': jobs_this_week,
                'jobs_in_progress': jobs_in_progress,
                'upcoming_jobs': upcoming_jobs_count,
            }
        }
        
        return render(request, 'employee_html/employee_schedule.html', context)
        
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('index')
    except Exception as e:
        messages.error(request, f"Error loading schedule: {str(e)}")
        return redirect('employee_dashboard')


#*********************************************************


def bulk_mark_unavailable(request):
    """Mark multiple dates as unavailable"""
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            date_str = request.POST.get('date')
            duration = request.POST.get('duration', 'single')
            reason = request.POST.get('reason', '').strip()
            
            if not date_str:
                messages.error(request, "Please select a date.")
                return redirect('employee_schedule')
            
            start_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Calculate end date based on duration
            if duration == 'week':
                end_date = start_date + timedelta(days=6)
            elif duration == 'month':
                # Add approximately one month
                if start_date.month == 12:
                    end_date = start_date.replace(year=start_date.year + 1, month=1)
                else:
                    end_date = start_date.replace(month=start_date.month + 1)
            else:
                end_date = start_date
            
            # Check for existing jobs in this period
            existing_jobs = JobRequest.objects.filter(
                employee=employee,
                proposed_date__range=[start_date, end_date],
                status__in=['accepted', 'pending']
            ).exists()
            
            if existing_jobs:
                messages.error(request, "Cannot mark dates as unavailable due to existing jobs.")
                return redirect('employee_schedule')
            
            # Mark dates as unavailable
            current_date = start_date
            while current_date <= end_date:
                # Skip if already has booked jobs
                has_jobs = JobRequest.objects.filter(
                    employee=employee,
                    proposed_date=current_date,
                    status__in=['accepted', 'pending']
                ).exists()
                
                if not has_jobs:
                    EmployeeAvailability.objects.update_or_create(
                        employee=employee,
                        date=current_date,
                        defaults={
                            'is_available': False,
                            'reason': reason or "Marked as unavailable by employee"
                        }
                    )
                
                current_date += timedelta(days=1)
            
            messages.success(request, f"Successfully marked dates as unavailable.")
            
        except Exception as e:
            messages.error(request, f"Error: {str(e)}")
    
    return redirect('employee_schedule')


#***************************************************************

# Employee Settings View
def employee_setting(request):
    # Check if employee is logged in
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        employee_login = EmployeeLogin.objects.get(employee=employee)
    except (Employee.DoesNotExist, EmployeeLogin.DoesNotExist):
        messages.error(request, "Employee not found.")
        return redirect('employee_dashboard')
    
    # Check for open password modal flag
    open_password_modal = request.session.pop('open_password_modal', False)
    
    context = {
        'employee': employee,
        'employee_name': employee.full_name,
        'employee_email': employee.email,
        'employee_profile_photo': employee.profile_image,
        'open_password_modal': open_password_modal,
        'SERVICE_RADIUS_CHOICES': Employee.SERVICE_RADIUS_CHOICES,
        'LANGUAGE_CHOICES': Employee.LANGUAGE_CHOICES,
        'CURRENCY_CHOICES': Employee.CURRENCY_CHOICES,
        'TIMEZONE_CHOICES': Employee.TIMEZONE_CHOICES,
        'AVAILABILITY_CHOICES': Employee.AVAILABILITY_CHOICES,
        'DATE_FORMAT_CHOICES': Employee.DATE_FORMAT_CHOICES,
        'PRIVACY_CHOICES': Employee.PRIVACY_CHOICES,
    }
    
    return render(request, 'employee_html/employee_settings.html', context)


#************************************************************

# Update About Me
def update_employee_bio(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            bio = request.POST.get('bio', '').strip()
            
            employee.bio = bio
            employee.save()
            
            messages.success(request, "About me updated successfully!")
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error updating bio: {str(e)}")
    
    return redirect('employee_profile')


#*********************************************************

# Add Experience
def add_employee_experience(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            job_title = request.POST.get('job_title', '').strip()
            company = request.POST.get('company', '').strip()
            start_date_str = request.POST.get('start_date', '')
            end_date_str = request.POST.get('end_date', '')
            currently_working = 'currently_working' in request.POST
            description = request.POST.get('description', '').strip()
            
            if not job_title or not company or not start_date_str:
                messages.error(request, "Job title, company, and start date are required.")
                return redirect('employee_profile')
            
            # Parse dates
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date() if end_date_str else None
            except ValueError:
                messages.error(request, "Invalid date format. Use YYYY-MM-DD format.")
                return redirect('employee_profile')
            
            # Create experience
            experience = EmployeeExperience.objects.create(
                employee=employee,
                job_title=job_title,
                company=company,
                start_date=start_date,
                end_date=end_date,
                currently_working=currently_working,
                description=description
            )
            
            messages.success(request, "Experience added successfully!")
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error adding experience: {str(e)}")
    
    return redirect('employee_profile')


#*********************************************************************


# Update Experience
def update_employee_experience(request, experience_id):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            experience = get_object_or_404(EmployeeExperience, id=experience_id, employee=employee)
            
            experience.job_title = request.POST.get('job_title', experience.job_title).strip()
            experience.company = request.POST.get('company', experience.company).strip()
            
            start_date_str = request.POST.get('start_date', '')
            end_date_str = request.POST.get('end_date', '')
            
            if start_date_str:
                try:
                    experience.start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                except ValueError:
                    pass
            
            if end_date_str:
                try:
                    experience.end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                except ValueError:
                    pass
            else:
                experience.end_date = None
            
            experience.currently_working = 'currently_working' in request.POST
            experience.description = request.POST.get('description', experience.description).strip()
            experience.save()
            
            messages.success(request, "Experience updated successfully!")
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error updating experience: {str(e)}")
    
    return redirect('employee_profile')


#*****************************************************************

# Delete Experience
def delete_employee_experience(request, experience_id):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            experience = get_object_or_404(EmployeeExperience, id=experience_id, employee=employee)
            experience.delete()
            
            messages.success(request, "Experience deleted successfully!")
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error deleting experience: {str(e)}")
    
    return redirect('employee_profile')


#**************************************************************

# Fixed Add Certificate View
def add_employee_certificate(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            name = request.POST.get('name', '').strip()
            issuer = request.POST.get('issuer', '').strip()
            issue_date_str = request.POST.get('issue_date', '')
            expiry_date_str = request.POST.get('expiry_date', '')
            description = request.POST.get('description', '').strip()
            upload_type = request.POST.get('upload_type', 'certificate')
            
            print(f"DEBUG Certificate: Upload type: {upload_type}")  # Debug log
            print(f"DEBUG Certificate: Files in request: {list(request.FILES.keys())}")  # Debug log
            
            if not name or not issuer or not issue_date_str:
                messages.error(request, "Certificate name, issuer, and issue date are required.")
                return redirect('employee_profile')
            
            # Parse dates
            try:
                issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d').date()
                expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d').date() if expiry_date_str else None
            except ValueError:
                messages.error(request, "Invalid date format. Use YYYY-MM-DD format.")
                return redirect('employee_profile')
            
            # Check for file - use 'image' field name
            if 'image' not in request.FILES:
                messages.error(request, "Please select a file to upload.")
                return redirect('employee_profile')
            
            certificate_file = request.FILES['image']
            print(f"DEBUG Certificate: File name: {certificate_file.name}, Size: {certificate_file.size}, Type: {certificate_file.content_type}")  # Debug log
            
            # Validate file size (max 10MB)
            max_size = 10 * 1024 * 1024  # 10MB
            if certificate_file.size > max_size:
                messages.error(request, f"File size should be less than {max_size//(1024*1024)}MB.")
                return redirect('employee_profile')
            
            # Validate file type
            allowed_types = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']
            if certificate_file.content_type not in allowed_types:
                messages.error(request, "Only PDF, JPEG, PNG files are allowed.")
                return redirect('employee_profile')
            
            # Create certificate
            certificate = EmployeeCertificate.objects.create(
                employee=employee,
                name=name,
                issuer=issuer,
                issue_date=issue_date,
                expiry_date=expiry_date,
                certificate_file=certificate_file,
                description=description
            )
            
            print(f"DEBUG Certificate: Certificate created - ID: {certificate.id}, Name: {certificate.name}")  # Debug log
            messages.success(request, "Certificate added successfully!")
            
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
            print("DEBUG Certificate: Employee not found")  # Debug log
        except Exception as e:
            error_msg = f"Error adding certificate: {str(e)}"
            print(f"DEBUG Certificate: {error_msg}")  # Debug log
            messages.error(request, error_msg)
    
    return redirect('employee_profile')


#**************************************************************************

# Update Certificate
def update_employee_certificate(request, certificate_id):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            certificate = get_object_or_404(EmployeeCertificate, id=certificate_id, employee=employee)
            
            certificate.name = request.POST.get('name', certificate.name).strip()
            certificate.issuer = request.POST.get('issuer', certificate.issuer).strip()
            
            issue_date_str = request.POST.get('issue_date', '')
            expiry_date_str = request.POST.get('expiry_date', '')
            
            if issue_date_str:
                try:
                    certificate.issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d').date()
                except ValueError:
                    pass
            
            if expiry_date_str:
                try:
                    certificate.expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d').date()
                except ValueError:
                    pass
            else:
                certificate.expiry_date = None
            
            certificate.description = request.POST.get('description', certificate.description).strip()
            
            # Handle file update
            if 'certificate_file' in request.FILES and request.FILES['certificate_file']:
                uploaded_file = request.FILES['certificate_file']
                
                # Validate file size (max 10MB)
                if uploaded_file.size > 10 * 1024 * 1024:
                    messages.error(request, "File size should be less than 10MB.")
                    return redirect('employee_profile')
                
                # Validate file type
                allowed_types = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']
                if uploaded_file.content_type not in allowed_types:
                    messages.error(request, "Only PDF, JPEG, PNG files are allowed.")
                    return redirect('employee_profile')
                
                # Delete old file if exists
                if certificate.certificate_file:
                    certificate.certificate_file.delete(save=False)
                
                certificate.certificate_file = uploaded_file
            
            certificate.save()
            messages.success(request, "Certificate updated successfully!")
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error updating certificate: {str(e)}")
    
    return redirect('employee_profile')


#**********************************************************


# Delete Certificate
def delete_employee_certificate(request, certificate_id):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            certificate = get_object_or_404(EmployeeCertificate, id=certificate_id, employee=employee)
            
            # Delete file if exists
            if certificate.certificate_file:
                certificate.certificate_file.delete(save=False)
            
            certificate.delete()
            messages.success(request, "Certificate deleted successfully!")
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error deleting certificate: {str(e)}")
    
    return redirect('employee_profile')


#************************************************************


# Download Certificate
def download_certificate(request, certificate_id):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee = Employee.objects.get(employee_id=request.session['employee_id'])
        certificate = get_object_or_404(EmployeeCertificate, id=certificate_id, employee=employee)
        
        if certificate.certificate_file:
            response = HttpResponse(certificate.certificate_file.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{certificate.name.replace(" ", "_")}.pdf"'
            return response
        else:
            messages.error(request, "Certificate file not found.")
            return redirect('employee_profile')
    except Exception as e:
        messages.error(request, f"Error downloading certificate: {str(e)}")
        return redirect('employee_profile')


#**********************************************************

def add_employee_portfolio(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            title = request.POST.get('title', '').strip()
            description = request.POST.get('description', '').strip()
            upload_type = request.POST.get('upload_type', 'portfolio')
            
            print(f"DEBUG: Upload type: {upload_type}")  # Debug log
            print(f"DEBUG: Files in request: {list(request.FILES.keys())}")  # Debug log
            
            if not title:
                messages.error(request, "Title is required.")
                return redirect('employee_profile')
            
            # Check for file - use 'image' field name
            if 'image' not in request.FILES:
                messages.error(request, "Please select an image to upload.")
                return redirect('employee_profile')
            
            image_file = request.FILES['image']
            print(f"DEBUG: File name: {image_file.name}, Size: {image_file.size}, Type: {image_file.content_type}")  # Debug log
            
            # Validate image size (max 5MB)
            max_size = 5 * 1024 * 1024  # 5MB
            if image_file.size > max_size:
                messages.error(request, f"Image size should be less than {max_size//(1024*1024)}MB.")
                return redirect('employee_profile')
            
            # Validate image type
            allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif', 'image/webp']
            if image_file.content_type not in allowed_types:
                messages.error(request, "Only JPEG, PNG, JPG, GIF, and WebP images are allowed.")
                return redirect('employee_profile')
            
            # Create portfolio item
            portfolio = EmployeePortfolio.objects.create(
                employee=employee,
                title=title,
                description=description,
                image=image_file
            )
            
            print(f"DEBUG: Portfolio created - ID: {portfolio.id}, Title: {portfolio.title}")  # Debug log
            messages.success(request, "Portfolio item added successfully!")
            
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
            print("DEBUG: Employee not found")  # Debug log
        except Exception as e:
            error_msg = f"Error adding portfolio item: {str(e)}"
            print(f"DEBUG: {error_msg}")  # Debug log
            messages.error(request, error_msg)
    
    return redirect('employee_profile')


#**************************************************************

# Update Portfolio Item
def update_employee_portfolio(request, portfolio_id):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            portfolio = get_object_or_404(EmployeePortfolio, id=portfolio_id, employee=employee)
            
            portfolio.title = request.POST.get('title', portfolio.title).strip()
            portfolio.description = request.POST.get('description', portfolio.description).strip()
            
            # Handle image update
            if 'image' in request.FILES and request.FILES['image']:
                new_image = request.FILES['image']
                
                # Validate image size (max 5MB)
                if new_image.size > 5 * 1024 * 1024:
                    messages.error(request, "Image size should be less than 5MB.")
                    return redirect('employee_profile')
                
                # Validate image type
                allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif', 'image/webp']
                if new_image.content_type not in allowed_types:
                    messages.error(request, "Only JPEG, PNG, JPG, GIF, and WebP images are allowed.")
                    return redirect('employee_profile')
                
                # Delete old image if exists
                if portfolio.image:
                    portfolio.image.delete(save=False)
                
                portfolio.image = new_image
            
            portfolio.save()
            messages.success(request, "Portfolio item updated successfully!")
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error updating portfolio item: {str(e)}")
    
    return redirect('employee_profile')


#******************************************************

# Delete Portfolio Item
def delete_employee_portfolio(request, portfolio_id):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            portfolio = get_object_or_404(EmployeePortfolio, id=portfolio_id, employee=employee)
            
            # Delete image file
            if portfolio.image:
                portfolio.image.delete(save=False)
            
            portfolio.delete()
            messages.success(request, "Portfolio item deleted successfully!")
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error deleting portfolio item: {str(e)}")
    
    return redirect('employee_profile')


#*************************************************************

# Add Skill
def add_employee_skill(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            # Method 1: Add skill to EmployeeSkill model
            skill_name = request.POST.get('skill_name', '').strip()
            
            if skill_name:
                # Check if skill already exists
                if not EmployeeSkill.objects.filter(employee=employee, skill_name=skill_name).exists():
                    EmployeeSkill.objects.create(
                        employee=employee,
                        skill_name=skill_name
                    )
                    messages.success(request, f"Skill '{skill_name}' added successfully!")
                else:
                    messages.warning(request, f"Skill '{skill_name}' already exists.")
            
            # Method 2: Update skills field in Employee model
            skills_text = request.POST.get('skills_text', '').strip()
            if skills_text:
                # Get existing skills
                existing_skills = []
                if employee.skills:
                    existing_skills = [s.strip() for s in employee.skills.split(',') if s.strip()]
                
                # Add new skills
                new_skills = [s.strip() for s in skills_text.split(',') if s.strip()]
                all_skills = list(set(existing_skills + new_skills))
                
                employee.skills = ','.join(all_skills)
                employee.save()
                messages.success(request, "Skills updated successfully!")
            
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error adding skill: {str(e)}")
    
    return redirect('employee_profile')


#************************************************************

# Remove Skill
def remove_employee_skill(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            skill_name = request.POST.get('skill_name', '').strip()
            
            if not skill_name:
                messages.error(request, "Skill name is required.")
                return redirect('employee_profile')
            
            # Method 1: Remove from EmployeeSkill model
            EmployeeSkill.objects.filter(employee=employee, skill_name=skill_name).delete()
            
            # Method 2: Remove from Employee skills field
            if employee.skills:
                skills_list = [s.strip() for s in employee.skills.split(',') if s.strip()]
                if skill_name in skills_list:
                    skills_list.remove(skill_name)
                    employee.skills = ','.join(skills_list)
                    employee.save()
            
            messages.success(request, f"Skill '{skill_name}' removed successfully!")
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error removing skill: {str(e)}")
    
    return redirect('employee_profile')

#******************************************************************

# Update Availability
def update_employee_availability(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            employee.availability_status = request.POST.get('availability_status', employee.availability_status)
            employee.response_time = request.POST.get('response_time', employee.response_time)
            employee.working_hours = request.POST.get('working_hours', employee.working_hours)
            employee.service_area = request.POST.get('service_area', employee.service_area)
            
            employee.save()
            messages.success(request, "Availability updated successfully!")
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error updating availability: {str(e)}")
    
    return redirect('employee_profile')


#******************************************************************

# Update Profile Image
def update_employee_profile_image(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            if 'profile_image' in request.FILES and request.FILES['profile_image']:
                profile_image = request.FILES['profile_image']
                
                # Validate image size (max 5MB)
                if profile_image.size > 5 * 1024 * 1024:
                    messages.error(request, "Image size should be less than 5MB.")
                    return redirect('employee_profile')
                
                # Validate image type
                allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif', 'image/webp']
                if profile_image.content_type not in allowed_types:
                    messages.error(request, "Only JPEG, PNG, JPG, GIF, and WebP images are allowed.")
                    return redirect('employee_profile')
                
                # Delete old profile image if exists
                if employee.profile_image:
                    employee.profile_image.delete(save=False)
                
                employee.profile_image = profile_image
                employee.save()
                
                messages.success(request, "Profile image updated successfully!")
            else:
                messages.error(request, "No image selected.")
                
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error updating profile image: {str(e)}")
    
    return redirect('employee_profile')


#*******************************************************************

# Update Cover Image 
def update_employee_cover_image(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    messages.info(request, "Cover image feature coming soon!")
    return redirect('employee_profile')


#**************************************************************************

# Update Professional Info
def update_employee_professional_info(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            employee.job_title = request.POST.get('job_title', employee.job_title).strip()
            employee.location = request.POST.get('location', employee.location).strip()
            employee.years_experience = int(request.POST.get('years_experience', employee.years_experience) or 0)
            
            employee.save()
            messages.success(request, "Professional information updated successfully!")
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except ValueError:
            messages.error(request, "Years of experience must be a number.")
        except Exception as e:
            messages.error(request, f"Error updating professional info: {str(e)}")
    
    return redirect('employee_profile')


#************************************************************************


# Update Profile Information (Settings)
def update_employee_profile(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            with transaction.atomic():
                employee = Employee.objects.get(employee_id=request.session['employee_id'])
                employee_login = EmployeeLogin.objects.get(employee=employee)
                
                # Update basic profile information
                employee.first_name = request.POST.get('first_name', employee.first_name)
                employee.last_name = request.POST.get('last_name', employee.last_name)
                
                # Phone uniqueness check
                new_phone = request.POST.get('phone')
                if new_phone and new_phone != employee.phone:
                    if Employee.objects.filter(phone=new_phone).exclude(employee_id=employee.employee_id).exists():
                        messages.error(request, "This phone number is already registered.")
                        return redirect('employee_setting')
                    employee.phone = new_phone
                
                # Update skills and bio
                employee.skills = request.POST.get('skills', employee.skills)
                employee.bio = request.POST.get('bio', employee.bio)
                employee.work_experience = request.POST.get('work_experience', employee.work_experience)
                
                # Handle profile image upload
                if 'profile_image' in request.FILES and request.FILES['profile_image']:
                    profile_image = request.FILES['profile_image']
                    
                    # Validate file size (max 5MB)
                    if profile_image.size > 5 * 1024 * 1024:
                        messages.error(request, "Image size should be less than 5MB.")
                        return redirect('employee_setting')
                    
                    # Validate file type
                    allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif', 'image/webp']
                    if profile_image.content_type not in allowed_types:
                        messages.error(request, "Only JPEG, PNG, JPG, GIF, and WebP images are allowed.")
                        return redirect('employee_setting')
                    
                    # Delete old profile image if exists
                    if employee.profile_image:
                        employee.profile_image.delete(save=False)
                    
                    employee.profile_image = profile_image
                
                # Handle remove photo
                remove_photo_flag = request.POST.get('remove_photo', '0')
                if remove_photo_flag == '1':
                    if employee.profile_image:
                        employee.profile_image.delete(save=False)
                    employee.profile_image = None
                
                # Update email in both tables if changed
                new_email = request.POST.get('email')
                if new_email and new_email != employee.email:
                    # Check if email already exists
                    if Employee.objects.filter(email=new_email).exclude(employee_id=employee.employee_id).exists():
                        messages.error(request, "This email is already registered.")
                        return redirect('employee_setting')
                    
                    # Update email in both tables
                    employee.email = new_email
                    employee_login.email = new_email
                    employee_login.save()
                
                employee.save()
                messages.success(request, "Profile updated successfully!")
                
                # Create Notification
                EmployeeNotification.objects.create(
                    employee=employee,
                    title="Profile Updated",
                    message="Your profile information has been successfully updated.",
                    notification_type='profile',
                    is_read=True
                )
                
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except EmployeeLogin.DoesNotExist:
            messages.error(request, "Login credentials not found.")
        except Exception as e:
            messages.error(request, f"Error updating profile: {str(e)}")
    
    return redirect('employee_setting')


#************************************************************


# Change Password
def change_employee_password(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            employee_login = EmployeeLogin.objects.get(employee=employee)
            
            current_password = request.POST.get('current_password')
            new_password = request.POST.get('new_password')
            confirm_password = request.POST.get('confirm_password')
            
            # 1. Validate inputs
            if not current_password or not new_password or not confirm_password:
                messages.error(request, 'All fields are required.')
                request.session['open_password_modal'] = True
                return redirect('employee_setting')
            
            # 2. Check current password
            if not check_password(current_password, employee_login.password):
                messages.error(request, 'Current password is incorrect.')
                request.session['open_password_modal'] = True
                return redirect('employee_setting')
            
            # 3. Check if new password is same as current password
            if check_password(new_password, employee_login.password):
                messages.error(request, 'New password cannot be the same as current password.')
                request.session['open_password_modal'] = True
                return redirect('employee_setting')
            
            # 4. Check if passwords match
            if new_password != confirm_password:
                messages.error(request, 'New passwords do not match.')
                request.session['open_password_modal'] = True
                return redirect('employee_setting')
            
            # 5. Update password
            employee_login.password = make_password(new_password)
            employee_login.save()
            
            # 6. Update last password change timestamp
            employee.last_password_change = timezone.now()
            employee.save()
            
            messages.success(request, 'Password changed successfully!')
            return redirect('employee_setting')
            
        except Employee.DoesNotExist:
            messages.error(request, 'Employee not found.')
            return redirect('employee_setting')
        except EmployeeLogin.DoesNotExist:
            messages.error(request, 'Login credentials not found.')
            return redirect('employee_setting')
        except Exception as e:
            messages.error(request, f'Error changing password: {str(e)}')
            request.session['open_password_modal'] = True
            return redirect('employee_setting')
    
    return redirect('employee_setting')


#********************************************************


# Update Privacy & Security Settings
def update_employee_privacy_security(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            # Update privacy settings
            employee.two_factor_auth = 'two_factor_auth' in request.POST
            employee.show_profile_to_employers = 'show_profile_to_employers' in request.POST
            employee.share_work_history = 'share_work_history' in request.POST
            employee.data_sharing_analytics = 'data_sharing_analytics' in request.POST
            employee.privacy_level = request.POST.get('privacy_level', employee.privacy_level)
            
            employee.save()
            messages.success(request, "Privacy settings updated successfully!")
            
            # Create Notification
            EmployeeNotification.objects.create(
                employee=employee,
                title="Privacy Settings Updated",
                message="Your privacy and security settings have been updated.",
                notification_type='security',
                is_read=True
            )
            
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error updating privacy settings: {str(e)}")
    
    return redirect('employee_setting')


#**********************************************************


# Update Location Settings
def update_employee_location(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            # Update location information
            employee.address = request.POST.get('address', employee.address)
            employee.city = request.POST.get('city', employee.city)
            employee.state = request.POST.get('state', employee.state)
            employee.zip_code = request.POST.get('zip_code', employee.zip_code)
            employee.country = request.POST.get('country', employee.country)
            
            # Update service radius
            service_radius = request.POST.get('service_radius')
            if service_radius:
                employee.service_radius = int(service_radius)
            
            employee.save()
            messages.success(request, "Location settings updated successfully!")
            
            # Create Notification
            EmployeeNotification.objects.create(
                employee=employee,
                title="Location Updated",
                message=f"Your location settings have been updated to {employee.city}, {employee.state}.",
                notification_type='profile',
                is_read=True
            )
            
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error updating location settings: {str(e)}")
    
    return redirect('employee_setting')


#*******************************************************************


def mark_notification_read(request, notification_id):
    """Mark a single notification as read"""
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
        
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        notification = get_object_or_404(EmployeeNotification, notification_id=notification_id, employee=employee)
        notification.is_read = True
        notification.save()
        
        # Return to previous page
        referer = request.META.get('HTTP_REFERER', 'employee_dashboard')
        return redirect(referer)
        
    except Exception as e:
        messages.error(request, f"Error updating notification: {str(e)}")
        return redirect('employee_dashboard')


#******************************************************

def mark_all_notifications_read(request):
    """Mark all notifications as read"""
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
        
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        EmployeeNotification.objects.filter(employee=employee, is_read=False).update(is_read=True)
        messages.success(request, "All notifications marked as read.")
        
        # Return to previous page
        referer = request.META.get('HTTP_REFERER', 'employee_dashboard')
        return redirect(referer)
        
    except Exception as e:
        messages.error(request, f"Error updating notifications: {str(e)}")
        return redirect('employee_dashboard')


#***************************************************************


def update_employee_notifications(request):
    """Update employee notification preferences"""
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            # Update notification preferences
            employee.job_alerts = 'job_alerts' in request.POST
            employee.message_notifications = 'message_notifications' in request.POST
            employee.payment_alerts = 'payment_alerts' in request.POST
            employee.platform_updates = 'platform_updates' in request.POST
            
            employee.save()
            messages.success(request, "Notification settings updated successfully!")
            
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error updating notification settings: {str(e)}")
    
    return redirect('employee_setting')


#***********************************************************************


# Update Account Preferences
def update_employee_preferences(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            
            # Update preferences
            employee.language = request.POST.get('language', employee.language)
            employee.currency = request.POST.get('currency', employee.currency)
            employee.timezone = request.POST.get('timezone', employee.timezone)
            employee.date_format = request.POST.get('date_format', employee.date_format)
            employee.availability = request.POST.get('availability', employee.availability)
            
            employee.save()
            messages.success(request, "Preferences updated successfully!")
            
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except Exception as e:
            messages.error(request, f"Error updating preferences: {str(e)}")
    
    return redirect('employee_setting')


#**************************************************************


# Deactivate Employee Account
def deactivate_employee_account(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            with transaction.atomic():
                employee = Employee.objects.get(employee_id=request.session['employee_id'])
                employee_login = EmployeeLogin.objects.get(employee=employee)
                
                # Get password from form
                password = request.POST.get('password', '')
                
                # Validate password
                if not password:
                    messages.error(request, "Please enter your password to confirm account deactivation.")
                    return redirect('employee_setting')
                
                # Verify password
                if not check_password(password, employee_login.password):
                    messages.error(request, "Incorrect password. Please try again.")
                    return redirect('employee_setting')
                
                # Set deactivation details
                employee.status = 'Inactive'
                employee.deactivation_date = timezone.now()
                employee.deactivation_reason = 'User requested deactivation'
                
                # Clear sensitive data (optional)
                if employee.profile_image:
                    employee.profile_image.delete(save=False)
                    employee.profile_image = None
                
                # Prefix phone number to make it unique
                if employee.phone:
                    employee.phone = f"DEACTIVATED_{employee.phone}"
                
                employee.save()
                
                # Deactivate login
                employee_login.status = 'Inactive'
                employee_login.save()
                
                # Clear session
                request.session.flush()
                
                messages.success(request, "Your account has been deactivated successfully.")
                return redirect('index')
                
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.")
        except EmployeeLogin.DoesNotExist:
            messages.error(request, "Login credentials not found.")
        except Exception as e:
            messages.error(request, f"Error deactivating account: {str(e)}")
    
    return redirect('employee_setting')


#**************************************************************

def get_employee_security_info(request):
    if 'employee_id' not in request.session:
        return {'error': 'Not authenticated', 'status': 401}
    
    try:
        employee = Employee.objects.get(employee_id=request.session['employee_id'])
        employee_login = EmployeeLogin.objects.get(employee=employee)
        
        security_info = {
            'last_password_change': employee.last_password_change.strftime('%B %d, %Y') if employee.last_password_change else 'Never',
            'account_created': employee.created_at.strftime('%B %d, %Y'),
            'login_count': employee.login_count,
            'last_login': employee_login.last_login.strftime('%B %d, %Y %H:%M') if employee_login.last_login else 'Never',
            'active_sessions': 1,
            'status': 200
        }
        
        return security_info
    except Employee.DoesNotExist:
        return {'error': 'Employee not found', 'status': 404}
    except EmployeeLogin.DoesNotExist:
        return {'error': 'Login info not found', 'status': 404}
    except Exception as e:
        return {'error': str(e), 'status': 500}
    

#*****************************************************************

def check_phone_availability(request):
    if 'employee_id' not in request.session:
        return {'available': False, 'error': 'Not authenticated', 'status': 401}
    
    try:
        employee = Employee.objects.get(employee_id=request.session['employee_id'])
        phone = request.GET.get('phone', '')
        
        if not phone:
            return {'available': True, 'status': 200}
        
        # Check if phone exists (excluding current user)
        exists = Employee.objects.filter(phone=phone).exclude(employee_id=employee.employee_id).exists()
        
        return {'available': not exists, 'status': 200}
        
    except Employee.DoesNotExist:
        return {'available': False, 'error': 'Employee not found', 'status': 404}
    except Exception as e:
        return {'available': False, 'error': str(e), 'status': 500}
    

#*********************************************************


def get_employee_stats(request):
    if 'employee_id' not in request.session:
        return {'error': 'Not authenticated', 'status': 401}
    
    try:
        employee = Employee.objects.get(employee_id=request.session['employee_id'])
        
        stats = {
            'total_earnings': float(employee.total_earnings),
            'total_jobs_done': employee.total_jobs_done,
            'rating': employee.rating,
            'service_radius': employee.service_radius,
            'availability': employee.availability,
            'status': 200
        }
        
        return stats
    except Employee.DoesNotExist:
        return {'error': 'Employee not found', 'status': 404}
    except Exception as e:
        return {'error': str(e), 'status': 500}


#*****************************************************************

def employee_schedule(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee = Employee.objects.get(employee_id=request.session['employee_id'])
        # Local import to avoid circular dependency if any
        from .models import EmployeeAvailability, JobRequest
        
        import calendar
        from datetime import date, datetime, timedelta
        
        # Handle POST (Toggle Availability)
        if request.method == 'POST':
            action = request.POST.get('action')
            date_str = request.POST.get('date')
            
            if action == 'toggle_availability' and date_str:
                try:
                    target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    
                    # Check if booked (cannot change availability if booked)
                    if JobRequest.objects.filter(employee=employee, proposed_date=target_date, status='accepted').exists():
                        messages.error(request, f"Cannot change availability for {target_date}, you have a booked job.")
                    else:
                        # Toggle availability
                        obj, created = EmployeeAvailability.objects.get_or_create(employee=employee, date=target_date)
                        # If it was just created (def=True/Available), we want to make it False/Unavailable.
                        # If it existed, toggle it.
                        if created:
                            obj.is_available = False # Toggle to unavailable
                        else:
                            obj.is_available = not obj.is_available
                            
                        obj.save()
                        
                        status_msg = "available" if obj.is_available else "unavailable"
                        messages.success(request, f"You are now {status_msg} on {target_date}")
                        
                except ValueError:
                    messages.error(request, "Invalid date format.")
                except Exception as e:
                    messages.error(request, f"Error updating availability: {str(e)}")
                    
            return redirect(f"{request.path}?month={request.GET.get('month', '')}&year={request.GET.get('year', '')}")

        # GET params or Today
        today = date.today()
        try:
            year = int(request.GET.get('year', today.year))
            month = int(request.GET.get('month', today.month))
        except (ValueError, TypeError):
            year, month = today.year, today.month
            
        # Calendar Setup (Sunday start)
        cal = calendar.Calendar(firstweekday=6) 
        month_days = cal.monthdatescalendar(year, month)
        
        # Fetch Data Range
        if month_days:
            start_date = month_days[0][0]
            end_date = month_days[-1][-1]
            
            booked_dates = set(JobRequest.objects.filter(
                employee=employee, 
                proposed_date__range=(start_date, end_date), 
                status='accepted'
            ).values_list('proposed_date', flat=True))
            
            # Fetch explicitly UNAVAILABLE dates (is_available=False)
            unavailable_records = EmployeeAvailability.objects.filter(
                employee=employee,
                date__range=(start_date, end_date),
                is_available=False
            ).values_list('date', flat=True)
            unavailable_dates = set(unavailable_records)
            
            # Prepare Calendar Data
            calendar_weeks = []
            for week in month_days:
                week_data = []
                for d in week:
                    status = 'available' # Default Green
                    css_class = 'bg-success' # Green
                    events = []
                    
                    is_booked = d in booked_dates
                    is_unavailable = d in unavailable_dates
                    
                    if is_booked:
                        status = 'booked'
                        css_class = 'bg-danger' # Red
                        events.append({'title': 'Booked Job', 'color': 'var(--danger)'})
                    elif is_unavailable:
                        status = 'unavailable'
                        css_class = 'bg-danger' # Red (Manual)
                        events.append({'title': 'Unavailable', 'color': 'var(--danger)'})
                    else:
                        # Available
                        css_class = 'bg-success' # Green
                        
                    week_data.append({
                        'date': d,
                        'day_number': d.day,
                        'in_month': d.month == month,
                        'is_today': d == today,
                        'status': status,
                        'events': events,
                        'css_class': css_class,
                        'date_str': d.strftime('%Y-%m-%d')
                    })
                calendar_weeks.append(week_data)
        else:
            calendar_weeks = []
            
        # Nav Links
        first = date(year, month, 1)
        prev_month_date = first - timedelta(days=1)
        next_month_date = (first + timedelta(days=32)).replace(day=1)
        
        # DEBUG PRINT
        if calendar_weeks:
            print("DEBUG SCHEDULE DATA SAMPLE:", calendar_weeks[0][0])

        first = date(year, month, 1)
        prev_month_date = first - timedelta(days=1)
        next_month_date = (first + timedelta(days=32)).replace(day=1)
        
        prev_month_url = f"?month={prev_month_date.month}&year={prev_month_date.year}"
        next_month_url = f"?month={next_month_date.month}&year={next_month_date.year}"
        today_url = f"?month={today.month}&year={today.year}"
        
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'calendar_weeks': calendar_weeks,
            'current_month_name': first.strftime('%B %Y'),
            'prev_month_url': prev_month_url,
            'next_month_url': next_month_url,
            'today_url': today_url,
        }
        return render(request, 'employee_html/employee_schedule.html', context)
        
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('index')
    except Exception as e:
        messages.error(request, f"Error loading schedule: {str(e)}")
        import traceback
        traceback.print_exc()
        return redirect('employee_dashboard')

# ************************************************************

def employee_notifications(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Get all notifications to display in main content
        all_notifications = EmployeeNotification.objects.filter(employee=employee).order_by('-created_at')
        
        context = {
            'employee': employee,
            'employee_name': employee.full_name,
            'employee_email': employee.email,
            'all_notifications': all_notifications,
        }
        
        return render(request, 'employee_html/employee_notifications.html', context)
        
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
        return redirect('index')
    except Exception as e:
        messages.error(request, f"Error loading notifications: {str(e)}")
        return redirect('employee_dashboard')

# ************************************************************

def mark_notification_read(request, notification_id):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee_id = request.session['employee_id']
            employee = Employee.objects.get(employee_id=employee_id)
            
            # Using notification_id as per model definition
            notification = get_object_or_404(EmployeeNotification, notification_id=notification_id, employee=employee)
            notification.is_read = True
            notification.save()
            
            messages.success(request, "Notification marked as read.")
            
        except Exception as e:
            messages.error(request, f"Error updating notification: {str(e)}")
            
    # Redirect back to the page they came from, or notifications page
    return redirect(request.META.get('HTTP_REFERER', 'employee_notifications'))


#**********************************************************************


def mark_all_notifications_read(request):
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee_id = request.session['employee_id']
            employee = Employee.objects.get(employee_id=employee_id)
            
            EmployeeNotification.objects.filter(employee=employee, is_read=False).update(is_read=True)
            
            messages.success(request, "All notifications marked as read.")
            
        except Exception as e:
            messages.error(request, f"Error updating notifications: {str(e)}")
            
    # Redirect back to the page they came from, or notifications page
    return redirect(request.META.get('HTTP_REFERER', 'employee_notifications'))


#**************************************************************

def employee_notifications_processor(request):
    """Context processor to make notifications available in all templates"""
    if 'employee_id' not in request.session:
        return {}
    
    try:
        employee_id = request.session['employee_id']
        employee = Employee.objects.get(employee_id=employee_id)
        
        # Get unread notification count
        unread_count = EmployeeNotification.objects.filter(
            employee=employee, 
            is_read=False
        ).count()
        
        # Get recent notifications (limit 20 for dropdown)
        notifications = EmployeeNotification.objects.filter(
            employee=employee
        ).order_by('-created_at')[:20]
        
        # Format for dropdown
        recent_notifications = []
        for n in notifications:
            # Map type to icon class suffix (simplifed for template)
            notif_type = 'system'
            icon = 'bell'
            action_url = n.link
            
            if n.notification_type == 'job': 
                notif_type = 'job_request' # or job_accepted/rejected based on title content usually
                icon = 'briefcase'
                if "Accepted" in n.title: notif_type = 'job_accepted'
                elif "Rejected" in n.title: notif_type = 'job_rejected'
                elif "Request" in n.title: notif_type = 'job_request'
            elif n.notification_type == 'payment': 
                notif_type = 'payment_received'
                icon = 'rupee-sign'
            elif n.notification_type == 'profile': 
                notif_type = 'system'
                icon = 'user'
            
            recent_notifications.append({
                'notification_id': n.notification_id,
                'title': n.title,
                'message': n.message,
                'time': n.created_at,
                'created_at': n.created_at,
                'is_read': n.is_read,
                'notification_type': notif_type,
                'icon': icon,
                'action_url': action_url
            })
            
        return {
            'unread_notifications_count': unread_count,
            'recent_notifications': recent_notifications
        }
        
    except Exception as e:
        # Fail silently to avoid breaking pages
        print(f"Error in notification context processor: {e}")
        return {}

#********************************************************************

def submit_employee_site_feedback(request):
    """Handle site feedback submission from employee"""
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        try:
            employee_id = request.session['employee_id']
            employee = Employee.objects.get(employee_id=employee_id)
            
            # Get form data
            feedback_type = request.POST.get('feedback_type', 'general_feedback')
            rating = request.POST.get('rating')
            title = request.POST.get('title', '').strip()
            feedback_text = request.POST.get('feedback_text', '').strip()
            areas = request.POST.getlist('areas')
            recommendation = request.POST.get('recommendation')
            contact_email = request.POST.get('contact_email', employee.email)
            
            # Validate required fields
            if not rating or not title or not feedback_text or not recommendation:
                messages.error(request, "Please fill in all required fields.", extra_tags='site_feedback danger')
                return redirect('employee_review_list')
            
            # Create SiteReview for feedback
            SiteReview.objects.create(
                employer=None,  # Null for employee reviews
                employee=employee,
                review_type='platform',
                rating=int(rating),
                title=title,
                review_text=feedback_text,
                areas=areas,
                recommendation=recommendation
            )
            
            messages.success(request, "Thank you for your feedback! It helps us improve WorkNest.", extra_tags='site_feedback success')
            
        except Employee.DoesNotExist:
            messages.error(request, "Employee not found.", extra_tags='site_feedback danger')
        except Exception as e:
            messages.error(request, f"Error submitting feedback: {str(e)}", extra_tags='site_feedback danger')
    
    return redirect(reverse('employee_review_list') + '?tab=feedback')



#*********************************************************

def send_rejection_message(request):
    """Send rejection message to employer via chat"""
    if 'employee_id' not in request.session:
        return JsonResponse({'success': False, 'error': 'Please login first.'})
    
    if request.method == 'POST':
        try:
            employee = Employee.objects.get(employee_id=request.session['employee_id'])
            job_id = request.POST.get('job_id')
            rejection_message = request.POST.get('rejection_message', '').strip()
            
            if not job_id:
                return JsonResponse({'success': False, 'error': 'Job ID is required.'})
            
            if not rejection_message:
                return JsonResponse({'success': False, 'error': 'Please enter a reason for rejection.'})
            
            job = JobRequest.objects.get(job_id=job_id, employee=employee)
            
            # Update job status
            job.status = 'rejected'
            job.rejection_reason = rejection_message
            job.updated_at = timezone.now()
            job.save()
            
            # Create action record
            JobAction.objects.create(
                job=job,
                employee=employee,
                action_type='rejected',
                notes=f"Job rejected by {employee.full_name}. Reason: {rejection_message}"
            )
            
            # Import message system models
            from message_system.models import ChatRoom, Message
            
            # Check if chat room exists, create if not
            chat_room, created = ChatRoom.objects.get_or_create(
                employer=job.employer,
                employee=employee,
                defaults={
                    'subject': f'Job #{job.job_id}: {job.title}',
                    'room_type': 'job',
                }
            )
            
            # Update chat room with job reference if not set
            if not chat_room.job:
                chat_room.job = job
                chat_room.save()
            
            # Create rejection message in chat
            message = Message.objects.create(
                room=chat_room,
                sender_type='employee',
                sender_employee=employee,
                content=f" **Job Rejected**\n\nI've declined the job '{job.title}'.\n\n**Reason:** {rejection_message}\n\nJob ID: #{job.job_id}\nProposed Date: {job.proposed_date}\nBudget: ₹{job.budget if job.budget else 'Negotiable'}",
                message_type='job_update',
                status='sent'
            )
            
            # Update chat room stats
            chat_room.message_count += 1
            chat_room.unread_employer += 1
            chat_room.last_message_time = timezone.now()
            chat_room.save()
            
            # Create chat notification for employer
            from message_system.models import ChatNotification
            ChatNotification.objects.create(
                user_type='employer',
                user_employer=job.employer,
                room=chat_room,
                message=message,
                notification_type='job_update',
                title='Job Rejected',
                message_preview=f"{employee.full_name} rejected your job '{job.title}'"
            )
            
            # Create Employee Notification
            EmployeeNotification.objects.create(
                employee=employee,
                title="Job Rejected",
                message=f"You rejected the job '{job.title}' and sent a message to the employer.",
                notification_type='job',
                is_read=True,
                link=f"/employee/job/details/{job.job_id}/"
            )
            
            return JsonResponse({
                'success': True, 
                'message': 'Job rejected and message sent successfully!',
                'redirect': reverse('employee_job_request')
            })
            
        except JobRequest.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Job request not found.'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method.'})

#****************************************************

def reject_job_with_message(request):
    """Handle job rejection with message"""
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    if request.method == 'POST':
        # This will now be handled by the AJAX function
        pass
    
    return redirect('employee_job_request')
