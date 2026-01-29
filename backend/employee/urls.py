# employee/urls.py

from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [

    # Employee Dashboard URLs
    path('dashboard', views.employee_dashboard, name='employee_dashboard'),

    
    path('earnings', views.employee_earnings, name='employee_earnings'),
     
     
     # Job History URLs
    path('job/history/', views.employee_job_history, name='employee_job_history'),
    path('job/history/payment/<int:job_id>/', views.job_history_payment, name='job_history_payment'),
    path('job/history/review/<int:job_id>/', views.job_history_review, name='job_history_review'),
    

    #job request
    path('job/request', views.employee_job_request, name='employee_job_request'),
    # 1. Job Details View
    path('job/details/<int:job_id>/', views.job_request_details, name='job_request_details'),
    # 2. Individual Accept/Reject (if used by your template)
    path('job/accept/', views.accept_job_request, name='accept_job_request'),
    path('job/reject/', views.reject_job_request, name='reject_job_request'),
    # 3. Generic Status Update (The one currently causing the error)
    path('job/update-status/', views.update_job_status, name='update_job_status'),
    # 4. Filter Jobs (if your template uses the filter function)
    path('job/filter/', views.filter_job_requests, name='filter_job_requests'),

    path('job/reject-with-message/', views.send_rejection_message, name='send_rejection_message'),



    #employee_review_list
    path('review/list', views.employee_review_list, name='employee_review_list'),
    path('submit-site-feedback/', views.submit_employee_site_feedback, name='submit_employee_site_feedback'),



    path('schedule', views.employee_schedule, name='employee_schedule'),
    path('schedule/bulk-unavailable/', views.bulk_mark_unavailable, name='bulk_mark_unavailable'),
    
    
        # Employee Profile URLs
    path('profile', views.employee_profile, name='employee_profile'),
    path('profile/update-bio/', views.update_employee_bio, name='update_employee_bio'),
    path('profile/add-experience/', views.add_employee_experience, name='add_employee_experience'),
    path('profile/update-experience/<int:experience_id>/', views.update_employee_experience, name='update_employee_experience'),
    path('profile/delete-experience/<int:experience_id>/', views.delete_employee_experience, name='delete_employee_experience'),
    path('profile/add-certificate/', views.add_employee_certificate, name='add_employee_certificate'),
    path('profile/update-certificate/<int:certificate_id>/', views.update_employee_certificate, name='update_employee_certificate'),
    path('profile/delete-certificate/<int:certificate_id>/', views.delete_employee_certificate, name='delete_employee_certificate'),
    path('profile/download-certificate/<int:certificate_id>/', views.download_certificate, name='download_certificate'),
    path('profile/add-portfolio/', views.add_employee_portfolio, name='add_employee_portfolio'),
    path('profile/update-portfolio/<int:portfolio_id>/', views.update_employee_portfolio, name='update_employee_portfolio'),
    path('profile/delete-portfolio/<int:portfolio_id>/', views.delete_employee_portfolio, name='delete_employee_portfolio'),
    path('profile/add-skill/', views.add_employee_skill, name='add_employee_skill'),
    path('profile/remove-skill/', views.remove_employee_skill, name='remove_employee_skill'),
    path('profile/update-availability/', views.update_employee_availability, name='update_employee_availability'),
    path('profile/update-profile-image/', views.update_employee_profile_image, name='update_employee_profile_image'),
    path('profile/update-cover-image/', views.update_employee_cover_image, name='update_employee_cover_image'),
    path('profile/update-professional-info/', views.update_employee_professional_info, name='update_employee_professional_info'),
    

    # Employee Settings URLs
    path('settings/', views.employee_setting, name='employee_setting'),
    path('settings/update-profile/', views.update_employee_profile, name='update_employee_profile'),
    path('settings/change-password/', views.change_employee_password, name='change_employee_password'),
    path('settings/update-privacy/', views.update_employee_privacy_security, name='update_employee_privacy'),
    path('settings/update-location/', views.update_employee_location, name='update_employee_location'),
    path('settings/update-notifications/', views.update_employee_notifications, name='update_employee_notifications'),
    path('settings/update-preferences/', views.update_employee_preferences, name='update_employee_preferences'),
    path('settings/deactivate-account/', views.deactivate_employee_account, name='deactivate_employee_account'),
    path('settings/security-info/', views.get_employee_security_info, name='get_employee_security_info'),
    path('settings/check-phone/', views.check_phone_availability, name='check_phone_availability'),
    path('settings/get-stats/', views.get_employee_stats, name='get_employee_stats'),


    path('notifications', views.employee_notifications, name='employee_notifications'),
    path('notifications/mark-read/<int:notification_id>/', views.mark_notification_read, name='mark_notification_read'),
    path('notifications/mark-all-read/', views.mark_all_notifications_read, name='mark_all_notifications_read'),


    

    

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)