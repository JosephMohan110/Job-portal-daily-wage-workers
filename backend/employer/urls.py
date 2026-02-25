from django.contrib import admin
from django.urls import path
from . import views
from .document_views import upload_employer_document, delete_employer_document
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    # Dashboard and main pages
    path('dashboard/', views.employer_dashboard, name='employer_dashboard'),
    
    # Settings pages
    path('settings/', views.employer_setting, name='employer_setting'),
    path('settings/update-profile/', views.update_profile, name='update_profile'),
    path('settings/update-privacy-security/', views.update_privacy_security, name='update_privacy_security'),
    path('settings/update-location/', views.update_location, name='update_location'),
    path('settings/update-notifications/', views.update_notifications, name='update_notifications'),
    path('settings/update-preferences/', views.update_preferences, name='update_preferences'),
    path('settings/change-password/', views.change_password, name='change_password'),
    path('settings/deactivate-account/', views.deactivate_account, name='deactivate_account'),
    
    # Document URLs
    path('settings/upload-document/', upload_employer_document, name='upload_employer_document'),
    path('settings/delete-document/', delete_employer_document, name='delete_employer_document'),
    
    # Other pages
    path('hiring-history/', views.employer_hiring_history, name="employer_hiring_history"),
    path('find-workers/', views.employer_find_workers, name="employer_find_workers"),
    path('employee-profile/<int:employee_id>/', views.view_employee_public_profile, name='view_employee_public_profile'),
    
    # Favorites URLs
    path('favorites/', views.employer_favorites, name="employer_favorites"),
    path('favorites/add/<int:employee_id>/', views.add_to_favorites, name="add_to_favorites"),
    path('favorites/remove/<int:favorite_id>/', views.remove_from_favorites, name="remove_favorite"),
    path('favorites/remove-employee/<int:employee_id>/', views.remove_from_favorites, name="remove_favorite_by_employee"),
    
    # Hiring URLs
    path('hire/<int:employee_id>/', views.hire_employee_view, name='hire_employee'),
    path('employee/availability/<int:employee_id>/', views.get_employee_availability, name='employee_availability'),
    
    # Contact employee
    path('contact/<int:employee_id>/', views.contact_employee, name='contact_employee'),
    

    # ==================== REVIEW SYSTEM URLS ====================
    path('reviews/dashboard/', views.employer_reviews_dashboard, name='employer_reviews_dashboard'),
    path('reviews/', views.employer_reviews_dashboard, name='employer_reviews'),
    
    # Give review for an employee (with optional job_id)
    path('give-review/<int:employee_id>/', views.give_employee_review, name='give_employee_review'),
    # path('give-review/<int:employee_id>/job/<int:job_id>/', views.give_employee_review, name='give_employee_review_with_job'),
    
    path('view-reviews/<int:employee_id>/', views.view_employee_reviews, name='view_employee_reviews'),
    path('delete-review/<int:review_id>/', views.delete_employee_review, name='delete_employee_review'),
    
    # Site Review and Report URLs
    path('submit-site-review/', views.submit_site_review, name='submit_site_review'),
    path('submit-report/', views.submit_report, name='submit_report'),
    path('view-reports/', views.view_previous_reports, name='view_previous_reports'),
    path('submit-site-feedback/', views.submit_site_feedback, name='submit_site_feedback'),
    path('submit-employee-report/', views.submit_employee_report, name='submit_employee_report'),
    path('submit-site-issue-report/', views.submit_site_issue_report, name='submit_site_issue_report'),
    

    # Hiring History URLs (including job-specific review)
    path('hired-employees/', views.employer_hired_employees, name='employer_hired_employees'),
    path('hiring-history/export/', views.export_hiring_history, name="export_hiring_history"),
    path('hiring-history/print-report/', views.print_hiring_report, name="print_hiring_report"),
    path('hiring-history/job/<int:job_id>/', views.view_job_details, name="view_job_details"),
    path('hiring-history/job/<int:job_id>/review/', views.add_job_review, name="add_job_review"),
    
    # Payment URLs
    path('payment-section/', views.employer_payment_section, name="employer_payment_section"),
    path('payment/initiate/<int:job_id>/', views.initiate_payment, name='initiate_payment'),
    path('payment/success/', views.payment_success_callback, name='payment_success_callback'),
    path('payment/failure/', views.payment_failure_callback, name='payment_failure_callback'),
    path('payment/verify/', views.verify_payment, name='verify_payment'),
    path('payment/history/', views.payment_history, name='payment_history'),
    path('payment/invoice/<int:payment_id>/', views.view_invoice, name='view_invoice'),
    path('payment/invoice/<int:payment_id>/download/', views.download_invoice, name='download_invoice'),
    path('payment/invoice/<int:payment_id>/send/', views.send_invoice_email, name='send_invoice_email'),
    path('payment/receipt/<int:payment_id>/', views.view_receipt, name='view_receipt'),
    path('payment/refund/<int:payment_id>/', views.initiate_refund, name='initiate_refund'),
    path('payment/transaction/<int:payment_id>/', views.transaction_details, name='transaction_details'),
    path('payment/job/<int:job_id>/', views.view_payment_for_job, name='view_payment_for_job'),
    path('payment/quick-pay/<int:employee_id>/', views.quick_payment, name='quick_payment'),

    path('payment/webhook/', views.razorpay_webhook, name='razorpay_webhook'),

    # Notifications
    path('notifications/mark-read/<int:notification_id>/', views.mark_notification_read, name='mark_notification_read'),
    path('notifications/mark-all-read/', views.mark_all_notifications_read, name='mark_all_notifications_read'),

]


# Add media URL configuration
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)