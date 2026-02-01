from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/login/', views.admin_login_view, name='admin_login'),
    path('admin/logout/', views.admin_logout_view, name='admin_logout'),
    
    path('admin/dashboard', views.admin_dashboard, name='admin_dashboard'),
    
    path('admin/analytics/prediction', views.analytics_prediction, name='analytics_prediction'),
    
    path('admin/bookings/', views.bookings, name='bookings'),
    path('admin/bookings/update-status/', views.update_booking_status, name='update_booking_status'),
    path('admin/bookings/process-refund/', views.process_refund, name='process_refund'),
    path('admin/bookings/export/', views.export_bookings_csv, name='export_bookings_csv'),


    path('admin/manage/employer/', views.manage_employer, name='manage_employer'),  # Single endpoint for all employer actions
    path('admin/manage/employer/export/', views.export_employers_csv, name='export_employers_csv'),
   
    path('admin/manage/workers/', views.manage_workers, name='manage_workers'),
    path('admin/manage/workers/export/', views.export_workers_csv, name='export_workers_csv'),


    
    path('admin/review/ratings/', views.review_ratings, name='review_ratings'),
    path('admin/review/action/', views.handle_review_action, name='handle_review_action'),
    path('admin/review/export/', views.export_reviews_csv, name='export_reviews_csv'),
    path('admin/review/filter/', views.apply_review_filters, name='apply_review_filters'),



    path('support/', views.support_page, name='support_page'),
    path('send-support-message/', views.send_support_message, name='send_support_message'),



    # Payment Dashboard URLs
    path('admin/payment/dashboard/', views.admin_payment_dashboard, name='admin_payment_dashboard'),
    path('admin/payment/dashboard/stats/', views.get_dashboard_stats, name='get_dashboard_stats'),
    path('admin/payment/payouts/create/', views.create_payout, name='create_payout'),
    path('admin/payment/payouts/verify-razorpay/', views.verify_razorpay_payout, name='verify_razorpay_payout'),
    path('admin/payment/payouts/create-ajax/', views.create_payout_ajax, name='create_payout_ajax'),
    path('admin/payment/commissions/create/', views.create_commissions, name='create_commissions'),
    path('admin/payment/commissions/', views.view_all_commissions, name='view_all_commissions'),
    path('admin/payment/commissions/<int:commission_id>/', views.commission_details, name='commission_details'),
    path('admin/payment/commissions/<int:commission_id>/update/', views.update_commission_status, name='update_commission_status'),
    path('admin/payment/commissions/export/', views.export_commissions, name='export_commissions'),
    
    path('admin/payment/payouts/create/', views.create_payout_view, name='create_payout'),
    path('admin/payment/payouts/', views.view_all_payouts, name='view_all_payouts'),
    path('admin/payment/payouts/<int:payout_id>/', views.view_payout_details, name='view_payout_details'),
    path('admin/payment/payouts/<int:payout_id>/process/', views.process_payout, name='process_payout'),
    path('admin/payment/payouts/<int:payout_id>/update/', views.update_payout_status, name='update_payout_status'),
    path('admin/payment/payouts/export/', views.export_payouts, name='export_payouts'),
    
    path('admin/payment/reports/', views.revenue_reports, name='revenue_reports'),



    path('algorithm/setting/', views.algorithm_setting, name='algorithm_setting'),
    path('algorithm/upload-model/', views.upload_ml_model, name='upload_ml_model'),
    path('algorithm/model/<int:model_id>/update-status/', views.update_model_status, name='update_model_status'),
    path('algorithm/model/<int:model_id>/delete/', views.delete_model, name='delete_model'),
    path('algorithm/model/<int:model_id>/download/', views.download_model_file, name='download_model_file'),
    path('algorithm/model/<int:model_id>/details/', views.get_model_details, name='get_model_details'),
    path('algorithm/export-data/', views.export_data_csv, name='export_data_csv'),
    path('algorithm/collect-data/', views.collect_data_now, name='collect_data_now'),
    path('algorithm/start-training/', views.start_self_training, name='start_self_training'),
    path('algorithm/train-upload/', views.train_from_uploaded_data, name='train_from_uploaded_data'),


    path('algorithm/old-models/', views.get_old_models_list, name='get_old_models_list'),
    
    # Real-time prediction API
    path('api/real-time-prediction/', views.real_time_prediction_api, name='real_time_prediction_api'),





]




if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)