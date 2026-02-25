from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.utils import timezone
from .models import Employer
import os
import traceback


@require_http_methods(["POST"])
def upload_employer_document(request):
    """
    Handle document upload for employers
    Validates file size (5MB max) and type (PDF, JPG, PNG only)
    """
    # Check session-based authentication
    if 'employer_id' not in request.session:
        return JsonResponse({'success': False, 'error': 'Please login first'}, status=401)
    
    try:
        # Get the employer profile
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
    except Employer.DoesNotExist:
        messages.error(request, "Employer profile not found.", extra_tags="document")
        return JsonResponse({'success': False, 'error': 'Employer profile not found'})
    except Exception as e:
        print(f"[ERROR] Failed to get employer: {e}")
        print(traceback.format_exc())
        messages.error(request, "An error occurred while retrieving your profile.", extra_tags="document")
        return JsonResponse({'success': False, 'error': str(e)})
    
    try:
        # Get the uploaded Aadhar file
        aadhar_file = request.FILES.get('aadhar_document')
        
        # Check if file is provided
        if not aadhar_file:
            messages.error(request, "Please select an Aadhar card file to upload.", extra_tags="document")
            return JsonResponse({'success': False, 'error': 'No file selected'})
        
        # Validate file size (5MB max)
        max_size = 5 * 1024 * 1024  # 5MB
        if aadhar_file.size > max_size:
            messages.error(request, "Aadhar card file size exceeds 5MB limit.", extra_tags="document")
            return JsonResponse({'success': False, 'error': 'File size exceeds 5MB'})
        
        # Validate file type
        allowed_types = ['application/pdf', 'image/jpeg', 'image/png']
        if aadhar_file.content_type not in allowed_types:
            messages.error(request, "Only PDF, JPG, and PNG files are allowed.", extra_tags="document")
            return JsonResponse({'success': False, 'error': 'Invalid file type'})
        
        # Delete old file if it exists
        if employer.aadhar_document:
            old_file = employer.aadhar_document
            if old_file.name:
                old_file.delete(save=False)
        
        # Save the new Aadhar document
        employer.aadhar_document = aadhar_file
        employer.aadhar_verified = False
        employer.aadhar_upload_date = timezone.now()
        employer.save()
        
        messages.success(request, "Documents uploaded successfully! They will be verified shortly.", extra_tags="document")
        return JsonResponse({'success': True})
        
    except Exception as e:
        print(f"[ERROR] Document upload failed: {e}")
        print(traceback.format_exc())
        messages.error(request, "Error uploading document. Please try again.", extra_tags="document")
        return JsonResponse({'success': False, 'error': str(e)})


@require_http_methods(["POST"])
def delete_employer_document(request):
    """
    Handle document deletion for employers
    """
    # Check session-based authentication
    if 'employer_id' not in request.session:
        return JsonResponse({'success': False, 'error': 'Please login first'}, status=401)
    
    try:
        # Get the employer profile
        employer = Employer.objects.get(employer_id=request.session['employer_id'])
    except Employer.DoesNotExist:
        messages.error(request, "Employer profile not found.", extra_tags="document")
        return JsonResponse({'success': False, 'error': 'Employer profile not found'})
    except Exception as e:
        print(f"[ERROR] Failed to get employer: {e}")
        print(traceback.format_exc())
        messages.error(request, "An error occurred while retrieving your profile.", extra_tags="document")
        return JsonResponse({'success': False, 'error': str(e)})
    
    try:
        document_type = request.POST.get('document_type', '')
        
        if document_type == 'business':
            # Delete the business document
            if employer.business_document:
                employer.business_document.delete(save=False)
                employer.business_verified = False
                employer.save()
                messages.success(request, "Business document deleted successfully.", extra_tags="document")
                return JsonResponse({'success': True})
            else:
                messages.warning(request, "No business document to delete.", extra_tags="document")
                return JsonResponse({'success': False, 'error': 'No document found'})
        
        elif document_type == 'aadhar':
            # Delete the Aadhar document
            if employer.aadhar_document:
                employer.aadhar_document.delete(save=False)
                employer.aadhar_verified = False
                employer.save()
                messages.success(request, "Aadhar card deleted successfully.", extra_tags="document")
                return JsonResponse({'success': True})
            else:
                messages.warning(request, "No Aadhar card to delete.", extra_tags="document")
                return JsonResponse({'success': False, 'error': 'No document found'})
        
        else:
            messages.error(request, "Invalid document type.", extra_tags="document")
            return JsonResponse({'success': False, 'error': 'Invalid document type'})
            
    except Exception as e:
        print(f"[ERROR] Document deletion failed: {e}")
        print(traceback.format_exc())
        messages.error(request, "Error deleting document. Please try again.", extra_tags="document")
        return JsonResponse({'success': False, 'error': str(e)})
