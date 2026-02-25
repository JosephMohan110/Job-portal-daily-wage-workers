# Document Upload Views
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from .models import Employee


def upload_employee_document(request):
    """Upload documents for employee (Aadhar card, certificates, etc)"""
    if request.method != 'POST':
        messages.error(request, "Invalid request method.")
        return redirect('employee_setting')
    
    if 'employee_id' not in request.session:
        messages.error(request, "Please login first.")
        return redirect('index')
    
    try:
        employee = Employee.objects.get(employee_id=request.session['employee_id'])
        
        # Handle Aadhar document upload
        if 'aadhar_document' in request.FILES:
            aadhar_file = request.FILES['aadhar_document']
            
            # Validate file size (max 5MB)
            if aadhar_file.size > 5 * 1024 * 1024:
                messages.error(request, "File size must be less than 5MB. Please try again.", extra_tags='document')
                return redirect('employee_setting')
            
            # Validate file type
            allowed_types = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']
            if aadhar_file.content_type not in allowed_types:
                messages.error(request, "Only PDF and image files (JPG, PNG) are allowed.", extra_tags='document')
                return redirect('employee_setting')
            
            # Save the document
            employee.aadhar_document = aadhar_file
            employee.aadhar_verified = False  # Mark as pending verification
            employee.save()
            
            messages.success(
                request, 
                "Aadhar card uploaded successfully! Verification may take 1-2 business days.",
                extra_tags='document'
            )
        
        return redirect('employee_setting')
    
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.", extra_tags='document')
        return redirect('index')
    except Exception as e:
        messages.error(request, f"Error uploading document: {str(e)}", extra_tags='document')
        return redirect('employee_setting')


def delete_employee_document(request):
    """Delete uploaded documents"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Invalid request method'})
    
    if 'employee_id' not in request.session:
        return JsonResponse({'success': False, 'error': 'Not authenticated'})
    
    try:
        employee = Employee.objects.get(employee_id=request.session['employee_id'])
        doc_type = request.POST.get('document_type', '')
        
        if doc_type == 'aadhar':
            # Delete the file if it exists
            if employee.aadhar_document:
                employee.aadhar_document.delete()
            
            employee.aadhar_document = None
            employee.aadhar_verified = False
            employee.save()
            
            messages.success(request, "Aadhar card deleted successfully.", extra_tags='document')
        else:
            return JsonResponse({'success': False, 'error': 'Invalid document type'})
        
        return redirect('employee_setting')
    
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.", extra_tags='document')
        return redirect('index')
    except Exception as e:
        messages.error(request, f"Error deleting document: {str(e)}", extra_tags='document')
        return redirect('employee_setting')
