import json
import re
import os
from django.conf import settings
from django.urls import reverse, resolve, Resolver404
from urllib.parse import urljoin

class VoiceAssistantCore:
    def __init__(self):
        self.current_page = None
        self.context = {}
        self.user_data = {}
        
        # Command patterns
        self.command_patterns = {
            'navigation': [
                r'(?:go to|open|take me to|show|navigate to|visit|access)\s+(.+)',
                r'(?:I want to see|I want to go to)\s+(.+)',
                r'(?:take|open)\s+(.+)',
                r'(?:goto)\s+(.+)'
            ],
            'form_fill': [
                r'(?:enter|type|fill|put)\s+(.+)\s+(?:in|into)\s+(.+)',
                r'(?:my|the)\s+(.+)\s+is\s+(.+)',
                r'(?:username|email|password|phone)\s+(?:is|:)\s+(.+)'
            ],
            'explain': [
                r'(?:explain|what is|tell me about|describe)\s+(.+)',
                r'(?:how does|how to)\s+(.+)',
                r'(?:what are|what do)\s+(.+)',
                r'explain this page'
            ],
            'action': [
                r'(?:click|press|tap|select)\s+(.+)',
                r'(?:submit|save|send|login|register)\s*(?:the form)?',
                r'(?:search for|find)\s+(.+)'
            ]
        }
        
        # URL mappings from all apps
        self.url_mappings = self.load_all_urls()
        
    def load_all_urls(self):
        """Load all URLs from all installed apps"""
        mappings = {}
        
        # Home app URLs
        mappings.update({
            'home': '/',
            'index': '/',
            'main': '/',
            'register': '/register/',
            'registration': '/register/',
            'sign up': '/register/',
            'contact': '/contact_us/',
            'contact us': '/contact_us/',
            'contact page': '/contact_us/',
            'about': '/about_us/',
            'about us': '/about_us/',
            'about page': '/about_us/',
            'privacy': '/privacy_terms/',
            'terms': '/privacy_terms/',
            'privacy terms': '/privacy_terms/',
            'login': '/#login-section',
            'sign in': '/#login-section',
            'employer login': '/#login-section',
            'employee login': '/#login-section',
            'worker login': '/#login-section',
            'admin login': '/admin/login/',
            'logout': '/logout/',
            'chat bot': '/chat-bot-icon/',
        })
        
        # Employee app URLs
        mappings.update({
            'employee dashboard': '/employee/dashboard',
            'my dashboard': '/employee/dashboard',
            'worker dashboard': '/employee/dashboard',
            'employee earnings': '/employee/earnings',
            'my earnings': '/employee/earnings',
            'salary': '/employee/earnings',
            'income': '/employee/earnings',
            'job history': '/employee/job/history/',
            'my jobs': '/employee/job/history/',
            'previous jobs': '/employee/job/history/',
            'job request': '/employee/job/request',
            'find job': '/employee/job/request',
            'available jobs': '/employee/job/request',
            'employee reviews': '/employee/review/list',
            'my reviews': '/employee/review/list',
            'ratings': '/employee/review/list',
            'employee schedule': '/employee/schedule',
            'my schedule': '/employee/schedule',
            'calendar': '/employee/schedule',
            'availability': '/employee/schedule',
            'employee profile': '/employee/profile',
            'my profile': '/employee/profile',
            'profile page': '/employee/profile',
            'employee settings': '/employee/settings/',
            'my settings': '/employee/settings/',
            'account settings': '/employee/settings/',
            'employee notifications': '/employee/notifications',
            'my notifications': '/employee/notifications',
            'alerts': '/employee/notifications',
        })
        
        # Employer app URLs
        mappings.update({
            'employer dashboard': '/employer/dashboard/',
            'business dashboard': '/employer/dashboard/',
            'hiring history': '/employer/hiring-history/',
            'past hires': '/employer/hiring-history/',
            'find workers': '/employer/find-workers/',
            'search workers': '/employer/find-workers/',
            'hire workers': '/employer/find-workers/',
            'favorites': '/employer/favorites/',
            'saved workers': '/employer/favorites/',
            'bookmarks': '/employer/favorites/',
            'payment': '/employer/payment-section/',
            'payments': '/employer/payment-section/',
            'billing': '/employer/payment-section/',
            'employer settings': '/employer/settings/',
            'business settings': '/employer/settings/',
            'employer reviews': '/employer/reviews/',
            'give review': '/employer/reviews/',
            'hired employees': '/employer/hired-employees/',
        })
        
        # Message system URLs
        mappings.update({
            'messages': '/messages/',
            'chat': '/messages/',
            'inbox': '/messages/',
            'conversations': '/messages/',
            'message dashboard': '/messages/',
        })
        
        # Chat bot URLs
        mappings.update({
            'chat bot main': '/chat-bot/',
            'assistant': '/chat-bot/',
            'help bot': '/chat-bot/',
        })
        
        return mappings
    
    def process_command(self, command, current_url=None):
        """Process voice command and return response"""
        self.current_page = current_url
        command = command.lower().strip()
        
        # Check for greeting - but don't STOP processing, just return a greeting action
        # Actually, for a web assistant, immediate action is better.
        # If user says "hello", we greet. If "open about", we open.
        
        if command in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']:
            return self.handle_greeting()
        
        # Check for help
        if 'help' in command or 'what can you do' in command:
            return self.handle_help()
        
        # Check for stop
        if any(stop in command for stop in ['stop', 'exit', 'quit', 'close', 'end']):
            return self.handle_stop()
        
        # Try to match navigation command
        nav_result = self.match_navigation(command)
        if nav_result:
            # Special handling for login switching
            if 'login' in command and 'employer' in command:
                return {
                    'type': 'navigation',
                    'message': "Showing Employer Login.",
                    'action': 'switch_login',
                    'target': 'employer',
                    'url': '/#login-section'
                }
            elif 'login' in command and 'employee' in command:
                 return {
                    'type': 'navigation',
                    'message': "Showing Employee Login.",
                    'action': 'switch_login',
                    'target': 'employee',
                    'url': '/#login-section'
                }
            return nav_result
        
        # Check for scroll commands
        if 'scroll' in command:
            if 'down' in command:
                 return {
                    'type': 'action',
                    'message': "Scrolling down.",
                    'action': 'scroll',
                    'direction': 'down'
                }
            elif 'up' in command:
                 return {
                    'type': 'action',
                    'message': "Scrolling up.",
                    'action': 'scroll',
                    'direction': 'up'
                }
            elif 'top' in command:
                 return {
                    'type': 'action',
                    'message': "Scrolling to top.",
                    'action': 'scroll',
                    'direction': 'top'
                }
            elif 'bottom' in command:
                 return {
                    'type': 'action',
                    'message': "Scrolling to bottom.",
                    'action': 'scroll',
                    'direction': 'bottom'
                }

        # Detect employee/employer username entry
        if 'enter' in command and 'username' in command:
            if 'employee' in command:
                return {
                    'type': 'action',
                    'message': "Starting Employee Login. Please say your email.",
                    'action': 'start_login_flow',
                    'login_type': 'employee'
                }
            elif 'employer' in command:
                return {
                    'type': 'action',
                    'message': "Starting Employer Login. Please say your email.",
                    'action': 'start_login_flow',
                    'login_type': 'employer'
                }
        
        # interactive login flow - generic
        if 'enter' in command and ('username' in command or 'email' in command) and 'password' in command:
             return {
                'type': 'action',
                'message': "Okay, let's log you in.",
                'action': 'start_login_flow'
            }
        
        # Try to match form fill command
        form_result = self.match_form_fill(command)
        if form_result:
            return form_result
        
        # Try to match explanation request
        explain_result = self.match_explanation(command)
        if explain_result:
            return explain_result
        
        # Try to match action command
        action_result = self.match_action(command)
        if action_result:
            return action_result
            
        # If no strict match, try to see if the command contains a page name directly
        # e.g. "about page" -> go to about page
        for page_name, url in self.url_mappings.items():
            if page_name in command:
                 return self.handle_navigation(page_name)
        
        # Default response
        return {
            'type': 'response',
            'message': "I didn't understand that command. Try saying 'go to login', 'about page', or 'help'.",
            'action': 'speak_only'
        }
    
    def match_navigation(self, command):
        """Match navigation commands"""
        for pattern in self.command_patterns['navigation']:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                page_name = match.group(1).strip()
                return self.handle_navigation(page_name)
        
        return None
    
    def handle_navigation(self, page_name):
        """Handle navigation to a page"""
        # Clean up page name
        page_name = page_name.replace('page', '').strip()
        
        # Find the URL for the page
        url = None
        
        # 1. Exact match on keys
        if page_name in self.url_mappings:
            url = self.url_mappings[page_name]
        
        # 2. Contained match
        if not url:
            for key, value in self.url_mappings.items():
                if key in page_name or page_name in key:
                    url = value
                    # Prefer exact matches or longer matches if possible, but first find is often okay.
                    # Let's refine: if we find multiple, maybe pick the one with most overlap?
                    # For now, simple containment is enough for "about" -> "about us"
                    break
        
        if not url:
            # Try to find similar page
            similar = self.find_similar_page(page_name)
            if similar:
                return {
                    'type': 'clarification',
                    'message': f"Did you mean '{similar}'?",
                    'options': [similar],
                    'action': 'clarify_navigation'
                }
            else:
                return {
                    'type': 'response',
                    'message': f"I couldn't find a page named '{page_name}'. Try saying 'help' to see available pages.",
                    'action': 'speak_only'
                }
        
        # Get page explanation
        explanation = self.get_page_explanation(url)
        
        return {
            'type': 'navigation',
            'message': f"Opening {page_name}.",
            'url': url,
            'page_name': page_name,
            'action': 'navigate_and_explain'
        }
    
    def find_similar_page(self, page_name):
        """Find similar page names using fuzzy matching"""
        import difflib
        pages = list(self.url_mappings.keys())
        matches = difflib.get_close_matches(page_name.lower(), pages, n=1, cutoff=0.6)
        return matches[0] if matches else None
    
    def get_page_explanation(self, url):
        """Get explanation for a page"""
        explanations = {
            '/register/': "Registration page.",
            '/employer/login/': "Employer login.",
            '/employee/login/': "Employee login.",
            '/contact_us/': "Contact us.",
            '/about_us/': "About us.",
            '/employee/dashboard': "Employee dashboard.",
            '/employer/dashboard/': "Employer dashboard.",
            '/employee/earnings': "Your earnings.",
            '/employee/job/history/': "Your job history.",
            '/employee/job/request': "Available jobs.",
            '/employer/find-workers/': "Find workers.",
            '/messages/': "Messages.",
            '/employee/settings/': "Settings.",
        }
        
        return explanations.get(url, "")
    
    def match_form_fill(self, command):
        """Match form filling commands"""
        for pattern in self.command_patterns['form_fill']:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    value, field = match.groups()
                    return self.handle_form_fill(field.strip(), value.strip())
                elif len(match.groups()) == 1:
                    # Handle patterns like "username is john"
                    parts = command.split()
                    if 'is' in parts:
                        idx = parts.index('is')
                        field = ' '.join(parts[:idx])
                        value = ' '.join(parts[idx+1:])
                        return self.handle_form_fill(field, value)
        
        # Direct field-value pairs
        common_fields = ['username', 'email', 'password', 'phone', 'name', 'first name', 'last name']
        for field in common_fields:
            if field in command:
                # Extract value after field
                pattern = rf'{field}\s+(?:is|:)?\s*(.+)'
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    return self.handle_form_fill(field, value)
        
        return None
    
    def handle_form_fill(self, field, value):
        """Handle form field filling"""
        field_mappings = {
            'username': ['username', 'user name', 'login'],
            'email': ['email', 'email address', 'e-mail'],
            'password': ['password', 'pass', 'secret'],
            'phone': ['phone', 'phone number', 'mobile', 'contact number'],
            'name': ['name', 'full name', 'first name', 'last name']
        }
        
        normalized_field = field.lower()
        for key, aliases in field_mappings.items():
            if any(alias in normalized_field for alias in aliases):
                return {
                    'type': 'form_fill',
                    'message': f"Entering {value} in {field}.",
                    'field': key,
                    'value': value,
                    'action': 'fill_field'
                }
        
        return {
            'type': 'form_fill',
            'message': f"Entering {value} in {field}.",
            'field': field,
            'value': value,
            'action': 'fill_field_generic'
        }
    
    def match_explanation(self, command):
        """Match explanation requests"""
        for pattern in self.command_patterns['explain']:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                if command.strip() == 'explain this page':
                    # Frontend should handle 'this page' by telling us the url, but simpler:
                    # We just return a generic 'explain' action and frontend passes current url context?
                    # Actually, better to just return an explanation action that triggers the frontend to read the page title/meta.
                    # Or we use the current_page passed in.
                   return {
                        'type': 'explanation',
                        'message': "I will explain this page.",
                        'topic': 'current',
                        'action': 'explain_current'
                    }
                
                topic = match.group(1).strip()
                return self.handle_explanation(topic)
        
        return None
    
    def handle_explanation(self, topic):
        """Handle explanation requests"""
        explanations = {
            'login page': "Scrolling to the login section. Choose Employer or Employee login.",
            'registration': "Create a new account here.",
            'dashboard': "Your main control center.",
            'job request': "Available jobs you can apply for.",
            'earnings': "Your income and payment history.",
            'schedule': "Your availability and upcoming jobs.",
            'profile': "Your skills, experience, and ratings.",
            'settings': "Update your account information and preferences.",
            'messages': "Communicate with employers or employees.",
            'hiring history': "Workers you've hired and job details.",
            'find workers': "Search for available employees.",
            'payment': "Financial transactions and invoices.",
            '/#login-section': "Here is the login section. Please select your role to login.",
        }
        
        for key, explanation in explanations.items():
            if key in topic.lower():
                return {
                    'type': 'explanation',
                    'message': explanation,
                    'topic': key,
                    'action': 'speak_only'
                }
        
        return {
            'type': 'explanation',
            'message': f"I'm not sure about '{topic}'.",
            'action': 'speak_only'
        }
    
    def match_action(self, command):
        """Match action commands"""
        for pattern in self.command_patterns['action']:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                if match.groups():
                    target = match.group(1).strip()
                    return self.handle_action(target, command)
                else:
                    return self.handle_action(None, command)
        
        # Direct actions
        actions = ['submit', 'login', 'register', 'search', 'find', 'apply', 'save', 'send']
        for action in actions:
            if action in command:
                return self.handle_action(action, command)
        
        return None
    
    def handle_action(self, target, command):
        """Handle action commands"""
        action_map = {
            'submit': "Submitting form.",
            'login': "Logging in.",
            'register': "Registering.",
            'search': "Searching.",
            'apply': "Applying.",
            'save': "Saving.",
            'send': "Sending."
        }
        
        for action, response in action_map.items():
            if action in command:
                return {
                    'type': 'action',
                    'message': response,
                    'action_name': action,
                    'target': target,
                    'action': 'perform_action'
                }
        
        return {
            'type': 'action',
            'message': f"Performing {command}.",
            'action_name': 'generic',
            'target': target,
            'action': 'perform_generic_action'
        }
    
    def handle_greeting(self):
        """Handle greetings"""
        return {
            'type': 'greeting',
            'message': "Hello! How can I help?",
            'action': 'speak_only'
        }
    
    def handle_help(self):
        """Handle help requests"""
        return {
            'type': 'help',
            'message': "You can say 'go to login', 'about page', or 'fill username John'.",
            'action': 'speak_only'
        }
    
    def handle_stop(self):
        """Handle stop commands"""
        return {
            'type': 'stop',
            'message': "Goodbye.",
            'action': 'stop'
        }