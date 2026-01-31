from django import template
from decimal import Decimal

register = template.Library()

@register.filter
def format_currency(value):
    """Format value as Indian Rupees with ₹ symbol"""
    try:
        if isinstance(value, (Decimal, float, int)):
            return f"₹{value:,.2f}"
        return value
    except (ValueError, TypeError):
        return "₹0.00"

# Keep your existing filters
@register.filter
def get_item(dictionary, key):
    """Get item from dictionary by key"""
    return dictionary.get(key, '')

@register.filter
def replace(value, arg):
    """Replace characters in string"""
    old, new = arg.split(',')
    return value.replace(old, new)

@register.filter
def add(value, arg):
    """Add two values"""
    try:
        return int(value) + int(arg)
    except (ValueError, TypeError):
        try:
            return float(value) + float(arg)
        except (ValueError, TypeError):
            return value