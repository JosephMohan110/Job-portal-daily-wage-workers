import os
import glob
import re

directory = r'c:\Users\LENOVO\Desktop\jobportal\backend\templates\employee_html'
files = glob.glob(os.path.join(directory, '*.html'))

for file in files:
    if 'employee_header.html' in file: continue
    
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original = content
    
    # Simple regex to move include employee_header.html into body
    # Find something like:
    # {% include 'employee_html/employee_header.html' %}
    # <body>
    # and replace with:
    # <body>
    # {% include 'employee_html/employee_header.html' %}
    content = re.sub(
        r'({%\s*include\s+[\'"]employee_html/employee_header\.html[\'"]\s*%})\s*(<body[^>]*>)',
        r'\2\n\1',
        content,
        flags=re.MULTILINE
    )
    
    # Also find if it's placed immediately after </head> and before <body> with no other tags:
    content = re.sub(
        r'(</head>)\s*({%\s*include\s+[\'"]employee_html/employee_header\.html[\'"]\s*%})\s*(<body[^>]*>)',
        r'\1\n\3\n\2',
        content,
        flags=re.MULTILINE
    )
    
    if content != original:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Fixed {os.path.basename(file)}')
