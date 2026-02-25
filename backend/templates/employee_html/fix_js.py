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
    
    # Remove menuToggle block
    content = re.sub(
        r'if\s*\(\s*menuToggle.*?\{\s*menuToggle\.addEventListener.*?\w+\.classList\.add\(\'open\'\);\s*;?.*?\}\s*\}',
        '',
        content,
        flags=re.DOTALL
    )
    
    # Remove sidebarClose block
    content = re.sub(
        r'if\s*\(\s*sidebarClose.*?\{\s*sidebarClose\.addEventListener.*?\w+\.classList\.remove\(\'open\'\);\s*;?.*?\}\s*\}',
        '',
        content,
        flags=re.DOTALL
    )
    
    # Remove overlay click block specifically tied to removing open
    content = re.sub(
        r'if\s*\(\s*overlay.*?\{\s*overlay\.addEventListener.*? sidebar\.classList\.remove\(\'open\'\);\s*;?.*?\}\s*\}',
        '',
        content,
        flags=re.DOTALL
    )
    
    if content != original:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Fixed JS in {os.path.basename(file)}')
