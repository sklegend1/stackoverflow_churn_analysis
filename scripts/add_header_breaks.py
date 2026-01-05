import re
import os

filepath = r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۵_مدل_LSTM_v2.md"

def add_page_breaks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    # Regex for Level 2 headers with numbering (e.g., ## 5.1, ## ۵.۱)
    # Matches: ##, whitespace, digits (English/Persian), dot, digits
    header_pattern = re.compile(r'^##\s+[\d\u06F0-\u06F9]+\.[\d\u06F0-\u06F9]+')
    
    # Page break marker
    page_break = '<div style="page-break-before: always;"></div>'

    for i, line in enumerate(lines):
        if header_pattern.match(line):
            # Check if we are at the very beginning of the file (unlikely for H2)
            # or if the previous line is already a page break
            
            # We'll just insert the break. 
            # To avoid double breaks if run multiple times, we can check the last added line.
            
            is_already_broken = False
            if new_lines:
                last_line = new_lines[-1].strip()
                if page_break in last_line:
                    is_already_broken = True
                # Also check if the break is a few lines up (empty lines in between)
                elif len(new_lines) > 1 and page_break in new_lines[-2]:
                     is_already_broken = True
            
            if not is_already_broken:
                new_lines.append(f'\n{page_break}\n\n')
            
            new_lines.append(line)
        else:
            new_lines.append(line)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Processed {file_path}")

add_page_breaks(filepath)
