import re
import os

files = [
    r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۲_مهندسی_داده_و_Event_Log.md",
    r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۳_سیستم_تفسیری_Rule_based.md",
    r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۴_Temporal_Drift_و_WA_پیشرفته.md"
]

def process_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Wrap Code Blocks
    code_block_pattern = re.compile(r'(```[\w]*\n[\s\S]*?\n```)')
    parts = code_block_pattern.split(content)
    
    new_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text part
            new_parts.append(part)
        else:
            # Code block part
            code_block = part
            prev_text = parts[i-1]
            
            # Check if already wrapped
            stripped_prev = prev_text.rstrip()
            is_wrapped = False
            if stripped_prev.endswith('<div dir="ltr">') or stripped_prev.endswith('<div dir="ltr" style="page-break-inside: avoid;">'):
                is_wrapped = True
            
            if not is_wrapped:
                # Wrap it
                wrapped_block = f'\n<div dir="ltr" style="page-break-inside: avoid;">\n\n{code_block}\n\n</div>\n'
                new_parts.append(wrapped_block)
            else:
                # Already wrapped. Check if we need to update style.
                if stripped_prev.endswith('<div dir="ltr">'):
                    # Update the previous part to include style
                    last_part = new_parts.pop()
                    # Find the last occurrence and replace
                    idx = last_part.rfind('<div dir="ltr">')
                    if idx != -1:
                        updated_last_part = last_part[:idx] + '<div dir="ltr" style="page-break-inside: avoid;">' + last_part[idx+15:]
                        new_parts.append(updated_last_part)
                    else:
                        new_parts.append(last_part) # Should not happen based on check
                    new_parts.append(code_block)
                else:
                    # Already has style or some other variation, leave it
                    new_parts.append(code_block)

    content_after_code_fix = "".join(new_parts)

    # 2. Wrap Tables
    lines = content_after_code_fix.splitlines(keepends=True)
    final_lines = []
    
    def is_table_line(line):
        s = line.strip()
        return s.startswith('|') and s.endswith('|')

    i = 0
    while i < len(lines):
        line = lines[i]
        
        if is_table_line(line):
            # Check if already wrapped
            # Look backwards in final_lines for a div start
            j = len(final_lines) - 1
            is_already_wrapped = False
            while j >= 0:
                if final_lines[j].strip() == "":
                    j -= 1
                    continue
                if '<div' in final_lines[j] and 'page-break-inside' in final_lines[j]:
                    is_already_wrapped = True
                # Also check for simple div if we missed it (though we shouldn't have)
                elif '<div' in final_lines[j]: 
                     # It might be a div without style, but we want to be safe.
                     # If it's a div, we assume it's wrapping the table.
                     is_already_wrapped = True
                break
            
            if is_already_wrapped:
                final_lines.append(line)
                i += 1
            else:
                # Collect table
                table_lines = []
                while i < len(lines) and is_table_line(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                
                final_lines.append('\n<div style="page-break-inside: avoid;">\n\n')
                final_lines.extend(table_lines)
                final_lines.append('\n</div>\n')
        else:
            final_lines.append(line)
            i += 1

    final_content = "".join(final_lines)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_content)
    print(f"Processed {filepath}")

for f in files:
    process_file(f)
