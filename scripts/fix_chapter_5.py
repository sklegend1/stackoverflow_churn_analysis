import re

filepath = r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۵_مدل_LSTM_v2.md"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Wrap Code Blocks
# We want to wrap ```...``` in <div dir="ltr" style="page-break-inside: avoid;">
# BUT only if it's not already wrapped.

# Regex to find code blocks
# (```[\w]*\n[\s\S]*?\n```)
code_block_pattern = re.compile(r'(```[\w]*\n[\s\S]*?\n```)')

def wrap_code(match):
    block = match.group(1)
    # Check if this block is already inside a div in the original content?
    # This is hard with regex substitution because we lose context.
    # Instead, let's split the content by code blocks and check surroundings.
    return block

# Better approach: Split by code blocks, then check the text before/after.
parts = code_block_pattern.split(content)
# parts[0] is text before first block
# parts[1] is first block
# parts[2] is text after first block (and before second)
# ...

new_parts = []
for i, part in enumerate(parts):
    if i % 2 == 0:
        # This is text (not code block)
        new_parts.append(part)
    else:
        # This is a code block
        code_block = part
        prev_text = parts[i-1]
        next_text = parts[i+1] if i+1 < len(parts) else ""
        
        # Check if already wrapped
        # We look at the end of prev_text for <div dir="ltr"...>
        # and start of next_text for </div>
        
        is_wrapped = False
        stripped_prev = prev_text.rstrip()
        if stripped_prev.endswith('<div dir="ltr">') or stripped_prev.endswith('<div dir="ltr" style="page-break-inside: avoid;">'):
            is_wrapped = True
        
        if not is_wrapped:
            # Wrap it
            # We add the div to the code block string itself
            wrapped_block = f'\n<div dir="ltr" style="page-break-inside: avoid;">\n\n{code_block}\n\n</div>\n'
            new_parts.append(wrapped_block)
        else:
            # Already wrapped, just keep it
            # But maybe update the style if it's missing page-break
            if stripped_prev.endswith('<div dir="ltr">'):
                # We need to update the PREVIOUS part to include the style
                # This is tricky because we already appended prev_text to new_parts
                # Let's modify the last element of new_parts
                last_part = new_parts.pop()
                # Replace the last occurrence of <div dir="ltr">
                updated_last_part = last_part[:last_part.rfind('<div dir="ltr">')] + '<div dir="ltr" style="page-break-inside: avoid;">' + last_part[last_part.rfind('<div dir="ltr">')+15:]
                new_parts.append(updated_last_part)
                new_parts.append(code_block)
            else:
                new_parts.append(code_block)

content_after_code_fix = "".join(new_parts)

# 2. Wrap Tables
# Similar logic. Find tables, check if wrapped.
# Table pattern: block of lines starting with |
# This is harder to regex perfectly, but let's try a line-by-line approach for tables since they are line-based.

lines = content_after_code_fix.splitlines(keepends=True)
final_lines = []
in_table = False
table_buffer = []

def is_table_line(line):
    s = line.strip()
    return s.startswith('|') and s.endswith('|')

def is_div_start(line):
    return '<div' in line

def is_div_end(line):
    return '</div>' in line

i = 0
while i < len(lines):
    line = lines[i]
    
    if is_table_line(line):
        # Start of potential table
        # Check if we are already inside a div?
        # We look at previous lines in final_lines
        
        # Check if the previous non-empty line was a div start
        j = len(final_lines) - 1
        is_already_wrapped = False
        while j >= 0:
            if final_lines[j].strip() == "":
                j -= 1
                continue
            if '<div' in final_lines[j]: # Simple check
                is_already_wrapped = True
            break
        
        if is_already_wrapped:
            final_lines.append(line)
            i += 1
        else:
            # Not wrapped. Collect the whole table.
            table_lines = []
            while i < len(lines) and is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1
            
            # Wrap the collected table
            final_lines.append('\n<div style="page-break-inside: avoid;">\n\n')
            final_lines.extend(table_lines)
            final_lines.append('\n</div>\n')
    else:
        final_lines.append(line)
        i += 1

final_content = "".join(final_lines)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(final_content)

print("Done processing Chapter 5.")
