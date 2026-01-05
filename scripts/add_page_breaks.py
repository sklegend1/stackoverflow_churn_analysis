import os
import re

files = [
    "فصل_۱_مقدمه_و_تعریف_مسئله.md",
    "فصل_۲_مهندسی_داده_و_Event_Log.md",
    "فصل_۳_سیستم_تفسیری_Rule_based.md",
    "فصل_۴_Temporal_Drift_و_WA_پیشرفته.md",
    "فصل_۵_مدل_LSTM_v2.md",
    "فصل_۶_جمع_بندی_و_نتیجه_گیری.md"
]
base_path = r"f:\Churn Pred Data\mysql data\docs_detailed"

def process_file(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    in_div = False
    in_table = False
    table_buffer = []

    def flush_table():
        nonlocal table_buffer
        if table_buffer:
            # Check if it really looks like a table (has at least 2 rows, one being separator)
            # But for simplicity, if it's a block of lines starting with |, we wrap it.
            new_lines.append('\n<div style="page-break-inside: avoid;">\n\n')
            new_lines.extend(table_buffer)
            new_lines.append('\n</div>\n')
            table_buffer = []

    for line in lines:
        # 1. Handle Code Block Divs (Update style)
        if '<div dir="ltr">' in line:
            # Add page-break-inside: avoid
            line = line.replace('<div dir="ltr">', '<div dir="ltr" style="page-break-inside: avoid;">')
            in_div = True
            if in_table: # Should not happen if structure is clean, but safety first
                flush_table()
                in_table = False
            new_lines.append(line)
            continue
        
        if '</div>' in line:
            in_div = False
            new_lines.append(line)
            continue

        # 2. Handle Tables (Only if not inside a div)
        if not in_div:
            stripped = line.strip()
            # Heuristic for table row: starts and ends with |
            # Also allow for some leniency if it's a separator row like |---|
            is_table_row = stripped.startswith('|') and stripped.endswith('|')
            
            if is_table_row:
                in_table = True
                table_buffer.append(line)
            else:
                if in_table:
                    flush_table()
                    in_table = False
                new_lines.append(line)
        else:
            # Inside div, just append
            new_lines.append(line)

    # Flush at end of file
    if in_table:
        flush_table()

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Processed {filepath}")

for fname in files:
    process_file(os.path.join(base_path, fname))
