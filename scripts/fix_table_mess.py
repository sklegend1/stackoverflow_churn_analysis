import os
import re

files = [
    r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۲_مهندسی_داده_و_Event_Log.md",
    r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۳_سیستم_تفسیری_Rule_based.md",
    r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۴_Temporal_Drift_و_WA_پیشرفته.md",
    r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۵_مدل_LSTM_v2.md" 
]
# Added chapter 5 just in case, though I think I fixed it separately. 
# But the user said "revert tables to previous state" for the ones I just touched.
# The user specifically mentioned "Chapter 2, 3, 4" in the previous request.
# I will stick to 2, 3, 4.

files_to_fix = [
    r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۲_مهندسی_داده_و_Event_Log.md",
    r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۳_سیستم_تفسیری_Rule_based.md",
    r"f:\Churn Pred Data\mysql data\docs_detailed\فصل_۴_Temporal_Drift_و_WA_پیشرفته.md"
]

def clean_file(filepath):
    if not os.path.exists(filepath):
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    
    # Step 1: Filter out the specific div lines that caused the mess
    # We look for <div style="page-break-inside: avoid;"> EXACTLY (or with whitespace)
    # We do NOT remove <div dir="ltr"...>
    
    temp_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == '<div style="page-break-inside: avoid;">':
            continue
        # We also need to remove the closing divs. 
        # This is the hard part: distinguishing </div> of table vs </div> of code.
        # However, the mess created nested divs.
        # If we remove ALL </div> lines that are NOT followed by a blank line? No.
        
        # Let's look at the context.
        # The bad divs were added around table parts.
        # If we just remove the bad opening divs, we are left with extra </div> tags.
        # And blank lines.
        
        temp_lines.append(line)

    # Now we have lines with extra </div> and blank lines between table rows.
    # Let's do a pass to remove blank lines between table rows.
    
    # Heuristic:
    # If line i is table row, and line i+1 is blank, and line i+2 is table row -> remove i+1
    # Also need to handle the extra </div> tags.
    # If line i is table row, i+1 is </div>, i+2 is table row -> remove i+1
    # If line i is table row, i+1 is </div>, i+2 is blank, i+3 is table row -> remove i+1, i+2
    
    # Actually, let's just rebuild the file.
    # We want to keep code blocks intact.
    
    final_lines = []
    i = 0
    while i < len(temp_lines):
        line = temp_lines[i]
        stripped = line.strip()
        
        # Check if it's a table row
        is_table = stripped.startswith('|') and stripped.endswith('|')
        
        if is_table:
            final_lines.append(line)
            # Look ahead
            j = i + 1
            while j < len(temp_lines):
                next_line = temp_lines[j]
                next_stripped = next_line.strip()
                
                if next_stripped == "":
                    j += 1
                    continue
                elif next_stripped == "</div>":
                    # If it's a closing div inside a table block (between rows), skip it
                    # But how do we know it's inside?
                    # If the NEXT non-empty line is a table row, then this </div> is garbage.
                    
                    # Look further ahead
                    k = j + 1
                    found_table_continuation = False
                    while k < len(temp_lines):
                        k_line = temp_lines[k].strip()
                        if k_line == "":
                            k += 1
                            continue
                        if k_line.startswith('|') and k_line.endswith('|'):
                            found_table_continuation = True
                        break
                    
                    if found_table_continuation:
                        j += 1 # Skip this </div>
                        continue
                    else:
                        # Not inside a table, so it's a real closing div (maybe end of table)
                        # But wait, we removed the opening <div style...>.
                        # So we should probably remove this closing div too if it matches the one we removed.
                        # But we don't know which one it matches.
                        
                        # If we assume code blocks are <div dir="ltr"> ... </div>
                        # We can count nesting?
                        
                        # Alternative: The bad script wrapped parts of tables.
                        # So we likely have:
                        # | header |
                        # </div> (maybe)
                        # | body |
                        # </div>
                        
                        # If we just remove </div> if it's immediately followed by a table row (ignoring blanks)?
                        pass
                        
                break
            
            # Now j points to the next content.
            # If we skipped lines, we need to advance i.
            # But we need to process the skipped lines? No, we want to discard them.
            # Wait, my logic above was just "looking", not consuming.
            
            # Let's restart the logic.
            i += 1
            continue

        # If it's a </div>
        if stripped == "</div>":
            # Check if it's closing a code block?
            # We can check if we are inside a code block.
            # But simpler: The bad divs were <div style="page-break-inside: avoid;">
            # We removed those lines.
            # So we have orphan </div> tags.
            # We should remove orphan </div> tags.
            # But how to identify orphans?
            # We can count open/close tags.
            pass
        
        final_lines.append(line)
        i += 1

    # Let's try a stack-based approach to remove the specific bad divs and their closing tags.
    # But since I already removed the opening tags in `temp_lines` logic (conceptually), I can't match them.
    
    # Let's go back to the original `lines`.
    
    cleaned_lines = []
    stack = [] # To track divs
    
    # We want to remove <div style="page-break-inside: avoid;"> AND its matching </div>
    # But keep <div dir="ltr"...> AND its matching </div>
    
    # Also remove blank lines that appear immediately after the removed opening div
    # or immediately before the removed closing div?
    
    skip_indices = set()
    
    # First pass: identify matching pairs
    div_stack = []
    
    for idx, line in enumerate(lines):
        s = line.strip()
        if s.startswith('<div'):
            if 'page-break-inside: avoid' in s and 'dir="ltr"' not in s:
                # This is the bad div
                div_stack.append((idx, 'bad'))
            else:
                # Good div (code or other)
                div_stack.append((idx, 'good'))
        elif s == '</div>':
            if div_stack:
                start_idx, type_ = div_stack.pop()
                if type_ == 'bad':
                    skip_indices.add(start_idx)
                    skip_indices.add(idx)
    
    # Second pass: Construct new content, skipping bad divs
    # And also merging table rows separated by blank lines
    
    pass2_lines = []
    for idx, line in enumerate(lines):
        if idx in skip_indices:
            continue
        pass2_lines.append(line)
        
    # Third pass: Remove blank lines between table rows
    pass3_lines = []
    i = 0
    while i < len(pass2_lines):
        line = pass2_lines[i]
        s = line.strip()
        
        if s.startswith('|') and s.endswith('|'):
            pass3_lines.append(line)
            # Look ahead for next table row, skipping blanks
            j = i + 1
            while j < len(pass2_lines):
                next_line = pass2_lines[j]
                if next_line.strip() == "":
                    j += 1
                    continue
                if next_line.strip().startswith('|') and next_line.strip().endswith('|'):
                    # Found next row. Skip the blanks in between.
                    i = j - 1 # Will be incremented to j in outer loop
                    break
                else:
                    # Not a table row, so the blanks were real (or end of table)
                    # We should keep the blanks? 
                    # Usually blank line ends table.
                    # But the bad script inserted blanks INSIDE the table (between header and body).
                    # So if we see |...| then BLANK then |...|, it's likely the split table.
                    # We should merge them.
                    
                    # But wait, what if it's two different tables?
                    # Unlikely to have two tables separated by just blank lines without text.
                    # Markdown treats |...|\n\n|...| as two tables.
                    # But here we want to restore the single table.
                    
                    # So yes, skip the blanks.
                    i = j - 1
                    break
        else:
            pass3_lines.append(line)
        i += 1
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(pass3_lines)
    print(f"Fixed {filepath}")

for f in files_to_fix:
    clean_file(f)
