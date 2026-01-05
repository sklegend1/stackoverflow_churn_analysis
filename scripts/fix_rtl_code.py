import os
import re

directory = r"f:\Churn Pred Data\mysql data\docs_detailed"
files = [
    "فصل_۱_مقدمه_و_تعریف_مسئله.md",
    "فصل_۲_مهندسی_داده_و_Event_Log.md",
    "فصل_۳_سیستم_تفسیری_Rule_based.md",
    "فصل_۴_Temporal_Drift_و_WA_پیشرفته.md",
    "فصل_۵_مدل_LSTM_v2.md",
    "فصل_۶_جمع_بندی_و_نتیجه_گیری.md"
]

def fix_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find code blocks
    # We look for ``` optionally followed by language, then content, then ```
    # We capture the whole block
    pattern = re.compile(r'(```[\w]*\n[\s\S]*?\n```)')
    
    def replacement(match):
        # Check if already wrapped
        # This is a simple check, might need more robustness if nested
        return f'\n<div dir="ltr">\n\n{match.group(1)}\n\n</div>\n'

    new_content = pattern.sub(replacement, content)
    
    # Avoid double wrapping if run multiple times (simple check)
    # If we see <div dir="ltr">\n\n<div dir="ltr">, we messed up. 
    # But since we are running once, it should be fine. 
    # To be safer, we could check if the match is already inside a div, but that's complex.
    # Let's just assume clean state or visual inspection.
    
    # Actually, let's clean up potential double wrapping just in case
    new_content = new_content.replace('<div dir="ltr">\n\n<div dir="ltr">', '<div dir="ltr">')
    new_content = new_content.replace('</div>\n\n</div>', '</div>')

    if content != new_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {filepath}")
    else:
        print(f"No changes: {filepath}")

for filename in files:
    fix_file(os.path.join(directory, filename))
