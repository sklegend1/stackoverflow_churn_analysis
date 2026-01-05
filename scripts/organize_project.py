import os
import shutil
import re

# Define the project structure
STRUCTURE = {
    "src/data_engineering": [
        "create_eventlog_v1.py", "create_eventlog_v2.py",
        "create_core_eventlog.py", "create_core_eventlog_splited.py",
        "create_feature_matrix.py", "create_seed_churn_labels.py",
        "comments_to_parquet.py", "extract_enhanced_persona_features.py"
    ],
    "src/analysis": [
        "activity_stats.py", "count_aged_users.py",
        "count_all_votes.py", "count_churned.py",
        "check_cuda.py"
    ],
    "src/models/rule_based": [
        "enhanced_interpretable_churn_rule.py",
        "estimate_tp_ap_per_persona.py",
        "calibrate_ap_advanced.py", "calibrate_ap_combined.py",
        "calibrate_ap_empirical_bayes.py", "calibrate_ap_quantile_matching.py",
        "calibrate_ap_ratio_scaling.py", "calibrate_ap_ratio_scaling_adjusted.py",
        "calibrate_ap_temperature.py", "adjust_ap_for_rate_ratio.py"
    ],
    "src/models/xgboost": [
        "ap_tp_xgboost_improved.py", "ap_tp_xgboost_improved2.py",
        "ap_tp_xgboost_v2.py"
    ],
    "src/models/lstm": [
        "ap_tp_lstm.py", "analyze_lstm_all_years.py",
        "extract_ap_tp_from_lstm.py", "ap_tp_optimize.py"
    ],
    "src/evaluation": [
        "evaluate_ap_tp_model.py"
    ],
    "docs/detailed_reports": [
        "فصل_۱_مقدمه_و_تعریف_مسئله.md",
        "فصل_۲_مهندسی_داده_و_Event_Log.md",
        "فصل_۳_سیستم_تفسیری_Rule_based.md",
        "فصل_۴_Temporal_Drift_و_WA_پیشرفته.md",
        "فصل_۵_مدل_LSTM_v2.md",
        "فصل_۶_جمع_بندی_و_نتیجه_گیری.md",
        "فصل_۶_جمع_بندی_جامع.md",
        "فصل_۶_نتیجه‌گیری_و_مقایسه.md",
        "فصل_۶_نتیجه‌گیری_و_مقایسه_v2.md"
    ],
    "data/raw": [
        "ActionTypes.csv", "ActionTypes copy.csv"
    ],
    "data/processed": [
        "*.parquet", "*.csv" # We will handle these with glob logic
    ],
    "results/best_params": [], # Will move best_params_*.csv here
    "results/calibration": [], # Will move calibrated_ap_*.csv here
    "results/evaluations": [], # Will move ap_tp_evaluation_*.csv here
    "checkpoints": [
        "*.pt"
    ]
}

ROOT_DIR = os.getcwd()

def create_structure():
    for folder in STRUCTURE.keys():
        os.makedirs(os.path.join(ROOT_DIR, folder), exist_ok=True)
    print("Created directory structure.")

def move_files():
    # 1. Move explicitly listed files
    for folder, files in STRUCTURE.items():
        for file in files:
            if '*' in file: continue # Skip wildcards for now
            
            src = os.path.join(ROOT_DIR, file)
            # Check if file exists in root (or docs_detailed for docs)
            if not os.path.exists(src):
                # Special case for docs which are currently in docs_detailed
                if "docs/detailed_reports" in folder:
                    src = os.path.join(ROOT_DIR, "docs_detailed", file)
                
            if os.path.exists(src):
                dst = os.path.join(ROOT_DIR, folder, file)
                shutil.move(src, dst)
                print(f"Moved {file} -> {folder}")

    # 2. Move Wildcard/Pattern files
    all_files = os.listdir(ROOT_DIR)
    
    for f in all_files:
        if os.path.isdir(f): continue
        
        # Data Processed (Parquet & CSVs that are data, not results)
        # Logic: If it's parquet, it goes to data/processed
        if f.endswith(".parquet"):
            shutil.move(f, os.path.join(ROOT_DIR, "data/processed", f))
            print(f"Moved {f} -> data/processed")
            continue

        # Checkpoints
        if f.endswith(".pt"):
            shutil.move(f, os.path.join(ROOT_DIR, "checkpoints", f))
            print(f"Moved {f} -> checkpoints")
            continue

        # Results - Best Params
        if f.startswith("best_params_") and f.endswith(".csv"):
            shutil.move(f, os.path.join(ROOT_DIR, "results/best_params", f))
            print(f"Moved {f} -> results/best_params")
            continue
            
        # Results - Calibration
        if f.startswith("calibrated_ap_") and f.endswith(".csv"):
            shutil.move(f, os.path.join(ROOT_DIR, "results/calibration", f))
            print(f"Moved {f} -> results/calibration")
            continue

        # Results - Evaluations
        if f.startswith("ap_tp_evaluation_") or f.startswith("eval_") or f.startswith("ap_tp_simple_eval"):
            if f.endswith(".csv"):
                shutil.move(f, os.path.join(ROOT_DIR, "results/evaluations", f))
                print(f"Moved {f} -> results/evaluations")
                continue

        # Other CSVs that are likely data
        # We need to be careful not to move source code or other things
        # Known data files:
        known_data_prefixes = ["core_activity_", "core_eventlog_", "churn_seed_", "activity_distribution_"]
        if f.endswith(".csv") and any(f.startswith(p) for p in known_data_prefixes):
             shutil.move(f, os.path.join(ROOT_DIR, "data/processed", f))
             print(f"Moved {f} -> data/processed")

def update_imports_and_paths():
    # We need to walk through all python files in src/
    src_dir = os.path.join(ROOT_DIR, "src")
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                update_single_file(filepath)

def update_single_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Fix Imports (Add sys.path hack)
    # We calculate how many levels deep we are relative to root
    rel_path = os.path.relpath(filepath, ROOT_DIR)
    depth = len(rel_path.split(os.sep)) - 1
    
    sys_path_code = "import sys\nimport os\n"
    sys_path_code += f"sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '{'../' * depth}')))\n"
    
    # Insert after existing imports or at top
    # Simple heuristic: Insert at top
    if "sys.path.append" not in content:
        content = sys_path_code + content

    # 2. Fix File Paths
    # We need to replace "file.csv" with "data/processed/file.csv" etc.
    # This is the tricky part. We use regex to find string literals that look like filenames.
    
    # Pattern: "filename.ext" or 'filename.ext'
    # We target specific extensions: .csv, .parquet, .pt
    
    def replace_path(match):
        full_match = match.group(0)
        quote = full_match[0]
        filename = full_match[1:-1]
        
        # Determine where this file moved to
        new_path = filename # Default: no change
        
        if filename.endswith(".parquet"):
            new_path = f"data/processed/{filename}"
        elif filename.endswith(".pt"):
            new_path = f"checkpoints/{filename}"
        elif filename.startswith("best_params_"):
            new_path = f"results/best_params/{filename}"
        elif filename.startswith("calibrated_ap_"):
            new_path = f"results/calibration/{filename}"
        elif filename.startswith("ap_tp_evaluation_") or filename.startswith("eval_"):
            new_path = f"results/evaluations/{filename}"
        elif filename in ["ActionTypes.csv", "ActionTypes copy.csv"]:
            new_path = f"data/raw/{filename}"
        elif filename.endswith(".csv"):
            # Check if it's in data/processed or results
            # Heuristic: if it looks like a result, it's result. Else data.
            # But we already handled specific result prefixes.
            # Let's assume other CSVs are data/processed unless they are output files being written.
            # This is risky. Let's stick to what we moved.
            if any(filename.startswith(p) for p in ["core_activity_", "core_eventlog_", "churn_seed_", "activity_distribution_"]):
                new_path = f"data/processed/{filename}"
        
        # If we are writing to a file, we might need to make sure the directory exists?
        # The user script won't do that automatically.
        # But for now, let's just update the path string.
        
        # IMPORTANT: If the path is already absolute or has folders, don't touch it?
        # Our regex is simple, it matches "string".
        if "/" in filename or "\\" in filename:
            return full_match # Already has path, assume correct or manual
            
        if new_path != filename:
            # We need to adjust for the fact that the script is running from src/xyz/
            # So "data/processed/file" should be "../../data/processed/file"
            
            # Calculate relative path from script dir to project root
            to_root = "../" * depth
            final_path = f"{to_root}{new_path}"
            return f"{quote}{final_path}{quote}"
            
        return full_match

    # Regex for string literals ending in our extensions
    # r"['\"][\w\-\.\s]+\.(csv|parquet|pt)['\"]"
    pattern = re.compile(r"['\"][\w\-\.\s]+\.(csv|parquet|pt)['\"]")
    content = pattern.sub(replace_path, content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated paths in {filepath}")

if __name__ == "__main__":
    create_structure()
    move_files()
    update_imports_and_paths()
    
    # Cleanup empty folders
    if os.path.exists("docs_detailed"):
        try:
            os.rmdir("docs_detailed") # Only if empty
        except:
            pass
