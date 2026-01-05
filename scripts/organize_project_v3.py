import os
import shutil
import re

# Define the project structure
STRUCTURE = {
    "src/data_engineering": [
        "extract_persona_features.py"
    ],
    "results/best_params": [
        "enhanced_wa_best_params.csv",
        "tp_ap_gridsearch_scalarWA.csv", # This looks like a result/param file
        "tp_ap_xgb_enhanced_best_params.csv",
        "tp_ap_xgb_enhanced_best_params_WA_enhanced.csv",
        "tp_ap_xgb_enhanced_best_params_WA_recency.csv",
        "tp_ap_xgb_enhanced_best_params_WA_simple.csv",
        "tp_ap_xgb_enhanced_best_params_WA_trend.csv",
        "tp_ap_xgb_joint_search.csv",
        "tp_ap_xgb_joint_search_sliding.csv"
    ],
    "data/processed": [
        "feature_matrix_sample.csv",
        "seed_churn_labels_per_user_sample.csv"
    ],
    "results": [
        "lstm_derived_ap_tp_2016.csv",
        "lstm_derived_ap_tp_2023.csv",
        "lstm_optimal_ap_tp_per_persona.csv",
        "temporal_year_multipliers.csv"
    ],
    "results/figures": [
        "Churn Analysis Diagram.png",
        "lstm_architecture_detailed.png",
        "lstm_architecture_detailed.svg",
        "lstm_architecture_visual.png",
        "lstm_architecture_visual.svg",
        "lstm_neural_network_diagram.png",
        "persona_classifier_confusion_matrix.png"
    ],
    "docs/diagrams": [
        "lstm_architecture_ascii.txt",
        "lstm_architecture_detailed.pdf",
        "lstm_architecture_interactive.html",
        "lstm_architecture_simple.txt",
        "lstm_architecture_visual.pdf",
        "lstm_feature_comparison.pdf",
        "lstm_model_explanation.pdf",
        "lstm_model_technical_documentation.pdf",
        "SO Churn Analysis Diagram.pdf",
        "Enhanced_Churn_Rule_Explanation_EN.pdf",
        "proposal.pdf"
    ],
    "scripts": [
        "organize_project.py",
        "organize_project_v2.py"
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
            src = os.path.join(ROOT_DIR, file)
            if os.path.exists(src):
                dst = os.path.join(ROOT_DIR, folder, file)
                shutil.move(src, dst)
                print(f"Moved {file} -> {folder}")

    # 2. Move Parquet Folders
    parquet_folders = [
        "comments_parquet", "filtered_posts_parquet", 
        "filtered_users_parquet", "posthistory_parquet", 
        "users_parquet"
    ]
    
    for folder in parquet_folders:
        src = os.path.join(ROOT_DIR, folder)
        if os.path.exists(src) and os.path.isdir(src):
            dst = os.path.join(ROOT_DIR, "data/processed", folder)
            # If destination exists, we might need to merge or skip. 
            # shutil.move will fail if dst exists and is a dir.
            if os.path.exists(dst):
                print(f"Skipping {folder}, destination already exists in data/processed")
            else:
                shutil.move(src, dst)
                print(f"Moved folder {folder} -> data/processed")

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
    
    # 1. Fix Imports (Add sys.path hack) - Same as before
    rel_path = os.path.relpath(filepath, ROOT_DIR)
    depth = len(rel_path.split(os.sep)) - 1
    
    sys_path_code = "import sys\nimport os\n"
    sys_path_code += f"sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '{'../' * depth}')))\n"
    
    if "sys.path.append" not in content:
        content = sys_path_code + content

    # 2. Fix File Paths
    def replace_path(match):
        full_match = match.group(0)
        quote = full_match[0]
        filename = full_match[1:-1]
        
        new_path = filename
        
        # Add new mappings
        if filename in ["feature_matrix_sample.csv", "seed_churn_labels_per_user_sample.csv"]:
            new_path = f"data/processed/{filename}"
        elif filename.endswith("_parquet"): # For folders
             new_path = f"data/processed/{filename}"
        elif filename in ["enhanced_wa_best_params.csv", "tp_ap_gridsearch_scalarWA.csv"] or filename.startswith("tp_ap_xgb_enhanced_best_params") or filename.startswith("tp_ap_xgb_joint_search"):
            new_path = f"results/best_params/{filename}"
        elif filename.startswith("lstm_derived_ap_tp_") or filename == "lstm_optimal_ap_tp_per_persona.csv" or filename == "temporal_year_multipliers.csv":
            new_path = f"results/{filename}"
            
        # Keep existing mappings from v2 (simplified)
        elif filename.endswith(".parquet"):
            new_path = f"data/processed/{filename}"
        elif filename.endswith(".pt") or filename.endswith(".pkl") or filename.endswith(".joblib"):
            new_path = f"checkpoints/{filename}"
        elif filename.startswith("best_params_"):
            new_path = f"results/best_params/{filename}"
        elif filename.startswith("calibrated_ap_"):
            new_path = f"results/calibration/{filename}"
        elif filename.startswith("ap_tp_evaluation_") or filename.startswith("eval_") or filename.startswith("lstm_eval_") or filename.startswith("lstm_simple_eval"):
            new_path = f"results/evaluations/{filename}"
        elif filename in ["ActionTypes.csv", "ActionTypes copy.csv"]:
            new_path = f"data/raw/{filename}"
        
        if "/" in filename or "\\" in filename:
            return full_match 
            
        if new_path != filename:
            to_root = "../" * depth
            final_path = f"{to_root}{new_path}"
            return f"{quote}{final_path}{quote}"
            
        return full_match

    # Regex for string literals
    # We need to match parquet folders too which don't have extensions usually in the code string, 
    # but here they are "name_parquet".
    pattern = re.compile(r"['\"][\w\-\.\s]+(_parquet|\.csv|\.parquet|\.pt|\.pkl|\.joblib)['\"]")
    content = pattern.sub(replace_path, content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated paths in {filepath}")

if __name__ == "__main__":
    create_structure()
    move_files()
    update_imports_and_paths()
