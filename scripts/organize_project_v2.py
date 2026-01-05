import os
import shutil
import re

# Define the project structure
STRUCTURE = {
    "src/data_engineering": [
        "create_eventlog_v1.py", "create_eventlog_v2.py",
        "create_core_eventlog.py", "create_core_eventlog_splited.py",
        "create_feature_matrix.py", "create_seed_churn_labels.py",
        "comments_to_parquet.py", "extract_enhanced_persona_features.py",
        "users_to_parquet.py", "votes_to_parquet.py", "posthistory_to_parquet.py",
        "filtered_posts_to_parquet.py", "import_filtered_posts.py",
        "import_filtered_users.py", "import_filtered_votes.py", "import_users.py",
        "sample_parquet_to_csv.py", "show_parquet.py", "sample_snipets.py"
    ],
    "src/analysis": [
        "activity_stats.py", "count_aged_users.py",
        "count_all_votes.py", "count_churned.py",
        "check_cuda.py", "temporal_drift_analysis.py",
        "feature_importance_by_xgboost.py", "oldvsnewpersonas.py",
        "nmf_clustering_clean.py", "nmf20_new_clustering.py",
        "recreate_nmf_clustering.py", "persona_matching_by_gain_similarity.py",
        "persona_models_comparison.py", "persona_clustering_improved.py",
        "persona_clustering_model.py", "persona_classifier_statistics_based.py"
    ],
    "src/models/rule_based": [
        "enhanced_interpretable_churn_rule.py",
        "estimate_tp_ap_per_persona.py",
        "calibrate_ap_advanced.py", "calibrate_ap_combined.py",
        "calibrate_ap_empirical_bayes.py", "calibrate_ap_quantile_matching.py",
        "calibrate_ap_ratio_scaling.py", "calibrate_ap_ratio_scaling_adjusted.py",
        "calibrate_ap_temperature.py", "adjust_ap_for_rate_ratio.py",
        "tp_ap_gridsearch.py", "tp_ap_logistic.py", "tp_ap_gridsearch_scalarWA.py"
    ],
    "src/models/xgboost": [
        "ap_tp_xgboost_improved.py", "ap_tp_xgboost_improved2.py",
        "ap_tp_xgboost_v2.py", "tp_ap_xgboost.py", "tp_ap_xgboost_enhanced.py",
        "train_persona_classifier.py", "train_persona_classifier_enhanced.py",
        "train_persona_classifier_nn.py", "train_persona_classifier_nn_enhanced.py",
        "predict_new_user_personas.py", "test_new_users_pipeline.py",
        "persona_predictor_production.py"
    ],
    "src/models/lstm": [
        "ap_tp_lstm.py", "analyze_lstm_all_years.py",
        "extract_ap_tp_from_lstm.py", "ap_tp_optimize.py",
        "lstm_ap_tp_predictor.py", "lstm_churn_v2_optimized.py",
        "lstm_churn_with_persona.py", "lstm_churn_with_persona_v_2.py",
        "lstm_temporal_ap_tp.py", "train_tp_ap_pytorch.py"
    ],
    "src/evaluation": [
        "evaluate_ap_tp_model.py", "simple_eval.py", "simple_eval_v2.py",
        "simple_eval_adjusted.py", "simple_eval_optimized.py",
        "simple_eval_temporal.py", "pragmatic_real_eval.py",
        "pragmatic_real_eval_by_grace.py", "lstm_eval_multi_year.py",
        "lstm_eval_temporal.py", "lstm_eval_v2.py", "lstm_simple_eval.py"
    ],
    "src/visualization": [
        "visualize_lstm_architecture.py", "visualize_lstm_architecture_v2.py",
        "visualize_lstm_architecture_v3.py", "visualize_lstm_nn_diagram.py",
        "generate_lstm_diagrams.py"
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
    "docs/other": [
        "Enhanced_Churn_Rule_Explanation_EN.md", "Enhanced_Churn_Rule_Explanation_FA.md",
        "LSTM_ARCHITECTURE_DIAGRAM.md", "LSTM_Complete_8Year_Analysis.md",
        "lstm_feature_comparison.md", "LSTM_Final_Results_Summary.md",
        "lstm_model_explanation.md", "lstm_model_technical_documentation.md",
        "LSTM_vs_XGBoost_Comparison.md", "Next_Steps_LSTM_Approach.md",
        "NMF_CLUSTERING_FINAL_VERDICT.md", "PERSONA_CLASSIFICATION_FINAL_SUMMARY.md",
        "PERSONA_CLASSIFIER_ENHANCED_RESULTS.md", "PERSONA_CLASSIFIER_RESULTS.md",
        "PERSONA_CLUSTERING_GUIDE.md", "PERSONA_CLUSTERING_GUIDE_FA.md",
        "Temporal_Drift_Solution_EN.md", "last_claude_conv.md"
    ],
    "data/raw": [
        "ActionTypes.csv", "ActionTypes copy.csv", "ActionTypeI1.docx", "ActionTypeId.docx", "ActionTypes.xlsx"
    ],
    "data/processed": [
        "*.parquet", "*.csv" # We will handle these with glob logic
    ],
    "results/best_params": [], # Will move best_params_*.csv here
    "results/calibration": [], # Will move calibrated_ap_*.csv here
    "results/evaluations": [], # Will move ap_tp_evaluation_*.csv here
    "checkpoints": [
        "*.pt", "*.pkl", "*.joblib"
    ],
    "scripts": [
        "fix_chapter_5.py", "fix_rtl_code.py", "fix_table_mess.py",
        "add_header_breaks.py", "add_page_breaks.py", "apply_formatting_robust.py"
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
        if f.endswith(".pt") or f.endswith(".pkl") or f.endswith(".joblib"):
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
        if f.startswith("ap_tp_evaluation_") or f.startswith("eval_") or f.startswith("ap_tp_simple_eval") or f.startswith("lstm_eval_") or f.startswith("lstm_simple_eval"):
            if f.endswith(".csv"):
                shutil.move(f, os.path.join(ROOT_DIR, "results/evaluations", f))
                print(f"Moved {f} -> results/evaluations")
                continue

        # Other CSVs that are likely data
        # We need to be careful not to move source code or other things
        # Known data files:
        known_data_prefixes = ["core_activity_", "core_eventlog_", "churn_seed_", "activity_distribution_", "nmf_", "persona_", "user_"]
        if f.endswith(".csv") and any(f.startswith(p) for p in known_data_prefixes):
             # Check if it's a result file that starts with persona_ or nmf_
             if "result" in f or "comparison" in f or "importance" in f or "f1" in f or "predictions" in f:
                 # It's likely a result
                 shutil.move(f, os.path.join(ROOT_DIR, "results", f))
                 print(f"Moved {f} -> results")
             else:
                 shutil.move(f, os.path.join(ROOT_DIR, "data/processed", f))
                 print(f"Moved {f} -> data/processed")
             continue
             
        # Move remaining CSVs to results if they look like results
        if f.endswith(".csv") and ("result" in f or "summary" in f or "comparison" in f or "estimates" in f):
             shutil.move(f, os.path.join(ROOT_DIR, "results", f))
             print(f"Moved {f} -> results")
             continue

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
    # We target specific extensions: .csv, .parquet, .pt, .pkl, .joblib
    
    def replace_path(match):
        full_match = match.group(0)
        quote = full_match[0]
        filename = full_match[1:-1]
        
        # Determine where this file moved to
        new_path = filename # Default: no change
        
        if filename.endswith(".parquet"):
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
        elif filename.endswith(".csv"):
            # Check if it's in data/processed or results
            if any(filename.startswith(p) for p in ["core_activity_", "core_eventlog_", "churn_seed_", "activity_distribution_", "nmf_", "persona_", "user_"]):
                 if "result" in filename or "comparison" in filename or "importance" in filename or "f1" in filename or "predictions" in filename:
                     new_path = f"results/{filename}"
                 else:
                     new_path = f"data/processed/{filename}"
            elif "result" in filename or "summary" in filename or "comparison" in filename or "estimates" in filename:
                 new_path = f"results/{filename}"
        
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
    # r"['\"][\w\-\.\s]+\.(csv|parquet|pt|pkl|joblib)['\"]"
    pattern = re.compile(r"['\"][\w\-\.\s]+\.(csv|parquet|pt|pkl|joblib)['\"]")
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
