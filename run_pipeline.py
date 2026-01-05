import subprocess
import os
import sys
import time

def run_script(script_path, description):
    """Runs a python script and handles errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running: {script_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(script_path):
        print(f"ERROR: File not found: {script_path}")
        return False

    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))
    script_name = os.path.basename(script_path)

    try:
        # Run the script using the current python interpreter
        # Set cwd to script_dir so relative paths (../../data/...) work correctly
        result = subprocess.run([sys.executable, script_name], cwd=script_dir, check=True)
        print(f"\n[SUCCESS] Finished {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed running {description}")
        print(f"Error code: {e.returncode}")
        return False

def run_xgboost_pipeline():
    """Executes the XGBoost/Rule-Based pipeline steps."""
    print("\nStarting XGBoost / Rule-Based Pipeline...")
    
    # Step 1: Create Feature Matrix
    step1 = os.path.join("src", "data_engineering", "create_feature_matrix.py")
    if not run_script(step1, "1. Create Feature Matrix"): return

    # Step 2: Create Core Event Log
    step2 = os.path.join("src", "data_engineering", "create_core_eventlog.py")
    if not run_script(step2, "2. Create Core Event Log"): return
    
    # Step 3: Run XGBoost Model
    step3 = os.path.join("src", "models", "xgboost", "tp_ap_xgboost.py")
    if not run_script(step3, "3. Run XGBoost Model (Train & Predict)"): return

    # Step 4: Evaluation
    step4 = os.path.join("src", "evaluation", "simple_eval.py")
    if not run_script(step4, "4. Evaluate Results"): return

    print("\n" + "*"*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Check the 'results' folder for output files.")
    print("*"*60)

def run_lstm_pipeline():
    """Executes the LSTM pipeline steps."""
    print("\nStarting LSTM Pipeline...")
    
    # Step 1: Create Feature Matrix
    step1 = os.path.join("src", "data_engineering", "create_feature_matrix.py")
    if not run_script(step1, "1. Create Feature Matrix"): return

    # Step 2: Create Core Event Log
    step2 = os.path.join("src", "data_engineering", "create_core_eventlog.py")
    if not run_script(step2, "2. Create Core Event Log"): return

    # Step 3: Run LSTM Temporal Model
    step3 = os.path.join("src", "models", "lstm", "lstm_temporal_ap_tp.py")
    if not run_script(step3, "3. Run LSTM Temporal Model"): return

    # Step 4: Evaluate LSTM Multi-Year
    step4 = os.path.join("src", "evaluation", "lstm_eval_multi_year.py")
    if not run_script(step4, "4. Evaluate LSTM Multi-Year"): return

    print("\nPIPELINE COMPLETED SUCCESSFULLY!")

def main():
    while True:
        print("\n" + "#"*40)
        print("   CHURN PREDICTION PROJECT RUNNER")
        print("#"*40)
        print("Select a pipeline to run:")
        print("1. XGBoost / Rule-Based Pipeline (EventLog -> XGBoost -> Eval)")
        print("2. LSTM Pipeline (Core EventLog -> LSTM -> Eval)")
        print("3. Data Engineering Only (Create EventLogs)")
        print("q. Quit")
        
        choice = input("\nEnter choice (1/2/3/q): ").strip().lower()
        
        if choice == '1':
            run_xgboost_pipeline()
        elif choice == '2':
            run_lstm_pipeline()
        elif choice == '3':
            run_script(os.path.join("src", "data_engineering", "create_eventlog_v2.py"), "Create EventLog V2")
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Ensure we are in the project root
    if not os.path.exists("src"):
        print("Error: Please run this script from the project root directory.")
    else:
        main()
