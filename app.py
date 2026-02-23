import streamlit as st
import subprocess
import sys
import os
import time

# Page Config
st.set_page_config(
    page_title="Churn Prediction Pipeline Runner",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS to fix cursor on sidebar
st.markdown("""
    <style>
        /* General sidebar cursor fix */
        [data-testid="stSidebar"] {
            cursor: default;
        }
        
        /* Fix text cursor on text elements */
        .stMarkdown, h1, h2, h3, p, label {
            cursor: default !important;
        }
        
        /* Fix cursor on Selectbox (make it look like a clickable dropdown) */
        div[data-baseweb="select"] > div {
            cursor: pointer !important;
        }
        
        /* Ensure input inside selectbox still allows typing if clicked (optional) */
        div[data-baseweb="select"] input {
            cursor: text !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Header
st.title("🤖 Churn Prediction Pipeline Runner")
st.markdown("---")

# Sidebar for Configuration
st.sidebar.header("Pipeline Configuration")
pipeline_type = st.sidebar.selectbox(
    "Select Pipeline to Run:",
    [
        "XGBoost / Rule-Based Pipeline",
        "LSTM (Deep Learning) Pipeline",
        "Data Engineering Only (Create Logs)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Instructions:**\n"
    "1. Select a pipeline above.\n"
    "2. Execute steps sequentially.\n"
    "3. View logs and results for each step."
)

# Initialize Session State Variables
if 'pipeline_selection' not in st.session_state or st.session_state['pipeline_selection'] != pipeline_type:
    st.session_state['pipeline_selection'] = pipeline_type
    st.session_state['current_step_index'] = 0
    st.session_state['completed_steps'] = set()
    st.session_state['step_logs'] = {} # Store logs per step index

# Reset Pipeline Button in Sidebar
if st.sidebar.button("🔄 Reset Pipeline", type="primary"):
    st.session_state['current_step_index'] = 0
    st.session_state['completed_steps'] = set()
    st.session_state['step_logs'] = {}
    st.rerun()

# Define Pipeline Steps Logic
pipelines = {
    "XGBoost / Rule-Based Pipeline": [
        {
            "name": "Step 1: Create Feature Matrix",
            "script": os.path.join("src", "data_engineering", "create_feature_matrix.py"),
            "desc": "Extracts features from event logs and creates `feature_matrix_v2.parquet`.",
            "output_file": os.path.join("data", "processed", "feature_matrix_v2.parquet")
        },
        {
            "name": "Step 2: Create Core Event Log",
            "script": os.path.join("src", "data_engineering", "create_core_eventlog_splited.py"),
            "desc": "Filters logs to core activities and creates `core_eventlog_train_v1.parquet`.",
            "output_file": os.path.join("data", "processed", "core_eventlog_train_v1.parquet")
        },
        {
            "name": "Step 3: Run XGBoost Model",
            "script": os.path.join("src", "models", "xgboost", "tp_ap_xgboost.py"),
            "desc": "Trains the XGBoost model and generates `tp_ap_xgb_joint_search_sliding.csv`.",
            "output_file": os.path.join("results","best_params", "tp_ap_xgb_joint_search_sliding.csv")
        },
        {
            "name": "Step 4: Evaluate Results",
            "script": os.path.join("src", "evaluation", "simple_eval.py"),
            "desc": "Calculates Accuracy, F1-Score and generates evaluation reports.",
            "output_file": os.path.join("results", "evaluations", "ap_tp_simple_eval.csv")
        }
    ],
    "LSTM (Deep Learning) Pipeline": [
        {
            "name": "Step 1: Create Feature Matrix",
            "script": os.path.join("src", "data_engineering", "create_feature_matrix.py"),
            "desc": "Extracts features.",
            "output_file": os.path.join("data", "processed", "feature_matrix.csv")
        },
        {
            "name": "Step 2: Create Core Event Log",
            "script": os.path.join("src", "data_engineering", "create_core_eventlog.py"),
            "desc": "Creates core event logs.",
            "output_file": os.path.join("data", "processed", "core_eventlog_train_v1.parquet")
        },
        {
            "name": "Step 3: Run LSTM Temporal Model",
            "script": os.path.join("src", "models", "lstm", "lstm_temporal_ap_tp.py"),
            "desc": "Trains LSTM model on temporal sequences.",
            "output_file": os.path.join("results", "evaluations", "lstm_temporal_results.csv")
        },
        {
            "name": "Step 4: Evaluate Multi-Year Results",
            "script": os.path.join("src", "evaluation", "lstm_eval_multi_year.py"),
            "desc": "Evaluates LSTM model performance across multiple years.",
            "output_file": os.path.join("results", "evaluations", "lstm_eval_multi_year.csv")
        }
    ],
    "Data Engineering Only (Create Logs)": [
        {
            "name": "Step 1: Create EventLog V2",
            "script": os.path.join("src", "data_engineering", "create_eventlog_v2.py"),
            "desc": "Processes raw data into `eventlog_v2.parquet`.",
            "output_file": os.path.join("data", "processed", "eventlog_v2.parquet")
        }
    ]
}

current_steps = pipelines[pipeline_type]

def run_script(script_relative_path):
    """Executes a python script and returns success status and output log."""
    script_path = os.path.abspath(script_relative_path)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    
    if not os.path.exists(script_path):
        return False, f"❌ Error: File not found: {script_path}"

    try:
        # Run process
        process = subprocess.run(
            [sys.executable, script_name],
            cwd=script_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        output_log = f"Standard Output:\n{process.stdout}\n"
        if process.stderr:
            output_log += f"\nProcess Logs (stderr):\n{process.stderr}"
        
        if process.returncode == 0:
            return True, output_log
        else:
            return False, output_log
            
    except Exception as e:
        return False, f"❌ Exception occurred: {str(e)}"

def open_file_in_explorer(file_path):
    """Opens the file in the default associated application by asking OS shell."""
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            st.warning(f"File not found: {abs_path}")
            return
        
        if sys.platform == 'win32':
            os.startfile(abs_path)
        elif sys.platform == 'darwin':
            subprocess.call(['open', abs_path])
        else:
            subprocess.call(['xdg-open', abs_path])
    except Exception as e:
        st.error(f"Could not open file: {e}")

# Display Pipeline Steps
st.subheader(f"📍 {pipeline_type}")

if len(current_steps) > 0:
    progress_val = len(st.session_state['completed_steps']) / len(current_steps)
    st.progress(progress_val)

for idx, step in enumerate(current_steps):
    # Determine Status
    is_completed = idx in st.session_state['completed_steps']
    
    # Card Styling
    with st.container():
        # Layout: Status Icon | Description | Action Button
        col1, col2, col3 = st.columns([0.05, 0.75, 0.2], gap="small")
        
        with col1:
            if is_completed:
                st.markdown("### ✅")
            else:
                st.markdown("### ⬜")
        
        with col2:
            st.markdown(f"**{step['name']}**")
            st.caption(step['desc'])
        
        with col3:
            # Logic: Always show run button if not completed. If completed, show re-run.
            if not is_completed:
                if st.button(f"▶ Run Step {idx+1}", key=f"btn_{idx}", type="primary"):
                    with st.spinner(f"Running '{step['script']}'... Please wait."):
                        success, log = run_script(step['script'])
                        
                        st.session_state['step_logs'][idx] = {"success": success, "log": log}
                        
                        if success:
                            st.session_state['completed_steps'].add(idx)
                            st.rerun()
            else:
                col_btn1, col_btn2 = st.columns([0.5, 0.5])
                with col_btn1:
                    if st.button("↺ Re-run", key=f"rerun_{idx}"):
                         with st.spinner(f"Re-running '{step['script']}'... Please wait."):
                            success, log = run_script(step['script'])
                            st.session_state['step_logs'][idx] = {"success": success, "log": log}
                            if success:
                                st.session_state['completed_steps'].add(idx)
                            st.rerun()
                with col_btn2:
                    if 'output_file' in step:
                         if st.button("📂 Open", key=f"open_{idx}", help="Open Output File"):
                             open_file_in_explorer(step['output_file'])

    # Show Logs for this step
    if idx in st.session_state['step_logs']:
        log_data = st.session_state['step_logs'][idx]
        status_text = "Success" if log_data["success"] else "Failed"
        
        # Determine if we should expand. Auto-expand if it's failed
        expanded = not log_data["success"]
        # expanded = True

        with st.expander(f"📝 Logs: {step['name']} ({status_text})", expanded=expanded):
            st.code(log_data["log"], language='bash')
            
    st.markdown("---")

# Success Message at End (Logs removed from bottom)
if len(st.session_state['completed_steps']) == len(current_steps):
    st.success("🎉 All steps in this pipeline have been completed successfully!")

