import os
import tempfile
import datetime
import re
import json
import concurrent.futures
from typing import Tuple, Dict, Any

import streamlit as st
# Import the new, updated review system
from review_system_3 import process_transcript_enhanced

# Load environment variables from Streamlit secrets
try:
    if 'openai' in st.secrets and 'api_key' in st.secrets.openai:
        os.environ["OPENAI_API_KEY"] = st.secrets.openai.api_key
    
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found in Streamlit secrets. Please configure it in the Streamlit Cloud settings.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading configuration: {str(e)}")
    st.stop()

# Streamlit page configuration
st.set_page_config(
    page_title="Mentor Review System",
    page_icon="🎓",
    layout="wide"
)

def save_uploaded_file(uploaded_file) -> str | None:
    """Save uploaded file to a temporary JSON and return its path."""
    try:
        # Use utf-8 encoding when writing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w', encoding='utf-8') as tmp:
            tmp.write(uploaded_file.getvalue().decode('utf-8'))
            return tmp.name
    except Exception as e:
        st.error(f"Error saving file {uploaded_file.name}: {e}")
        return None

def safe_get(data: Dict[str, Any], *keys, default=None):
    """Safely get nested dictionary values."""
    temp = data
    for key in keys:
        if isinstance(temp, dict):
            temp = temp.get(key)
        else:
            return default
    return temp if temp is not None else default

def extract_email_content(result: dict) -> Tuple[str, str, str, dict]:
    """Extract email content, summary, and checklist from result."""
    email_content = safe_get(result, "feedback_email", default="")
    overall_summary = safe_get(result, "overall_session_summary", default="No summary available.")
    checklist_data = safe_get(result, "session_checklist", default={})
    checklist_formatted = format_checklist_display(checklist_data)
    
    return email_content, overall_summary, checklist_formatted, checklist_data

def format_checklist_display(checklist_data: dict) -> str:
    """Format checklist data into a readable string."""
    if not checklist_data or "checklist" not in checklist_data:
        return "No checklist data available."
    
    output = ""
    for item in checklist_data.get("checklist", []):
        question = item.get("question", "")
        answer = item.get("answer", "UNCLEAR")
        explanation = item.get("explanation", "")
        
        if answer.upper() == "YES":
            emoji = "✅"
        elif answer.upper() == "NO":
            emoji = "❌"
        else:
            emoji = "❓"
        
        output += f"{emoji} **{question}**\n"
        output += f"**Answer:** {answer}\n"
        output += f"*{explanation}*\n\n"
    
    return output

def extract_overall_score(result: dict) -> str:
    """Extract overall score from the new result structure."""
    score = safe_get(result, "overall_score")
    if isinstance(score, (int, float)):
        return str(round(score, 1))
    return "N/A"

def extract_detailed_scores(result: dict) -> dict:
    """Extract detailed guideline scores from the new result structure."""
    scores = safe_get(result, "scores", default={})
    
    detailed_scores = {
        "Professionalism": safe_get(scores, "professionalism", default="N/A"),
        "Session Flow": safe_get(scores, "session_flow", default="N/A"),
        "Guideline Compliance": safe_get(scores, "guideline_compliance", default="N/A"),
    }
    
    return detailed_scores

def process_single_file(
    config_dict: dict, 
    guidelines_path: str, 
    file_name: str
) -> Tuple[bool, str, dict]:
    """Process a single transcript file and return results."""
    try:
        transcript_path = config_dict["path"]
        session_date = config_dict["date"]
        session_time = config_dict["time"]

        result = process_transcript_enhanced(
            transcript_path=transcript_path,
            guidelines_path=guidelines_path,
            session_date=session_date,
            session_time=session_time
        )
        
        if not result.get("success", False):
            return False, file_name, {"error": result.get("error", "Unknown error")}
        
        email_content, overall_summary, checklist_formatted, checklist_data = extract_email_content(result)
        overall_score = extract_overall_score(result)
        detailed_scores = extract_detailed_scores(result)
        
        return True, file_name, {
            "email_content": email_content,
            "overall_summary": overall_summary,
            "checklist_formatted": checklist_formatted,
            "checklist_data": checklist_data,
            "overall_score": overall_score,
            "detailed_scores": detailed_scores,
            "mentor_name": result.get("mentor_name", "Mentor"),
            "result": result  # Pass the full result for JSON download
        }
    except Exception as e:
        return False, file_name, {"error": str(e)}

        return False, file_name, {"error": str(e)}


def main():
    # Add custom CSS for better UI
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            border-radius: 8px;
        }
        .stProgress > div > div > div > div {
            background-color: #1E88E5;
        }
        .stAlert {
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'ready_to_process' not in st.session_state:
        st.session_state.ready_to_process = False
    if 'file_configs' not in st.session_state:
        st.session_state.file_configs = {}

    # Sidebar with app info
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("""
        **Mentor Review System ** analyzes mentorship sessions 
        based on guideline adherence using a multi-step AI process.
        
        ### Features:
        - Evidence-based scoring from a powerful Assessor AI.
        - AI-drafted, context-aware feedback emails.
        - 📊 Guideline-based scoring (Professionalism, Session Flow, Compliance).
        - ✅ Evidence-based session checklist.
        
        ### How to use:
        1.  Upload one or more transcript JSON files.
        2.  Set the individual **Session Date** and **Session Time** for each file.
        3.  Click **"Process Files"** at the bottom of the list.
        4.  View and download results, which will appear below.
        """)
        
        if st.button("🔄 Start New Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Main content
    st.title("🎓 Mentor Review System")
    st.markdown("""
    Upload mentorship session transcripts to generate comprehensive, AI-powered reviews
    based on **Analytics Vidhya mentorship guidelines**.
    """)

    # Ensure guidelines PDF exists
    GUIDELINES_PATH = "Guidelines.pdf"
    if not os.path.exists(GUIDELINES_PATH):
        st.warning("Guidelines.pdf not found. Creating a placeholder. Please upload the correct file for accurate analysis.")
        with open(GUIDELINES_PATH, "w") as f:
            f.write("Standard mentorship guidelines: Be professional. Be helpful. Structure the session. Encourage participation.")

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose transcript JSON files",
        type=["json"],
        help="Upload one or more transcript files in JSON format",
        accept_multiple_files=True
    )

    # Step 1: Upload and configure
    if uploaded_files and not st.session_state.get("ready_to_process"):
        
        st.info("Please configure the session date and time for each uploaded file.")
        
        # Use a form to hold all inputs
        with st.form("file_config_form"):
            
            now = datetime.datetime.now()
            
            for uploaded_file in uploaded_files:
                st.markdown(f"--- \n#### 📄 {uploaded_file.name}")
                col1, col2 = st.columns(2)
                with col1:
                    st.date_input(
                        "Session Date", 
                        value=now.date(), 
                        key=f"date_{uploaded_file.name}"
                    )
                with col2:
                    st.time_input(
                        "Session Time", 
                        value=now.time(), 
                        key=f"time_{uploaded_file.name}"
                    )

            submitted = st.form_submit_button(
                "🚀 Process Files", 
                use_container_width=True, 
                type="primary"
            )
            
            if submitted:
                if not uploaded_files:
                    st.warning("Please upload at least one file.")
                    st.stop()
                    
                file_configs = {}
                for file in uploaded_files:
                    tmp_path = save_uploaded_file(file)
                    if tmp_path:
                        date_val = st.session_state[f"date_{file.name}"]
                        time_val = st.session_state[f"time_{file.name}"]
                        file_configs[file.name] = {
                            "path": tmp_path,
                            "date": str(date_val),
                            "time": str(time_val.strftime("%H:%M:%S"))
                        }
                
                if not file_configs:
                    st.error("Failed to save one or more files. Please try again.")
                    st.stop()
                    
                st.session_state["file_configs"] = file_configs
                st.session_state["ready_to_process"] = True
                st.rerun()

    # Step 2: Process and display
    if st.session_state.get("ready_to_process"):
        file_configs = st.session_state.get("file_configs", {})
        
        if not file_configs:
            st.error("No valid files to process.")
            st.session_state["ready_to_process"] = False
            st.stop()
            
        results = {}
        status_text = st.empty()
        status_text.info("Starting processing... This may take a few minutes.")
        
        # This will now show chunk processing progress from the review system
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(process_single_file, config, GUIDELINES_PATH, name): name
                for name, config in file_configs.items()
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                name = future_to_file[future]
                try:
                    success, file_name, result = future.result()
                    results[file_name] = result
                    if success:
                        status_text.text(f"✅ Completed: {file_name}")
                    else:
                        status_text.text(f"❌ Error processing {file_name}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    status_text.text(f"❌ Critical Error processing {name}: {str(e)}")
                    results[name] = {"error": str(e)}
        
        # Clean up temporary files
        for config in file_configs.values():
            try:
                os.remove(config["path"])
            except OSError:
                pass
        
        # Display summary
        successful = len([r for r in results.values() if 'error' not in r])
        status_text.success(f"✅ Successfully processed {successful}/{len(results)} files!")
        
        # Show each result in an expander
        for file_name, result in results.items():
            with st.expander(f"📄 {file_name}", expanded=True):
                if "error" in result:
                    st.error(f"Error processing {file_name}: {result['error']}")
                    continue
                    
                email_content = result.get("email_content", "")
                overall_summary = result.get("overall_summary", "")
                checklist_formatted = result.get("checklist_formatted", "")
                checklist_data = result.get("checklist_data", {})
                overall_score = result.get("overall_score", "N/A")
                detailed_scores = result.get("detailed_scores", {})
                full_result = result.get("result", {})
                mentor_name = result.get("mentor_name", "Mentor")
                
                session_date = full_result.get("session_date", "N/A")
                session_time = full_result.get("session_time", "N/A")

                # Header
                st.markdown(f"### 📝 {file_name}")
                st.caption(f"**Session Time:** {session_date} at {session_time}")

                
                # Display scores
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Score", f"{overall_score}/100" if overall_score != "N/A" else "N/A")
                with col2:
                    prof_score = detailed_scores.get("Professionalism", "N/A")
                    st.metric("Professionalism", f"{prof_score}/10" if prof_score != "N/A" else "N/A")
                with col3:
                    flow_score = detailed_scores.get("Session Flow", "N/A")
                    st.metric("Session Flow", f"{flow_score}/10" if flow_score != "N/A" else "N/A")
                with col4:
                    comp_score = detailed_scores.get("Guideline Compliance", "N/A")
                    st.metric("Compliance", f"{comp_score}/10" if comp_score != "N/A" else "N/A")
                
                st.markdown("---")
                
                # Tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📧 Feedback Email", 
                    "📊 AI Assessment", 
                    "✅ Session Checklist",
                    "📥 Downloads"
                ])
                
                with tab1:
                    st.markdown("#### AI-Generated Feedback Email")
                    st.text_area(
                        label=f"Feedback for {file_name}",
                        value=email_content,
                        height=400,
                        key=f"email_{file_name}",
                        disabled=False, # Make it editable
                        label_visibility="collapsed"
                    )
                
                with tab2:
                    st.markdown("#### AI-Powered Assessment Details")
                    
                    st.markdown("##### Overall Summary")
                    st.markdown(overall_summary)
                    st.markdown("---")

                    assessment_data = safe_get(full_result, "final_assessment", default={})
                    
                    st.markdown("##### Key Strengths (Identified by AI)")
                    key_strengths = safe_get(assessment_data, "key_strengths", default=[])
                    if key_strengths:
                        for s in key_strengths:
                            st.markdown(f"- {s}")
                    else:
                        st.info("No specific key strengths identified.")

                    st.markdown("##### Key Improvements (Identified by AI)")
                    key_improvements = safe_get(assessment_data, "key_improvements", default=[])
                    if key_improvements:
                        for imp in key_improvements:
                            st.markdown(f"**{imp.get('title', 'Improvement Area')}:**")
                            st.markdown(f"> {imp.get('suggestion', 'No specific suggestion.')}")
                    else:
                        st.info("No specific improvement areas identified.")

                with tab3:
                    st.markdown("#### Evidence-Based Session Checklist")
                    if checklist_formatted:
                        st.markdown(checklist_formatted)
                    else:
                        st.info("No checklist data available for this transcript.")
                
                with tab4:
                    st.markdown("#### Download Options")
                    
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    with col_dl1:
                        st.download_button(
                            label="📥 Download Feedback",
                            data=email_content,
                            file_name=f"{os.path.splitext(file_name)[0]}_feedback.txt",
                            mime="text/plain",
                            key=f"dl_feedback_{file_name}",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        summary_report = f"Overall Summary:\n{overall_summary}\n\nChecklist:\n{checklist_formatted}"
                        st.download_button(
                            label="📥 Download Summary",
                            data=summary_report,
                            file_name=f"{os.path.splitext(file_name)[0]}_summary.txt",
                            mime="text/plain",
                            key=f"dl_summary_{file_name}",
                            use_container_width=True
                        )
                    
                    with col_dl3:
                        st.download_button(
                            label="📥 Download Checklist",
                            data=checklist_formatted,
                            file_name=f"{os.path.splitext(file_name)[0]}_checklist.txt",
                            mime="text/plain",
                            key=f"dl_checklist_{file_name}",
                            use_container_width=True
                        )
                    
                    st.download_button(
                        label="📥 Download Complete Report (JSON)",
                        data=json.dumps(full_result, indent=2, ensure_ascii=False),
                        file_name=f"{os.path.splitext(file_name)[0]}_complete_report.json",
                        mime="application/json",
                        key=f"dl_json_{file_name}",
                        use_container_width=True
                    )
                
                st.markdown("---")
        
        # Reset state
        st.session_state["ready_to_process"] = False
        st.session_state["file_configs"] = {}
        
        st.session_state["file_configs"] = {}
        

if __name__ == "__main__":
    main()
