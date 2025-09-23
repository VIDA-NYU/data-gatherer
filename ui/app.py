import streamlit as st
from data_gatherer.data_gatherer import DataGatherer
from dotenv import load_dotenv
import pandas as pd
import altair as alt
import os
import io
import xlsxwriter
import uuid
import threading
import time
import json
from collections import defaultdict
from datetime import datetime
import re

# Configure for better multi-user support
st.set_page_config(
    page_title="Data Gatherer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

linux = os.path.exists('/.dockerenv')

# Global user management for concurrent requests only
if 'active_requests' not in st.session_state:
    st.session_state.active_requests = defaultdict(int)

user_locks = defaultdict(threading.Lock)
MAX_CONCURRENT_REQUESTS = 2
MAX_PMCIDS_PER_REQUEST = 10

def get_user_id():
    """Get or create a unique user ID for this session"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id

def check_rate_limit(user_id):
    """Check if user is within concurrent request limits"""
    with user_locks[user_id]:
        current_requests = st.session_state.active_requests[user_id]
        if current_requests >= MAX_CONCURRENT_REQUESTS:
            return False, f"Too many active requests ({current_requests}). Please wait."
        return True, "OK"

def start_request(user_id):
    """Mark the start of a request for a user"""
    with user_locks[user_id]:
        st.session_state.active_requests[user_id] += 1

def end_request(user_id):
    """Mark the end of a request for a user"""
    with user_locks[user_id]:
        st.session_state.active_requests[user_id] = max(0, st.session_state.active_requests[user_id] - 1)

def app_process_url(orch, url, **kwargs):
    """
    Wrapper around DataGatherer.process_url for concurrent user support
    """
    user_id = get_user_id()
    
    # Add small delay to prevent overwhelming the system
    time.sleep(0.5)
    
    try:
        # Call the original process_url method
        return orch.process_url(url, **kwargs)
    except Exception as e:
        # Log error with user context
        orch.logger.error(f"Error in app_process_url for user {user_id[:8]}: {e}")
        raise

def sanitize_sheet_name(name):
    name = re.sub(r'[:\\/*?\[\]]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name[:31]

def save_user_feedback(feedback_text):
    """Save user feedback to a file"""
    feedback_dir = "feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": get_user_id()[:12],  # Only store partial ID for privacy
        "feedback": feedback_text
    }
    
    feedback_file = os.path.join(feedback_dir, "user_feedback.jsonl")
    with open(feedback_file, "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")

# Custom CSS
st.markdown("""
    <style>
    [data-testid="stExpander"] p {
        font-size: 20px !important;
    }
    
    /* Better mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    .feedback-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- USER SESSION MANAGEMENT ---
user_id = get_user_id()

# Display title and status
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📄 Data Gatherer – Dataset Reference Extractor")
with col2:
    current_requests = st.session_state.active_requests[user_id]
    if current_requests > 0:
        st.info(f"🔄 Processing ({current_requests} active)")
    else:
        st.success("✅ Ready")

# --- PARAMETER SELECTION UI ---
st.sidebar.header("⚙️ Extraction Settings")

MODEL_OPTIONS = [
    "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-2.5-flash"
]
PROMPT_MODEL_OPTIONS = {
    'FDR': 'GPT_from_full_input_Examples',
    'RTR': 'retrieve_datasets_simple_JSON',
}

metadata_prompt_name = 'portkey_gemini_metadata_extract'

model_name = st.sidebar.selectbox("Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index('gemini-2.0-flash'))
use_portkey = True

extraction_mode = st.sidebar.radio(
    "Extraction Mode",
    options=["Full Document Read", "Retrieve Then Read"],
    index=0
)

full_document_read = extraction_mode == "Full Document Read"
prompt_name = PROMPT_MODEL_OPTIONS['FDR'] if full_document_read else PROMPT_MODEL_OPTIONS['RTR']

# Load environment variables from .env file
load_dotenv()

st.markdown("Enter one or more **PMCIDs** to extract dataset references from open-access publications.")
st.info(f"Maximum {MAX_PMCIDS_PER_REQUEST} PMCIDs per request. For larger batches, please make multiple requests.")

pmc_input = st.text_area("🔢 Enter PMCIDs (comma or newline separated)", height=100)

# Initialize extraction completion flag
if 'extraction_completed' not in st.session_state:
    st.session_state.extraction_completed = False

# Main processing logic
if st.button("🚀 Run Extraction", type="primary"):
    if not pmc_input.strip():
        st.warning("Please enter at least one PMCID.")
    else:
        orch = None
        # Rate limiting check
        allowed, message = check_rate_limit(user_id)
        if not allowed:
            st.error(message)
            # Log rate limit exceeded (no orch yet)
            print(f"[LOG] Extraction denied for user {user_id[:8]}: {message}")
            st.stop()
        pmcids = [pmcid.strip() for pmcid in pmc_input.replace(",", "\n").splitlines() if pmcid.strip()]
        if len(pmcids) > MAX_PMCIDS_PER_REQUEST:
            st.warning(f"Limited to {MAX_PMCIDS_PER_REQUEST} PMCIDs. Processing first {MAX_PMCIDS_PER_REQUEST}.")
            pmcids = pmcids[:MAX_PMCIDS_PER_REQUEST]
        start_request(user_id)
        try:
            with st.spinner("Extraction in progress..."):
                log_placeholder = st.empty()
                driver_path = '/usr/local/bin/geckodriver' if linux else None
                orch = DataGatherer(
                    llm_name=model_name, 
                    process_entire_document=full_document_read, 
                    log_level="INFO",
                    load_from_cache=True, 
                    save_to_cache=True, 
                    driver_path=driver_path,
                    log_file_override="ui/data_gatherer.log",
                    clear_previous_logs=False
                )
                orch.logger.info(f"[FLOW] Extraction button pressed by user {user_id[:8]}")
                orch.logger.info(f"[FLOW] Checking rate limit for user {user_id[:8]}")
                orch.logger.info(f"[FLOW] PMCIDs to process: {pmcids}")
                orch.logger.info(f"[FLOW] Extraction started for user {user_id[:8]}")
                excel_buffer = io.BytesIO()
                excel_tabs = {}
                all_supp_rows = []
                all_avail_rows = []
                st.markdown("<h2 style='margin-top: 1.5em;'>Results</h2>", unsafe_allow_html=True)
                for idx, article_id in enumerate(pmcids):
                    log_placeholder.info(f"Processing article {idx+1} of {len(pmcids)}: {article_id}")
                    orch.logger.info(f"[FLOW] Processing article {idx+1}/{len(pmcids)}: {article_id}")
                    try:
                        pmcid = orch.data_fetcher.url_to_pmcid(article_id)
                        doi = orch.data_fetcher.url_to_doi(article_id, pmcid)
                        url = orch.preprocess_url(article_id)
                        orch.logger.info(f"[FLOW] Preprocessed URL: {url}")
                        result = app_process_url(
                            orch, url, 
                            driver_path=driver_path, 
                            use_portkey_for_gemini=use_portkey, 
                            prompt_name=prompt_name,
                            semantic_retrieval=True
                        )
                        orch.logger.info(f"[FLOW] Result received for {article_id}: {None if result is None else 'DataFrame'}")
                        if result is None or result.empty:
                            st.warning(f"No results found for {article_id}")
                            orch.logger.info(f"[FLOW] No results found for {article_id}")
                            continue
                        title = result["pub_title"]
                        if isinstance(title, pd.Series):
                            title = title.iloc[0]
                        elif isinstance(title, list):
                            title = title[0]
                        files_with_extension = result[result["file_extension"].notna()] if "file_extension" in result else pd.DataFrame(columns=["download_link", "description", "file_extension"])
                        files_with_repo = result[result["data_repository"].notna()] if "data_repository" in result else pd.DataFrame(columns=["data_repository", "dataset_identifier", "dataset_webpage"])
                        supp_df = files_with_extension[["download_link", "description"]].copy() if not files_with_extension.empty else pd.DataFrame(columns=["download_link", "description"])
                        supp_df["Source PMCID"] = pmcid
                        supp_df["Source DOI"] = doi
                        supp_df["Source Title"] = title
                        if not supp_df.empty:
                            supp_df = supp_df.drop_duplicates()
                            all_supp_rows.append(supp_df)
                        avail_cols = ["data_repository", "dataset_identifier", "dataset_webpage"]
                        avail_df = files_with_repo[avail_cols].copy() if not files_with_repo.empty else pd.DataFrame(columns=avail_cols)
                        avail_df["Source PMCID"] = pmcid
                        avail_df["Source DOI"] = doi
                        avail_df["Source Title"] = title
                        if not avail_df.empty:
                            avail_df = avail_df.drop_duplicates()
                            all_avail_rows.append(avail_df)
                        with st.expander(f"{title}", expanded=False):
                            st.markdown(f"<h2 style='text-align: left; font-size: 2em; font-weight: bold; margin-bottom: 0.5em;'>{title}</h2>", unsafe_allow_html=True)
                            summary = orch.summarize_result(result)
                            supp_df = files_with_extension[["download_link", "description"]]
                            if not supp_df.empty:
                                supp_df = supp_df.drop_duplicates()
                                files_with_extension = files_with_extension.drop_duplicates(
                                    subset=["download_link", "description"],
                                    keep='first'
                                )
                            avail_df = files_with_repo[["data_repository", "dataset_identifier", "dataset_webpage"]]
                            file_exts = pd.DataFrame.from_dict(
                                files_with_extension["file_extension"].value_counts().to_dict(),
                                orient="index", columns=["Count"]
                            ).reset_index().rename(columns={"index": "File Type"})
                            repos = pd.DataFrame.from_dict(summary["frequency_of_data_repository"], orient="index",
                                                           columns=["Count"]).reset_index().rename(columns={"index": "Repository"})
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(
                                    "<h3 style='text-align: center; font-size: 1.3em;'>Supplementary File Types</h3>",
                                    unsafe_allow_html=True
                                )
                                bar_chart1 = (
                                    alt.Chart(file_exts)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X("File Type:N", sort="-y", axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=15)),
                                        y=alt.Y("Count:Q"),
                                        tooltip=["File Type", "Count"]
                                    )
                                    .properties(width=300, height=200)
                                    .configure_axis(labelFontSize=13, titleFontSize=15)
                                )
                                st.markdown(
                                    "<div style='display: flex; justify-content: center; align-items: center;'>",
                                    unsafe_allow_html=True
                                )
                                st.altair_chart(bar_chart1, use_container_width=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                            with col2:
                                st.markdown(
                                    "<h3 style='text-align: center; font-size: 1.3em;'>Available Data Repositories</h3>",
                                    unsafe_allow_html=True
                                )
                                bar_chart2 = (
                                    alt.Chart(repos)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X("Repository:N", sort="-y", axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=15)),
                                        y=alt.Y("Count:Q"),
                                        tooltip=["Repository", "Count"]
                                    )
                                    .properties(width=300, height=200)
                                    .configure_axis(labelFontSize=13, titleFontSize=15)
                                )
                                st.markdown(
                                    "<div style='display: flex; justify-content: center; align-items: center;'>",
                                    unsafe_allow_html=True
                                )
                                st.altair_chart(bar_chart2, use_container_width=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                            st.markdown("### Supplementary Material")
                            st.markdown("#### Supplementary File Types")
                            st.dataframe(file_exts, width='content')
                            st.markdown("#### Supplementary Files")
                            st.dataframe(supp_df, width='content')
                            st.markdown("### Available datasets")
                            if files_with_repo.empty:
                                st.info("No datasets found.")
                            else:
                                st.dataframe(avail_df, width='content')
                                for j, data_item in files_with_repo.iterrows():
                                    dataset_label = f"**{data_item['dataset_identifier']}** ({data_item['dataset_webpage']})"
                                    st.markdown(f"- {dataset_label}")
                                    dataset_webpage = data_item["dataset_webpage"] if "dataset_webpage" in data_item and pd.notna(data_item["dataset_webpage"]) else ""
                                    with st.spinner(
                                        f"Fetching metadata from repo: {data_item['data_repository']}... {dataset_webpage}"):
                                        try:
                                            preview_result = orch.process_metadata(
                                                data_item, interactive=False, return_metadata=True,
                                                write_raw_metadata=False,
                                                use_portkey_for_gemini=use_portkey,
                                                prompt_name=metadata_prompt_name,
                                                timeout=15
                                            )
                                            if not preview_result or not isinstance(preview_result, list) or len(preview_result) == 0:
                                                st.warning("No data preview available.")
                                                orch.logger.warning(f"No data preview available for {data_item['dataset_identifier']}, dataset_webpage: {dataset_webpage}")
                                                continue
                                            item = preview_result[0]
                                        except Exception as e:
                                            st.warning(f"No data preview available.")
                                            orch.logger.warning(f"No data preview available  dataset_webpage: {dataset_webpage}")
                                            continue
                                        display_item = None
                                        if isinstance(item, dict):
                                            unwanted = {'', 'na', 'n/a', 'nan', 'unavailable', 'none', '0'}
                                            pairs = [
                                                (k, v) for k, v in item.items()
                                                if str(v).strip().lower() not in unwanted and str(v).strip() != ''
                                            ]
                                            display_item = pd.DataFrame(pairs, columns=["Field", "Value"])
                                            display_item['Value'] = display_item['Value'].astype(str)
                                            display_item = display_item[display_item["Value"].astype(str).str.strip() != ""]
                                        if display_item is not None and not display_item.empty:
                                            st.dataframe(display_item, width='content')
                                            orch.logger.info(f"[FLOW] Metadata preview available for {data_item['dataset_identifier']}, dataset_webpage: {dataset_webpage}")
                                            safe_dataset_identifier = sanitize_sheet_name(str(data_item["dataset_identifier"]))
                                            orch.logger.info(f"[FLOW] Adding metadata preview tab for {pmcid} - {safe_dataset_identifier}")
                                            sheet_name = sanitize_sheet_name(f"{pmcid}_meta_{safe_dataset_identifier}")
                                            orch.logger.info(f"[FLOW] Sheet name sanitized to {sheet_name}")
                                            excel_tabs[sheet_name] = display_item
                                        else:
                                            st.warning("No data preview available.")
                                            orch.logger.warning(f"No data preview available for {data_item['dataset_identifier']}, error occurred: {str(e)}")
                    except Exception as e:
                        st.error(f"Error processing {article_id}: {str(e)}")
                        orch.logger.info(f"[FLOW] Error processing {article_id}: {str(e)}")
                        continue
                log_placeholder.success("Extraction complete.")
                st.session_state.extraction_completed = True
                orch.logger.info(f"[FLOW] Extraction complete for user {user_id[:8]}")
                if all_supp_rows:
                    supp_summary_df = pd.concat(all_supp_rows, ignore_index=True)
                else:
                    supp_summary_df = pd.DataFrame(columns=["download_link", "description", "Source PMCID", "Source Title"])
                if all_avail_rows:
                    avail_summary_df = pd.concat(all_avail_rows, ignore_index=True)
                else:
                    avail_summary_df = pd.DataFrame(columns=["data_repository", "dataset_identifier", "dataset_webpage", "Source PMCID", "Source Title"])
                if xlsxwriter and (excel_tabs or not supp_summary_df.empty or not avail_summary_df.empty):
                    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                        supp_summary_df.to_excel(writer, sheet_name="Supplementary Material", index=False)
                        worksheet = writer.sheets["Supplementary Material"]
                        for i, col in enumerate(supp_summary_df.columns):
                            max_len = max(
                                supp_summary_df[col].astype(str).map(len).max() if not supp_summary_df.empty else 0,
                                len(str(col)),
                                15
                            )
                            worksheet.set_column(i, i, max_len + 2)
                        avail_summary_df.to_excel(writer, sheet_name="Data Availability", index=False)
                        worksheet = writer.sheets["Data Availability"]
                        for i, col in enumerate(avail_summary_df.columns):
                            max_len = max(
                                avail_summary_df[col].astype(str).map(len).max() if not avail_summary_df.empty else 0,
                                len(str(col)),
                                15
                            )
                            worksheet.set_column(i, i, max_len + 2)
                        for sheet_name, sections in excel_tabs.items():
                            if isinstance(sections, list):
                                startrow = 0
                                worksheet = None
                                for title, df in sections:
                                    df = df.copy()
                                    df_rows, df_cols = df.shape
                                    df_cols = max(df_cols, 1)
                                    df_startcol = 0
                                    df_endcol = df_cols - 1
                                    df.to_excel(writer, sheet_name=sheet_name, startrow=startrow+1, index=False, header=True)
                                    worksheet = writer.sheets[sheet_name]
                                    worksheet.merge_range(startrow, df_startcol, startrow, df_endcol, title, writer.book.add_format({
                                        'bold': True, 'align': 'center', 'valign': 'vcenter', 'bg_color': '#D9E1F2', 'border': 1
                                    }))
                                    for i, col in enumerate(df.columns):
                                        max_len = max(
                                            df[col].astype(str).map(len).max() if not df.empty else 0,
                                            len(str(col)),
                                            15
                                        )
                                        worksheet.set_column(i, i, max_len + 2)
                                    startrow += len(df) + 4
                            else:
                                df = sections
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                                worksheet = writer.sheets[sheet_name]
                                for i, col in enumerate(df.columns):
                                    max_len = max(
                                        df[col].astype(str).map(len).max() if not df.empty else 0,
                                        len(str(col)),
                                        15
                                    )
                                    worksheet.set_column(i, i, max_len + 2)
                    st.download_button(
                        label="📥 Download Excel Summary",
                        data=excel_buffer.getvalue(),
                        file_name="data_gatherer_summary.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        except Exception as e:
            log_placeholder.error(f"Extraction failed: {e}")
            st.error(f"Extraction failed: {e}")
            if orch is not None:
                orch.logger.info(f"[FLOW] Extraction failed for user {user_id[:8]}: {e}")
        finally:
            end_request(user_id)
            if orch is not None:
                orch.logger.info(f"[FLOW] Request ended for user {user_id[:8]}")

# Feedback section - only show after extraction is completed
if st.session_state.extraction_completed:
    st.markdown("---")
    st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
    st.subheader("💬 Your Feedback")
    st.markdown("Help us improve the Data Gatherer tool! Share your experience, report issues, or suggest improvements.")
    
    feedback_text = st.text_area(
        "Share your feedback:",
        placeholder="Tell us about your experience, any issues you encountered, or suggestions for improvement...",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📤 Submit Feedback", type="secondary"):
            if feedback_text.strip():
                save_user_feedback(feedback_text.strip())
                st.success("Thank you for your feedback! Your input helps us improve the tool.")
                # Clear the feedback text
                st.session_state.extraction_completed = False
                st.rerun()
            else:
                st.warning("Please enter some feedback before submitting.")
    
    with col2:
        if st.button("Skip Feedback", type="secondary"):
            st.session_state.extraction_completed = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)