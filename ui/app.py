# High-level: Streamlit UI for extracting dataset references from PMCIDs, showing results, and collecting user feedback.

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

# --- Streamlit page configuration ---
st.set_page_config(
    page_title="Data Gatherer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

linux = os.path.exists('/.dockerenv')

# --- Session state and concurrency management ---
if 'active_requests' not in st.session_state:
    st.session_state.active_requests = defaultdict(int)

user_locks = defaultdict(threading.Lock)
MAX_CONCURRENT_REQUESTS = 2
MAX_PMCIDS_PER_REQUEST = 10

def get_user_id():
    """Assign a unique user/session ID for concurrency and feedback tracking."""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id

def check_rate_limit(user_id):
    """Limit concurrent requests per user."""
    with user_locks[user_id]:
        current_requests = st.session_state.active_requests[user_id]
        if current_requests >= MAX_CONCURRENT_REQUESTS:
            return False, f"Too many active requests ({current_requests}). Please wait."
        return True, "OK"

def start_request(user_id):
    """Increment active request count for user."""
    with user_locks[user_id]:
        st.session_state.active_requests[user_id] += 1

def end_request(user_id):
    """Decrement active request count for user."""
    with user_locks[user_id]:
        st.session_state.active_requests[user_id] = max(0, st.session_state.active_requests[user_id] - 1)

def app_process_url(orch, url, **kwargs):
    """
    Wrapper for DataGatherer.process_url with user context and error logging.
    """
    user_id = get_user_id()
    time.sleep(0.5)  # Prevent system overload
    try:
        return orch.process_url(url, **kwargs)
    except Exception as e:
        orch.logger.error(f"Error in app_process_url for user {user_id[:8]}: {e}")
        raise

def sanitize_sheet_name(name):
    """Sanitize Excel sheet names for compatibility."""
    name = re.sub(r'[:\\/*?\[\]]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name[:31]

def save_user_feedback(feedback_text):
    """Persist user feedback to a file with timestamp and user ID."""
    feedback_dir = "feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_file = os.path.join(feedback_dir, datetime.now().isoformat() + "_" + get_user_id() + ".txt")
    with open(feedback_file, "w") as f:
        f.write(feedback_text)

# --- Custom CSS for UI tweaks ---
st.markdown("""
    <style>
    [data-testid="stExpander"] p {
        font-size: 20px !important;
    }
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

# --- User/session management and UI header ---
user_id = get_user_id()
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📄 Data Gatherer – Dataset Reference Extractor")
with col2:
    current_requests = st.session_state.active_requests[user_id]
    if current_requests > 0:
        st.info(f"🔄 Processing ({current_requests} active)")
    else:
        st.success("✅ Ready")

# --- Sidebar: Extraction settings ---
st.sidebar.header("⚙️ Extraction Settings")
MODEL_OPTIONS = [
    "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-2.5-flash"
]
PROMPT_MODEL_OPTIONS = {
    'FDR': 'GPT_FDR_FewShot_shortDescr',
    'RTR': 'GPT_FDR_FewShot_shortDescr',
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
extract_keywords_from_supp_files = True

# --- Main input: PMCIDs ---
load_dotenv()
st.markdown("Enter one or more **PMCIDs** to extract dataset references from open-access publications.")
st.info(f"Maximum {MAX_PMCIDS_PER_REQUEST} PMCIDs per request.")
pmc_input = st.text_area("🔢 Enter PMCIDs (comma or newline separated)", height=100)

# --- Extraction state initialization ---
if 'extraction_completed' not in st.session_state:
    st.session_state.extraction_completed = False

# --- Main extraction logic: runs on button click ---
if st.button("🚀 Run Extraction", type="primary"):
    st.session_state.extraction_completed = False  # Reset feedback flag
    if not pmc_input.strip():
        st.warning("Please enter at least one PMCID.")
    else:
        orch = None
        allowed, message = check_rate_limit(user_id)
        if not allowed:
            st.error(message)
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
                    log_level="DEBUG",
                    load_from_cache=not True, 
                    save_to_cache=not True, 
                    driver_path=driver_path,
                    log_file_override="ui/data_gatherer.log",
                    clear_previous_logs=not False
                )
                orch.logger.info(f"[FLOW] Extraction button pressed by user {user_id[:8]}")
                orch.logger.info(f"[FLOW] Checking rate limit for user {user_id[:8]}")
                orch.logger.info(f"[FLOW] PMCIDs to process: {pmcids}")
                orch.logger.info(f"[FLOW] Extraction started for user {user_id[:8]}")
                excel_tabs = {}
                all_supp_rows = []
                all_avail_rows = []
                per_article_results = []
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
                            use_portkey=use_portkey, 
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
                        
                        orch.logger.info(f"[FLOW] Before LLM: files_with_extension has {len(files_with_extension)} rows, columns: {files_with_extension.columns.tolist()}")
                        
                        # Process supplementary files with LLM to enhance descriptions
                        if not files_with_extension.empty and extract_keywords_from_supp_files:
                            orch.logger.info(f"[FLOW] Processing supplementary files with LLM for {article_id}")
                            files_with_extension = orch.parser.extract_supplementary_file_info_from_refs(files_with_extension)
                            orch.logger.info(f"[FLOW] After LLM: files_with_extension has {len(files_with_extension)} rows, columns: {files_with_extension.columns.tolist()}")
                            if 'supplementary_file_keywords' in files_with_extension.columns:
                                orch.logger.info(f"[FLOW] supplementary_file_keywords column exists! Sample values: {files_with_extension['supplementary_file_keywords'].head(2).tolist()}")
                            else:
                                orch.logger.warning(f"[FLOW] supplementary_file_keywords column NOT FOUND after LLM processing!")
                        else:
                            orch.logger.info(f"[FLOW] Skipping LLM processing: empty={files_with_extension.empty}, extract_keywords={extract_keywords_from_supp_files}")
                        
                        supp_df = files_with_extension.copy() if not files_with_extension.empty else pd.DataFrame(columns=["download_link", "supplementary_file_keywords", "description"])
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
                        summary = orch.summarize_result(result)
                        per_article_results.append({
                            "article_id": article_id,
                            "pmcid": pmcid,
                            "doi": doi,
                            "title": title,
                            "files_with_extension": files_with_extension,
                            "files_with_repo": files_with_repo,
                            "supp_df": supp_df,
                            "avail_df": avail_df,
                            "summary": summary,
                        })
                    except Exception as e:
                        st.error(f"Error processing {article_id}: {str(e)}")
                        orch.logger.info(f"[FLOW] Error processing {article_id}: {str(e)}")
                        continue
                # --- Prepare summary dataframes and persist results ---
                supp_summary_df = pd.concat(all_supp_rows, ignore_index=True) if all_supp_rows else pd.DataFrame(columns=["download_link", "description", "Source PMCID", "Source DOI", "Source Title"])
                avail_summary_df = pd.concat(all_avail_rows, ignore_index=True) if all_avail_rows else pd.DataFrame(columns=["data_repository", "dataset_identifier", "dataset_webpage", "Source PMCID", "Source DOI", "Source Title"])
                st.session_state.supp_summary_df = supp_summary_df
                st.session_state.avail_summary_df = avail_summary_df
                st.session_state.excel_tabs = excel_tabs
                st.session_state.pmcids = pmcids
                st.session_state.per_article_results = per_article_results
                st.session_state.results_ready = True
                st.session_state.orch = orch
                st.session_state.extraction_completed = True
                # --- Excel export ---
                excel_buffer = io.BytesIO()
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
                    st.session_state.excel_buffer = excel_buffer.getvalue()
        except Exception as e:
            log_placeholder.error(f"Extraction failed: {e}")
            st.error(f"Extraction failed: {e}")
            if orch is not None:
                orch.logger.info(f"[FLOW] Extraction failed for user {user_id[:8]}: {e}")
        finally:
            end_request(user_id)
            if orch is not None:
                orch.logger.info(f"[FLOW] Request ended for user {user_id[:8]}")

# --- Results display: show results if available in session state ---
if st.session_state.get("results_ready", False):
    # Only show Results heading once per run (prevents flicker/duplication)
    if not st.session_state.get("results_displayed", False):
        st.session_state.results_displayed = True
        st.markdown("<h2 style='margin-top: 1.5em;'>Results</h2>", unsafe_allow_html=True)
    # Results tables, charts, and per-article expanders
    supp_summary_df = st.session_state.supp_summary_df
    avail_summary_df = st.session_state.avail_summary_df
    excel_tabs = st.session_state.excel_tabs
    pmcids = st.session_state.pmcids
    per_article_results = st.session_state.per_article_results
    orch_display = st.session_state.get("orch", None)
    if orch_display is None:
        st.warning("Session expired or no extraction context. Please rerun extraction.")
    else:
        for idx, article in enumerate(per_article_results):
            title = article["title"]
            files_with_extension = article["files_with_extension"]
            files_with_repo = article["files_with_repo"]
            supp_df = article["supp_df"]
            avail_df = article["avail_df"]
            summary = article["summary"]
            pmcid = article["pmcid"]
            with st.expander(f"{title}", expanded=False):
                st.markdown(f"<h2 style='text-align: left; font-size: 2em; font-weight: bold; margin-bottom: 0.5em;'>{title}</h2>", unsafe_allow_html=True)
                if not supp_df.empty:
                    supp_df = supp_df.drop_duplicates()
                    files_with_extension = files_with_extension.drop_duplicates(
                        subset=["download_link", "description"],
                        keep='first'
                    )
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
                
                # --- Unified Dataset Table with Checkboxes ---
                st.markdown("### Datasets Found")
                
                # Create unified dataset table
                unified_datasets = []
                
                # Add repository datasets (from avail_df)
                for j, row in files_with_repo.iterrows():
                    # Handle description - join list elements with " | "
                    desc_value = row.get("dataset_keywords", "n/a") if "dataset_keywords" in row else "n/a"
                    if isinstance(desc_value, list):
                        desc_value = " | ".join(str(item) for item in desc_value)
                    
                    dataset_entry = {
                        "source_type": "repository",
                        "source_index": j,
                        "Repository": row.get("data_repository", "n/a"),
                        "Dataset ID": row.get("dataset_identifier", "n/a"),
                        "Description": desc_value,
                        "webpage": row.get("dataset_webpage", "n/a"),
                        "original_data": row
                    }
                    unified_datasets.append(dataset_entry)
                
                # Add supplementary files (from supp_df) 
                for j, row in supp_df.iterrows():
                    # Only add if it has meaningful content
                    if pd.notna(row.get("download_link")):
                        # Extract ID from download link or use index
                        file_id = row.get("id", row.get('download_link', f"supp_{j}")).split("/")[-1] if pd.notna(row.get("download_link")) else f"supp_{j}"
                        
                        orch_display.logger.info(f"[FLOW] Row {j}: {row.to_dict()}")

                        # Use supplementary_file_keywords if available, otherwise fall back to description
                        desc_value = row.get("supplementary_file_keywords", row.get("description", "n/a"))
                        if isinstance(desc_value, list):
                            desc_value = " | ".join(str(item) for item in desc_value)
                        
                        dataset_entry = {
                            "source_type": "supplementary",
                            "source_index": j,
                            "Repository": "PMC",
                            "Dataset ID": file_id,
                            "Description": desc_value,
                            "webpage": row.get("download_link", "n/a"),
                            "original_data": row
                        }
                        unified_datasets.append(dataset_entry)
                
                if not unified_datasets:
                    st.info("No datasets found.")
                else:
                    # Initialize session state for checkbox selections
                    checkbox_key = f"selected_datasets_{pmcid}_{idx}"
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = {}
                    
                    st.markdown("**Select datasets to fetch metadata:**")
                    
                    # Create columns for the table layout (widened Repository column)
                    col_select, col_repo, col_id, col_desc = st.columns([0.5, 2.5, 2, 4])
                    
                    # Header row
                    with col_select:
                        st.markdown("**Select**")
                    with col_repo:
                        st.markdown("**Repository**")
                    with col_id:
                        st.markdown("**Dataset ID**")
                    with col_desc:
                        st.markdown("**Description**")
                    
                    st.markdown("---")
                    
                    # Data rows with checkboxes
                    for i, dataset in enumerate(unified_datasets):
                        dataset_key = f"{pmcid}_{dataset['source_type']}_{dataset['source_index']}"
                        
                        # Initialize checkbox state
                        if dataset_key not in st.session_state[checkbox_key]:
                            # Auto-select repository datasets, don't auto-select supplementary files
                            st.session_state[checkbox_key][dataset_key] = (dataset['source_type'] == 'repository')
                        
                        col_select, col_repo, col_id, col_desc = st.columns([0.5, 2.5, 2, 4])
                        
                        with col_select:
                            is_selected = st.checkbox(
                                "✓",
                                value=st.session_state[checkbox_key][dataset_key],
                                key=f"checkbox_{dataset_key}",
                                label_visibility="collapsed"
                            )
                            st.session_state[checkbox_key][dataset_key] = is_selected
                        
                        with col_repo:
                            st.text(dataset["Repository"])
                        
                        with col_id:
                            st.text(dataset["Dataset ID"][:50] + "..." if len(str(dataset["Dataset ID"])) > 50 else dataset["Dataset ID"])
                        
                        with col_desc:
                            desc_text = str(dataset["Description"])[:100] + "..." if len(str(dataset["Description"])) > 100 else str(dataset["Description"])
                            st.text(desc_text)
                    
                    st.markdown("---")
                    
                    # Button to fetch metadata for selected datasets
                    if st.button(f"🔍 Fetch Metadata for Selected Datasets", key=f"fetch_metadata_{pmcid}_{idx}"):
                        # Get selected datasets
                        selected_datasets = [
                            dataset for i, dataset in enumerate(unified_datasets)
                            if st.session_state[checkbox_key].get(f"{pmcid}_{dataset['source_type']}_{dataset['source_index']}", False)
                        ]
                        
                        if not selected_datasets:
                            st.warning("Please select at least one dataset to fetch metadata.")
                        else:
                            # Process only repository datasets (supplementary files don't have metadata to fetch)
                            repo_datasets = [d for d in selected_datasets if d['source_type'] == 'repository']
                            
                            if not repo_datasets:
                                st.info("Only supplementary files selected. Metadata fetching is only available for repository datasets.")
                            else:
                                for dataset in repo_datasets:
                                    data_item = dataset['original_data']
                                    dataset_label = f"**{data_item['dataset_identifier']}** ({data_item['dataset_webpage']})"
                                    st.markdown(f"- {dataset_label}")
                                    dataset_webpage = data_item["dataset_webpage"] if "dataset_webpage" in data_item and pd.notna(data_item["dataset_webpage"]) else ""
                                    
                                    with st.spinner(f"Fetching metadata from repo: {data_item['data_repository']}... {dataset_webpage}"):
                                        try:
                                            preview_result = orch_display.process_metadata(
                                                data_item, interactive=False, return_metadata=True,
                                                write_raw_metadata=False,
                                                use_portkey=use_portkey,
                                                prompt_name=metadata_prompt_name,
                                                timeout=15
                                            )
                                            if not preview_result or not isinstance(preview_result, list) or len(preview_result) == 0:
                                                st.warning("No data preview available.")
                                                orch_display.logger.warning(f"No data preview available for {data_item['dataset_identifier']}, dataset_webpage: {dataset_webpage}")
                                                continue
                                            item = preview_result[0]
                                        except Exception as e:
                                            st.warning(f"No data preview available.")
                                            orch_display.logger.warning(f"No data preview available  dataset_webpage: {dataset_webpage}")
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
                                            orch_display.logger.info(f"[FLOW] Metadata preview available for {data_item['dataset_identifier']}, dataset_webpage: {dataset_webpage}")
                                            safe_dataset_identifier = sanitize_sheet_name(str(data_item["dataset_identifier"]))
                                            orch_display.logger.info(f"[FLOW] Adding metadata preview tab for {pmcid} - {safe_dataset_identifier}")
                                            sheet_name = sanitize_sheet_name(f"{pmcid}_meta_{safe_dataset_identifier}")
                                            orch_display.logger.info(f"[FLOW] Sheet name sanitized to {sheet_name}")
                                            excel_tabs[sheet_name] = display_item
                                        else:
                                            st.warning("No data preview available.")
                                            orch_display.logger.warning(f"No data preview available for {data_item['dataset_identifier']}, error occurred: {str(e)}")
        # Download button (use the saved excel_buffer)
        if "excel_buffer" in st.session_state:
            st.download_button(
                label="📥 Download Excel Summary",
                data=st.session_state.excel_buffer,
                file_name="data_gatherer_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# --- Feedback section: shown after extraction is completed ---
if st.session_state.get("extraction_completed", False):
    st.markdown("---")
    st.markdown("Help us improve the Data Gatherer tool! Did it miss a dataset or report any false positives?")
    # Feedback text area and state management
    if "feedback_text" not in st.session_state:
        st.session_state.feedback_text = ""
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    feedback_text = st.text_area(
        "Share your feedback:",
        value=st.session_state.feedback_text,
        placeholder="Tell us about your experience, any issues you encountered, or suggestions for improvement...",
        height=100,
        key="feedback_text_area"
    )
    st.session_state.feedback_text = feedback_text
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📤 Submit Feedback", type="secondary"):
            if st.session_state.feedback_text.strip():
                save_user_feedback(st.session_state.feedback_text.strip())
                st.session_state.feedback_submitted = True
                st.session_state.feedback_text = ""
                st.rerun()
            else:
                st.warning("Please enter some feedback before submitting.")
    with col2:
        if st.button("Skip Feedback", type="secondary"):
            st.session_state.extraction_completed = False
            st.session_state.feedback_submitted = False
            st.session_state.feedback_text = ""
            st.rerun()
    if st.session_state.feedback_submitted:
        st.success("Thank you for your feedback! Your input helps us improve the tool.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- End of script: reset results_displayed flag for next run ---
if "results_displayed" in st.session_state:
    st.session_state.results_displayed = False