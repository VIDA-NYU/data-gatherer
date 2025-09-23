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

st.set_page_config(page_title="Data Gatherer", layout="wide")

st.markdown("""
    <style>
    [data-testid="stExpander"] p {
        font-size: 20px !important;  /* Change this to desired size */
    }
    </style>
""", unsafe_allow_html=True)

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

st.title("📄 Data Gatherer – Dataset Reference Extractor")

st.markdown("Enter one or more **PMCIDs** to extract dataset references from open-access publications.")

pmc_input = st.text_area("🔢 Enter PMCIDs (comma or newline separated)", height=100)

if st.button("🚀 Run Extraction"):
    if not pmc_input.strip():
        st.warning("Please enter at least one PMCID.")
    else:
        pmcids = [pmcid.strip() for pmcid in pmc_input.replace(",", "\n").splitlines() if pmcid.strip()]

        # --- Spinner and log area ---
        with st.spinner("Extraction in progress..."):
            log_placeholder = st.empty()
            try:
                driver_path = '/usr/local/bin/geckodriver' if linux else None
                orch = DataGatherer(llm_name=model_name, process_entire_document=full_document_read, log_level="INFO",
                                    load_from_cache=True, save_to_cache=True, driver_path=driver_path)

                excel_buffer = io.BytesIO()
                excel_tabs = {}

                # --- Prepare global summary tables ---
                all_supp_rows = []
                all_avail_rows = []

                # --- Results header ---
                st.markdown("<h2 style='margin-top: 1.5em;'>Results</h2>", unsafe_allow_html=True)

                for idx, article_id in enumerate(pmcids):
                    log_placeholder.info(f"Processing article {idx+1} of {len(pmcids)}: {article_id}")
                    pmcid = orch.data_fetcher.url_to_pmcid(article_id)
                    doi = orch.data_fetcher.url_to_doi(article_id, pmcid)
                    url = orch.preprocess_url(article_id)

                    result = orch.process_url(
                        url, driver_path=driver_path, use_portkey_for_gemini=use_portkey, prompt_name=prompt_name,
                        semantic_retrieval=True
                    )

                    # --- Robustly extract title ---
                    title = result["pub_title"]
                    if isinstance(title, pd.Series):
                        title = title.iloc[0]
                    elif isinstance(title, list):
                        title = title[0]

                    # --- Safely handle missing columns ---
                    files_with_extension = result[result["file_extension"].notna()] if "file_extension" in result else pd.DataFrame(columns=["download_link", "description", "file_extension"])
                    files_with_repo = result[result["data_repository"].notna()] if "data_repository" in result else pd.DataFrame(columns=["data_repository", "dataset_identifier", "dataset_webpage"])

                    # Supplementary Material rows
                    supp_df = files_with_extension[["download_link", "description"]].copy() if not files_with_extension.empty else pd.DataFrame(columns=["download_link", "description"])
                    supp_df["Source PMCID"] = pmcid
                    supp_df["Source DOI"] = doi
                    supp_df["Source Title"] = title
                    if not supp_df.empty:
                        supp_df = supp_df.drop_duplicates()
                        all_supp_rows.append(supp_df)

                    # Available Datasets rows (handle missing 'dataset_webpage')
                    avail_cols = ["data_repository", "dataset_identifier", "dataset_webpage"]
                    avail_df = files_with_repo[avail_cols].copy() if not files_with_repo.empty else pd.DataFrame(columns=avail_cols)
                    avail_df["Source PMCID"] = pmcid
                    avail_df["Source DOI"] = doi
                    avail_df["Source Title"] = title
                    if not avail_df.empty:
                        avail_df = avail_df.drop_duplicates()
                        all_avail_rows.append(avail_df)

                    # --- Top-level expander for each article (no nested expanders) ---
                    with st.expander(f"{title}", expanded=False):
                        # Make the title larger and bold at the top
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

                        # --- Show bar charts side by side at the top ---
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
                            # Center the chart using HTML/CSS flexbox
                            st.markdown(
                                "<div style='display: flex; justify-content: center; align-items: center;'>",
                                unsafe_allow_html=True
                            )
                            st.altair_chart(bar_chart1, use_container_width=False)
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
                            st.altair_chart(bar_chart2, use_container_width=False)
                            st.markdown("</div>", unsafe_allow_html=True)

                        # --- Supplementary Material section ---
                        st.markdown("### Supplementary Material")
                        st.markdown("#### Supplementary File Types")
                        st.dataframe(file_exts, use_container_width=True)
                        st.markdown("#### Supplementary Files")
                        st.dataframe(supp_df, use_container_width=True)

                        # --- Available datasets section ---
                        st.markdown("### Available datasets")
                        if files_with_repo.empty:
                            st.info("No datasets found.")
                        else:
                            st.dataframe(avail_df, use_container_width=True)
                            for j, data_item in files_with_repo.iterrows():
                                dataset_label = f"**{data_item['dataset_identifier']}** ({data_item['dataset_webpage']})"
                                st.markdown(f"- {dataset_label}")
                                dataset_webpage = data_item["dataset_webpage"] if "dataset_webpage" in data_item and pd.notna(data_item["dataset_webpage"]) else ""
                                with st.spinner(
                                    f"Fetching metadata from repo: {data_item['data_repository']}... {dataset_webpage}"):
                                    try:
                                        try:
                                            preview_result = orch.process_metadata(
                                                data_item, interactive=False, return_metadata=True,
                                                write_raw_metadata=False,
                                                use_portkey_for_gemini=use_portkey,
                                                prompt_name=metadata_prompt_name,
                                                timeout=15  # <-- Increase timeout from 3 to 15 seconds
                                            )
                                            if not preview_result or not isinstance(preview_result, list) or\
                                                    len(preview_result) == 0:
                                                st.warning("No data preview available.")
                                                continue
                                            item = preview_result[0]
                                        except Exception as e:
                                            st.warning(f"No data preview available: {e}")
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
                                            st.dataframe(display_item, use_container_width=True)
                                            # --- ADDITION: Save per-dataset metadata DataFrame for Excel ---
                                            safe_dataset_identifier = data_item["dataset_identifier"].replace("/", "_")
                                            sheet_name = f"{pmcid}_meta_{safe_dataset_identifier}"
                                            # Excel sheet names max 31 chars
                                            if len(sheet_name) > 31:
                                                sheet_name = sheet_name[:31]
                                            # Store DataFrame in excel_tabs
                                            excel_tabs[sheet_name] = display_item
                                        else:
                                            st.warning("No data preview available.")
                                    except Exception as e:
                                        st.error(f"Error fetching metadata: {e}")

                log_placeholder.success("Extraction complete.")

                # --- Combine all supplementary and dataset rows for summary tabs ---
                if all_supp_rows:
                    supp_summary_df = pd.concat(all_supp_rows, ignore_index=True)
                else:
                    supp_summary_df = pd.DataFrame(columns=["download_link", "description", "Source PMCID", "Source Title"])

                if all_avail_rows:
                    avail_summary_df = pd.concat(all_avail_rows, ignore_index=True)
                else:
                    avail_summary_df = pd.DataFrame(columns=["data_repository", "dataset_identifier", "dataset_webpage", "Source PMCID", "Source Title"])

                # --- Excel writing ---
                if xlsxwriter and (excel_tabs or not supp_summary_df.empty or not avail_summary_df.empty):
                    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                        # Write Supplementary Material summary tab
                        supp_summary_df.to_excel(writer, sheet_name="Supplementary Material", index=False)
                        worksheet = writer.sheets["Supplementary Material"]
                        for i, col in enumerate(supp_summary_df.columns):
                            max_len = max(
                                supp_summary_df[col].astype(str).map(len).max() if not supp_summary_df.empty else 0,
                                len(str(col)),
                                15
                            )
                            worksheet.set_column(i, i, max_len + 2)

                        # Write Data Availability summary tab
                        avail_summary_df.to_excel(writer, sheet_name="Data Availability", index=False)
                        worksheet = writer.sheets["Data Availability"]
                        for i, col in enumerate(avail_summary_df.columns):
                            max_len = max(
                                avail_summary_df[col].astype(str).map(len).max() if not avail_summary_df.empty else 0,
                                len(str(col)),
                                15
                            )
                            worksheet.set_column(i, i, max_len + 2)

                        # Write per-dataset meta tabs as before
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
