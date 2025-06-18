import streamlit as st
from data_gatherer.orchestrator import Orchestrator
from dotenv import load_dotenv
import pandas as pd
import altair as alt
import os
import io
import xlsxwriter

linux = os.path.exists('/.dockerenv')

use_portkey = True
prompt_name = 'retrieve_datasets_simple_JSON'
model_name = 'gemini-2.0-flash'
metadata_prompt_name = 'portkey_gemini_metadata_extract'
full_document_read = False

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Data Gatherer", layout="wide")
st.title("📄 Data Gatherer – Dataset Reference Extractor")

st.markdown("Enter one or more **PMCIDs** to extract dataset references from open-access publications.")

# Input box for PMCIDs (comma or newline separated)
pmc_input = st.text_area("🔢 Enter PMCIDs (comma or newline separated)", height=100)

# Process button
if st.button("🚀 Run Extraction"):
    if not pmc_input.strip():
        st.warning("Please enter at least one PMCID.")
    else:
        # Split input into list of PMCIDs
        pmcids = [pmcid.strip() for pmcid in pmc_input.replace(",", "\n").splitlines() if pmcid.strip()]
        st.info(f"Processing {len(pmcids)} PMCID(s)...")

        try:
            orch = Orchestrator(llm_name=model_name, process_entire_document=full_document_read,)

            if linux:
                driver_path = '/usr/local/bin/geckodriver'
            else:
                driver_path = None

            orch.setup_data_fetcher('url_list', driver_path=driver_path)

            results = orch.process_articles(pmcids, driver_path=driver_path, use_portkey_for_gemini=use_portkey,
                                            prompt_name=prompt_name)

            st.success("Extraction complete.")

            excel_buffer = io.BytesIO()
            excel_tabs = {}  # sheet_name: DataFrame

            for pmcid, result in results.items():
                st.subheader(f"Results for {pmcid}")
                pmcid = orch.data_fetcher.url_to_pmcid(pmcid)
                # assume summary is already generated:
                summary = orch.summarize_result(result)
                file_exts = pd.DataFrame.from_dict(summary["frequency_of_file_extensions"], orient="index",
                                                   columns=["Count"]).reset_index().rename(
                    columns={"index": "File Type"})
                repos = pd.DataFrame.from_dict(summary["frequency_of_data_repository"], orient="index",
                                               columns=["Count"]).reset_index().rename(columns={"index": "Repository"})

                # Add summary tables to excel
                excel_tabs[f"{pmcid}_file_types"] = file_exts
                excel_tabs[f"{pmcid}_repositories"] = repos

                st.markdown("### 📊 Visual Summary")

                col1, col2 = st.columns(2)

                files_with_extension = result[result["file_extension"].notna()]
                files_with_repo = result[result["data_repository"].notna()]

                with col1:
                    st.markdown("#### Supplementary Material Summary")
                    bar_chart1 = alt.Chart(file_exts).mark_bar().encode(
                        x=alt.X("File Type:N", sort="-y"),
                        y="Count:Q",
                        tooltip=["File Type", "Count"]
                    ).properties(width=300, height=200)
                    st.altair_chart(bar_chart1, use_container_width=False)
                    supp_df = files_with_extension[["download_link", "description"]]
                    st.dataframe(supp_df, use_container_width=True)
                    # Add to excel
                    excel_tabs[f"{pmcid}_supplementary"] = supp_df

                with col2:
                    st.markdown("#### Available Data Summary")
                    bar_chart2 = alt.Chart(repos).mark_bar().encode(
                        x=alt.X("Repository:N", sort="-y"),
                        y="Count:Q",
                        tooltip=["Repository", "Count"]
                    ).properties(width=300, height=200)
                    st.altair_chart(bar_chart2, use_container_width=False)
                    avail_df = files_with_repo[["data_repository", "dataset_identifier", "dataset_webpage"]]
                    st.dataframe(avail_df, use_container_width=True)
                    # Add to excel
                    excel_tabs[f"{pmcid}_available_data"] = avail_df

                    for j, data_item in files_with_repo.iterrows():
                        with st.spinner(
                                f"Fetching metadata from repo: {data_item['data_repository']}... {data_item['dataset_webpage']}"):
                            item = orch.get_data_preview(data_item, interactive=False, return_metadata=True,
                                                         write_raw_metadata=False,
                                                         use_portkey_for_gemini=use_portkey,
                                                         prompt_name=metadata_prompt_name)[0]

                        st.markdown(f"#### Dataset {data_item['dataset_identifier']} ({data_item['data_repository']}) metadata")
                        display_item = None
                        if isinstance(item, dict):
                            unwanted = {'', 'na', 'n/a', 'nan', 'unavailable', 'none', '0'}
                            pairs = [
                                (k, v) for k, v in item.items()
                                if str(v).strip().lower() not in unwanted and str(v).strip() != ''
                            ]
                            display_item = pd.DataFrame(pairs, columns=["Field", "Value"])
                            display_item['Value'] = display_item['Value'].astype(str)
                            # Drop rows where Value is empty after filtering
                            display_item = display_item[display_item["Value"].astype(str).str.strip() != ""]

                        if display_item is not None and not display_item.empty:
                            st.dataframe(display_item, use_container_width=True)
                            # Add to excel
                            sheet_name = f"{pmcid}_meta_{data_item['dataset_identifier']}"
                            # Excel sheet names must be <= 31 chars and not contain []:*?/\
                            invalid_chars = set(r'[]:*?/\\')
                            sanitized = ''.join('_' if c in invalid_chars else c for c in str(sheet_name))
                            sanitized = sanitized[:31]
                            excel_tabs[sanitized] = display_item
                        else:
                            st.warning(f"No data preview available for item {j + 1}")

            # Write all collected DataFrames to Excel
            if xlsxwriter and excel_tabs:
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    for sheet_name, df in excel_tabs.items():
                        # Ensure sheet name is valid
                        invalid_chars = set(r'[]:*?/\\')
                        sanitized = ''.join('_' if c in invalid_chars else c for c in str(sheet_name))
                        sanitized = sanitized[:31]
                        df.to_excel(writer, sheet_name=sanitized, index=False)
                st.download_button(
                    label="📥 Download Excel Summary",
                    data=excel_buffer.getvalue(),
                    file_name="data_gatherer_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"Extraction failed: {e}")
