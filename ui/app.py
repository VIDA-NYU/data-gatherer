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
prompt_name = 'GPT_from_full_input_Examples'
model_name = 'gemini-2.0-flash'
metadata_prompt_name = 'portkey_gemini_metadata_extract'
full_document_read = True

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Data Gatherer", layout="wide")
st.title("📄 Data Gatherer – Dataset Reference Extractor")

st.markdown("Enter one or more **PMCIDs** to extract dataset references from open-access publications.")

pmc_input = st.text_area("🔢 Enter PMCIDs (comma or newline separated)", height=100)

if st.button("🚀 Run Extraction"):
    if not pmc_input.strip():
        st.warning("Please enter at least one PMCID.")
    else:
        pmcids = [pmcid.strip() for pmcid in pmc_input.replace(",", "\n").splitlines() if pmcid.strip()]
        st.info(f"Processing {len(pmcids)} PMCID(s)...")

        try:
            orch = Orchestrator(llm_name=model_name, process_entire_document=full_document_read,)
            driver_path = '/usr/local/bin/geckodriver' if linux else None
            orch.setup_data_fetcher('url_list', driver_path=driver_path)

            results = orch.process_articles(
                pmcids, driver_path=driver_path, use_portkey_for_gemini=use_portkey, prompt_name=prompt_name
            )

            st.success("Extraction complete.")

            excel_buffer = io.BytesIO()
            excel_tabs = {}

            for pmcid, result in results.items():
                st.subheader(f"Results for {pmcid}")
                pmcid = orch.data_fetcher.url_to_pmcid(pmcid)
                summary = orch.summarize_result(result)

                files_with_extension = result[result["file_extension"].notna()]
                files_with_repo = result[result["data_repository"].notna()]

                supp_df = files_with_extension[["download_link", "description"]]
                # Drop duplicates in supplementary files
                if not supp_df.empty:
                    supp_df = supp_df.drop_duplicates()
                    # Also drop duplicates from files_with_extension for correct file type counts
                    files_with_extension = files_with_extension.drop_duplicates(subset=["download_link", "description"])

                avail_df = files_with_repo[["data_repository", "dataset_identifier", "dataset_webpage"]]

                # Now compute file_exts after deduplication
                file_exts = pd.DataFrame.from_dict(
                    files_with_extension["file_extension"].value_counts().to_dict(),
                    orient="index", columns=["Count"]
                ).reset_index().rename(columns={"index": "File Type"})

                repos = pd.DataFrame.from_dict(summary["frequency_of_data_repository"], orient="index",
                                               columns=["Count"]).reset_index().rename(columns={"index": "Repository"})

                # --- Web display ---
                st.markdown("### 📊 Visual Summary")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Supplementary Material Summary")
                    bar_chart1 = alt.Chart(file_exts).mark_bar().encode(
                        x=alt.X("File Type:N", sort="-y"),
                        y="Count:Q",
                        tooltip=["File Type", "Count"]
                    ).properties(width=300, height=200)
                    st.altair_chart(bar_chart1, use_container_width=False)
                    st.dataframe(supp_df, use_container_width=True)

                with col2:
                    st.markdown("#### Available Data Summary")
                    bar_chart2 = alt.Chart(repos).mark_bar().encode(
                        x=alt.X("Repository:N", sort="-y"),
                        y="Count:Q",
                        tooltip=["Repository", "Count"]
                    ).properties(width=300, height=200)
                    st.altair_chart(bar_chart2, use_container_width=False)
                    st.dataframe(avail_df, use_container_width=True)

                # --- Excel summary sheet ---
                # Create a single summary sheet per PMCID with both sides, but visually separate tables using merged cells and spacing

                # Prepare the summary tables
                file_exts_table = file_exts.copy()
                repos_table = repos.copy()
                supp_table = supp_df.rename(columns={"download_link": "Download Link", "description": "Description"})
                avail_table = avail_df.rename(columns={
                    "data_repository": "Repository",
                    "dataset_identifier": "Dataset Identifier",
                    "dataset_webpage": "Dataset Webpage"
                })

                # We'll build a list of (df, title) for each section
                summary_sections = [
                    ("Supplementary File Types", file_exts_table),
                    ("Available Data Repositories", repos_table),
                    ("Supplementary Files Table", supp_table),
                    ("Available Data Table", avail_table)
                ]

                # Store for writing to Excel
                excel_tabs[f"{pmcid}_summary"] = summary_sections

                # --- Metadata tabs for each available dataset ---
                try:
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
                            display_item = display_item[display_item["Value"].astype(str).str.strip() != ""]

                        if display_item is not None and not display_item.empty:
                            st.dataframe(display_item, use_container_width=True)
                            # Add to excel
                            sheet_name = f"{pmcid}_meta_{data_item['dataset_identifier']}"
                            invalid_chars = set(r'[]:*?/\\')
                            sanitized = ''.join('_' if c in invalid_chars else c for c in str(sheet_name))
                            sanitized = sanitized[:31]
                            excel_tabs[sanitized] = display_item
                        else:
                            st.warning(f"No data preview available for item {j + 1}")

                except Exception as e:
                    st.error(f"Error fetching metadata: {e}")

            # Write all collected DataFrames to Excel
            if xlsxwriter and excel_tabs:
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    for sheet_name, sections in excel_tabs.items():
                        if isinstance(sections, list):
                            # This is our summary sheet with multiple tables
                            startrow = 0
                            worksheet = None
                            for title, df in sections:
                                df = df.copy()
                                # Write section title with merged cells
                                df_rows, df_cols = df.shape
                                df_cols = max(df_cols, 1)
                                df_startcol = 0
                                df_endcol = df_cols - 1
                                # Write the title
                                df.to_excel(writer, sheet_name=sheet_name, startrow=startrow+1, index=False, header=True)
                                worksheet = writer.sheets[sheet_name]
                                # Merge cells for the section title
                                worksheet.merge_range(startrow, df_startcol, startrow, df_endcol, title, writer.book.add_format({
                                    'bold': True, 'align': 'center', 'valign': 'vcenter', 'bg_color': '#D9E1F2', 'border': 1
                                }))
                                # Set column widths
                                for i, col in enumerate(df.columns):
                                    max_len = max(
                                        df[col].astype(str).map(len).max() if not df.empty else 0,
                                        len(str(col)),
                                        15
                                    )
                                    worksheet.set_column(i, i, max_len + 2)
                                # Add spacing between tables
                                startrow += len(df) + 4  # 4 rows: 1 for title, 1 for header, 2 for spacing
                        else:
                            # Metadata tab: just write as before
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
            st.error(f"Extraction failed: {e}")
