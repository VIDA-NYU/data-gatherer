import streamlit as st
from data_gatherer.orchestrator import Orchestrator
from dotenv import load_dotenv
import pandas as pd
import altair as alt

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
            orch = Orchestrator()

            orch.setup_data_fetcher('url_list')

            results = orch.process_articles(pmcids)

            st.success("Extraction complete.")
            for pmcid, result in results.items():
                st.subheader(f"Results for {pmcid}")
                # assume summary is already generated:
                summary = orch.summarize_result(result)
                file_exts = pd.DataFrame.from_dict(summary["frequency_of_file_extensions"], orient="index",
                                                   columns=["Count"]).reset_index().rename(
                    columns={"index": "File Type"})
                repos = pd.DataFrame.from_dict(summary["frequency_of_data_repository"], orient="index",
                                               columns=["Count"]).reset_index().rename(columns={"index": "Repository"})

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
                    st.dataframe(files_with_extension[["download_link", "description"]],
                                 use_container_width=True)

                with col2:
                    st.markdown("#### Available Data Summary")
                    bar_chart2 = alt.Chart(repos).mark_bar().encode(
                        x=alt.X("Repository:N", sort="-y"),
                        y="Count:Q",
                        tooltip=["Repository", "Count"]
                    ).properties(width=300, height=200)
                    st.altair_chart(bar_chart2, use_container_width=False)
                    st.dataframe(files_with_repo[["data_repository", "dataset_identifier", "dataset_webpage"]],
                                 use_container_width=True)

                    for j, data_item in files_with_repo.iterrows():
                        with st.spinner(
                                f"Fetching metadata from repo: {data_item['data_repository']}... {data_item['dataset_webpage']}"):
                            item = orch.get_data_preview(data_item, interactive=False, return_metadata=True,
                                                         write_raw_metadata=True)[0]

                        st.markdown(f"#### Data Preview for item {j + 1}")
                        display_item = None
                        if isinstance(item, dict):
                            unwanted = {'', 'na', 'n/a', 'nan', 'unavailable', 'none', '0'}
                            pairs = [
                                (k, v) for k, v in item.items()
                                if str(v).strip().lower() not in unwanted and str(v).strip() != ''
                            ]
                            display_item = pd.DataFrame(pairs, columns=["Field", "Value"])
                            # Drop rows where Value is empty after filtering
                            display_item = display_item[display_item["Value"].astype(str).str.strip() != ""]

                        if display_item is not None and not display_item.empty:
                            st.dataframe(display_item, use_container_width=True)
                        else:
                            st.warning(f"No data preview available for item {j + 1}")


        except Exception as e:
            st.error(f"Extraction failed: {e}")