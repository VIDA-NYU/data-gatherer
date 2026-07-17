import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime

def load_pmc_files_from_html_xml_dir_to_dataframe_fetch_file(src_dir,raw_HTML_data_filepath):
    """
    Loads all files from the specified HTML/XML directory into a DataFrame and saves it as a parquet file.
    Args:
        raw_HTML_data_filepath (str): The path where the DataFrame will be saved as a parquet file.
    """
    # find all the files in the html_xml_dir directory
    files_df = []
    for root, dirs, file_names in os.walk(src_dir):
        for file_name in file_names:
            file_format = None
            pmcid = None
            basename = os.path.basename(file_name)

            if not(basename.startswith('PMC')):
                print(f"Filename does not start with 'PMC': {basename}. Skipping this file.")
                
            if '__' in basename:
                pmcid = basename.split('__')[0]
                pub_title = basename.split('__')[1]
            else:
                print(f"Basename does not contain '__': {basename}")

            if file_name.endswith('.xml'):
                file_format = 'xml'
                content = open(os.path.join(root, file_name), 'r', encoding='utf-8').read()
            elif file_name.endswith('.html'):
                file_format = 'html'
                content = open(os.path.join(root, file_name), 'r', encoding='utf-8').read()
            else:
                print(f"Skipping unsupported file format: {file_name}")
                continue

            if not pmcid:
                print(f"PMCID not found in filename: {file_name}. Skipping this file. PMCID was {pmcid}")

            files_df.append({
                'pub_title': pub_title,
                'file_name': file_name,
                'raw_cont': content,
                'format': file_format,
                'length': len(content),
                'path': os.path.join(root, file_name),
                'publication': pmcid.lower() if pmcid else None,
            })

    files_df = pd.DataFrame(files_df)
    print(f"Loaded {len(files_df)} files from {src_dir}")

    if os.path.exists(raw_HTML_data_filepath):
        print(f"File {raw_HTML_data_filepath} already exists. Loading existing DataFrame.")
        old_data = pd.read_parquet(raw_HTML_data_filepath)
        union_df = pd.concat([old_data, files_df]).drop_duplicates(subset=['file_name']).reset_index(drop=True).drop_duplicates()
        print(f"Combined DataFrame has {len(union_df)} unique entries after merging.")
    else:
        union_df = files_df
        print(f"No existing file found. Using loaded DataFrame with {len(union_df)} entries.")

    print(f"Saving DataFrame to {raw_HTML_data_filepath}")
    union_df.to_parquet(raw_HTML_data_filepath, index=False)

def PMID_to_doi(pmid,pmid_doi_mapping):
    if pmid in pmid_doi_mapping:
        return pmid_doi_mapping[pmid]

    else:
        print(f"PMID {pmid} not found in mapping file. Querying API...")

        base = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        params = {"tool": "mytool", "email": "myemail@example.com", "ids": pmid, "format": "json"}

        response = requests.get(base, params=params)

        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])
            if records and "doi" in records[0]:
                print(f"PMID: {pmid}, DOI: {records[0]['doi']}")
                pmid_doi_mapping[pmid] = records[0]["doi"]  # Store in mapping
                return records[0]["doi"]
            else:
                print(f"PMID: {pmid}, DOI: xxxx")  # No doi found
                return None  # No doi found
        else:
            print(f"API request failed for PMID {pmid}")
            return None  # Request failed

def url_to_doi(url,pmid_doi_mapping):
    if "dx.doi.org" in url:
        match = re.search(r'dx.doi.org/([a-zA-Z0-9./-]+)', url)
        if match:
            return match.group(1)
        else:
            print(f"DOI not found in {url}")
            return None
    elif "pubmed" in url:
        match = re.search(r'pubmed/([0-9]+)', url)
        if match:
            return PMID_to_doi(match.group(1),pmid_doi_mapping)
        else:
            print(f"PMID not found in {url}")
            return None

def PMID_to_url(pmid):
    base_url = "https://www.ncbi.nlm.nih.gov/pubmed/"
    return base_url + str(pmid)

def PMIDs_list_to_urls(pmids_list):
    if not isinstance(pmids_list, list):
        raise ValueError("Input must be a list of PMIDs!")
    if not all(isinstance(pmid, str) for pmid in pmids_list):
        raise ValueError("All elements in the list must be strings!")
    return ["https://www.ncbi.nlm.nih.gov/pubmed/" + pmid for pmid in pmids_list]

def url_to_pmid(url):
    if "pubmed" in url:
        match = re.search(r'pubmed/([0-9]+)', url)
        if match:
            return match.group(1)
        else:
            print(f"PMID not found in {url}")
            return None
    else:
        print(f"URL does not contain a PMID: {url}") if 'doi' not in url else None
        return None

def batch_PMID_to_doi(pmids, batch_size=100):
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    results = {}

    for i in range(0, len(pmids), batch_size):
        progress = i / len(pmids) * 100
        print(f"Processing batch {i}-{i + batch_size} ({progress:.2f}%)")
        batch = pmids[i:i + batch_size]  # Get a batch of PMIDs
        params = {"tool": "mytool", "email": "myemail@example.com", "ids": ",".join(batch), "format": "json"}

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])

            for record in records:
                pmid = record.get("pmid")
                doi = record.get("doi", None)  # Get DOI if available

                if pmid and pmid not in results:
                    results[pmid] = doi  # Store in dictionary

        else:
            print(f"API request failed for batch {i}-{i + batch_size}: {response.status_code}")

        time.sleep(0.1)  # Prevent hitting API rate limits (adjust as needed)

    return results


def batch_PMID_to_PMCID(pmids, batch_size=100):
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    results = {}

    for i in range(0, len(pmids), batch_size):
        progress = i / len(pmids) * 100
        print(f"Processing batch {i}-{i + batch_size} ({progress:.2f}%)")

        batch = pmids[i:i + batch_size]  # Get a batch of PMIDs
        params = {
            "tool": "mytool",
            "email": "myemail@example.com",
            "ids": ",".join(batch),  # Join PMIDs into a comma-separated string
            "format": "json"
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])

            for record in records:
                pmid = record.get("pmid")
                pmcid = record.get("pmcid", None)  # Get PMCID if available

                if pmid:
                    results[pmid] = pmcid  # Map PMID to PMCID

        else:
            print(f"API request failed for batch {i}-{i + batch_size}: {response.status_code}")

        time.sleep(0.1)  # Prevent hitting API rate limits (adjust as needed)

    return results

# pmid_doi_mapping update to get also the pmcids from the api call
def batch_doi_to_PMCID(ids, batch_size=150):
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    results = {}

    for i in range(0, len(ids), batch_size):
        progress = i / len(ids) * 100
        print(f"Processing batch {i}-{i + batch_size} ({progress:.2f}%)")

        batch = ids[i:i + batch_size]  # Get a batch of DOIs
        params = {
            "tool": "mytool",
            "email": "myemail@example.com",
            "ids": ",".join(batch),  # Join DOIs into a comma-separated string
            "format": "json"
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])

            for record in records:
                doi = record.get("doi")
                pmcid = record.get("pmcid", None)  # Get PMCID if available

                if doi:
                    results[doi] = pmcid  # Map DOI to PMCID

        else:
            print(f"API request failed for batch {i}-{i + batch_size}: {response.status_code}")

        time.sleep(0.1)  # Prevent hitting API rate limits (adjust as needed)

    return results

def batch_doi_to_PMID(dois, batch_size=150):
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    results = {}

    for i in range(0, len(dois), batch_size):
        progress = i / len(dois) * 100
        print(f"Processing batch {i}-{i + batch_size} ({progress:.2f}%)")

        batch = dois[i:i + batch_size]  # Get a batch of DOIs
        params = {
            "tool": "mytool",
            "email": "myemail@example.com",
            "ids": ",".join(batch),  # Join DOIs into a comma-separated string
            "format": "json"
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])

            for record in records:
                doi = record.get("doi")
                pmid = record.get("pmid", None)  # Get PMCID if available

                if doi:
                    results[doi] = pmid  # Map DOI to PMCID

        else:
            print(f"API request failed for batch {i}-{i + batch_size}: {response.status_code}")

        time.sleep(0.1)  # Prevent hitting API rate limits (adjust as needed)

    return results

def compare_dataframes(df1, df2):
    differences = {}
    print(df1.columns)
    print(df2.columns)
    # Column Differences
    differences['columns_only_in_df1'] = list(set(df1.columns) - set(df2.columns))
    differences['columns_only_in_df2'] = list(set(df2.columns) - set(df1.columns))

    # Row Count Differences
    differences['row_count_df1'] = df1.shape[0]
    differences['row_count_df2'] = df2.shape[0]

    # Row Differences (rows unique to each DataFrame)
    diff_rows = df1.merge(df2, indicator=True, how='outer').query('_merge != "both"')

    # Value Differences in Common Rows
    common_columns = list(set(df1.columns) & set(df2.columns))
    df1_common = df1[common_columns].sort_values(by=common_columns).reset_index(drop=True)
    df2_common = df2[common_columns].sort_values(by=common_columns).reset_index(drop=True)

    value_diff = df1_common.compare(df2_common) if not df1_common.equals(df2_common) else None

    return differences, diff_rows, value_diff

def fetch_GEO_data(IDs, request_url, start, stop):
    params = {
        "db": "gds",
        "id": ",".join(IDs[start:stop]),  # Query window
        "retmode": "json"
    }
    response = requests.get(request_url, params=params)

    try:
        data = response.json()
    except:
        raise ValueError("Failed to parse JSON response! Please check the response content.")

    return data


def extract_publication_ids_from_PX_export(filtered_df,
                                           pmid_doi_mapping, pmid_pmcid_mapping, doi_pmid_mapping, doi_pmcid_mapping,
                                           ret_missing_values=False):
    doi_pmid_none, doi_pmcid_none, pmid_doi_none, pmid_pmcid_none = [] , [] , [] , []

    for i, row in filtered_df.iterrows():
        publication_link = str(row['citing_publications_links'])
        if "www.ncbi.nlm.nih.gov/pubmed" in publication_link:
            pmid = publication_link.split('/')[-1]  # Extract PMID from URL
            pmcid = pmid_pmcid_mapping.get(pmid)  # Get PMCID from mapping
            if pmcid is None:
                pmid_pmcid_none.append(pmid)
            doi = pmid_doi_mapping.get(pmid)  # Get DOI from mapping
            if doi is None:
                pmid_doi_none.append(pmid)
        elif "dx.doi.org" in publication_link:
            doi = publication_link.split('dx.doi.org/')[-1]  # Extract DOI from URL
            doi = ''.join(doi)  # Join DOI parts
            pmid = doi_pmid_mapping.get(doi)
            if pmid is None:
                doi_pmid_none.append(doi)
            pmcid = doi_pmcid_mapping.get(doi)
            if pmcid is None:
                doi_pmcid_none.append(doi)
        else:
            print(f"Unknown link format: {publication_link} of type {type(publication_link)}")

        filtered_df.at[i, 'PMID'] = pmid
        filtered_df.at[i, 'PMCID'] = pmcid
        filtered_df.at[i, 'DOI'] = doi

    if ret_missing_values:
        return filtered_df, doi_pmid_none, doi_pmcid_none, pmid_doi_none, pmid_pmcid_none

    else:
        return filtered_df


def add_citing_publication_link_columns1(dataframe):
    dataframe["citing_publications_links"] = dataframe.apply(lambda row: [
        f"https://dx.doi.org/{row['DOI']}" if pd.notna(row["DOI"]) else None,
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{row['PMCID']}" if pd.notna(row["PMCID"]) else None
    ], axis=1)

    # Remove None values from lists
    dataframe["citing_publications_links"] = dataframe["citing_publications_links"].apply(
        lambda x: [link for link in x if link is not None])

    # Explode to create multiple rows for each publication link
    dataframe = dataframe.explode("citing_publications_links", ignore_index=True)
    # Remove Nan values
    dataframe = dataframe[dataframe['citing_publications_links'].astype(bool)]  # Drop rows with empty lists

    # Display result
    print(len(dataframe))
    return dataframe

def add_citing_publication_link_columns(dataframe):
    def create_links(row):
        #print(f"Processing row {row.name}: DOI={row['DOI']}, PMCID={row['PMCID']}")  # Debugging

        # Ensure values are strings before using them
        doi_link = f"https://dx.doi.org/{row['DOI']}" if isinstance(row["DOI"], str) else None
        pmcid_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{row['PMCID']}" if isinstance(row["PMCID"], str) else None

        return [link for link in [doi_link, pmcid_link] if link is not None]

    # Apply transformation
    dataframe["citing_publications_links"] = dataframe.apply(create_links, axis=1)

    # Print before exploding to check structure
    print("Before explode sample:", dataframe[["DOI", "PMCID", "citing_publications_links"]].head(10))

    # Explode to create multiple rows if there are multiple links
    dataframe = dataframe.explode("citing_publications_links", ignore_index=True)

    # Drop rows with empty or NaN values in the citing_publications_links column
    dataframe = dataframe.dropna(subset=["citing_publications_links"])

    # Display final row count after processing
    print(f"Final dataframe length: {len(dataframe)}")

    return dataframe

def df_summary(dataframe):
    # Summary of Missing Values & Unique Counts
    summary_stats = pd.DataFrame({
        "Missing Values": dataframe.isna().sum(),
        "Unique Values": dataframe.nunique()
    })

    # Summary for Numeric Columns (if any exist)
    numeric_summary = dataframe.describe()

    # Display results
    print("Column Summary:")
    print(summary_stats)

    print("\nNumeric Column Summary:")
    print(numeric_summary)

def add_example_to_merged_df(row, raw_html):
    # handle uid also when comma-separated, then split and extract smallest element
    if 'identifier' in row:
        uid = row['identifier']
    elif 'dataset_uid' in row:
        uid = row['dataset_uid']
    if ',' in uid:
        uids = uid.split(',')
        elements = []
        for uid in uids:
            elm_i = extract_all_elements_with_UID(raw_html, uid)
            if elm_i in elements: # no dupes
                continue
            else:
                elements.append(elm_i)
        return elements
    else:
        return extract_all_elements_with_UID(raw_html, uid)


def extract_all_elements_with_UID(source_html, uid):
    print(f"Extracting elements with UID: {uid}")

    soup = BeautifulSoup(source_html, "html.parser")

    matching_elements = []

    for p in soup.find_all(["table", "p"]):  # Find only <p> elements
        text = p.get_text(strip=True)

        if re.search(uid, text, re.IGNORECASE):  # Check if UID is in the text
            matching_elements.append((str(p), len(text)))  # Store element and length

    # If multiple matches, return the **smallest** one
    if matching_elements:
        # smallest_p, _ = min(matching_elements, key=lambda x: x[1])  # Find smallest
        return matching_elements  # Pretty-print the raw HTML for debugging # smallest_p

    return [None]  # No match found

def evaluate_performance(predict_df, ground_truth, orchestrator, false_positives_file, false_negatives_file=None,
                         repo_return=False, gt_base=None):
    """ Evaluates dataset extraction performance using precision, recall, and F1-score. """

    recall_list, false_positives_output, false_negatives_output = [], [], []
    total_precision, total_recall, num_sources = 0, 0, 0

    if gt_base is None:
        gt_base = predict_df['source_url'].unique()

    # Pre-build GT index: pmcid (lower) → set of identifiers — avoids O(n) scan per article
    gt_index = {}
    for pmcid, identifier in zip(ground_truth['pmcid'].str.lower(), ground_truth['identifier'].fillna('').str.lower()):
        if pmcid not in gt_index:
            gt_index[pmcid] = set()
        for id_part in identifier.split(','):
            id_part = id_part.strip()
            if id_part:
                gt_index[pmcid].add(id_part)

    # Pre-build predict index: pmcid → (set of identifiers, subset df for FP lookup)
    predict_df = predict_df.copy()
    # .astype(str) first — an empty predict_df (a config with zero predictions) leaves
    # these columns as float64, and .str accessor raises on non-string dtype regardless
    # of row count; casting up front keeps this correct for both the empty and non-empty case.
    predict_df['_pub_id'] = (
        predict_df['source_url'].astype(str).replace('nan', '')
        .str.lower()
        .str.extract(r'(pmc\d+)', expand=False)
    )
    predict_df['_id_lower'] = predict_df['dataset_identifier'].astype(str).replace('nan', '').str.lower()
    predict_index = {
        pub_id: grp
        for pub_id, grp in predict_df.groupby('_pub_id', sort=False)
    }

    t_start = time.time()
    n_total = len(gt_base)

    for i, source_page in enumerate(gt_base):
        pub_id = source_page.split('/')[-1].lower() if not source_page.endswith('/') else source_page.split('/')[-2].lower()

        if i % 500 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_total - i) / rate if rate > 0 else float('inf')
            orchestrator.logger.info(
                f"[eval] {i}/{n_total} articles ({i/n_total:.0%})  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )

        orchestrator.logger.debug(f"Evaluating pub_id: {pub_id}")
        gt_datasets = gt_index.get(pub_id, set())

        orchestrator.logger.debug(f"# of elements in gt_data: {len(gt_datasets)}. Element IDs: {gt_datasets}")

        num_sources += 1

        # O(1) lookup via pre-built index
        eval_data = predict_index.get(pub_id, predict_df.iloc[0:0])
        eval_datasets = set(eval_data['_id_lower'].dropna())
        # Remove invalid entries
        eval_datasets.discard('n/a')
        eval_datasets.discard('')

        orchestrator.logger.info(f"# of Extracted Datasets: {len(eval_datasets)}. Evaluation datasets: {eval_datasets}")

        # Handle cases where both ground truth and evaluation are empty
        if not gt_datasets and not eval_datasets or (len(gt_datasets) == 0 and len(eval_datasets) == 0):
            orchestrator.logger.info("No datasets in both ground truth and evaluation. Perfect precision and recall.")
            total_precision += 1
            total_recall += 1
            continue

        # Match Extraction Logic
        matched_gt, matched_eval = set(), set()

        # Exact Matches
        exact_matches = gt_datasets & eval_datasets  # Intersection of ground truth and extracted datasets
        matched_gt.update(exact_matches)
        matched_eval.update(exact_matches)

        # Partial Matches (Aliased Identifiers)
        for eval_id in eval_datasets - matched_eval:
            for gt_id in gt_datasets:
                orchestrator.logger.debug(f"Comparing eval_id='{eval_id}' with gt_id='{gt_id}'")
                if eval_id in gt_id or gt_id in eval_id:
                    orchestrator.logger.info(f"Partial or alias match found: eval_id={eval_id}, gt_id={gt_id}")
                    matched_gt.add(gt_id)
                    matched_eval.add(eval_id)
                    break
                else:
                    orchestrator.logger.debug(f"No match: eval_id='{eval_id}' gt_id='{gt_id}'")

        # **False Positives (Unmatched extracted datasets)**
        FP = eval_datasets - matched_eval
        false_positives_output.extend([false_p, eval_data[eval_data['_id_lower'] == false_p]['data_repository'].values[0] if 'data_repository' in eval_data.columns and len(eval_data[eval_data['_id_lower'] == false_p]) > 0 else 'unknown', pub_id] for false_p in FP)

        # **False Negatives (Unmatched ground truth datasets)**
        FN = gt_datasets - matched_gt
        false_negatives_output.extend((FN, pub_id)) if len(FN) > 0 else None

        # **Precision and Recall Calculation**
        true_positives = len(matched_gt)
        false_positives = len(FP)
        false_negatives = len(FN)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        if true_positives + false_negatives == 0:
            orchestrator.logger.info(f"No ground truth datasets for {source_page}. Setting recall to 1.")
            recall = 1.0

        orchestrator.logger.info(f"Precision for {source_page}: {precision}")
        orchestrator.logger.info(f"Recall for {source_page}: {recall}")

        if recall == 0:
            recall_list.append(source_page)

        # Accumulate totals
        total_precision += precision
        total_recall += recall

    # **Compute Overall Metrics**
    average_precision = total_precision / num_sources if num_sources > 0 else 0
    average_recall = total_recall / num_sources if num_sources > 0 else 0
    f1_score = (
        2 * (average_precision * average_recall) / (average_precision + average_recall)
        if (average_precision + average_recall) > 0
        else 0
    )

    orchestrator.logger.info(f"\nPerformance evaluation completed for {num_sources} source pages.")

    # **Save false positives**
    with open(false_positives_file, 'w') as f:
        for item in false_positives_output:
            f.write("%s\n" % item)

    if false_negatives_file:
        with open(false_negatives_file, 'w') as f:
            for item in false_negatives_output:
                f.write("%s\n" % item)

    return {
        "average_precision": average_precision,
        "average_recall": average_recall,
        "f1_score": f1_score
    }

#
# filtered_df = filtered_df.explode('citing_publications_links', ignore_index=True)  # Split lists into rows
# print(f"Length: {len(filtered_df)}")
#
# doi_pmid_none = []
# doi_pmcid_none = []
# pmid_doi_none = []
# pmid_pmcid_none = []
#
# for i, row in filtered_df.iterrows():
#     publication_link = str(row['citing_publications_links'])
#     if "www.ncbi.nlm.nih.gov/pubmed" in publication_link:
#         pmid = publication_link.split('/')[-1]  # Extract PMID from URL
#         pmcid = pmid_pmcid_mapping.get(pmid)  # Get PMCID from mapping
#         if pmcid is None:
#             pmid_pmcid_none.append(pmid)
#         doi = pmid_doi_mapping.get(pmid)  # Get DOI from mapping
#         if doi is None:
#             pmid_doi_none.append(pmid)
#     elif "dx.doi.org" in publication_link:
#         doi = publication_link.split('dx.doi.org/')[-1]  # Extract DOI from URL
#         doi = ''.join(doi)  # Join DOI parts
#         pmid = doi_pmid_mapping.get(doi)
#         if pmid is None:
#             doi_pmid_none.append(doi)
#         pmcid = doi_pmcid_mapping.get(doi)
#         if pmcid is None:
#             doi_pmcid_none.append(doi)
#     else:
#         print(f"Unknown link format: {publication_link} of type {type(publication_link)}")
#
#     filtered_df.at[i, 'PMID'] = pmid
#     filtered_df.at[i, 'PMCID'] = pmcid
#     filtered_df.at[i, 'DOI'] = doi
#
# filtered_df[['citing_publications_links','DOI','PMID','PMCID']].sample(10)
# #%%
# print(len(doi_pmid_none), len(doi_pmcid_none), len(pmid_doi_none), len(pmid_pmcid_none) )
# #%%
# results = batch_PMID_to_doi(pmid_doi_none, batch_size=20)
# added = 0
# for key, value in results.items():
#     if key not in pmid_doi_mapping or pmid_doi_mapping[key] is None:
#         pmid_doi_mapping[key] = value
#         added += 1
#
# added
# #%%
# results = batch_PMID_to_PMCID(pmid_pmcid_none, batch_size=20)
# added = 0
# for key, value in results.items():
#     if key not in pmid_pmcid_mapping or pmid_pmcid_mapping[key] is None:
#         pmid_pmcid_mapping[key] = value
#         added += 1
# added
# #%%
# results = batch_doi_to_PMID(doi_pmid_none, batch_size=3)
# added = 0
# for key, value in results.items():
#     if key not in doi_pmid_mapping or doi_pmid_mapping[key] is None:
#         doi_pmid_mapping[key] = value
#         added += 1
#
# added
# #%%
# doi_pmid_mapping
# #%%
# results = batch_doi_to_PMCID(doi_pmcid_none, batch_size=20)
# added = 0
# for key, value in results.items():
#     if key not in doi_pmcid_mapping or doi_pmcid_mapping[key] is None:
#         doi_pmcid_mapping[key] = value
#         added += 1
#
# added
# #%%
# # save to file
# with open(pmid_doi_mapping_file, "w") as f:
#     json.dump(pmid_doi_mapping, f, indent=4) # Save to JSON
# #%%
# # save to file
# with open(pmid_pmcid_mapping_file, "w") as f:
#     json.dump(pmid_pmcid_mapping, f, indent=4) # Save to JSON
# #%%
# # save to file
# with open(doi_pmid_mapping_file, "w") as f:
#     json.dump(doi_pmid_mapping, f, indent=4) # Save to JSON
# #%%
# # save to file
# with open(doi_to_pmcid_mapping_file, "w") as f:
#     json.dump(doi_pmcid_mapping, f, indent=4) # Save to JSON
# #%% md

def parse_job_duration_from_log(run_log_path):
    """
    Get a job's actual wall-clock duration for free from k8s_processor.py's own
    run.log timestamps (LOG_FMT = "%(asctime)s - ..."), instead of inferring it
    from how many times a sampler happened to poll during the run.

    Args:
        run_log_path (str): Path to a run.log file.

    Returns:
        float: duration in hours, from the first to the last logged timestamp.
    """
    ts_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
    first_ts, last_ts = None, None
    with open(run_log_path) as f:
        for line in f:
            m = ts_pattern.match(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f')
            if first_ts is None:
                first_ts = ts
            last_ts = ts

    if first_ts is None or last_ts is None:
        raise ValueError(f"No timestamped log lines found in {run_log_path}")

    return (last_ts - first_ts).total_seconds() / 3600


def compute_gpu_energy_wh(power_log_csv, sample_interval_s=5, duration_hours=None):
    """
    Estimate GPU energy consumption from an nvidia-smi power draw log, as produced by
    k8s/deterministic-batch-job-template.yaml's background sampler:
    `nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l <sample_interval_s>`.

    By default this integrates power over the sampler's own sample count
    (n_samples * sample_interval_s), which requires polling often enough to cover
    the whole job — and frequent polling can itself perturb latency-sensitive GPU
    workloads (see parse_job_duration_from_log for a lower-overhead alternative).

    If duration_hours is supplied (e.g. from parse_job_duration_from_log, which reads
    it for free from run.log's own timestamps), energy is instead computed as
    mean(power) * duration_hours — letting the power log be sampled sparsely (just
    enough for a representative average) without needing to track wall-clock time
    via polling frequency at all.

    Args:
        power_log_csv (str or list[str]): Path to a gpu_power.csv file, or a list of
            paths (e.g. one per k8s job slice) to sum energy across a multi-GPU run.
        sample_interval_s (int): Seconds between samples. Only used when duration_hours
            is not supplied. Must match the -l value used when the log was recorded
            (default in the job template is 5s).
        duration_hours (float or list[float], optional): Actual job duration(s) (e.g.
            from parse_job_duration_from_log). If given, energy = mean(power) *
            duration_hours per file, summed across files, instead of the sample-count
            based integration.

    Returns:
        dict with 'n_samples', 'duration_min', 'energy_wh', 'energy_kwh'.
    """
    paths = [power_log_csv] if isinstance(power_log_csv, str) else list(power_log_csv)
    durations = None
    if duration_hours is not None:
        durations = [duration_hours] * len(paths) if isinstance(duration_hours, (int, float)) else list(duration_hours)
        if len(durations) != len(paths):
            raise ValueError("duration_hours list must match the number of power_log_csv paths")

    total_energy_wh = 0.0
    total_samples = 0
    total_duration_hours = 0.0
    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        power_col = next((c for c in df.columns if c.startswith('power.draw')), None)
        if power_col is None:
            raise ValueError(f"No 'power.draw' column found in {path}. Columns: {list(df.columns)}")

        power_w = df[power_col].astype(str).str.replace(' W', '', regex=False).astype(float)
        total_samples += len(df)

        if durations is not None:
            total_energy_wh += power_w.mean() * durations[i]
            total_duration_hours += durations[i]
        else:
            total_energy_wh += power_w.sum() * sample_interval_s / 3600
            total_duration_hours += len(df) * sample_interval_s / 3600

    return {
        'n_samples': total_samples,
        'duration_min': total_duration_hours * 60,
        'energy_wh': total_energy_wh,
        'energy_kwh': total_energy_wh / 1000,
    }
