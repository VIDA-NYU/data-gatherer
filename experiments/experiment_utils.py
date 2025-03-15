import re
import time
import requests
import pandas as pd

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
