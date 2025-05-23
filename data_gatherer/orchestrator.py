import logging

import numpy as np
import requests
from data_gatherer.logger_setup import setup_logging
from data_gatherer.data_fetcher import *
from data_gatherer.parser import RuleBasedParser, LLMParser
from data_gatherer.classifier import LLMClassifier
import json
from data_gatherer.selenium_setup import create_driver
import pandas as pd
import cloudscraper
import time
from data_gatherer.resources_loader import load_config
import ipywidgets as widgets
from IPython.display import display, clear_output
import textwrap

class Orchestrator:
    """
    This class orchestrates the data gathering process by coordinating the data fetcher, parser, and classifier in a
    single workflow.
    """
    def __init__(self, config_path='config.json', log_file_override=None):
        """
        Initializes the Orchestrator with the given configuration file and sets up logging.

        :param config_path: Path to the configuration file.
        """
        self.config = load_config(config_path)
        self.parser_config = load_config(self.config['parser_config_path'])
        log_file = log_file_override or self.config.get('log_file', 'logs/scraper.log')
        self.logger = setup_logging('orchestrator', log_file)
        self.classifier = LLMClassifier(self.config['retrieval_patterns'], self.logger)
        self.data_fetcher = None
        self.parser = None
        self.raw_data_format = None
        self.data_checker = DataCompletenessChecker(self.config, self.logger)
        self.full_DOM = self.parser_config['llm_model'] in self.parser_config['entire_document_models'] and self.parser_config['process_entire_document']
        self.logger.info(f"Data_Gatherer Orchestrator initialized. Extraction step Model: {self.parser_config['llm_model']}")
        self.downloadables = []

    def setup_data_fetcher(self):
        """
        Sets up either a web scraper or API client based on the config.
        """
        self.logger.debug("Setting up data fetcher...")

        # Close previous driver if exists
        if hasattr(self, 'data_fetcher') and hasattr(self.data_fetcher, 'scraper_tool'):
            try:
                self.data_fetcher.scraper_tool.quit()
                self.logger.info("Previous driver quit.")
            except Exception as e:
                self.logger.warning(f"Failed to quit previous driver: {e}")

        if self.config['search_method'] == 'url_list' and self.config['dataframe_fetch']:
            self.data_fetcher = DatabaseFetcher(self.config, self.logger)
            return

        elif self.config['search_method'] == 'url_list':
            self.data_fetcher = WebScraper(None, self.config, self.logger)

        elif self.config['search_method'] == 'cloudscraper':
            driver = cloudscraper.create_scraper()
            self.data_fetcher = WebScraper(driver, self.config, self.logger)

        elif self.config['search_method'] == 'google_scholar':
            driver = create_driver(self.config['DRIVER_PATH'], self.config['BROWSER'], self.config['HEADLESS'], self.logger)
            self.data_fetcher = WebScraper(driver, self.config, self.logger)

        else:
            raise ValueError(f"Invalid search method: {self.config['search_method']}")

        self.logger.info("Data fetcher setup completed.")

        return self.data_fetcher.scraper_tool


    def process_url(self, url, save_staging_table=False):
        """
        Orchestrates the process for a single given source URL (publication).
        param url: The URL to process.
        param save_staging_table: Flag to save the staging table.
        """
        self.logger.info(f"Processing URL: {url}")
        self.current_url = url
        self.publisher = self.data_fetcher.url_to_publisher_domain(url)
        self.local_data = self.config['dataframe_fetch']

        if not self.local_data:
            self.data_fetcher = self.data_fetcher.update_DataFetcher_settings(url, self.full_DOM, self.logger)
            # Step 1: Use DataFetcher (WebScraper or APIClient) to fetch raw data
            self.logger.debug(f"data_fetcher.fetch_source = {self.data_fetcher.fetch_source}")

        try:
            self.logger.debug("Fetching Raw content...")
            raw_data = None
            parsed_data = None
            additional_data = None

            # if model processes the entire document, fetch the entire document and go to the parsing step
            if self.full_DOM:
                self.logger.info("Fetching entire document for processing.")
                self.raw_data_format = "full_HTML"
                raw_data = self.data_fetcher.fetch_data(url)
                raw_data = self.data_fetcher.remove_cookie_patterns(raw_data)

            # if model processes chunks of the document, fetch the relevant sections and go to the parsing step
            else:

                if "API" in self.data_fetcher.fetch_source:
                    self.logger.debug(f"Using {self.data_fetcher.fetch_source} to fetch data.")
                    self.raw_data_format = "XML"
                    self.config['search_method'] = 'api'
                else:
                    self.logger.debug("Using WebScraper to fetch data.")
                    self.raw_data_format = "HTML"

                raw_data = self.data_fetcher.fetch_data(url)
                self.logger.info(f"Raw data fetched from source: {self.data_fetcher.fetch_source}")

                if "API" not in self.data_fetcher.fetch_source:
                    raw_data = self.data_fetcher.scraper_tool.page_source

                elif not self.data_checker.is_xml_data_complete(raw_data, url):
                    self.raw_data_format = "HTML"
                    self.parser_mode = "LLMParser"
                    self.logger.info(f"Fallback to HTML data fetcher for {url}.")
                    self.data_fetcher = self.data_fetcher.update_DataFetcher_settings(url, self.full_DOM, self.logger, HTML_fallback=True)
                    raw_data = self.data_fetcher.fetch_data(url)
                    raw_data = self.data_fetcher.remove_cookie_patterns(raw_data)

            if self.config['write_htmls_xmls']:
                directory = self.config['html_xml_dir'] + self.publisher + '/'
                self.logger.info(f"Raw Data is {self.raw_data_format}.")
                if self.raw_data_format == "HTML" or self.raw_data_format == "full_HTML":
                    self.data_fetcher.download_html(directory, self.raw_data_format)
                    self.logger.info(f"Raw HTML saved to: {directory}")
                elif self.raw_data_format == "XML":
                    self.data_fetcher.download_xml(directory, raw_data)
                    self.logger.info(f"Raw XML saved in {directory} directory")

            self.logger.info("Successfully fetched Raw content.")

            # Step 2: Use RuleBasedParser to parse and extract HTML elements and rule-based matches
            if self.raw_data_format == "HTML" and self.parser_mode == "RuleBasedParser":
                self.logger.info("Using RuleBasedParser to parse data.")
                self.parser = RuleBasedParser(self.config['parser_config_path'], self.logger)
                parsed_data = self.parser.parse_data(raw_data, self.publisher, self.current_url)

                parsed_data['rule_based_classification'] = 'n/a'
                self.logger.info(f"Parsed data extraction completed. Links collected: {len(parsed_data)}")
                rule_based_matches = self.data_fetcher.get_rule_based_matches(self.publisher)
                #            print(f"rule_based_matches pre: {rule_based_matches}")
                #            rule_based_matches = self.data_fetcher.normalize_links(rule_based_matches)
                #            print(f"rule_based_matches post: {rule_based_matches}")

                self.logger.info(f"rule_based_matches: {rule_based_matches}")
                # create a new null column in the parsed_data DataFrame to store the rule_based_matches
                # iterate through the rule_based_matches and update the parsed_data DataFrame
                for key, value in rule_based_matches.items():
                    self.logger.debug(f"key: {key}, value: {value}")
                    parsed_data.loc[parsed_data['link'].str.contains(key), 'rule_based_classification'] = value

                parsed_data.to_csv('staging_table/parsed_data.csv', index=False) if save_staging_table else None

            elif self.raw_data_format == "XML" and raw_data is not None:
                self.logger.info("Using LLMParser to parse data.")
                self.parser = LLMParser(self.config['parser_config_path'], self.logger)

                if additional_data is None:
                    parsed_data = self.parser.parse_data(raw_data, self.publisher, self.current_url)

                else:
                    self.logger.info(f"Processing additional data. # of items: {len(additional_data)}")
                    # add the additional data to the parsed_data
                    add_data = self.parser.parse_data(raw_data, self.publisher, self.current_url,
                                                         additional_data=additional_data)
                    self.logger.info(type(add_data))

                    parsed_data = pd.concat([parsed_data, add_data], ignore_index=True).drop_duplicates()

                parsed_data['source_url'] = url
                self.logger.info(f"Parsed data extraction completed. Elements collected: {len(parsed_data)}")
                if self.logger.level == logging.DEBUG:
                    parsed_data.to_csv('staging_table/parsed_data_from_XML.csv', index=False) if save_staging_table else None

            elif self.raw_data_format == "full_HTML" or self.parser_mode == "LLMParser":
                self.logger.info("Using LLMParser to parse data.")
                self.parser = LLMParser(self.config['parser_config_path'], self.logger)
                parsed_data = self.parser.parse_data(raw_data, self.publisher, self.current_url, raw_data_format="full_HTML")
                parsed_data['source_url'] = url
                self.logger.info(f"Parsed data extraction completed. Elements collected: {len(parsed_data)}")
                if self.logger.level == logging.DEBUG:
                    parsed_data.to_csv('staging_table/parsed_data_from_XML.csv', index=False) if save_staging_table else None

            else:
                self.logger.error(f"Unsupported raw data format: {self.raw_data_format}. Cannot parse data.")
                return None

            self.logger.info("Raw Data parsing completed.")

            # skip unstructured files and filter out unstructured files from dataframe
            if self.config['skip_unstructured_files'] and 'file_extension' in parsed_data.columns:
                for ext in self.config['skip_file_extensions']:
                    # filter out unstructured files
                    parsed_data = parsed_data[parsed_data['file_extension'] != ext]

            # Step 3: Use Classifier to classify Extracted and Parsed elements
            if parsed_data is not None:
                if self.raw_data_format == "HTML" and self.parser_mode != "LLMParser":
                    classified_links = self.classifier.classify_anchor_elements_links(parsed_data)
                    self.logger.info("Link classification completed.")
                elif self.raw_data_format == "XML":
                    self.logger.info("XML element classification not needed. Using parsed_data.")
                    classified_links = parsed_data
                elif self.raw_data_format == "full_HTML":
                    classified_links = parsed_data
                    self.logger.info("Full HTML element classification not supported. Using parsed_data.")
                elif self.parser_mode == "LLMParser":
                    classified_links = parsed_data
                    self.logger.info("Full HTML element classification not supported. Using parsed_data.")
                else:
                    self.logger.error(f"Unsupported raw data format and parser mode combination.")
                    return None
            else:
                raise ValueError("Parsed data is None. Cannot classify links.")

            # add the deduplication step here
            #classified_links = self.classifier.deduplicate_links(classified_links)

            return classified_links

        except Exception as e:
            self.logger.error(f"Error processing URL {url}: {e}", exc_info=True)
            return None

    def deduplicate_links(self, classified_links):
        """
        Deduplicates the classified links based on the link / download_link itself. If two entry share the same link
        or if download link of record A is the same as link of record B, merge rows.

        :param classified_links: DataFrame of classified links.
        """
        self.logger.info(f"Deduplicating {len(classified_links)} classified links.")
        classified_links['link'] = classified_links['link'].str.strip()
        classified_links['download_link'] = classified_links['download_link'].str.strip()

        # Deduplicate based on link column
        #classified_links = classified_links.drop_duplicates(subset=['link', 'download_link'], keep='last')

        self.logger.info(f"Deduplication completed. {len(classified_links)} unique links found.")
        return classified_links

    def process_urls(self, url_list, log_modulo=10):
        """
        Processes a list of URLs and returns classified data.

        :param url_list: List of URLs to process.

        :param log_modulo: Frequency of logging progress (useful when url_list is long).
        """
        self.logger.debug("Starting to process URL list...")
        start_time = time.time()
        total_iters = len(url_list)
        results = {}

        for iteration, url in enumerate(url_list):
            self.logger.info(f"{iteration}th function call: self.process_url({url})")
            results[url] = self.process_url(url)

            if iteration % log_modulo == 0:
                elapsed = time.time() - start_time  # Time elapsed since start
                avg_time_per_iter = elapsed / (iteration + 1)  # Average time per iteration
                remaining_iters = total_iters - (iteration + 1)
                estimated_remaining = avg_time_per_iter * remaining_iters  # Estimated time remaining
                self.logger.info(
                    f"\nProgress: {iteration+1}/{total_iters} ({(iteration+1)/total_iters*100:.2f}%) "
                    f"| Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} "
                    f"| ETA: {time.strftime('%H:%M:%S', time.gmtime(estimated_remaining))}\n"
                )
        self.logger.debug("Completed processing all URLs.")
        return results

    def load_urls_from_config(self):
        """
        Loads URLs from the input file specified in the config.
        """
        self.logger.debug(f"Loading URLs from file: {self.config['input_urls_filepath']}")
        try:
            with open(self.config['input_urls_filepath'], 'r') as file:
                url_list = [line.strip() for line in file]
            self.logger.info(f"Loaded {len(url_list)} URLs from file.")
            self.url_list = url_list
            return url_list
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Create file with input links! File not found: {self.config['input_urls_filepath']}\n\n{e}\n")

    def get_data_preview(self, combined_df, display_type='console', interactive=True, return_metadata=False):
        """
        Shows user a preview of the data they are about to download.
        -- future release
        """
        self.already_previewed = []
        self.metadata_parser = LLMParser(self.config['parser_config_path'], self.logger)
        self.data_fetcher = self.data_fetcher.update_DataFetcher_settings('any_url', self.full_DOM, self.logger)

        if isinstance(self.data_fetcher, WebScraper):
            self.logger.info("Found WebScraper to fetch data.")

        if return_metadata:
            ret_list = []

        for i, row in combined_df.iterrows():
            self.logger.info(f"Row # {i}")
            self.logger.debug(f"Row keys: {row}")

            dataset_webpage = row.get('dataset_webpage', None)
            download_link = row.get('download_link', None)

            if dataset_webpage is None and download_link is None:
                self.logger.info(f"Row {i} does not contain 'dataset_webpage' or 'download_link'. Skipping...")
                continue

            # skip if already added
            if (dataset_webpage is not None and dataset_webpage in self.already_previewed) or (
                    download_link is not None and download_link in self.already_previewed):
                self.logger.info(f"Duplicate dataset. Skipping...")
                continue

            # identify those that may be datasets
            if dataset_webpage is None or not isinstance(dataset_webpage, str) or len(dataset_webpage) <= 5:
                if (row.get('file_extension', None) is not None and 'data' not in row['source_section'] and row['file_extension'] not
                        in ['xlsx', 'csv', 'json', 'xml', 'zip']):
                    self.logger.info(f"Skipping row {i} as it does not contain a valid dataset webpage or file extension.")
                    continue
                else:
                    self.logger.info(f"Potentially a valid dataset, displaying hardscraped metadata")
                    #metadata = self.metadata_parser.parse_metadata(row['source_section'])
                    hardscraped_metadata = {k:v for k,v in row.items() if v is not None and v not in ['nan', 'None', '', 'n/a', np.nan, 'NaN', 'na']}
                    self.already_previewed.append(download_link)
                    if self.config['download_data_for_description_generation']:
                        split_source_url = hardscraped_metadata.get('source_url').split('/')
                        paper_id = split_source_url[-1] if len(split_source_url[-1]) > 0 else split_source_url[-2]
                        self.data_fetcher.download_file_from_url(download_link, "output/suppl_files", paper_id)
                        hardscraped_metadata['data_description_generated'] = self.metadata_parser.generate_dataset_description(download_link)
                    self.display_data_preview(hardscraped_metadata, display_type=display_type, interactive=interactive)
                    continue

            else:
                self.logger.info(f"LLM scraped metadata")
                repo_mapping_key = row['repository_reference'].lower() if 'repository_reference' in row else row['data_repository'].lower()
                resolved_key = self.parser.resolve_data_repository(repo_mapping_key)
                if ('javascript_load_required' in self.parser_config['repos'][resolved_key]):
                    self.logger.info(f"JavaScript load required for {repo_mapping_key} dataset webpage. Using WebScraper.")
                    html = self.data_fetcher.fetch_data(row['dataset_webpage'], delay=3.5)
                    if "informative_html_metadata_tags" in self.parser_config['repos'][resolved_key]:
                        html = self.data_fetcher.normalize_HTML(html, self.parser_config['repos'][resolved_key]['informative_html_metadata_tags'])
                    if self.config['write_raw_metadata']:
                        self.logger.info(f"Saving raw metadata to: {self.config['html_xml_dir']+ 'raw_metadata/'}")
                        self.data_fetcher.download_html(self.config['html_xml_dir'] + 'raw_metadata/')
                else:
                    html = requests.get(row['dataset_webpage']).text
                metadata = self.metadata_parser.parse_metadata(html)
                metadata['source_url_for_metadata'] = row['dataset_webpage']
                metadata['access_mode'] = row.get('access_mode', None)
                metadata['source_section'] = row.get('source_section', row.get('section_class', None))
                metadata['download_link'] = row.get('download_link', None)
                metadata['accession_id'] = row.get('dataset_id', row.get('dataset_identifier', None))
                metadata['data_repository'] = repo_mapping_key
                self.already_previewed.append(row['dataset_webpage'])

            metadata['paper_with_dataset_citation'] = row['source_url']

            if return_metadata:
                ret_list.append(metadata)

            self.display_data_preview(metadata, display_type=display_type, interactive=interactive)

        return ret_list if return_metadata else None

    def flatten_json(self, y, parent_key='', sep='.'):
        """
        Flatten nested JSON into dot notation with list index support.
        """
        items = []
        if isinstance(y, dict):
            for k, v in y.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(self.flatten_json(v, new_key, sep=sep))
        elif isinstance(y, list):
            for i, v in enumerate(y):
                new_key = f"{parent_key}[{i}]"
                items.extend(self.flatten_json(v, new_key, sep=sep))
        else:
            items.append((parent_key, y))
        return items

    def display_data_preview(self, metadata, display_type='console', interactive=True):
        """
        Display extracted metadata as a clean table in both Jupyter and terminal environments.
        -- future release
        """
        self.logger.info("Displaying metadata preview")

        if not isinstance(metadata, dict):
            self.logger.warning("Metadata is not a dictionary. Cannot display properly.")
            return

        if not interactive:
            self.logger.info("Skipping interactive preview. Change the interactive flag to True to enable.")
            return

        if display_type == 'console':
            # Prepare rows
            rows = []
            flat_metadata = []
            for key, value in metadata.items():
                if value is not None and str(value).strip() not in ['nan', 'None', '', 'NaN', 'na', 'unavailable', '0']:
                    if isinstance(value, (dict, list)):
                        flat_metadata.extend(self.flatten_json(value, parent_key=key))
                    else:
                        flat_metadata.append((key, value))

            for key, value in flat_metadata:
                pretty_val = str(value)
                wrapped_lines = textwrap.wrap(pretty_val, width=80) or [""]
                rows.append((key.strip(), wrapped_lines))

            if not rows:
                preview = "No usable metadata found."
            else:
                # Compute dynamic widths
                max_key_len = max(len(k) for k, _ in rows)
                sep = f"+{'-' * (max_key_len + 2)}+{'-' * 80}+"
                lines = [sep]
                lines.append(f"| {'Field'.ljust(max_key_len)} | {'Value'.ljust(80)} |")
                lines.append(sep)
                for key, wrapped in rows:
                    lines.append(f"| {key.ljust(max_key_len)} | {wrapped[0].ljust(80)} |")
                    for cont in wrapped[1:]:
                        lines.append(f"| {' '.ljust(max_key_len)} | {cont.ljust(80)} |")
                lines.append(sep)
                preview = "\n".join(lines)

            # Final question to user
            user_input = input(
                f"\nDataset preview:\n{preview}\n\nDo you want to proceed with downloading this dataset? [y/N]: "
            ).strip().lower()

            if user_input not in ["y", "yes"]:
                self.logger.info("User declined to download the dataset.")
            else:
                self.downloadables.append(metadata)
                self.logger.info("User confirmed download. Proceeding...")

        elif display_type == 'ipynb':

            # Clean and prepare rows
            rows = []
            for key, value in metadata.items():
                if value and str(value).strip() not in ['nan', 'None', '', 'NaN', 'na', 'unavailable', '0']:
                    val_str = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
                    rows.append({'Field': key, 'Value': val_str})

            if not rows:
                print("No usable metadata found.")
                return

            # Display metadata table
            df = pd.DataFrame(rows)
            display(df)
            time.sleep(1)  # Allow UI to render before proceeding

            # Widgets for user confirmation
            checkbox = widgets.Checkbox(description="✅ Download this dataset?", value=False)
            confirm_button = widgets.Button(description="Confirm", button_style='success')
            output = widgets.Output()

            def confirm_handler():
                with output:
                    clear_output()
                    if checkbox.value:
                        self.downloadables.append(metadata)
                        self.logger.info("User confirmed download. Dataset queued.")
                        print("Queued for download.")
                    else:
                        self.logger.info("User declined download.")
                        print("Skipped.")

            confirm_button.on_click(lambda _: confirm_handler())

            # Show the checkbox + button
            ui_box = widgets.VBox([checkbox, confirm_button, output])
            display(ui_box)
            time.sleep(1)

        else:
            self.logger.warning(f"Unsupported display type: {display_type}. Cannot display metadata preview.")
            return

    def download_previewed_data_resources(self, output_root="output/suppl_files"):
        """
        Function to download all the files
        -- future release
        """
        self.logger.info(f"Downloading {len(self.downloadables)} previewed data resources.")
        for metadata in self.downloadables:
            download_link = metadata.get('download_link', None)
            if download_link is not None:
                split_source_url = metadata.get('source_url').split('/')
                paper_id = split_source_url[-1] if len(split_source_url) > 0 else split_source_url[-2]
                self.data_fetcher.download_file_from_url(download_link, output_root=output_root, paper_id=paper_id)
            else:
                self.logger.warning(f"No valid download_link found for metadata: {metadata}")

    def get_internal_id(self, metadata):
        """
        Function to get the internal ID of the dataset from metadata (utils).
        """
        self.logger.info(f"Getting internal ID for {metadata}")
        if 'source_url_for_metadata' in metadata and metadata['source_url_for_metadata'] is not None and metadata[
            'source_url_for_metadata'] not in ['nan', 'None', '', np.nan]:
            return metadata['source_url_for_metadata']
        elif 'dataset_webpage' in metadata and metadata['dataset_webpage'] is not None and metadata[
            'dataset_webpage'] not in ['nan', 'None', '', np.nan]:
            return metadata['dataset_webpage']
        elif 'download_link' in metadata and metadata['download_link'] is not None:
            return metadata['download_link']
        else:
            self.logger.warning("No valid internal ID found in metadata.")
            return None

    def run(self):
        """
        Main method to run the Orchestrator simple workflow:
        1. Setup data fetcher (web scraper or API client)
        2. Load URLs from config
        3. Process each URL and return results as a dictionary like source_url: DataFrame_of_data_links
        4. Write results to output file specified in configuration file
        """
        self.logger.debug("Orchestrator run started.")
        try:
            # Setup data fetcher (web scraper or API client)
            self.setup_data_fetcher()

            # Load URLs from config
            urls = self.load_urls_from_config()

            # Process each URL and return results as a dictionary like source_url: DataFrame_of_data_links
            results = self.process_urls(urls)

            # return the union of all the results
            combined_df = pd.DataFrame()
            for url, df in results.items():
                combined_df = pd.concat([combined_df, df], ignore_index=True)

            # evaluate the performance if ground_truth is provided
            #if 'ground_truth' in self.config:
               # self.logger.info("Evaluating performance...")
               # self.classifier.evaluate_performance(combined_df, self.config['ground_truth'])

            if self.config['data_resource_preview']:
                self.get_data_preview(combined_df)

                if self.config['download_previewed_data_resources']:
                    self.download_previewed_data_resources()

            combined_df.to_csv(self.config['full_output_file'], index=False)

            self.logger.info(f"Output written to file: {self.config['full_output_file']}")

            self.logger.info(f"File Download Schedule: {self.downloadables}")

            self.logger.debug("Orchestrator run completed.")

            return combined_df

        except Exception as e:
            self.logger.error(f"Error in orchestrator run: {e}", exc_info=True)
            return None

        finally:
            # Quit the driver to close the browser and free up resources
            if isinstance(self.data_fetcher, WebScraper):
                self.logger.info("Quitting the WebDriver.")
                self.data_fetcher.scraper_tool.quit()

            if isinstance(self.data_fetcher, APIClient):
                self.logger.info("Closing the APIClient.")
                self.data_fetcher.api_client.close()
