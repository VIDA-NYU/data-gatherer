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

class Orchestrator:
    def __init__(self, config_path):
        self.config = json.load(open(config_path))
        self.XML_config = json.load(open(self.config['navigation_config']))
        self.logger = setup_logging('orchestrator', self.config['log_file'])  # Initialize orchestrator logger
        self.classifier = LLMClassifier(self.config['retrieval_patterns'], self.logger)
        self.data_fetcher = None
        self.parser = None
        self.raw_data_format = None
        self.data_checker = DataCompletenessChecker(self.config, self.logger)
        self.full_DOM = (self.XML_config['llm_model'] in self.XML_config['entire_document_models']) and self.XML_config['process_entire_document']
        self.logger.info(f"Data_Gatherer Orchestrator initialized. Extraction step Model: {self.XML_config['llm_model']}")
        self.downloadables = []

    def setup_data_fetcher(self):
        """Sets up either a web scraper or API client based on the config."""
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
            driver = create_driver(self.config['DRIVER_PATH'], self.config['BROWSER'], self.config['HEADLESS'])
            self.data_fetcher = WebScraper(driver, self.config, self.logger)

        elif self.config['search_method'] == 'cloudscraper':
            driver = cloudscraper.create_scraper()
            self.data_fetcher = WebScraper(driver, self.config, self.logger)

        elif self.config['search_method'] == 'google_scholar':
            driver = create_driver(self.config['DRIVER_PATH'], self.config['BROWSER'], self.config['HEADLESS'])
            self.data_fetcher = WebScraper(driver, self.config, self.logger)

        elif self.config['search_method'] == 'api':
            self.logger.error("API data source not yet implemented.")

        else:
            raise ValueError(f"Invalid search method: {self.config['search_method']}")

        self.logger.info("Data fetcher setup completed.")

        return self.data_fetcher.scraper_tool


    def process_url(self, url):
        """Orchestrates the process for a given source URL (publication)."""
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
            if (self.XML_config['llm_model'] in self.XML_config['entire_document_models'] and self.XML_config['process_entire_document']):
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
                self.logger.info(f"Raw data fetched as: {raw_data}")

                self.logger.info(f"Fetcher source: {self.data_fetcher.fetch_source}")
                if "API" not in self.data_fetcher.fetch_source:
                    raw_data = self.data_fetcher.scraper_tool.page_source

                else:
                    additional_data = self.data_checker.ensure_data_sections(raw_data, url)
                    self.logger.debug(f"Additional data fetched as: {additional_data}")

            if self.config['write_htmls_xmls']:
                directory = self.config['html_xml_dir'] + self.publisher + '/'
                self.logger.info(f"Raw Data is {self.raw_data_format}.")
                if self.raw_data_format == "HTML" or self.raw_data_format == "full_HTML":
                    self.data_fetcher.download_html(directory)
                    self.logger.info(f"HTML saved to: {directory}")
                    # for XML files, debug print done by parser

            self.logger.info("Successfully fetched Raw content.")

            # Step 2: Use RuleBasedParser to parse and extract HTML elements and rule-based matches
            if self.raw_data_format == "HTML":
                self.logger.info("Using RuleBasedParser to parse data.")
                self.parser = RuleBasedParser(self.XML_config, self.logger)
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

                parsed_data.to_csv('staging_table/parsed_data.csv', index=False)

            elif self.raw_data_format == "XML" and raw_data is not None:
                self.logger.info("Using LLMParser to parse data.")
                self.parser = LLMParser(self.XML_config, self.logger)

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
                    parsed_data.to_csv('staging_table/parsed_data_from_XML.csv', index=False)  # save parsed data to a file

            elif self.raw_data_format == "full_HTML":
                self.logger.info("Using LLMParser to parse data.")
                self.parser = LLMParser(self.XML_config, self.logger)
                parsed_data = self.parser.parse_data(raw_data, self.publisher, self.current_url, raw_data_format="full_HTML")
                parsed_data['source_url'] = url
                self.logger.info(f"Parsed data extraction completed. Elements collected: {len(parsed_data)}")
                if self.logger.level == logging.DEBUG:
                    parsed_data.to_csv('staging_table/parsed_data_from_XML.csv', index=False)

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
                if self.raw_data_format == "HTML":
                    classified_links = self.classifier.classify_anchor_elements_links(parsed_data)
                    self.logger.info("Link classification completed.")
                elif self.raw_data_format == "XML":
                    self.logger.info("XML element classification not needed. Using parsed_data.")
                    classified_links = parsed_data
                elif self.raw_data_format == "full_HTML":
                    classified_links = parsed_data
                    self.logger.info("Full HTML element classification not supported. Using parsed_data.")
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
        """
        self.logger.info(f"Deduplicating {len(classified_links)} classified links.")
        classified_links['link'] = classified_links['link'].str.strip()
        classified_links['download_link'] = classified_links['download_link'].str.strip()

        # Deduplicate based on link column
        #classified_links = classified_links.drop_duplicates(subset=['link', 'download_link'], keep='last')

        self.logger.info(f"Deduplication completed. {len(classified_links)} unique links found.")
        return classified_links

    def process_urls(self, url_list, log_modulo=10):
        """Processes a list of URLs and returns classified data."""
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
        """Loads URLs from the input file specified in the config."""
        self.logger.debug(f"Loading URLs from file: {self.config['input_urls_filepath']}")
        with open(self.config['input_urls_filepath'], 'r') as file:
            url_list = [line.strip() for line in file]
        self.logger.info(f"Loaded {len(url_list)} URLs from file.")
        self.url_list = url_list
        return url_list

    def get_data_preview(self, combined_df):
        """Shows user a preview of the data they are about to download."""
        self.already_previewed = []
        self.metadata_parser = LLMParser(self.XML_config, self.logger)
        scraper_tool = create_driver(self.config['DRIVER_PATH'], self.config['BROWSER'], self.config['HEADLESS'])

        if isinstance(self.data_fetcher, WebScraper):
            self.data_fetcher.quit()

        self.data_fetcher = WebScraper(scraper_tool, self.config, self.logger)

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
                    hardsraped_metadata = {k:v for k,v in row.items() if v is not None and v not in ['nan', 'None', '', 'n/a', np.nan, 'NaN', 'na']}
                    self.display_data_preview(hardsraped_metadata)
                    continue

            else:
                self.logger.info(f"LLM scraped metadata")
                repo_mapping_key = row['repository_reference'].lower() if 'repository_reference' in row else row['data_repository'].lower()
                if ('javascript_load_required' in self.XML_config['repos'][self.parser.repo_domain_to_name_mapping[repo_mapping_key]]):
                    self.logger.info(f"JavaScript load required for {repo_mapping_key} dataset webpage. Using WebScraper.")
                    html = self.data_fetcher.fetch_data(row['dataset_webpage'])
                else:
                    html = requests.get(row['dataset_webpage']).text
                metadata = self.metadata_parser.parse_metadata(html)
                metadata['source_url_for_metadata'] = row['dataset_webpage']
                metadata['access_mode'] = row.get('access_mode', None)
                metadata['source_section'] = row.get('source_section', row.get('section_class', None))
                metadata['download_link'] = row.get('download_link', None)
                metadata['accession_id'] = row.get('dataset_id', row.get('dataset_identifier', None))
                metadata['data_repository'] = repo_mapping_key

            metadata['paper_with_dataset_citation'] = row['source_url']
            self.display_data_preview(metadata)
        self.data_fetcher.quit()

    def display_data_preview(self, metadata):
        """
        Display extracted metadata and ask the user whether to proceed with download.
        """

        if not isinstance(metadata, dict):
            self.logger.warning("Metadata is not a dictionary. Cannot display properly.")
            return

        self.logger.debug("Iterating over metadata items to show non-null fields:")
        for key, value in metadata.items():
            if value is not None and value not in ['nan', 'None', '', np.nan, 'NaN', 'na', 'unavailable', 0]:
                print(f"{key}: {value}")
        time.sleep(0.1)

        user_input = input("\nDo you want to proceed with downloading this dataset? [y/N]: \n"
                           "__________________________________________________________________  ").strip().lower()
        if user_input not in ["y", "yes"]:
            self.logger.info("User declined to download the dataset.")
        else:
            self.downloadables.append(metadata)
            self.already_previewed.append(self.get_internal_id(metadata))
            self.logger.info(f"Added {self.get_internal_id(metadata)} to self.already_previewed.")
            self.logger.info("User confirmed download. Proceeding...")

    def get_internal_id(self, metadata):
        self.logger.info(f"Getting internal ID for {metadata}")
        if 'source_url_for_metadata' in metadata and metadata['source_url_for_metadata'] is not None and metadata[
            'source_url_for_metadata'] not in ['nan', 'None', '', np.nan]:
            return metadata['source_url_for_metadata']
        elif 'download_link' in metadata and metadata['download_link'] is not None:
            return metadata['download_link']
        else:
            self.logger.warning("No valid internal ID found in metadata.")
            return None

    def run(self):
        """Main method to run the Orchestrator."""
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
