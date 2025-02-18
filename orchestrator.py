from logger_setup import setup_logging
from data_fetcher import *
from parser import HTMLParser, XMLParser
from classifier import LLMClassifier
import json
from selenium_setup import create_driver
import pandas as pd

class Orchestrator:
    def __init__(self, config_path):
        self.config = json.load(open(config_path))
        self.XML_config = json.load(open(self.config['navigation_config']))
        self.logger = setup_logging('orchestrator', self.config['log_file'])  # Initialize orchestrator logger
        self.classifier = LLMClassifier(self.config['classification_patterns'], self.logger)
        self.data_fetcher = None
        self.parser = None
        self.raw_data_format = None
        self.data_checker = DataCompletenessChecker(self.config, self.logger)
        self.full_DOM = (self.XML_config['llm_model'] in self.XML_config['entire_document_models']) and self.XML_config['process_entire_document']
        self.logger.info("Data_Gatherer Orchestrator initialized.")

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

        if self.config['search_method'] == 'url_list':
            driver = create_driver(self.config['DRIVER_PATH'], self.config['BROWSER'], self.config['HEADLESS'])
            self.data_fetcher = WebScraper(driver, self.config, self.logger)

        elif self.config['search_method'] == 'google_scholar':
            driver = create_driver(self.config['DRIVER_PATH'], self.config['BROWSER'], self.config['HEADLESS'])
            self.data_fetcher = WebScraper(driver, self.config, self.logger)

        elif self.config['search_method'] == 'api':
            self.logger.error("API data source not yet implemented.")

        self.logger.info("Data fetcher setup completed.")

        return self.data_fetcher.scraper_tool


    def process_url(self, url):
        """Orchestrates the process for a given source URL (publication)."""
        self.logger.info(f"Processing URL: {url}")
        self.current_url = url
        self.publisher = self.data_fetcher.url_to_publisher_domain(url)

        self.data_fetcher = self.data_fetcher.update_DataFetcher_settings(url, self.full_DOM, self.logger)

        try:
            self.logger.debug("Fetching Raw content...")
            raw_data = None
            parsed_data = None
            additional_data = None
            process_everything_as_additional_data = True #additional data is getting processed without link-based prompts

            # Step 1: Use DataFetcher (WebScraper or APIClient) to fetch raw data
            self.logger.debug(f"data_fetcher.fetch_source = {self.data_fetcher.fetch_source}")

            # if model processes the entire document, fetch the entire document and go to the parsing step
            if (self.XML_config['llm_model'] in self.XML_config['entire_document_models'] and self.XML_config['process_entire_document']):
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
                    if self.raw_data_format == "HTML":
                        self.data_fetcher.download_html(directory)
                        self.logger.info(f"HTML saved to: {directory}")
                    # for XML files, debug print done by parser

            self.logger.info("Successfully fetched Raw content.")

            # Step 2: Use HTMLParser to parse and extract HTML elements and rule-based matches
            if self.raw_data_format == "HTML":
                self.logger.info("Using HTML_Parser to parse data.")
                self.parser = HTMLParser(self.config, self.logger)
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
                self.logger.info("Using XMLParser to parse data.")
                self.parser = XMLParser(self.XML_config, self.logger)

                if additional_data is not None:
                    self.logger.info(f"Processing additional data: {len(additional_data)}")
                    parsed_data = self.parser.parse_data(raw_data, self.publisher, self.current_url,
                                                         additional_data=additional_data)
                    self.logger.info(type(parsed_data))
                else:
                    self.logger.info(f"Parser. {self.parser}")
                    parsed_data = self.parser.parse_data(raw_data, self.publisher, self.current_url)

                parsed_data['source_url'] = url
                self.logger.info(f"Parsed data extraction completed. Elements collected: {len(parsed_data)}")
                parsed_data.to_csv('staging_table/parsed_data_from_XML.csv', index=False)  # save parsed data to a file

            elif self.raw_data_format == "full_HTML":
                self.logger.info("Using XMLParser to parse data.")
                self.parser = XMLParser(self.XML_config, self.logger)
                parsed_data = self.parser.parse_data(raw_data, self.publisher, self.current_url, raw_data_format="full_HTML")
                parsed_data['source_url'] = url
                self.logger.info(f"Parsed data extraction completed. Elements collected: {len(parsed_data)}")
                parsed_data.to_csv('staging_table/parsed_data_from_XML.csv', index=False)  # save parsed data to a file

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

    def process_urls(self, url_list):
        """Processes a list of URLs and returns classified data."""
        self.logger.debug("Starting to process URL list...")
        results = {}
        for url in url_list:
            results[url] = self.process_url(url)
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

            combined_df.to_csv(self.config['full_output_file'], index=False)

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
