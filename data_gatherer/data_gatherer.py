import os.path
import os
import re
import json
import time
import threading
import pandas as pd
import ipywidgets as widgets
import textwrap
import cloudscraper
from typing import Dict, Any
from data_gatherer.logger_setup import setup_logging
from data_gatherer.data_fetcher import *
from data_gatherer.parser.html_parser import *
from data_gatherer.parser.xml_parser import *
from data_gatherer.parser.pdf_parser import *
from data_gatherer.parser.grobid_pdf_parser import *
from data_gatherer.llm.response_schema import *
from data_gatherer.classifier import LLMClassifier
from data_gatherer.env import CACHE_BASE_DIR
from data_gatherer.selenium_setup import create_driver
from data_gatherer.resources_loader import load_config
from IPython.display import display, clear_output


class DataGatherer:
    """
    This class orchestrates the data gathering process by coordinating the data fetcher, parser, and classifier in a
    single workflow.
	Initializes the DataGatherer with the given configuration file and sets up logging.

    :param process_entire_document: Flag to indicate if the model processes the entire document.

    :param log_file_override: Optional log file path to override the default logging configuration.

    :param write_htmls_xmls: Flag to indicate if raw HTML/XML files should be saved.

    :param article_file_dir: Directory to save the raw HTML/XML/PDF files.

    :param full_output_file: Path to the output file where results will be saved.

    :param download_data_for_description_generation: Flag to indicate if data should be downloaded for description generation.

    :param data_resource_preview: Flag to indicate if a preview of data resources should be generated.

    :param download_previewed_data_resources: Flag to indicate if previewed data resources should be downloaded.

    :param log_level: Logging level for the logger.

    :param clear_previous_logs: Flag to clear previous logs before setting up logging.

    :param retrieval_patterns_file: Path to the JSON file containing retrieval patterns for classification.

    :param load_from_cache: Flag to indicate if results should be loaded from cache.

    :param save_to_cache: Flag to indicate if results should be saved to cache.

    :param driver_path: Path to the WebDriver executable for the data fetcher (if applicable).

    :param save_dynamic_prompts: Flag to indicate if dynamically generated prompts should be saved.

    """

    def __init__(self, llm_name='gpt-4o-mini', process_entire_document=False, log_file_override=None,
                 write_htmls_xmls=False, article_file_dir='tmp/raw_files/', full_output_file='output/result.csv',
                 download_data_for_description_generation=False, data_resource_preview=False,
                 download_previewed_data_resources=False, log_level=logging.ERROR, clear_previous_logs=True,
                 retrieval_patterns_file='retrieval_patterns.json', load_from_cache=False, save_to_cache=False,
                 driver_path=None, save_dynamic_prompts=False
                 ):

        self.open_data_repos_ontology = load_config('open_bio_data_repos.json')

        log_file = log_file_override or 'logs/data_gatherer.log'
        self.logger = setup_logging('orchestrator', log_file, level=log_level,
                                    clear_previous_logs=clear_previous_logs)

        self.classifier = LLMClassifier(self.logger, retrieval_patterns_file)
        self.data_fetcher = None
        self.parser = None
        self.raw_data_format = None
        self.setup_data_fetcher(driver_path=driver_path)
        self.fetcher_driver_path = driver_path
        self.data_checker = DataCompletenessChecker(self.logger)

        self.write_htmls_xmls = write_htmls_xmls
        self.article_file_dir = article_file_dir
        self.load_from_cache = load_from_cache
        self.save_to_cache = save_to_cache
        self.save_dynamic_prompts = save_dynamic_prompts

        self.download_data_for_description_generation = download_data_for_description_generation

        entire_document_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-2.0-flash",
                                  "gemini-2.5-flash", "gpt-4o", "gpt-4o-mini", "gpt-5-nano", "gpt-5-mini", "gpt-5"]
        self.full_document_read = llm_name in entire_document_models and process_entire_document
        self.llm = llm_name

        self.search_method = 'url_list'  # Default search method

        self.full_output_file = full_output_file

        self._processing_semaphore = threading.Semaphore(2)  # Max 2 concurrent operations per instance
        self._last_request_time = 0
        self._min_delay = 1.0  # Minimum 1 second between requests
        self.data_resource_preview = data_resource_preview
        self.download_previewed_data_resources = download_previewed_data_resources
        self.downloadables = []
        self.logger.info(f"DataGatherer orchestrator initialized. Extraction Model: {llm_name}")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def fetch_data(self, urls, search_method='url_list', driver_path=None, browser=None, headless=True,
                   HTML_fallback=False, local_fetch_file=None, write_htmls_xmls=False, article_file_dir='tmp/raw_files/',
                   write_to_df_path=False):
        """
        Fetches data from the given URL using the configured data fetcher (WebScraper or EntrezFetcher).

        :param urls: The list of URLs to fetch data from.

        :param search_method: Optional method to override the default search method. Supported values are 'url_list', 'cloudscraper', 'google_scholar'.

        :param driver_path: Path to your local WebDriver executable (if applicable). When set to None, Webdriver manager will be used.

        :param browser: Browser to use for scraping (if applicable). Supported values are 'Firefox', 'Chrome'.

        :param headless: Whether to run the browser in headless mode (if applicable).

        :param HTML_fallback: Flag to indicate if HTML fallback should be used when fetching data. This will override any other fetching resource (i.e. API).

        :param local_fetch_file: Optional file containing data to be used in the fetching process. Supported format is 'parquet' file.

        :param write_htmls_xmls: Flag to indicate if raw HTML/XML files should be saved. Overwrites the default setting.

        :param article_file_dir: Directory to save the raw HTML/XML/PDF files. Overwrites the default setting.

        :param write_to_df_path: Optional path to save the fetched data as a DataFrame in Parquet format.

        :return: Dictionary with URLs as keys and raw data as values.

        """
        single_article = False

        if not isinstance(urls, str) and not isinstance(urls, list):
            raise ValueError("URL must be a string or a list of strings.")

        if isinstance(urls, str):
            urls = [urls]
            single_article = True

        complete_publication_fetches = {}
        i = None
        HTML_fallback_priority_list = ['HTTPGetRequest', 'Selenium']

        while len(complete_publication_fetches) < len(urls):
            HTML_fallback = False if i is None else HTML_fallback_priority_list[i]
            self.logger.info(f"Fetch attempt with HTML_fallback={HTML_fallback}...")
            i = 0 if i is None else i + 1
            for pub_link in urls:
                self.logger.info(f"length of complete fetches < urls: {len(complete_publication_fetches)} < {len(urls)}")
                if pub_link in complete_publication_fetches:
                    continue

                # Update fetcher settings for this method and publication
                self.data_fetcher = self.data_fetcher.update_DataFetcher_settings(
                    pub_link,
                    self.full_document_read,
                    self.logger,
                    local_fetch_file=local_fetch_file,
                    HTML_fallback=HTML_fallback,
                    driver_path=driver_path,
                    browser=browser,
                    headless=headless
                )

                # Fetch data
                fetched_data = self.data_fetcher.fetch_data(pub_link)
                self.logger.info(f"Raw_data_format: {self.data_fetcher.raw_data_format}, Type of fetched data: {type(fetched_data)}")
                completeness_check = self.data_checker.is_fulltext_complete(fetched_data, pub_link, self.data_fetcher.raw_data_format)

                if completeness_check:
                    self.logger.info(f"Fetch complete {self.data_fetcher.raw_data_format} data from {pub_link}.")
                    complete_publication_fetches[pub_link] = {
                        'fetched_data': fetched_data,
                        'raw_data_format': self.data_fetcher.raw_data_format
                    }
                    continue
                elif HTML_fallback == 'Selenium':
                    self.logger.info(f"Selenium fetch the final fulltext {pub_link}.")
                    complete_publication_fetches[pub_link] = {
                        'fetched_data': fetched_data, 
                        'raw_data_format': self.data_fetcher.raw_data_format
                        }
                    continue
                else:
                    self.logger.info(f"{self.data_fetcher.raw_data_format} Data from {pub_link} is incomplete.")

                # Optionally save HTML/XMLs if requested
                if write_htmls_xmls:
                    publisher = self.data_fetcher.url_to_publisher_domain(pub_link)
                    directory = os.path.join(article_file_dir, publisher)
                    if HTML_fallback == 'Selenium':
                        self.data_fetcher.html_page_source_download(directory, pub_link)
                    elif self.data_fetcher.raw_data_format == "HTML" and completeness_check:
                        self.data_fetcher.html_page_source_download(directory, pub_link, fetched_data)
                    elif self.data_fetcher.raw_data_format == "XML" and completeness_check:
                        self.data_fetcher.download_xml(directory, fetched_data, pub_link)
                    elif self.data_fetcher.raw_data_format == "PDF":
                        self.data_fetcher.download_pdf(directory, fetched_data, pub_link)
                    else:
                        self.logger.warning(f"Unsupported raw data format: {self.data_fetcher.raw_data_format}.")

        # Clean up driver if needed
        if hasattr(self.data_fetcher, 'scraper_tool'):
            self.data_fetcher.scraper_tool.quit()

        if write_to_df_path and write_to_df_path.endswith('.parquet'):
            df = pd.DataFrame.from_dict(complete_publication_fetches, orient='index')
            df.to_parquet(write_to_df_path, index=True)

        if single_article:
            return complete_publication_fetches[urls[0]]['fetched_data']

        return complete_publication_fetches

    def init_parser_by_input_type(self, raw_data_format, raw_data=None):
        if raw_data_format.upper() == "XML":
            if raw_data is not None:
                router = XMLRouter(self.open_data_repos_ontology, self.logger, full_document_read=self.full_document_read,
                                llm_name=self.llm, save_dynamic_prompts=self.save_dynamic_prompts)
                self.parser = router.get_parser(raw_data)
            else:
                self.parser = XMLParser(self.open_data_repos_ontology, self.logger, full_document_read=self.full_document_read,
                                llm_name=self.llm, save_dynamic_prompts=self.save_dynamic_prompts)

        elif raw_data_format.upper() == "HTML":
            self.parser = HTMLParser(self.open_data_repos_ontology, self.logger, full_document_read=self.full_document_read,
                               llm_name=self.llm, save_dynamic_prompts=self.save_dynamic_prompts)

        elif raw_data_format.upper() == "PDF" and grobid_for_pdf:
            self.parser = GrobidPDFParser(self.open_data_repos_ontology, self.logger, full_document_read=self.full_document_read,
                               llm_name=self.llm, save_dynamic_prompts=self.save_dynamic_prompts)

        elif raw_data_format.upper() == "PDF":
            self.parser = PDFParser(self.open_data_repos_ontology, self.logger, full_document_read=self.full_document_read,
                               llm_name=self.llm, save_dynamic_prompts=self.save_dynamic_prompts)
        else:
            raise ValueError(f"Unsupported raw data format: {raw_data_format}")

    def parse_data(self, raw_data, publisher=None, current_url_address=None, additional_data=None,
                   raw_data_format='XML', parsed_data_dir='tmp/parsed_articles/', grobid_for_pdf=False,
                   process_DAS_links_separately=False, full_document_read=False, semantic_retrieval=False, top_k=5,
                   prompt_name='GPT_FewShot', use_portkey=True, section_filter=None,
                   response_format=dataset_response_schema_gpt):
        """
        Parses the raw data fetched from the source URL using the appropriate parser.

        :param raw_data: The raw data to parse, typically string formatted as HTML or XML content.

        :param publisher: The publisher domain or identifier for the data source.

        :param current_url_address: The URL of the current data source being processed.

        :param additional_data: Optional additional data to include in the parsing process, such as metadata or supplementary information.

        :param raw_data_format: The format of the raw data (e.g., 'HTML', 'XML').

        :param parsed_data_dir: Directory to save the parsed HTML/XML/PDF files.

        :param grobid_for_pdf: Flag to indicate if Grobid should be used for PDF parsing. Read more on GROBID PDF Parser here: https://grobid.readthedocs.io/en/latest/

        :param process_DAS_links_separately: Flag to indicate if DAS links should be processed separately.

        :param full_document_read: Flag to indicate if the model processes the entire document.

        :param semantic_retrieval: Flag to indicate if semantic retrieval should be used.

        :param top_k: Number of top relevant sections to retrieve if semantic retrieval is enabled.

        :param prompt_name: Name of the prompt to use for LLM parsing.

        :param use_portkey: Flag to use Portkey for Gemini LLM.

        :param section_filter: Optional filter to apply to the sections (supplementary_material', 'data_availability_statement').

        :param response_format: The response schema to use for parsing the data.

        :return: Parsed data as a DataFrame or dictionary, depending on the parser used.
        """
        self.logger.info(f"Parsing data from URL: {current_url_address} with publisher: {publisher}")

        if raw_data_format.upper() == "XML":
            router = XMLRouter(self.open_data_repos_ontology, self.logger, full_document_read=full_document_read,
                               llm_name=self.llm, use_portkey=use_portkey,
                                     save_dynamic_prompts=self.save_dynamic_prompts)
            self.parser = router.get_parser(raw_data)

        elif raw_data_format.upper() == "HTML":
            self.parser = HTMLParser(self.open_data_repos_ontology, self.logger, full_document_read=full_document_read,
                                     llm_name=self.llm, use_portkey=use_portkey,
                                     save_dynamic_prompts=self.save_dynamic_prompts)

        elif raw_data_format.upper() == "PDF" and grobid_for_pdf:
            self.parser = GrobidPDFParser(self.open_data_repos_ontology, self.logger, full_document_read=full_document_read,
                                    llm_name=self.llm, use_portkey=use_portkey,
                                     save_dynamic_prompts=self.save_dynamic_prompts)

        elif raw_data_format.upper() == "PDF":
            self.parser = PDFParser(self.open_data_repos_ontology, self.logger, full_document_read=full_document_read,
                                    llm_name=self.llm, use_portkey=use_portkey,
                                     save_dynamic_prompts=self.save_dynamic_prompts)

        else:
            raise ValueError(f"Unsupported raw data format: {raw_data_format}")

        if isinstance(raw_data, dict):
            cont = raw_data.values()
            cont = list(cont)[0]

        elif isinstance(raw_data, str):
            cont = raw_data

        else:
            cont = raw_data

        ret = self.parser.parse_data(cont,
                                      publisher=publisher,
                                      current_url_address=current_url_address,
                                      raw_data_format=raw_data_format,
                                      prompt_name=prompt_name,
                                      use_portkey=use_portkey,
                                      article_file_dir=parsed_data_dir,
                                      additional_data=additional_data,
                                      process_DAS_links_separately=process_DAS_links_separately,
                                      semantic_retrieval=semantic_retrieval,
                                      top_k=top_k,
                                      section_filter=section_filter,
                                      response_format=response_format
                                      )

        ret['raw_data_format'] = raw_data_format

        return ret

    def setup_data_fetcher(self, search_method='url_list', driver_path='', browser='Firefox', headless=True,
                           raw_HTML_data_fp=None):
        """
        Sets up either an empty web scraper, one with scraper_tool, or an API client based on the config.
        """

        if search_method is not None:
            self.search_method = search_method

        self.logger.info("Setting up data fetcher...")

        # Close previous driver if exists
        if hasattr(self, 'data_fetcher') and hasattr(self.data_fetcher, 'scraper_tool'):
            try:
                self.data_fetcher.scraper_tool.quit()
                self.logger.info("Previous driver quit.")
            except Exception as e:
                self.logger.warning(f"Failed to quit previous driver: {e}")

        elif self.search_method == 'url_list':
            self.data_fetcher = WebScraper(None, self.logger, driver_path=driver_path, browser=browser,
                                           headless=headless)

        elif self.search_method == 'cloudscraper':
            driver = cloudscraper.create_scraper()
            self.data_fetcher = WebScraper(driver, self.logger)

        elif self.search_method == 'google_scholar':
            driver = create_driver(driver_path, browser, headless, self.logger)
            self.data_fetcher = WebScraper(driver, self.logger, driver_path=driver_path, browser=browser,
                                           headless=headless)

        else:
            raise ValueError(f"Invalid search method: {self.search_method}")

        self.logger.info("Data fetcher setup completed.")

        return self.data_fetcher.scraper_tool

    def PMCID_to_URL(self, pmcid):
        pmcid = pmcid.strip().upper()
        if not pmcid.startswith("PMC"):
            raise ValueError("Invalid PMCID format. Must start with 'PMC'.")

        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"

    def preprocess_url(self, url):
        if url.upper().startswith("PMC"):
            return self.PMCID_to_URL(url)
        elif url.lower().startswith("https://"):
            return url
        elif os.path.isfile(url):
            return url
        else:
            raise ValueError(f"Invalid URL format: {url}. Must start with 'PMC' or 'https://'.")

    def process_url(self, url, save_staging_table=False, article_file_dir='tmp/raw_files/', use_portkey=True,
                    driver_path=None, browser='Firefox', headless=True, prompt_name='GPT_FewShot',
                    semantic_retrieval=False, section_filter=None, response_format=dataset_response_schema_gpt,
                    HTML_fallback=False, grobid_for_pdf=False, write_htmls_xmls=False):
        """
        Orchestrates the process for a single given source URL (publication).

        1. Fetches raw data using the data fetcher (WebScraper or EntrezFetcher).

        2. Parses the raw data using the parser (LLMParser).

        3. Collects Metadata.

        4. Classifies the parsed data using the classifier (LLMClassifier).

        :param url: The URL to process.

        :param save_staging_table: Flag to save the staging table.

        :param article_file_dir: Directory to save the raw HTML/XML/PDF files.

        :param use_portkey: Flag to use Portkey for Gemini LLM.

        :param driver_path: Path to your local WebDriver executable (if applicable). When set to None, Webdriver manager will be used.

        :param browser: Browser to use for scraping (if applicable). Supported values are 'Firefox', 'Chrome'.

        :param headless: Whether to run the browser in headless mode (if applicable).

        :param prompt_name: Name of the prompt to use for LLM parsing.

        :param semantic_retrieval: Flag to indicate if semantic retrieval should be used.

        :param section_filter: Optional filter to apply to the sections (supplementary_material', 'data_availability_statement').

        :param response_format: The response schema to use for parsing the data.

        :param HTML_fallback: Flag to indicate if HTML fallback should be used when fetching data. This will override any other fetching resource (i.e. API).

        :param grobid_for_pdf: Flag to indicate if GROBID should be used for PDF processing.

        :param write_htmls_xmls: Flag to indicate if raw HTML/XML files should be saved. Overwrites the default setting.

        :return: DataFrame of classified links or None if an error occurs.
        """
        self.logger.info(f"Processing URL: {url}")
        self.current_url = url
        self.write_htmls_xmls = write_htmls_xmls or self.write_htmls_xmls
        self.publisher = self.data_fetcher.url_to_publisher_domain(url)

        self.data_fetcher = self.data_fetcher.update_DataFetcher_settings(url, self.full_document_read, self.logger,
                                                                          driver_path=driver_path, browser=browser,
                                                                          headless=headless, HTML_fallback=HTML_fallback)
        self.logger.info(f"Type of data_fetcher {self.data_fetcher.__class__.__name__}")

        article_id = self.url_to_article_id(url)
        process_id = self.llm + "-FDR-" + article_id if self.full_document_read else self.llm + "-RTR-" + article_id
        if os.path.exists(os.path.join(CACHE_BASE_DIR, "process_url_cache.json")) and self.load_from_cache:
            cache = json.load(open(os.path.join(CACHE_BASE_DIR, "process_url_cache.json"), 'r'))
            if process_id in cache:
                self.logger.info(f"Loading results from cache for process ID: {process_id}")
                return pd.DataFrame(cache[process_id])

        try:
            self.logger.debug("Fetching Raw content...")
            raw_data = None
            parsed_data = None
            additional_data = None
            filepath = None

            if os.path.isfile(url):
                filepath = url
                self.raw_data_format = str.split(filepath, '.')[-1].lower()
                raw_data = filepath if self.raw_data_format.upper() == 'PDF' else open(filepath, 'r', encoding='utf-8').read()
                self.logger.info(f"Local file {filepath} detected. Using it as raw data.")

            else:
                raw_data = self.data_fetcher.fetch_data(url)
                self.raw_data_format = self.data_fetcher.raw_data_format
            
            if filepath is None:

                if not self.data_checker.is_fulltext_complete(raw_data, url, self.raw_data_format) and not (
                    self.data_fetcher.__class__.__name__ == "WebScraper"
                ):
                    self.logger.info(f"Fallback to Selenium WebScraper data fetcher.")
                    self.raw_data_format = "HTML"
                    self.data_fetcher = self.data_fetcher.update_DataFetcher_settings(url,
                                                                                        self.full_document_read,
                                                                                        self.logger,
                                                                                        HTML_fallback=True,
                                                                                        driver_path=driver_path,
                                                                                        browser=browser,
                                                                                        headless=headless)
                    raw_data = self.data_fetcher.fetch_data(url)

                else:
                    self.logger.info(f"{self.raw_data_format} data is complete for {url}.")

                raw_data = self.data_fetcher.remove_cookie_patterns(raw_data) if self.raw_data_format == "HTML" else raw_data

                self.logger.info(f"Raw {self.raw_data_format} data fetched from {url} is ready for parsing.")

                if self.write_htmls_xmls and not isinstance(self.data_fetcher, DatabaseFetcher):
                    directory = article_file_dir + self.publisher + '/'
                    self.logger.info(f"Raw Data is {self.raw_data_format}.")
                    if isinstance(self.data_fetcher, WebScraper):
                        self.data_fetcher.html_page_source_download(directory, url)
                        self.logger.info(f"Raw HTML saved to: {directory}")
                    elif isinstance(self.data_fetcher, EntrezFetcher):
                        self.data_fetcher.download_xml(directory, raw_data, url)
                        self.logger.info(f"Raw XML saved in {directory} directory")
                    elif self.raw_data_format.upper() == 'PDF':
                        # For PDF, raw_data should already be a file path, just log the location
                        self.logger.info(f"Raw PDF file location: {raw_data}")
                    else:
                        self.logger.warning(f"Unsupported raw data format: {self.raw_data_format}.")
                else:
                    self.logger.info(f"Skipping raw HTML/XML/PDF saving. Param write_htmls_xmls set to {self.write_htmls_xmls}.")

                self.data_fetcher.quit() if hasattr(self.data_fetcher, 'scraper_tool') else None

            # Step 2: Use HTMLParser/XMLParser
            self.logger.info("Parsing Raw content from format: " + self.raw_data_format)
            if self.raw_data_format.upper() == "XML" and raw_data is not None:
                self.logger.info("Using XMLParser to parse data.")
                self.parser = XMLParser(self.open_data_repos_ontology, self.logger,
                                        llm_name=self.llm,
                                        full_document_read=self.full_document_read,
                                        use_portkey=use_portkey,
                                        save_dynamic_prompts=self.save_dynamic_prompts)

                if additional_data is None:
                    self.logger.info("No additional data provided. Parsing raw data only.")
                    parsed_data = self.parser.parse_data(raw_data, self.publisher, self.current_url,
                                                         prompt_name=prompt_name, semantic_retrieval=semantic_retrieval,
                                                         section_filter=section_filter, response_format=response_format)

                else:
                    self.logger.info(f"Processing additional data. # of items: {len(additional_data)}")
                    add_data = self.parser.parse_data(raw_data, self.publisher, self.current_url,
                                                      additional_data=additional_data, prompt_name=prompt_name,
                                                      semantic_retrieval=semantic_retrieval,
                                                      section_filter=section_filter, response_format=response_format)
                    self.logger.debug(f"Type of additional data{type(add_data)}")

                    parsed_data = pd.concat([parsed_data, add_data], ignore_index=True).drop_duplicates()

            elif self.raw_data_format.upper() == 'HTML':
                self.logger.info("Using HTMLParser to parse data.")
                self.parser = HTMLParser(self.open_data_repos_ontology, self.logger,
                                         llm_name=self.llm,
                                         full_document_read=self.full_document_read,
                                         use_portkey=use_portkey,
                                         save_dynamic_prompts=self.save_dynamic_prompts)
                parsed_data = self.parser.parse_data(raw_data, self.publisher, self.current_url,
                                                     raw_data_format=self.raw_data_format, prompt_name=prompt_name,
                                                     semantic_retrieval=semantic_retrieval,
                                                     section_filter=section_filter, response_format=response_format)
                parsed_data['source_url'] = url
                parsed_data['pub_title'] = self.parser.extract_publication_title(raw_data)
                self.logger.info(f"Parsed data extraction completed. Elements collected: {len(parsed_data)}")
            
            elif self.raw_data_format.upper() == 'PDF':
                self.logger.info("Using PDFParser to parse data.")
                if grobid_for_pdf:
                    self.logger.info("GROBID PDF parsing enabled.")
                    self.parser = GrobidPDFParser(self.open_data_repos_ontology, self.logger,
                                                 llm_name=self.llm,
                                                 full_document_read=self.full_document_read,
                                                 use_portkey=use_portkey,
                                                 save_dynamic_prompts=self.save_dynamic_prompts,
                                                 write_XML=write_htmls_xmls)
                else:
                    self.parser = PDFParser(self.open_data_repos_ontology, self.logger,
                                            llm_name=self.llm,
                                            full_document_read=self.full_document_read,
                                            use_portkey=use_portkey,
                                            save_dynamic_prompts=self.save_dynamic_prompts)
                # For PDF, raw_data should be the file path
                parsed_data = self.parser.parse_data(raw_data, 
                                                     publisher=self.publisher, 
                                                     current_url_address=self.current_url,
                                                     raw_data_format=self.raw_data_format, 
                                                     prompt_name=prompt_name,
                                                     semantic_retrieval=semantic_retrieval,
                                                     section_filter=section_filter,
                                                     response_format=response_format)
                self.logger.info(f"PDF parsing completed. Elements collected: {len(parsed_data)}")

            else:
                self.logger.error(f"Unsupported raw data format: {self.raw_data_format}. Cannot parse data.")
                return None

            self.logger.info("Raw Data parsing completed.")
            parsed_data.to_csv('staging_table/parsed_data.csv', index=False) if save_staging_table else None

            # Step 3: Use Classifier to classify Parsed data
            if parsed_data is not None:
                if self.raw_data_format.upper() == "XML":
                    self.logger.info("XML element classification not needed. Using parsed_data.")
                    classified_links = parsed_data
                elif 'HTML' in self.raw_data_format.upper():
                    classified_links = parsed_data
                    self.logger.info("HTML element classification not supported. Using parsed_data.")
                elif self.raw_data_format.upper() == 'PDF':
                    classified_links = parsed_data
                    self.logger.info("PDF element classification not needed. Using parsed_data.")
                else:
                    self.logger.error(f"Unsupported raw data format and parser mode combination.")
                    return None
            else:
                raise ValueError("Parsed data is None. Cannot classify links.")

            # add the deduplication step here
            #classified_links = self.classifier.deduplicate_links(classified_links)

            if self.save_to_cache:
                self.save_func_output_to_cache(classified_links.to_dict(orient='records'), process_id, 'process_url')
            return classified_links

        except Exception as e:
            self.logger.error(f"Error processing URL {url}: {e}", exc_info=True)
            return None

    def app_process_url(self, url, save_staging_table=False, article_file_dir='tmp/raw_files/', 
                       use_portkey=True, driver_path=None, browser='Firefox', headless=True, 
                       prompt_name='GPT_FewShot', semantic_retrieval=False, section_filter=None):
        """
        Application wrapper for process_url with concurrent user support.
        This method handles rate limiting and resource management for multi-user scenarios.
        """
        with self._processing_semaphore:
            # Simple rate limiting
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_delay:
                wait_time = self._min_delay - elapsed
                self.logger.info(f"Rate limiting: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            self._last_request_time = time.time()
            
            try:
                # Call the original process_url method unchanged
                return self.process_url(
                    url=url,
                    save_staging_table=save_staging_table,
                    article_file_dir=article_file_dir,
                    use_portkey=use_portkey,
                    driver_path=driver_path,
                    browser=browser,
                    headless=headless,
                    prompt_name=prompt_name,
                    semantic_retrieval=semantic_retrieval,
                    section_filter=section_filter
                )
            except Exception as e:
                self.logger.error(f"Error in app_process_url wrapper: {e}")
                raise

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

    def process_articles(self, url_list, log_modulo=10, save_staging_table=False, article_file_dir='tmp/raw_files/',
                         driver_path=None, browser='Firefox', headless=True, use_portkey=True, response_format=dataset_response_schema_gpt,
                         prompt_name='GPT_FewShot', semantic_retrieval=False, section_filter=None, grobid_for_pdf=False, write_htmls_xmls=False):
        """
        Processes a list of article URLs and returns parsed data.

        :param url_list: List of URLs/PMCIDs to process.

        :param log_modulo: Frequency of logging progress (useful when url_list is long).

        :param save_staging_table: Flag to save the staging table.

        :param article_file_dir: Directory to save the raw HTML/XML/PDF files.

        :param driver_path: Path to your local WebDriver executable (if applicable). When set to None, Webdriver manager will be used.

        :param browser: Browser to use for scraping (if applicable). Supported values are 'Firefox', 'Chrome'.

        :param headless: Whether to run the browser in headless mode (if applicable).

        :param use_portkey: Flag to use Portkey for Gemini LLM.

        :param response_format: The response schema to use for parsing the data.

        :param prompt_name: Name of the prompt to use for LLM parsing.

        :param semantic_retrieval: Flag to indicate if semantic retrieval should be used.

        :param section_filter: Optional filter to apply to the sections (supplementary_material', 'data_availability_statement').

        :param grobid_for_pdf: Flag to indicate if GROBID should be used for PDF processing.

        :return: Dictionary with URLs as keys and DataFrames of classified data as values.
        """

        self.logger.debug("Starting to process URL list...")
        start_time = time.time()
        total_iters = len(url_list)
        results = {}

        for iteration, url in enumerate(url_list):
            url = self.preprocess_url(url)
            self.logger.info(f"#{iteration} function call: self.process_url({url})")

            results[url] = self.process_url(
                url,
                save_staging_table=save_staging_table,
                article_file_dir=article_file_dir,
                driver_path=driver_path,
                browser=browser,
                headless=headless,
                use_portkey=use_portkey,
                prompt_name=prompt_name,
                semantic_retrieval=semantic_retrieval,
                section_filter=section_filter,
                response_format=response_format,
                grobid_for_pdf=grobid_for_pdf,
                write_htmls_xmls=write_htmls_xmls
            )

            if iteration % log_modulo == 0:
                elapsed = time.time() - start_time  # Time elapsed since start
                avg_time_per_iter = elapsed / (iteration + 1)  # Average time per iteration
                remaining_iters = total_iters - (iteration + 1)
                estimated_remaining = avg_time_per_iter * remaining_iters  # Estimated time remaining
                self.logger.info(
                    f"\nProgress: {iteration + 1}/{total_iters} ({(iteration + 1) / total_iters * 100:.2f}%) "
                    f"| Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} "
                    f"| ETA: {time.strftime('%H:%M:%S', time.gmtime(estimated_remaining))}\n"
                )
        self.logger.debug("Completed processing all URLs.")
        # rename 'dataset_id', 'repository_reference' to 'dataset_identifier', 'data_repository' respectively
        for url, df in results.items():
            if df is not None and not df.empty:
                if 'dataset_id' in df.columns:
                    df.rename(columns={'dataset_id': 'dataset_identifier'}, inplace=True)
                if 'repository_reference' in df.columns:
                    df.rename(columns={'repository_reference': 'data_repository'}, inplace=True)
        return results
    

    def summarize_result(self, df):
        """
        Summarizes the result of 1 processed URL.

        :param df: Dataframe of candidate datasets from source article.

        :return: Summary dict with URL, number of classified links, and additional metadata.
        """
        self.logger.info(f"Summarizing results...{df.columns}")
        if df is not None and not df.empty:
            if 'file_extension' in df.columns:
                file_ext_counts = df['file_extension'].dropna().value_counts().to_dict()
            else:
                file_ext_counts = {}
            repo_counts = df[
                'data_repository'].dropna().value_counts().to_dict() if 'data_repository' in df.columns else {}

            summary = {
                'number_of_data_objects_extracted': len(df),
                'frequency_of_file_extensions': file_ext_counts,
                'frequency_of_data_repository': repo_counts,
            }

            return summary

        else:
            empty_summary = {
                'number_of_data_objects_extracted': 0,
                'frequency_of_file_extensions': {},
                'frequency_of_data_repository': {},
            }
            return empty_summary

    def load_urls_from_input(self, input_file):
        """
        Loads URLs from the input file.

        :param input_file: Path to the input file containing URLs.

        :return: List of URLs loaded from the file.
        """
        self.logger.debug(f"Loading URLs from file: {input_file}")
        if not os.path.exists(str(input_file)):
            if isinstance(input_file, str):
                return [input_file.strip()]
            elif isinstance(input_file, list):
                return input_file
        try:
            with open(input_file, 'r') as file:
                url_list = [line.strip() for line in file]
            self.logger.info(f"Loaded {len(url_list)} URLs from file.")
            self.url_list = url_list
            return url_list
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Create file with input links! File not found: {input_file}\n\n{e}\n")

    def process_metadata(self, combined_df, display_type='console', interactive=True, return_metadata=False,
                         write_raw_metadata=False, article_file_dir='tmp/raw_files/', use_portkey=True,
                         prompt_name='gpt_metadata_extract', timeout=1):
        """
        This method iterates through the combined_df DataFrame, checks for dataset webpages or download links,

        :param combined_df: DataFrame containing the data to preview. It should contain columns like 'dataset_webpage', 'download_link', etc.

        :param display_type: Type of display for the preview. Options are 'console', 'html', or 'json'.

        :param interactive: If True, allows user interaction for displaying data previews.

        :param return_metadata: If True, returns a list of metadata dictionaries instead of displaying them.

        :param write_raw_metadata: If True, saves raw metadata to the specified directory.

        :param article_file_dir: Directory to save raw HTML/XML files if write_raw_metadata is True.

        :param use_portkey: If True, uses Portkey for Gemini LLM.

        :param prompt_name: Name of the prompt to use for LLM parsing.

        :param timeout: Timeout for requests to fetch dataset webpages.

        :return: If return_metadata is True, returns a list of metadata dictionaries. Otherwise, displays the data preview.
        """

        self.logger.info(f"Processing metadata for preview to display metadata preview in {display_type} format.")

        self.already_previewed = []
        self.metadata_parser = HTMLParser(self.open_data_repos_ontology, self.logger, full_document_read=True,
                                          llm_name=self.llm, use_portkey=use_portkey)

        self.data_fetcher = self.data_fetcher.update_DataFetcher_settings('any_url',
                                                                          self.full_document_read,
                                                                          self.logger,
                                                                          driver_path=None,
                                                                          browser='Firefox',
                                                                          headless=True)

        if isinstance(self.data_fetcher, WebScraper):
            self.logger.info("Found WebScraper to fetch data.")
        else:
            raise TypeError(f"DataFetcher must be an instance of WebScraper to fetch dataset webpages. Found {type(self.data_fetcher).__name__} instead.")

        if return_metadata:
            ret_list = []

        if isinstance(combined_df, pd.Series):
            combined_df = combined_df.to_frame().T

        for i, row in combined_df.iterrows():
            self.logger.info(f"Row # {i}")
            self.logger.debug(f"Row keys: {row}")

            dataset_webpage = row.get('dataset_webpage', None)
            download_link = row.get('download_link', None)
            dataset_webpage_id = self.url_to_article_id(dataset_webpage) if dataset_webpage is not None else None

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
                if (row.get('file_extension', None) is not None and 'data' not in row['source_section'] and row[
                    'file_extension'] not
                        in ['xlsx', 'csv', 'json', 'xml', 'zip']):
                    self.logger.info(
                        f"Skipping row {i} as it does not contain a valid dataset webpage or file extension.")
                    continue
                else:
                    self.logger.info(f"Potentially a valid dataset, displaying hardscraped metadata")
                    #metadata = self.metadata_parser.parse_datasets_metadata(row['source_section'])
                    hardscraped_metadata = {k: v for k, v in row.items() if
                                            v is not None and v not in ['nan', 'None', '', 'n/a', np.nan, 'NaN', 'na']}
                    self.already_previewed.append(download_link)
                    if self.download_data_for_description_generation:
                        split_source_url = hardscraped_metadata.get('source_url').split('/')
                        paper_id = split_source_url[-1] if len(split_source_url[-1]) > 0 else split_source_url[-2]
                        self.data_fetcher.download_file_from_url(download_link, "scripts/downloads/suppl_files", paper_id)
                        hardscraped_metadata[
                            'data_description_generated'] = self.metadata_parser.generate_dataset_description(
                            download_link)
                    self.display_metadata(hardscraped_metadata, display_type=display_type, interactive=interactive)
                    continue

            else:
                self.logger.info(f"LLM scraped metadata")
                repo_mapping_key = row['repository_reference'].lower() if 'repository_reference' in row else row[
                    'data_repository'].lower()
                resolved_key = self.metadata_parser.resolve_data_repository(repo_mapping_key)

                # caching: load_from_cache
                skip, cache = False, {}
                process_id = self.llm + "-" + dataset_webpage_id
                if self.load_from_cache and os.path.exists(os.path.join(CACHE_BASE_DIR, "process_metadata_cache.json")):
                    cache = json.load(open(os.path.join(CACHE_BASE_DIR, "process_metadata_cache.json"), 'r'))
                    if process_id in cache:
                        metadata, skip = cache[process_id], True

                if ('javascript_load_required' in self.open_data_repos_ontology['repos'][resolved_key]) and not skip:
                    self.logger.info(
                        f"JavaScript load required for {repo_mapping_key} dataset webpage. Using WebScraper.")
                    html = self.data_fetcher.fetch_data(row['dataset_webpage'], delay=5)
                    if "informative_html_metadata_tags" in self.open_data_repos_ontology['repos'][resolved_key]:
                        html = self.metadata_parser.normalize_HTML(html, self.open_data_repos_ontology['repos'][
                            resolved_key]['informative_html_metadata_tags'])
                    else:
                        html = self.metadata_parser.normalize_HTML(html)
                    if write_raw_metadata:
                        self.logger.info(f"Saving raw metadata to: {article_file_dir + 'raw_metadata/'}")
                        self.data_fetcher.html_page_source_download(article_file_dir + 'raw_metadata/')
                elif not skip:
                    if 'informative_html_metadata_tags' in self.open_data_repos_ontology['repos'][resolved_key]:
                        keep_sect = self.open_data_repos_ontology['repos'][resolved_key][
                            'informative_html_metadata_tags']
                    else:
                        keep_sect = None
                    response = requests.get(row['dataset_webpage'], timeout=timeout)
                    html = self.metadata_parser.normalize_HTML(response.text, keep_tags=keep_sect)

                if not skip:
                    metadata = self.metadata_parser.parse_datasets_metadata(html,
                                                                            use_portkey=use_portkey,
                                                                            prompt_name=prompt_name)
                    metadata['source_url_for_metadata'] = row['dataset_webpage']
                    metadata['access_mode'] = row.get('access_mode', None)
                    metadata['source_section'] = row.get('source_section', row.get('section_class', None))
                    metadata['download_link'] = row.get('download_link', None)
                    metadata['accession_id'] = row.get('dataset_id', row.get('dataset_identifier', None))
                    metadata['data_repository'] = repo_mapping_key
                    self.already_previewed.append(row['dataset_webpage'])

            metadata['paper_with_dataset_citation'] = row['source_url']

            if self.save_to_cache:
                self.logger.debug(f"Saving metadata to cache for process ID: {process_id}")
                self.save_func_output_to_cache(metadata, process_id, 'process_metadata')

            if return_metadata:
                flat_metadata = self.metadata_parser.flatten_metadata_dict(metadata)
                ret_list.append(flat_metadata)

            self.display_metadata(metadata, display_type=display_type, interactive=interactive)

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

    def display_metadata(self, metadata, display_type='console', interactive=True):
        """
        Display extracted metadata as a clean table in both Jupyter and terminal environments.

        :param metadata: Dictionary containing metadata to display.

        :param display_type: Type of display for the preview. Options are 'console' or 'ipynb'.

        :param interactive: If True, allows user interaction for displaying data previews.
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
                self.logger.info("No usable metadata found.")
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
                        self.logger.info("Queued for download.")
                    else:
                        self.logger.info("User declined download.")
                        self.logger.info("Skipped.")

            confirm_button.on_click(lambda _: confirm_handler())

            # Show the checkbox + button
            ui_box = widgets.VBox([checkbox, confirm_button, output])
            display(ui_box)
            time.sleep(1)

        else:
            self.logger.warning(f"Unsupported display type: {display_type}. Cannot display metadata preview.")
            return

    def download_data_resources(self, output_root="scripts/downloads/suppl_files"):
        """
        Function to download all the files that were previewed and confirmed for download.

        :param output_root: Root directory where the files will be downloaded.

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
        Function to get the internal ID of the dataset from metadata.

        :param metadata: Dictionary containing metadata of the dataset.

        :return: Internal ID of the dataset if found, otherwise None.
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

    def check_required_sections(self, raw_data, url, required_sections):
        """
        Checks if the raw data contains all the required sections.

        :param raw_data: Raw data fetched from the source URL.

        :param url: Source URL from which the raw data was fetched.

        :param required_sections: List of required sections to check in the raw data.

        :return: True if all required sections are present, False otherwise.

        """
        required_sections = [sect + "_sections" for sect in required_sections]
        return self.data_checker.is_xml_data_complete(raw_data, url, required_sections)

    def url_to_article_id(self, url):
        url = re.sub(r'^https?://', '', url)
        article_id = re.sub(r'[^A-Za-z0-9]', '_', url)
        if article_id.endswith('_'):
            article_id = article_id[:-1]
        return article_id

    def save_func_output_to_cache(self, output, process_id, function_name):
        """
        Save output to cache file in a thread/process-safe and atomic way.
        Uses filelock for locking and atomic write via .tmp file and os.replace.
        """
        from filelock import FileLock
        cache_file = os.path.join(CACHE_BASE_DIR, function_name + "_cache.json")
        lock_file = cache_file + ".lock"
        tmp_file = cache_file + ".tmp"
        self.logger.info(f"Saving results to cache with process_id: {process_id}")
        os.makedirs(CACHE_BASE_DIR, exist_ok=True)
        with FileLock(lock_file):
            cache = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Could not read cache file {cache_file}: {e}. Overwriting.")
                    cache = {}
            if process_id not in cache:
                self.logger.info(f"Saving results to cache with process_id: {process_id}")
                cache[process_id] = output
                try:
                    with open(tmp_file, 'w') as f:
                        json.dump(cache, f, indent=4)
                    os.replace(tmp_file, cache_file)  # atomic move
                except Exception as e:
                    self.logger.error(f"Error writing cache file {cache_file}: {e}")
            else:
                self.logger.debug(f"Process ID {process_id} already exists in cache. Skipping save.")

    def run(self, input_file='input/test_input.txt', semantic_retrieval=False, section_filter=None,
            prompt_name='GPT_FewShot'):
        """
        This method orchestrates the entire data gathering process by performing the following steps:

        1. Setup data fetcher (web scraper or API client)

        2. Load URLs from input_file

        3. Process each URL and return results as a dictionary like source_url: DataFrame_of_data_links

        4. Write results to output file specified in configuration file

        :param input_file: Path to the input file containing URLs or PMCIDs to process.

        :param semantic_retrieval: Flag to indicate if semantic retrieval should be used.

        :param section_filter: Optional filter to apply to the sections (supplementary_material', 'data_availability_statement').

        :param prompt_name: Name of the prompt to use for LLM parsing.

        :return: Combined DataFrame of all processed data links.

        """
        self.logger.debug("DataGatherer run started.")
        try:

            # Load URLs from input file
            urls = self.load_urls_from_input(input_file)

            # Process each URL and return results as a dictionary like source_url: DataFrame_of_data_links
            results = self.process_articles(urls, semantic_retrieval=semantic_retrieval, section_filter=section_filter,
                                            prompt_name=prompt_name, driver_path=self.fetcher_driver_path)

            # return the union of all the results
            combined_df = pd.DataFrame()
            for url, df in results.items():
                combined_df = pd.concat([combined_df, df], ignore_index=True)

            # evaluate the performance if ground_truth is provided
            #if 'ground_truth' in self.config:
            # self.logger.info("Evaluating performance...")
            # self.classifier.evaluate_performance(combined_df, self.config['ground_truth'])

            if self.data_resource_preview:
                self.process_metadata(combined_df)

                if self.download_previewed_data_resources:
                    self.download_data_resources()

            combined_df.to_csv(self.full_output_file, index=False)

            self.logger.info(f"Output written to file: {self.full_output_file}")

            self.logger.info(f"File Download Schedule: {self.downloadables}")

            self.logger.debug("DataGatherer run completed.")

            return combined_df

        except Exception as e:
            self.logger.error(f"Error in orchestrator run: {e}", exc_info=True)
            return None

        finally:
            # Quit the driver to close the browser and free up resources
            if isinstance(self.data_fetcher, WebScraper):
                self.logger.info("Quitting the WebDriver.")
                self.data_fetcher.scraper_tool.quit()

            if isinstance(self.data_fetcher, EntrezFetcher):
                self.logger.info("Closing the EntrezFetcher.")
                self.data_fetcher.api_client.close()

    def run_integrated_batch_processing(
        self,
        url_list,
        batch_file_path,
        output_file_path=None,
        api_provider='openai',
        prompt_name='GPT_FewShot',
        response_format=None,
        temperature=0.0,
        semantic_retrieval=False,
        section_filter=None,
        submit_immediately=True,
        wait_for_completion=False,
        poll_interval=60,
        batch_description=None,
        grobid_for_pdf=False,
        use_portkey=True):
        """
        Complete integrated batch processing using LLMClient batch functionality.
        
        This method leverages the new LLMClient batch processing capabilities for
        improved performance and proper separation of concerns.
        
        :param url_list: List of URLs/PMCIDs to process
        :param batch_file_path: Path for the batch JSONL file
        :param output_file_path: Path for the results file (auto-generated if None)
        :param api_provider: 'openai' or 'portkey'
        :param prompt_name: Name of the prompt template
        :param response_format: Response schema
        :param temperature: Model temperature
        :param semantic_retrieval: Enable semantic retrieval
        :param section_filter: Section filter
        :param submit_immediately: Whether to submit the batch job immediately
        :param wait_for_completion: Whether to wait for batch completion
        :param poll_interval: Seconds between status checks
        :param batch_description: Optional description for the batch job
        :return: Dictionary with batch information and results
        """

        self.logger.info(f"Starting integrated batch processing for {len(url_list)} URLs")
        
        try:
            # Step 1: Fetch data
            self.logger.info("Step 1: Fetching data...")
            fetched_data = self.fetch_data(url_list)
            
            # Count raw_data_format frequencies and store URLs for parser reuse optimization
            format_counts = {}
            for url, data in fetched_data.items():
                if data and 'raw_data_format' in data:
                    fmt = data['raw_data_format']
                    if fmt not in format_counts:
                        format_counts[fmt] = {'count': 0, 'urls': []}
                    format_counts[fmt]['count'] += 1
                    format_counts[fmt]['urls'].append(url)
            
            # Log format frequencies (counts only for readability)
            frequency_summary = {fmt: info['count'] for fmt, info in format_counts.items()}
            self.logger.info(f"Fetched {len(fetched_data)} Papers. Format frequencies: {frequency_summary}")
            self.logger.debug(f"Detailed fetched data: {fetched_data}")
            
            # Step 2: Prepare batch requests for LLMClient (parser per URL)
            
            batch_requests, cnt, last_url_raw_data_format = [], 0, False
            for url_raw_data_format, vals in format_counts.items():
                for url in vals['urls']:
                    try:                        
                        if cnt != 0 and url_raw_data_format == last_url_raw_data_format:
                            self.logger,info(f"Reusing existing parser of name: {self.parser.__class__.__name__}")
                        else:
                            if url_raw_data_format.upper() == "XML":
                                self.logger.info("Initializing XMLParser to parse data.")
                                self.parser = XMLParser(
                                    self.open_data_repos_ontology,
                                    self.logger,
                                    llm_name=self.llm,
                                    full_document_read=self.full_document_read,
                                    use_portkey=use_portkey,
                                    save_dynamic_prompts=self.save_dynamic_prompts
                                    )

                            elif url_raw_data_format.upper() == 'HTML':
                                self.logger.info("Initializing HTMLParser to parse data.")
                                self.parser = HTMLParser(
                                    self.open_data_repos_ontology, 
                                    self.logger,
                                    llm_name=self.llm,
                                    full_document_read=self.full_document_read,
                                    use_portkey=use_portkey,
                                    save_dynamic_prompts=self.save_dynamic_prompts
                                    )

                            elif url_raw_data_format.upper() == 'PDF':
                                self.logger.info("Using PDFParser to parse data.")
                                if grobid_for_pdf:
                                    self.logger.info("GROBID PDF parsing enabled.")
                                    self.parser = GrobidPDFParser(
                                        self.open_data_repos_ontology,
                                        self.logger,
                                        llm_name=self.llm,
                                        full_document_read=self.full_document_read,
                                        use_portkey=use_portkey,
                                        save_dynamic_prompts=self.save_dynamic_prompts,
                                        write_XML=write_htmls_xmls
                                        )
                                else:
                                    self.parser = PDFParser(
                                        self.open_data_repos_ontology, 
                                        self.logger,
                                        llm_name=self.llm,
                                        full_document_read=self.full_document_read,
                                        use_portkey=use_portkey,
                                        save_dynamic_prompts=self.save_dynamic_prompts
                                        )
                                         
                        data = fetched_data[url]
                        
                        # Generate unique custom_id
                        article_id = self.url_to_article_id(url)
                        timestamp = int(time.time() * 1000)
                        custom_id = f"{self.llm}_{article_id}_{timestamp}"
                        custom_id = re.sub(r'[^a-zA-Z0-9_-]', '_', custom_id)[:64]
                        
                        # Normalize input data based on actual format
                        if url_raw_data_format.upper() == 'XML':
                            normalized_input = (self.parser.normalize_XML(data['fetched_data']) 
                                            if hasattr(self.parser, 'normalize_XML') 
                                            else data['fetched_data'])
                        elif url_raw_data_format.upper() == 'HTML':
                            normalized_input = (self.parser.normalize_HTML(data['fetched_data']) 
                                            if hasattr(self.parser, 'normalize_HTML') 
                                            else data['fetched_data'])
                        elif url_raw_data_format.upper() == 'PDF':
                            normalized_input = data['fetched_data']
                        else:
                            raise ValueError(f"Unsupported raw data format: {url_raw_data_format}")
                        
                        # Render prompt using the correct parser
                        static_prompt = self.parser.prompt_manager.load_prompt(prompt_name)
                        messages = self.parser.prompt_manager.render_prompt(
                            static_prompt,
                            entire_doc=self.full_document_read,
                            content=normalized_input,
                            repos=', '.join(self.parser.repo_names) if hasattr(self.parser, 'repo_names') else '',
                            url=url,
                            section_filter=section_filter
                        )
                        
                        # Create batch request for LLMClient
                        batch_request = {
                            'custom_id': custom_id,
                            'messages': messages,
                            'metadata': {
                                'url': url,
                                'article_id': article_id,
                                'raw_data_format': url_raw_data_format
                            }
                        }
                        
                        batch_requests.append(batch_request)
                        
                    except Exception as e:
                        self.logger.error(f"Error preparing request for {url}: {e}")
                        continue

                    last_data_format = url_raw_data_format
                    cnt+=1
            
            self.logger.info(f"Prepared {len(batch_requests)} batch requests")
            
            # Step 3: Use LLMClient to handle batch processing
            self.logger.info("Step 3: Creating batch file using LLMClient...")
            
            # Use LLMClient's batch processing capabilities
            batch_result = self.parser.llm_client._handle_batch_mode(
                batch_requests=batch_requests,
                batch_file_path=batch_file_path,
                temperature=temperature,
                response_format=response_format,
                api_provider=api_provider
            )
            
            result = {
                'batch_file_created': batch_result,
                'fetched_data_count': len(fetched_data),
                'processed_requests': len(batch_requests),
                'api_provider': api_provider,
                'model': self.llm
            }
            
            # Step 4: Submit batch job if requested
            if submit_immediately:
                self.logger.info("Step 4: Submitting batch job...")
                
                submission_result = self.parser.llm_client.submit_batch_job(
                    batch_file_path=batch_file_path,
                    api_provider=api_provider,
                    batch_description=batch_description
                )
                
                result['batch_submission'] = submission_result
                batch_id = submission_result['batch_id']
                
                self.logger.info(f"Batch job submitted successfully. ID: {batch_id}")
                
                # Step 5: Wait for completion if requested
                if wait_for_completion:
                    self.logger.info(f"Step 5: Waiting for batch completion (polling every {poll_interval}s)...")
                    
                    while True:
                        status_info = self.parser.llm_client.check_batch_status(
                            batch_id=batch_id,
                            api_provider=api_provider
                        )
                        
                        status = status_info['status']
                        self.logger.info(f"Batch status: {status}")
                        
                        if status == 'completed':
                            # Download results
                            if not output_file_path:
                                output_file_path = batch_file_path.replace('.jsonl', '_results.jsonl')
                            
                            download_result = self.parser.llm_client.download_batch_results(
                                batch_id=batch_id,
                                output_file_path=output_file_path,
                                api_provider=api_provider
                            )
                            
                            result['batch_results'] = download_result
                            result['output_file_path'] = output_file_path
                            
                            self.logger.info(f"Batch processing completed successfully. Results saved to: {output_file_path}")
                            break
                            
                        elif status in ['failed', 'expired', 'cancelled']:
                            self.logger.error(f"Batch job failed with status: {status}")
                            result['error'] = f"Batch job failed with status: {status}"
                            break
                            
                        else:
                            # Still processing, wait and check again
                            time.sleep(poll_interval)
                    
                    result['final_status'] = status_info
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in integrated batch processing: {e}", exc_info=True)
            raise

    def split_jsonl_and_submit(self, 
                              batch_file_path: str,
                              max_file_size_mb: float = 200.0,
                              api_provider: str = 'openai',
                              wait_between_submissions: int = 30,
                              batch_description: str = None) -> Dict[str, Any]:
        """
        Simple function to chunk large JSONL files and submit them sequentially.
        
        This function ONLY handles chunking and submission - no monitoring or result combination.
        Use llm_client methods for monitoring batch completion and combining results.
        
        :param batch_file_path: Path to the large JSONL batch file
        :param max_file_size_mb: Maximum size per chunk in MB (default: 200MB for OpenAI)
        :param api_provider: API provider ('openai' or 'portkey')
        :param wait_between_submissions: Seconds to wait between chunk submissions
        :param batch_description: Description for the batch jobs
        :return: Dictionary with submission results
        """
        from data_gatherer.llm.batch_storage import BatchStorageManager
        
        self.logger.info(f"Starting split_jsonl_and_submit for file: {batch_file_path}")
        
        # Initialize batch storage manager
        batch_manager = BatchStorageManager(logger=self.logger)
        
        # Check if file exists and get size
        if not os.path.exists(batch_file_path):
            raise FileNotFoundError(f"Batch file not found: {batch_file_path}")
        
        file_size_mb = os.path.getsize(batch_file_path) / 1024 / 1024
        self.logger.info(f"Batch file size: {file_size_mb:.2f} MB")
        
        # Chunk
        self.logger.info("Chunking and submitting batches...")
        batches_chunked = batch_manager.chunk_batch_file(
            large_batch_file_path=batch_file_path,
            max_file_size_mb=max_file_size_mb,
        )

        if self.parser is None:
            self.parser = XMLParser(self.open_data_repos_ontology, self.logger, llm_name=self.llm)

        submission_results = []
        # Submit
        for batch in batches_chunked:
            chunk_info = batch['chunk_info']
            submission_results.append(
                self.parser.llm_client.submit_batch_job(
                    chunk_info['chunk_file_path'], 
                    api_provider=api_provider,
                    batch_description= f'''
                    chunk_number: {chunk_info['chunk_number']}, 
                    total_chunks: {chunk_info['total_chunks']},
                    chunk_file_path: {chunk_info['chunk_file_path']},
                    requests_in_chunk: {chunk_info['requests_in_chunk']},
                    chunk_size_mb: {chunk_info['chunk_size_mb']}
                    '''
                    )
                )
            1/0
        
        # Prepare result
        successful_submissions = [r for r in submission_results if 'batch_id' in r]
        failed_submissions = [r for r in submission_results if 'error' in r]
        
        result = {
            'original_file': batch_file_path,
            'original_size_mb': file_size_mb,
            'chunks_created': len(submission_results),
            'chunks_submitted': len(successful_submissions),
            'chunks_failed': len(failed_submissions),
            'submission_results': submission_results,
            'metadata_file': f"{os.path.splitext(batch_file_path)[0]}_chunking_metadata.json",
            'batch_ids': [r['batch_id'] for r in successful_submissions]
        }
        
        # Log submission summary
        self.logger.info(f"Chunking and submission complete:")
        self.logger.info(f"  Created {result['chunks_created']} chunks")
        self.logger.info(f"  Successfully submitted {result['chunks_submitted']} batches")
        self.logger.info(f"  Failed submissions: {result['chunks_failed']}")
        
        for i, sub_result in enumerate(successful_submissions):
            chunk_info = sub_result.get('chunk_info', {})
            self.logger.info(f"  Chunk {i+1}: Batch ID {sub_result['batch_id']} "
                           f"({chunk_info.get('requests_in_chunk', 'N/A')} requests)")
        
        return result

    def from_batch_resp_file_to_df(self, batch_results_file: str):
        """
        Convert a batch response JSONL file to a pandas DataFrame.
        This method processes batch API results and converts them to the standard DataFrame format.

        :param batch_results_file: Path to the JSONL batch results file.
        :return: DataFrame containing the processed dataset information.
        """
        self.logger.info(f"Converting batch response file to DataFrame: {batch_results_file}")
        
        try:
            # Step 1: Process batch responses using LLMClient
            batch_raw_resps = self.parser.llm_client.process_batch_responses(
                batch_results_file=batch_results_file,
                expected_key="datasets"
            )
            
            # Step 2: Process each response using the parser's post-processing logic
            processed_datasets = []
            
            for batch_item in batch_raw_resps['processed_results']:
                self.logger.debug(f"Processing batch item: {batch_item.keys()}")
                
                # Extract metadata
                custom_id = batch_item.get('custom_id', 'N/A')
                status = batch_item.get('status', 'unknown')
                
                if status != 'success':
                    self.logger.warning(f"Skipping failed batch item {custom_id}: {batch_item.get('error', 'Unknown error')}")
                    continue
                
                # Process the LLM response using parser's post-processing method
                processed_response = batch_item.get('processed_response', [])
                datasets = self.parser.process_datasets_response(processed_response)
                
                # Enhance each dataset with metadata
                for dataset in datasets:
                    # Add custom_id to track source
                    dataset['custom_id'] = custom_id
                    
                    # Reconstruct source URL if it's a PMC article
                    if re.search(r'_PMC\d+', custom_id, re.IGNORECASE):
                        pmc_match = re.search(r'PMC(\d+)', custom_id, re.IGNORECASE)
                        if pmc_match:
                            dataset['source_url'] = f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_match.group(1)}/'
                    
                    processed_datasets.append(dataset)
            
            # Step 3: Convert to DataFrame
            if processed_datasets:
                df = pd.DataFrame(processed_datasets)
                self.logger.info(f"Successfully converted batch results to DataFrame with {len(df)} rows")
                
                # Standardize column names (ensure compatibility with existing pipeline)
                if 'dataset_id' in df.columns:
                    df.rename(columns={'dataset_id': 'dataset_identifier'}, inplace=True)
                if 'repository_reference' in df.columns:
                    df.rename(columns={'repository_reference': 'data_repository'}, inplace=True)
                
                return df
            else:
                self.logger.warning("No valid datasets found in batch results")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error converting batch response file to DataFrame: {e}", exc_info=True)
            raise
        