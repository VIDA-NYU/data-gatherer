from abc import ABC, abstractmethod
import re
import logging
import numpy as np
from selenium.webdriver.common.by import By
import os
import time
import requests
from lxml import etree as ET
from data_gatherer.selenium_setup import create_driver
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd
from data_gatherer.retriever.xml_retriever import xmlRetriever
import tempfile

# Abstract base class for fetching data
class DataFetcher(ABC):
    def __init__(self, logger, src='WebScraper', driver_path=None, browser='firefox', headless=True,
                 raw_HTML_data_filepath=None):
        self.dataframe_fetch = True  # Flag to indicate dataframe fetch supported or not
        self.raw_HTML_data_filepath = raw_HTML_data_filepath
        self.logger = logger
        self.logger.warning(
            "DataFetcher raw_HTML_data_filepath set no None") if raw_HTML_data_filepath is None else None
        self.logger.debug("DataFetcher initialized.")
        self.driver_path = driver_path
        self.browser = browser
        self.headless = headless

    @abstractmethod
    def fetch_data(self, url, retries=3, delay=2):
        """
        Abstract method to fetch data from a given URL.

        :param url: The URL to fetch data from.

        :param retries: Number of retries in case of failure.

        :param delay: Delay time between retries.

        :return: The raw content of the page.
        """
        pass

    def url_to_publisher_domain(self, url):
        """
        Extracts the publisher domain from a given URL.
        """
        self.logger.debug(f"URL: {url}")
        if re.match(r'^https?://www\.ncbi\.nlm\.nih\.gov/pmc', url) or re.match(r'^https?://pmc\.ncbi\.nlm\.nih\.gov/', url):
            return 'PMC'
        if re.match(r'^https?://pubmed\.ncbi\.nlm\.nih\.gov/[\d]+', url):
            self.logger.info("Publisher: pubmed")
            return 'pubmed'
        match = re.match(r'^https?://(?:\w+\.)?([\w\d\-]+)\.\w+', url)
        if match:
            domain = match.group(1)
            self.logger.info(f"Publisher: {domain}")
            return domain
        else:
            return self.url_to_publisher_root(url)

    def url_to_publisher_root(self, url):
        """
        Extracts the root domain from a given URL.
        """
        self.logger.debug(f"Function call url_to_publisher_root: {url}")
        match = re.match(r'https?://([\w\.]+)/', url, re.IGNORECASE)
        if match:
            root = match.group(1)
            self.logger.info(f"Root: {root}")
            return root
        else:
            self.logger.warning("No valid root extracted from URL. This may cause issues with data gathering.")
            return 'Unknown Publisher'

    def url_to_pmcid(self, url):
        """
        Extracts the PMC ID from a given URL.

        :param url: The URL to extract the PMC ID from.

        :return: The extracted PMC ID or None if not found.
        """
        match = re.search(r'PMC(\d+)', url)
        if match:
            pmcid = f"PMC{match.group(1)}"
            self.logger.info(f"Extracted PMC ID: {pmcid}")
            return pmcid
        else:
            self.logger.warning(f"No PMC ID found in URL: {url}")
            return None

    def url_to_doi(self, url : str, candidate_pmcid=None):
        # Extract DOI from the URL
        url = url.lower()

        # url_doi mappings for different publishers
        url = re.sub(r'www=\.nature\.com/articles', '10.1038', url, re.IGNORECASE) # nature

        match = re.search(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', url, re.IGNORECASE)

        if match:
            doi = match.group(1)
            self.logger.info(f"DOI: {doi}")
            return doi

        elif candidate_pmcid is not None:
            return self.PMCID_to_doi(candidate_pmcid)

        else:
            return None

    def PMCID_to_doi(self, pmid):
        base_url = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=__ID__&format=json"

        url_request = base_url.replace('__ID__', pmid)
        response = requests.get(url_request, headers={"User-Agent": "Mozilla/5.0"})

        if response.status_code == 200:
            data = response.json()
            records = data.get("records")
            id = records[0] if records else None
            if 'doi' in id:
                doi = id['doi']
                return doi
        else:
            self.logger.info(f"Failed to fetch DOI for PMCID {pmid}. Status code: {response.status_code}")
            return None

    def url_to_filename(self, url):
        parsed_url = urlparse(url)
        return os.path.basename(parsed_url.path)

    def pmid_to_url(self, pubmed_id):
        return f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"

    def PMCID_to_url(self, PMCID):
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{PMCID}"

    def update_DataFetcher_settings(self, url, entire_doc_model, logger, HTML_fallback=False, driver_path=None,
                                    browser='firefox', headless=True, raw_HTML_data_filepath=None):
        """
        Sets up either a web scraper or API client based on the URL domain.
        Also used to avoid re_instantiating another selenium webdriver.

        :param url: The URL to fetch data from.

        :param entire_doc_model: Flag to indicate if the entire document model is being used.

        :param logger: The logger instance for logging messages.

        :param HTML_fallback: Flag to indicate if HTML fallback is needed.

        :return: An instance of the appropriate data fetcher (WebScraper or EntrezFetcher).
        """
        self.logger.debug(f"update_DataFetcher_settings for current URL")

        API = None

        if not HTML_fallback:
            API = self.url_to_api_root(url)

        self.raw_HTML_data_filepath = raw_HTML_data_filepath if raw_HTML_data_filepath else self.raw_HTML_data_filepath
        self.logger.info(f"raw_HTML_data_filepath: {self.raw_HTML_data_filepath}")

        if self.raw_HTML_data_filepath and self.dataframe_fetch and self.url_in_dataframe(url, self.raw_HTML_data_filepath):
            self.logger.info(f"URL {url} found in DataFrame. Using DatabaseFetcher.")
            return DatabaseFetcher(logger, self.raw_HTML_data_filepath)

        if API == 'PMC':
            self.logger.info(f"Initializing EntrezFetcher({'requests', 'self.config'})")
            return EntrezFetcher(requests, logger)

        # Reuse existing driver if we already have one
        if isinstance(self, WebScraper) and self.scraper_tool is not None:
            self.logger.info(f"Reusing existing WebScraper driver: {self.scraper_tool}")
            return self  # Reuse current instance

        if self.url_is_pdf(url):
            self.logger.info(f"URL {url} is a PDF. Using PdfFetcher.")
            return PdfFetcher(logger)

        self.logger.info(f"WebScraper instance: {isinstance(self, WebScraper)}")
        self.logger.info(f"EntrezFetcher instance: {isinstance(self, EntrezFetcher)}")
        self.logger.info(f"scraper_tool attribute: {hasattr(self,'scraper_tool')}")

        self.logger.info(f"Initializing new selenium driver {driver_path}, {browser}, {headless}.")
        driver = create_driver(driver_path, browser, headless, self.logger)
        return WebScraper(driver, logger)

    def url_in_dataframe(self, url, raw_HTML_data_filepath):
        """
        Checks if the given doi / pmcid is present in the DataFrame.

        :param url: The URL to check.

        :return: True if the URL is found, False otherwise.
        """
        pmcid = re.search(r'PMC\d+', url, re.IGNORECASE)
        pmcid = pmcid.group(0) if pmcid else None

        df_fetch = pd.read_parquet(raw_HTML_data_filepath)

        return True if pmcid and pmcid.lower() in df_fetch['publication'].values else False

    def url_to_api_root(self, url):

        API_supported_url_patterns = {
            'https://www.ncbi.nlm.nih.gov/pmc/articles/': 'PMC',
            'https://pmc.ncbi.nlm.nih.gov/': 'PMC'
        }

        # Check if the URL corresponds to any API_supported_url_patterns
        for ptr, src in API_supported_url_patterns.items():
            self.logger.debug(f"Checking {src} with pattern {ptr}")
            match = re.match(ptr, url)
            if match:
                self.logger.debug(f"URL detected as {src}.")
                return src
        self.logger.debug("No API pattern matched.")

    def url_is_pdf(self, url):
        """
        Checks if the given URL points to a PDF file.

        :param url: The URL to check.

        :return: True if the URL points to a PDF file, False otherwise.
        """
        self.logger.debug(f"Checking if URL is a PDF: {url}")
        if url.lower().endswith('.pdf'):
            self.logger.info(f"URL {url} ends with .pdf")
            return True
        elif re.search(r'arxiv\.org/pdf/', url, re.IGNORECASE):
            return True
        return False

    def download_file_from_url(self, url, output_root="output/suppl_files", paper_id=None):
        output_dir = os.path.join(output_root, paper_id)
        os.makedirs(output_dir, exist_ok=True)
        filename = url.split("/")[-1]
        path = os.path.join(output_dir, filename)

        headers = {
            "User-Agent": "Mozilla/5.0",
            # Add cookies or headers if needed
        }

        r = requests.get(url, stream=True, headers=headers)

        if "Preparing to download" in r.text[:100]:  # Detect anti-bot response
            raise ValueError("Page blocked or JS challenge detected.")

        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
            self.logger.info(f"Downloaded {filename} to {path}")

        return path


# Implementation for fetching data via web scraping
class WebScraper(DataFetcher):
    """
    Class for fetching data from web pages using Selenium.
    """
    def __init__(self, scraper_tool, logger, retrieval_patterns_file=None, driver_path=None, browser='firefox',
                 headless=True, local_fetch_fp=None):
        super().__init__(logger, src='WebScraper', raw_HTML_data_filepath=local_fetch_fp, driver_path=driver_path,
                         browser=browser, headless=headless)
        self.scraper_tool = scraper_tool  # Inject your scraping tool (Selenium)
        self.driver_path = driver_path
        self.browser = browser
        self.headless = headless
        self.logger.debug("WebScraper initialized.")

    def fetch_data(self, url, retries=3, delay=2):
        """
        Fetches data from the given URL, by simulating user scroll, waiting some delay time and doing multiple retries
        to allow for page load and then get the source html

        :param url: The URL to fetch data from.

        :param retries: Number of retries in case of failure.

        :param delay: Delay time between retries.

        :return: The raw HTML content of the page.
        """
        # Use the scraper tool to fetch raw HTML from the URL
        self.raw_data_format = 'HTML'  # Default format for web scraping
        self.logger.debug(f"Fetching data with function call: self.scraper_tool.get(url)")
        self.scraper_tool.get(url)
        self.logger.debug(f"http get complete, now waiting {delay} seconds for page to load")
        self.simulate_user_scroll(delay)
        self.title = self.extract_publication_title()
        return self.scraper_tool.page_source

    def remove_cookie_patterns(self, html: str):
        pattern = r'<img\s+alt=""\s+src="https://www\.ncbi\.nlm\.nih\.gov/stat\?.*?"\s*>'
        # the patterns that change every time you visit the page and are not relevant to data-gatherer
        # ;cookieSize = 93 & amp;
        # ;jsperf_basePage = 17 & amp;
        # ;ncbi_phid = 993
        # CBBA47A4F74F305BBA400333DB8BA.m_1 & amp;

        if re.search(pattern, html):
            self.logger.info("Removing cookie pattern 1 from HTML")
            html = re.sub(pattern, 'img_alt_subst', html)
        else:
            self.logger.info("No cookie pattern 1 found in HTML")
        return html

    def simulate_user_scroll(self, delay=2, scroll_wait=0.5):
        time.sleep(delay)
        last_height = self.scraper_tool.execute_script("return document.body.scrollHeight")
        while True:
            # Scroll down to bottom
            self.scraper_tool.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(scroll_wait + np.random.random())

            # Calculate new height and compare with last height
            new_height = self.scraper_tool.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def html_page_source_download(self, directory):
        """
        Downloads the HTML page source as html file in the specified directory.

        :param directory: The directory where the HTML file will be saved.

        """
        if os.path.exists(directory):
            logging.info(f"Dir {directory} exists")
        else:
            os.makedirs(directory, exist_ok=True)

        if hasattr(self, 'extract_publication_title'):
            pub_name = self.extract_publication_title()

        else:
            raise Exception("Pubblication name extraction is only supported for WebScraper instances.")
        pub_name = re.sub(r'[\\/:*?"<>|]', '_', pub_name)  # Replace invalid characters in filename

        fn = directory + pub_name + '.html'
        self.logger.info(f"Downloading HTML page source to {fn}")

        self.logger.debug(f"scraper_tool: {self.scraper_tool}, page_source: {getattr(self.scraper_tool, 'page_source', None)}")

        if hasattr(self, 'scraper_tool') and self.scraper_tool.page_source:
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(self.scraper_tool.page_source)
        else:
            raise RuntimeError(f"Error saving HTML page source to {fn}. scraper_tool or page_source not available.")

    def extract_publication_title(self):
        """
        Extracts the publication name from the WebDriver's current page title. **Remark**: this should be called after
        scraper_tool.get(url) to ensure the page is loaded.

        :return: The publication name as a string.

        """
        self.logger.debug(f"Extracting publication title from page source")
        publication_name_pointer = self.scraper_tool.find_element(By.TAG_NAME, 'title')
        self.logger.debug(f"Publication name pointer: {publication_name_pointer}")
        if publication_name_pointer is not None and publication_name_pointer.text:
            publication_name = publication_name_pointer.text
            publication_name = re.sub(r"\n+", "", publication_name)
            publication_name = re.sub(r"^\s+", "", publication_name)
            self.logger.info(f"Paper name: {publication_name}")
            return publication_name
        else:
            self.logger.warning("Publication name not found in the page title.")
            return "No title found"

    def get_PMCID_from_pubmed_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        # Extract PMC ID
        pmc_tag = soup.find("a", {"data-ga-action": "PMCID"})
        pmc_id = pmc_tag.text.strip() if pmc_tag else None  # Extract text safely
        self.logger.info(f"PMCID: {pmc_id}")
        return pmc_id

    def get_doi_from_pubmed_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        # Extract DOI
        doi_tag = soup.find("a", {"data-ga-action": "DOI"})
        doi = doi_tag.text.strip() if doi_tag else None  # Extract text safely
        self.logger.info(f"DOI: {doi}")
        return doi

    def get_opendata_from_pubmed_id(self, pmid):
        """
        Given a PubMed ID, fetches the corresponding PMC ID and DOI from PubMed.

        :param pmid: The PubMed ID to fetch data for.

        :return: A tuple containing the PMC ID and DOI.

        """
        url = self.pmid_to_url(pmid)
        self.logger.info(f"Reconstructed URL: {url}")

        html = self.fetch_data(url)
        # Parse PMC ID and DOI from the HTML content

        # Extract PMC ID
        pmc_id = self.get_PMCID_from_pubmed_html(html)

        # Extract DOI
        doi = self.get_doi_from_pubmed_html(html)

        return pmc_id, doi

    def convert_url_to_doi(self, url : str):
        # Extract DOI from the URL
        url = url.lower()
        match = re.search(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', url, re.IGNORECASE)
        if match:
            doi = match.group(1)
            self.logger.info(f"DOI: {doi}")
            return doi
        else:
            return None

    def download_file_from_url(self, url, output_root, paper_id):
        """
        Downloads a file from the given URL and saves it to the specified directory.

        :param url: The URL to download the file from.

        :param output_root: The root directory where the file will be saved.

        :param paper_id: The ID of the paper, used to create a subdirectory.

        """

        # Set download dir in profile beforehand when you create the driver
        self.logger.info(f"Using Selenium to fetch download: {url}")

        driver = create_driver(self.driver_path, self.browser,
                               self.headless, self.logger,
                               download_dir=output_root + "/" + paper_id)
        driver.get(url)
        time.sleep(1.5)
        driver.quit()
        time.sleep(0.5)

    def quit(self):
        if self.scraper_tool:
            self.scraper_tool.quit()
            self.logger.info("WebScraper driver quit.")


class DatabaseFetcher(DataFetcher):
    """
    Class for fetching data from a DataFrame.
    """
    def __init__(self, logger, raw_HTML_data_filepath):
        super().__init__(logger, src='DatabaseFetcher', raw_HTML_data_filepath=raw_HTML_data_filepath)
        self.dataframe = pd.read_parquet(self.raw_HTML_data_filepath)
        self.logger.debug("DatabaseFetcher initialized.")

    def fetch_data(self, url_key, retries=3, delay=2):
        """
        Fetches data from a local file or database.

        :param url_key: The key to identify the data in the database.

        :returns: The raw HTML content of the page.
        """
        split_source_url = url_key.split('/')
        key = (split_source_url[-1] if len(split_source_url[-1]) > 0 else split_source_url[-2]).lower()
        self.logger.info(f"Fetching data for {key}")
        self.logger.debug(f"Data file: {self.dataframe.columns}")
        self.logger.debug(f"Data file: {self.dataframe[self.dataframe['publication'] == key]}")
        self.logger.info(f"Fetching data from {self.raw_HTML_data_filepath}")
        for i, row in self.dataframe[self.dataframe['publication'] == key].iterrows():
            self.raw_data_format = row['format']
            return row['raw_cont']

    def remove_cookie_patterns(self, html: str):
        pattern = r'<img\s+alt=""\s+src="https://www\.ncbi\.nlm\.nih\.gov/stat\?.*?"\s*>'

        if re.search(pattern, html):
            self.logger.info("Removing cookie pattern 1 from HTML")
            html = re.sub(pattern, 'img_alt_subst', html)
        else:
            self.logger.info("No cookie pattern 1 found in HTML")
        return html

# Implementation for fetching data from an API
class EntrezFetcher(DataFetcher):
    """
    Class for fetching data from an API using the requests library for ncbi e-utilities API.
    """
    def __init__(self, api_client, logger, local_fetch_fp=None):
        """
        Initializes the EntrezFetcher with the specified API client.

        :param api_client: The API client to use (e.g., requests).

        :param API: The API to use (e.g., PMC).


        :param logger: The logger instance for logging messages.

        """
        super().__init__(logger, src='EntrezFetcher', raw_HTML_data_filepath=local_fetch_fp)
        self.api_client = api_client.Session()
        self.raw_data_format = 'XML'
        self.base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=__PMCID__&retmode=xml'
        self.publisher = 'PMC'
        self.logger.debug("EntrezFetcher initialized.")


    def fetch_data(self, article_id, retries=3, delay=2):
        """
        Fetches data from the API using the provided article URL.

        :param article_id: The URL of the article to fetch data for.

        """
        try:
            # Extract the PMC ID from the article URL, ignore case
            PMCID = re.search(r'PMC\d+', article_id, re.IGNORECASE).group(0)
            self.PMCID = PMCID

            # Construct the API call using the PMC ID
            api_call = re.sub('__PMCID__', PMCID, self.base)
            self.logger.info(f"Fetching data from request: {api_call}")

            # Retry logic for API calls
            for attempt in range(retries):
                response = self.api_client.get(api_call)

                # Check if request was successful
                if response.status_code == 200:
                    self.logger.debug(f"Successfully fetched data for {PMCID}")
                    # Parse and return XML response
                    xml_content = response.content
                    root = ET.fromstring(xml_content)
                    return root  # Returning the parsed XML tree

                # Handle common issues
                elif response.status_code == 400:
                    self.logger.error(f"400 Bad Request for {PMCID}: {response.text}")
                    time.sleep(delay)
                    #break  # Stop retrying if it's a client-side error (bad request)

                # Log and retry for 5xx server-side errors or 429 (rate limit)
                elif response.status_code in [500, 502, 503, 504, 429]:
                    self.logger.warning(f"Server error {response.status_code} for {PMCID}, retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to fetch data for {PMCID}, Status code: {response.status_code}")
                    return None

            # If retries exhausted and request still failed
            self.logger.error(f"Exhausted retries. Failed to fetch data for {PMCID} after {retries} attempts.")
            return None

        except requests.exceptions.RequestException as req_err:
            # Catch all request-related errors (timeouts, network errors, etc.)
            self.logger.error(f"Network error fetching data for {article_id}: {req_err}")
            return None
        except Exception as e:
            # Log any other exceptions
            self.logger.error(f"Error fetching data for {article_id}: {e}")
            return None

    def download_xml(self, directory, api_data):
        """
        Downloads the XML data to a specified directory.

        :param directory: The directory where the XML file will be saved.

        :param api_data: The XML data to be saved.
        """

        os.makedirs(directory, exist_ok=True)  # Ensure directory exists
        # Construct the file path
        fn = os.path.join(directory, f"{self.extract_publication_title(api_data)}.xml")

        # Check if the file already exists
        if os.path.exists(fn):
            self.logger.info(f"File already exists: {fn}. Skipping download.")
            return

        # Write the XML data to the file
        ET.ElementTree(api_data).write(fn, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        self.logger.info(f"Downloaded XML file: {fn}")

    def extract_publication_title(self, xml_data):
        """
        Extracts the publication title from the XML data.

        :param xml_data: The XML data to extract the title from.

        :return: The publication title as a string.
        """
        # Use XPath to find the title element
        title_element = xml_data.find('.//article-title')
        if title_element is not None:
            return title_element.text.strip()
        else:
            self.logger.warning("No article title found in XML data.")
            return "No Title Found"

class PdfFetcher(DataFetcher):
    """
    Class for fetching PDF files from URLs.
    """
    def __init__(self, logger, driver_path=None, browser='firefox', headless=True):
        super().__init__(logger, src='PdfFetcher', driver_path=driver_path, browser=browser, headless=headless)
        self.logger.debug("PdfFetcher initialized.")

    def fetch_data(self, url, return_temp=True, **kwargs):
        """
        Fetches PDF data from the given URL.

        :param url: The URL to fetch data from.

        :return: The raw content of the PDF file.
        """
        self.raw_data_format = 'PDF'
        self.logger.info(f"Fetching PDF data from {url}")
        response = requests.get(url)
        if response.status_code == 200:
            if return_temp:
                # Write the PDF content to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(response.content)
                    self.logger.info(f"PDF data written to temporary file: {temp_file.name}")
                    return temp_file.name
            else:
                return response.content
        else:
            self.logger.error(f"Failed to fetch PDF data from {url}, status code: {response.status_code}")
            return None

    def download_pdf(self, directory, raw_data, src_url):
        """
        Downloads the PDF data to a specified directory.
        """
        fn = os.path.join(directory, f"{self.url_to_filename(src_url)}.pdf")

        self.logger.info(f"Downloading PDF from {src_url}")
        with open(fn, 'wb') as file:
            file.write(raw_data)
            self.logger.info(f"PDF data written to file: {file}")

class DataCompletenessChecker:
    """
    Class to check the completeness of data sections in API responses.
    """
    def __init__(self, logger, publisher='PMC', retrieval_patterns_file='retrieval_patterns.json'):
        """
        Initializes the DataCompletenessChecker with the specified logger.

        :param logger: The logger instance for logging messages.

        :param publisher: The publisher to check for (default is 'PMC').

        """
        self.logger = logger
        self.retriever = xmlRetriever(logger, publisher, retrieval_patterns_file)
        self.logger.debug("DataCompletenessChecker initialized.")

    def is_xml_data_complete(self, raw_data, url,
                             required_sections=["data_availability_sections", "supplementary_data_sections"]) -> bool:
        """
        Check if required sections are present in the raw_data.
        Return True if all required sections are present.

        :param raw_data: Raw XML data.

        :param url: The URL of the article.

        :param required_sections: List of required sections to check.

        :return: True if all required sections are present, False otherwise.
        """
        return self.retriever.is_xml_data_complete(raw_data, url, required_sections)