from abc import ABC, abstractmethod
import re
import logging
import numpy as np
from selenium.webdriver.common.by import By
import asyncio
import os
import time
import requests
from lxml import etree as ET
from data_gatherer.selenium_setup import create_driver
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd
from data_gatherer.retriever.xml_retriever import xmlRetriever
from data_gatherer.retriever.html_retriever import htmlRetriever
import tempfile


# Singleton backup data store for all fetchers
class BackupDataStore:
    """
    Lightweight singleton that provides backup data access for all fetchers.
    This acts as a supplementary layer, not a replacement for live fetching.
    """
    _instance = None
    _dataframe = None
    _filepath = None
    _timestamp = None
    _ttl = 1800  # 30 minutes
    
    def __new__(cls, filepath=None, logger=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, filepath=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        if filepath and (self._filepath != filepath or not self._is_valid()):
            self._load_dataframe(filepath)
            self.logger.debug(f"BackupDataStore loaded from {filepath}, entries: {len(self._dataframe) if self._dataframe is not None else 0}")
    
    def _load_dataframe(self, filepath):
        """Load DataFrame from file with error handling."""
        try:
            self._dataframe = pd.read_parquet(filepath)
            self._filepath = filepath
            self._timestamp = time.time()
            return True
        except Exception:
            self._dataframe = None
            self._filepath = None
            self._timestamp = None
            return False
    
    def _is_valid(self):
        """Check if cached data is still valid."""
        return (self._dataframe is not None and 
                self._timestamp is not None and
                (time.time() - self._timestamp) < self._ttl)
    
    def has_publication(self, identifier):
        """Check if publication exists in backup store."""
        if not self._is_valid() or self._dataframe is None:
            return False
        if 'publication' in self._dataframe.keys():
            return identifier.lower() in self._dataframe['publication'].str.lower().values
        else:
            return identifier.lower() in self._dataframe.index.str.lower().values
    
    def get_publication_data(self, identifier):
        """Retrieve publication data if available."""
        if not self.has_publication(identifier):
            return None
        self.logger.debug(f"Fetching publication {identifier} from backup store")
        if 'publication' in self._dataframe.columns:
            row = self._dataframe[self._dataframe['publication'].str.lower() == identifier.lower()]
        else:
            row = self._dataframe[self._dataframe.index.str.lower() == identifier.lower()]
        self.logger.info(f"Fetched {len(row)} records from backup store")
        if len(row) > 0:
            return {
                'content': row.iloc[0]['raw_cont'],
                'format': row.iloc[0]['format'].upper()  # Ensure format is uppercase (HTML/XML)
            }
        return None
    
    def get_stats(self):
        """Get backup store statistics."""
        return {
            'valid': self._is_valid(),
            'filepath': self._filepath,
            'size': len(self._dataframe) if self._dataframe is not None else 0,
            'age_seconds': time.time() - self._timestamp if self._timestamp else None
        }

# Abstract base class for fetching data
class DataFetcher(ABC):
    def __init__(self, logger, src='WebScraper', driver_path=None, browser='firefox', headless=True, 
                 backup_file='scripts/exp_input/Local_fulltext_pub_REV.parquet'):
        self.logger = logger
        self.logger.debug(f"DataFetcher ({src}) initialized.")
        self.driver_path = driver_path
        self.browser = browser
        self.headless = headless
        self.src = src
        self.local_data_used = False
        self.redirect_mapping = {}

        self.logger.debug(f"Setting up BackupDataStore with file: {backup_file}")
        
        if hasattr(self, 'backup_store') and self.backup_store is not None:
            self.logger.debug("Using existing BackupDataStore instance.")
        else:
            if backup_file and os.path.exists(backup_file):
                self.backup_store = BackupDataStore(filepath=backup_file, logger=self.logger)
                stats = self.backup_store.get_stats()
                self.logger.info(f"Backup data store initialized: {stats['size']} publications, valid: {stats['valid']}")
            else:
                self.logger.info(f"No backup data available at {backup_file}")
    
    def try_backup_fetch(self, identifier):
        """
        Try to fetch data from backup store as fallback.
        
        :param identifier: Publication identifier (PMC ID, DOI, etc.)
        :return: Raw data if found, None otherwise
        """
        if not self.backup_store:
            return None
            
        data = self.backup_store.get_publication_data(identifier)
        if data:
            self.logger.info(f"Found {identifier} in backup data store (format: {data['format']})")
            # Set the raw_data_format based on backup data
            self.raw_data_format = data['format']
            self.local_data_used = True
            
            # For XML data, parse it into lxml Element tree to match live fetching behavior
            if data['format'].upper() == 'XML':
                try:
                    xml_content = data['content']
                    if isinstance(xml_content, str):
                        if xml_content.startswith('<?xml'):
                            xml_content = xml_content.encode('utf-8') # Ensure bytes for parsing
                    root = ET.fromstring(xml_content)
                    self.logger.debug(f"Parsed backup XML data into Element tree for {identifier}")
                    return root
                except Exception as e:
                    self.logger.warning(f"Failed to parse backup XML for {identifier}: {e}")
                    return data['content']
            else:
                return data['content']
        return None

    def remove_cookie_patterns(self, html: str):
        pattern = r'<img\s+alt=""\s+src="https://www\.ncbi\.nlm\.nih\.gov/stat\?.*?"\s*>'

        if re.search(pattern, html):
            self.logger.info("Removing cookie pattern 1 from HTML")
            html = re.sub(pattern, 'img_alt_subst', html)
        else:
            self.logger.info("No cookie pattern 1 found in HTML")
        return html

    def remove_cookie_patterns(self, html: str):
        pattern = r'<img\s+alt=""\s+src="https://www\.ncbi\.nlm\.nih\.gov/stat\?.*?"\s*>'

        if re.search(pattern, html):
            self.logger.info("Removing cookie pattern 1 from HTML")
            html = re.sub(pattern, 'img_alt_subst', html)
        else:
            self.logger.info("No cookie pattern 1 found in HTML")
        return html

    @abstractmethod
    def fetch_data(self, url, retries=3, delay=2, **kwargs):
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
        if re.match(r'^https?://www\.ncbi\.nlm\.nih\.gov/pmc', url) or \
            re.match(r'^https?://pmc\.ncbi\.nlm\.nih\.gov/', url) or \
                re.match(r'^https?://ncbi\.nlm\.nih\.gov/pmc', url):
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

    def url_to_article_id(self, url, return_only_known_ids=False):
        """
        Extracts the PMC ID from a given URL.

        :param url: The URL to extract the PMC ID from.

        :return: The extracted PMC ID or None if not found.
        """
        match = re.search(r'PMC(\d+)', url, re.IGNORECASE)
        doi = re.search(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', url, re.IGNORECASE)
        pmid = re.search(r'ncbi\.nlm\.nih\.gov/.*/(\d+)', url, re.IGNORECASE) 
        if match:
            pmcid = f"PMC{match.group(1)}"
            self.logger.info(f"Extracted PMC ID: {pmcid}")
            self.current_article_id = pmcid
            return pmcid

        elif doi:
            self.current_article_id = doi.group(1)
            return doi.group(1)

        elif pmid:
            self.current_article_id = pmid.group(1)
            return pmid.group(1)

        else:
            if not return_only_known_ids:
                article_id, i = '', 0
                while len(article_id) < 6:
                    i+=1
                    article_id = url.split("/")[-i]

                self.current_article_id = article_id
                
                return self.current_article_id
            self.logger.warning(f"No PMC ID found in URL: {url}")
            self.current_article_id = None
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

    def update_DataFetcher_settings(self, url, HTML_fallback=False, driver_path=None,
                                    browser='firefox', headless=True, local_fetch_file=None):
        """
        Creates appropriate data fetcher with BackupDataStore integration. 
        All fetchers now automatically check backup data first, then fall back to live fetching.

        :param url: The URL to fetch data from.
        :param HTML_fallback: If False, use simple HTTP. If True, use Selenium. If 'HTTPGetRequest', force HTTP. If 'Playwright', force Playwright.
        :return: An instance of the appropriate data fetcher with backup capability.
        """
        self.logger.info(f"update_DataFetcher_settings for URL: {url}, current instance: {self.__class__.__name__}, HTML_fallback={HTML_fallback}, BackupDataFile={local_fetch_file}")
        self.local_data_used = False

        # Determine backup data file
        backup_file = local_fetch_file or 'scripts/exp_input/Local_fulltext_pub_REV.parquet'

        self.logger.debug(f"Using backup data file: {backup_file}, current backup store file: {getattr(self, 'backup_store', None)}")

        if hasattr(self, 'backup_store') and (self.backup_store is None or self.backup_store._filepath != backup_file):
            self.backup_store = BackupDataStore(filepath=backup_file, logger=self.logger)
            stats = self.backup_store.get_stats()
            self.logger.info(f"Backup data store re-initialized: {stats['size']} publications, valid: {stats['valid']}")
        
        # Check if it's a PDF first
        if self.url_is_pdf(url):
            self.logger.info(f"URL {url} is a PDF. Using PdfFetcher.")
            if isinstance(self, PdfFetcher):
                self.logger.info(f"Reusing existing PdfFetcher instance.")
                return self
            return PdfFetcher(self.logger, driver_path=driver_path, browser=browser, headless=headless, backup_file=local_fetch_file)

        # Detect API type for optimal fetcher selection
        API = None
        if not HTML_fallback:
            API = self.url_to_api_root(url)
        self.logger.info(f"API detected: {API} ")

        # Choose fetcher based on content type and availability, all with backup support
        if API == 'PMC':
            # For PMC content, use EntrezFetcher (XML) with backup fallback
            if isinstance(self, EntrezFetcher):
                self.logger.info(f"Reusing existing EntrezFetcher instance with backup support.")
                return self
            self.logger.info(f"Creating EntrezFetcher with backup support")
            return EntrezFetcher(requests, self.logger, backup_file=local_fetch_file)

        # For HTTP GET requests (simpler, faster for static content)
        if type(HTML_fallback) == str and HTML_fallback == 'HTTPGetRequest':
            self.logger.info(f"Using HttpGetRequest with backup support for URL: {url}")
            if isinstance(self, HttpGetRequest):
                self.logger.info(f"Reusing existing HttpGetRequest instance.")
                return self
            return HttpGetRequest(self.logger, backup_file=local_fetch_file)
        
        use_playwright = False
        if type(HTML_fallback) == str and HTML_fallback == 'Playwright':
            use_playwright = True
        
        # Default case: check if we need complex JS rendering or simple HTTP
        if not HTML_fallback:
            # Start with simple HTTP GET (faster, backup-first)
            self.logger.info(f"Using HttpGetRequest (backup-first) for URL: {url}")
            return HttpGetRequest(self.logger, backup_file=local_fetch_file)

        # For complex dynamic content, use Playwright (with asyncio detection)
        if use_playwright:
            # Check if we're in an asyncio event loop (Jupyter notebook)
            in_event_loop = False
            try:
                asyncio.get_running_loop()
                in_event_loop = True
                self.logger.info("Detected asyncio event loop - will use Playwright Async API")
            except RuntimeError:
                self.logger.info("No asyncio event loop detected - will use Playwright Sync API")
            
            if isinstance(self, PlaywrightFetcher):
                self.logger.info(f"Reusing existing PlaywrightFetcher instance with backup support")
                return self
            
            self.logger.info(f"Creating new PlaywrightFetcher with backup support: {browser}, headless={headless}, async={in_event_loop}")
            return PlaywrightFetcher(self.logger, browser_type=browser, headless=headless, use_async=in_event_loop, backup_file=local_fetch_file)
        
        # Default: Use traditional Selenium WebScraper with backup support
        # Reuse existing driver if available
        if isinstance(self, WebScraper) and hasattr(self, 'scraper_tool') and self.scraper_tool is not None:
            self.logger.info(f"Reusing existing WebScraper driver with backup support")
            return self

        self.logger.info(f"Creating new WebScraper with backup support: {browser}, headless={headless}")
        driver = create_driver(driver_path, browser, headless, self.logger)
        return WebScraper(driver, self.logger, driver_path=driver_path, browser=browser, headless=headless, backup_file=local_fetch_file)

    def url_in_dataframe(self, url, idx='pmcid'):
        """
        Checks if the given doi / pmcid is present in the backup data store.

        :param url: The URL to check.
        :return: True if the URL is found, False otherwise.
        """
        if not self.backup_store:
            return False
        
        if idx == 'pmcid':
            pmcid = re.search(r'PMC\d+', url, re.IGNORECASE)
            if pmcid:
                return self.backup_store.has_publication(pmcid.group(0))
        elif idx == 'doi':
            doi = re.search(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', url, re.IGNORECASE)
            if doi:
                return self.backup_store.has_publication(doi.group(1))
        else:
            # using url as i
            return self.backup_store.has_publication(url)
        
        return False
    
    def url_to_api_root(self, url):

        API_ptrs = {
            r'PMC\d+': 'PMC'
        }

        if not url:
            return None

        if not url.startswith('http'):
            for ptr, src in API_ptrs.items():
                match = re.match(ptr, url, re.IGNORECASE)
                if match:
                    self.logger.info(f"URL detected as {src}.")
                    return src

        API_supported_url_patterns = {
            r'https*://www.ncbi.nlm.nih.gov/pmc/articles/': 'PMC',
            r'https*://www.ncbi.nlm.nih.gov/labs/pmc/': 'PMC',
            r'https*://pmc.ncbi.nlm.nih.gov/': 'PMC',
            r'https*://ncbi.nlm.nih.gov/pmc/': 'PMC',
        }

        # Check if the URL corresponds to any API_supported_url_patterns
        for ptr, src in API_supported_url_patterns.items():
            self.logger.debug(f"Checking {src} with pattern {ptr}")
            match = re.match(ptr, url, re.IGNORECASE)
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
        if not url:
            return None
        if url.lower().endswith('.pdf'):
            self.logger.info(f"URL {url} ends with .pdf")
            return True
        elif re.search(r'arxiv\.org/pdf/', url, re.IGNORECASE):
            return True
        elif re.search(r'aclanthology\.org/', url, re.IGNORECASE):
            return True
        elif re.search(r'doi\.org/10\.18653', url, re.IGNORECASE):
            return True
        elif re.search(r'doi\.org/10\.48550', url, re.IGNORECASE):
            return True
        elif re.search(r'doi\.org/10\.23977', url, re.IGNORECASE):
            return True
        elif re.search(r'doi\.org/10\.5121', url, re.IGNORECASE):
            return True
        elif re.search(r'publica.fraunhofer.de', url, re.IGNORECASE):
            return True
        elif re.search(r'ojs.aaai.org', url, re.IGNORECASE):
            return True
        return False

    def download_file_from_url(self, url, output_root="scripts/downloads/suppl_files", paper_id=None):
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

    def remove_cookie_patterns(self, html: str):
        pattern = r'<img\s+alt=""\s+src="https://www\.ncbi\.nlm\.nih\.gov/stat\?.*?"\s*>'

        if re.search(pattern, html):
            self.logger.info("Removing cookie pattern 1 from HTML")
            html = re.sub(pattern, 'img_alt_subst', html)
        else:
            self.logger.info("No cookie pattern 1 found in HTML")
        return html

    def get_PMCID_from_pubmed_html(self, html):
        try:
            self.logger.debug(f"html: {html}")
            soup = BeautifulSoup(html, 'html.parser')
            # Extract PMC ID
            pmc_tag = soup.find("a", {"data-ga-action": "PMCID"})
            pmc_id = pmc_tag.text.strip() if pmc_tag else None  # Extract text safely
            self.logger.info(f"PMCID: {pmc_id}")
            return pmc_id
        except Exception as e:
            self.logger.error(f"Error extracting PMCID from HTML: {e}")
            return None

    def get_doi_from_pubmed_html(self, html):
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # Extract DOI
            doi_tag = soup.find("a", {"data-ga-action": "DOI"})
            doi = doi_tag.text.strip() if doi_tag else None  # Extract text safely
            self.logger.info(f"DOI: {doi}")
            return doi
        except Exception as e:
            self.logger.error(f"Error extracting DOI from HTML: {e}")
            return None

    def redirect_if_needed(self, url):
        """
        Follows redirects for the given URL and returns the final URL.

        :param url: The initial URL.
        :return: The final URL after following redirects.
        """
        # Handle None or empty URLs
        if not url:
            return url
        
        self.logger.info(f"url to redirect: {url}")
        
        # if pumbmed url, follow redirects to get final URL
        if re.match(r'^https?://pubmed\.ncbi\.nlm\.nih\.gov/[\d]+', url) or re.match(
            r'^https?://www\.ncbi\.nlm\.nih\.gov/pubmed/[\d]+', url) or re.match(
            r'^https?://www\.ncbi\.nlm\.nih\.gov/pmc/articles/pmid/[\d]+', url) or re.match(
            r'^https?://pmc\.ncbi\.nlm\.nih\.gov/pmc/articles/pmid/[\d]+', url) or re.match(
            r'^https?://www\.ncbi\.nlm\.nih\.gov/labs/pmc/articles/', url):
            try:
                self.logger.info(f"1")
                response = requests.get(url, timeout=3)
                self.logger.info(f"2: {response.url}")
                html = response.text
                pmc_id = self.get_PMCID_from_pubmed_html(html)
                if pmc_id:
                    final_url = self.PMCID_to_url(pmc_id)
                    self.logger.info(f"Redirected PubMed URL to PMC URL: {final_url}")
                    self.redirect_mapping[url] = final_url
                    return final_url
                doi = self.get_doi_from_pubmed_html(html)
                if doi:
                    final_url = f"https://doi.org/{doi}"
                    self.logger.info(f"Redirected PubMed URL to DOI URL: {final_url}")
                    self.redirect_mapping[url] = final_url
                    return final_url

            except requests.RequestException as e:
                self.logger.warning(f"Failed to follow redirects for PubMed URL {url}: {e}")

        return url

class HttpGetRequest(DataFetcher):
    "class for fetching data via HTTP GET requests using the requests library."
    def __init__(self, logger, backup_file='scripts/exp_input/Local_fulltext_pub_REV.parquet'):
        super().__init__(logger, src='HttpGetRequest', backup_file=backup_file)
        self.session = requests.Session()
        # Keep default requests User-Agent - many sites (like PubMed) allow it
        # but block fake browser User-Agents with incomplete fingerprints
        self.logger.debug("HttpGetRequest initialized with default requests headers.")
        self.raw_data_format = 'HTML'

    def fetch_data(self, url, retries=3, delay=0.2, **kwargs):
        """
        Fetches data from the given URL. First tries backup data (fast), then HTTP GET if needed.

        :param url: The URL to fetch data from.
        :param retries: Number of retries in case of failure.
        :param delay: Delay time between retries.
        :return: The raw content of the page.
        """
        # Try backup data FIRST (microsecond lookup)
        article_id = self.url_to_article_id(url)
        self.article_id = article_id
        if article_id and hasattr(self, 'backup_store') and self.backup_store is not None:
            backup_data = self.try_backup_fetch(article_id)
            if backup_data:
                self.logger.info(f"Found {article_id} in local backup data (fast path, format: {self.raw_data_format})")
                return backup_data
        
        # Fallback to live HTTP fetch (slow path)
        self.logger.info(f"Local data not found, fetching live from {url}")

        if self.url_is_pdf(url):
            self.logger.info(f"URL {url} is a PDF. Using PdfFetcher.")
            pdf_fetcher = PdfFetcher(self.logger)
            return pdf_fetcher.fetch_data(url, retries=retries, delay=delay)
            
        attempt = 0
        while attempt < retries:
            time.sleep(delay/2)
            try:
                self.logger.info(f"HTTP GET attempt {attempt + 1} of {retries}")
                # Session already has headers set in __init__, no need to override
                response = self.session.get(url, timeout=30)
                response.raise_for_status()  # Raise an error for bad responses
                self.raw_data_format = 'HTML' if 'text/html' in response.headers.get('Content-Type', '') else 'Other'
                return response.text
            except requests.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                attempt += 1
                time.sleep(delay*2)
        
        self.logger.error(f"Failed to fetch data from {url} after {retries} attempts and no backup data available.")

    
    def html_page_source_download(self, directory, url, fetched_data):
        """
        Downloads the HTML page source as html file in the specified directory.

        :param directory: The directory where the HTML file will be saved.

        """
        if os.path.exists(directory):
            logging.info(f"Dir {directory} exists")
        else:
            os.makedirs(directory, exist_ok=True)

        if hasattr(self, 'extract_publication_title'):
            pub_name = self.extract_publication_title(fetched_data)

        else:
            pub_name = url.split("/")[-1] if url.split("/")[-1] != '' else url.split("/")[-2]
        pub_name = re.sub(r'[\\/:*?"<>|]', '_', pub_name)  # Replace invalid characters in filename
        pub_name = re.sub(r'[\s-]+PMC\s*$', '', pub_name)

        if directory[-1] != '/':
            directory += '/'

        article_id = self.url_to_filename(self.url_to_article_id(url))
        pub_fname = self.url_to_filename(pub_name)

        fn = directory + f"{article_id}__{pub_fname}.html"
        self.logger.info(f"Downloading HTML page source to {fn}")

        with open(fn, 'w', encoding='utf-8') as f:
            f.write(self.fetch_data(url))
        
    def extract_publication_title(self, html=None):
        """
        Extracts the publication name from the HTML content (not Selenium).
        :param html: The HTML content as a string.
        :return: The publication name as a string.
        """
        self.logger.debug("Extracting publication title from HTML (HttpGetRequest)")
        try:
            if html is None:
                self.logger.warning("No HTML provided to extract_publication_title.")
                return "No title found"

            soup = BeautifulSoup(html, "html.parser")

            # Try <title> tag first
            title_tag = soup.find("title")
            if title_tag and title_tag.text.strip():
                publication_name = title_tag.text.strip()
                self.logger.info(f"Paper name (from <title> tag): {publication_name}")
                return publication_name

            # Fallback: <meta name="citation_title">
            meta_title = soup.find("meta", attrs={"name": "citation_title"})
            if meta_title and meta_title.get("content"):
                publication_name = meta_title["content"].strip()
                self.logger.info(f"Paper name (from meta citation_title): {publication_name}")
                return publication_name

            # Fallback: <h1 class="article-title">
            h1 = soup.find("h1", class_="article-title")
            if h1:
                publication_name = h1.get_text(strip=True)
                self.logger.info(f"Paper name (from h1.article-title): {publication_name}")
                return publication_name

            self.logger.warning("Publication name not found in the HTML.")
            return "No title found"
        except Exception as e:
            self.logger.error(f"Error extracting publication title: {e}")
            return "No title found"
    
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


# Implementation for fetching data via web scraping
class WebScraper(DataFetcher):
    """
    Class for fetching data from web pages using Selenium.
    """
    def __init__(self, scraper_tool, logger, retrieval_patterns_file=None, driver_path=None, browser='firefox',
                 headless=True, backup_file='scripts/exp_input/Local_fulltext_pub_REV.parquet'):
        super().__init__(logger, src='WebScraper', driver_path=driver_path, browser=browser, headless=headless, backup_file=backup_file)
        self.scraper_tool = scraper_tool  # Inject your scraping tool (Selenium)
        self.driver_path = driver_path
        self.browser = browser
        self.headless = headless
        self.backup_file = backup_file
        self.logger.debug("WebScraper initialized.")

    def fetch_data(self, url, retries=3, delay=2, update_redirect_map=False, **kwargs):
        """
        Fetches data from the given URL. First tries backup data (fast), then live web scraping if needed.

        :param url: The URL to fetch data from.
        :param retries: Number of retries in case of failure.
        :param delay: Delay time between retries.
        :return: The raw HTML content of the page.
        """
        self.raw_data_format = 'HTML'  # Default format for web scraping
        
        # Try backup data FIRST (microsecond lookup)
        article_id = self.url_to_article_id(url)
        self.article_id = article_id
        if article_id and hasattr(self, 'backup_store') and self.backup_store is not None:
            backup_data = self.try_backup_fetch(article_id)
            if backup_data:
                self.logger.info(f"Found {article_id} in local backup data (fast path, format: {self.raw_data_format})")
                return backup_data
        
        # Fallback to live web scraping (slow path)
        self.logger.info(f"Local data not found, fetching live from {url}")
        try:
            self.logger.debug(f"Fetching data with function call: self.scraper_tool.get(url)")
            self.scraper_tool.get(url)
            self.logger.debug(f"http get complete, now waiting {delay} seconds for page to load")
            self.simulate_user_scroll(delay)
            self.title = self.extract_publication_title()
            if update_redirect_map:
                final_url = self.scraper_tool.current_url
                if final_url != url and url not in self.redirect_mapping:
                    self.logger.info(f"URL redirected from {url} to {final_url}")
                    self.redirect_mapping[url] = final_url
            return self.scraper_tool.page_source
        
        except Exception as e:
            self.logger.error(f"Live web scraping failed for {url}: {e}")
            raise e

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

    def html_page_source_download(self, directory, pub_link):
        """
        Downloads the HTML page source as html file in the specified directory.

        :param directory: The directory where the HTML file will be saved.

        :param pub_link: The URL of the publication page.

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
        pub_name = re.sub(r'[\s-]+PMC\s*$', '', pub_name)

        if directory[-1] != '/':
            directory += '/'

        pmcid = self.url_to_article_id(pub_link)

        fn = directory + f"{pmcid}__{pub_name}.html"
        self.logger.info(f"Downloading HTML page source to {fn}")

        self.logger.debug(f"scraper_tool: {self.scraper_tool}, page_source: {getattr(self.scraper_tool, 'page_source', None)}")

        if hasattr(self, 'scraper_tool') and self.scraper_tool.page_source:
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(self.scraper_tool.page_source)
        else:
            raise RuntimeError(f"Error saving HTML page source to {fn}. scraper_tool or page_source not available.")

    def extract_publication_title(self):
        """
        Extracts the publication name from the WebDriver's current page title or meta tags.
        Should be called after scraper_tool.get(url) to ensure the page is loaded.

        :return: The publication name as a string.
        """
        self.logger.debug("Extracting publication title from page source")
        try:
            # Try Selenium's title property first (most robust)
            page_title = self.scraper_tool.title
            if page_title and page_title.strip():
                publication_name = page_title.strip()
                self.logger.info(f"Paper name (from Selenium .title): {publication_name}")
                return publication_name

            # Fallback: Try to find <title> tag via Selenium
            publication_name_pointer = self.scraper_tool.find_element(By.TAG_NAME, 'title')
            if publication_name_pointer is not None and publication_name_pointer.text:
                publication_name = publication_name_pointer.text.strip()
                self.logger.info(f"Paper name (from <title> tag): {publication_name}")
                return publication_name

            # Fallback: Parse page source for <meta name="citation_title">
            soup = BeautifulSoup(self.scraper_tool.page_source, "html.parser")
            meta_title = soup.find("meta", attrs={"name": "citation_title"})
            if meta_title and meta_title.get("content"):
                publication_name = meta_title["content"].strip()
                self.logger.info(f"Paper name (from meta citation_title): {publication_name}")
                return publication_name

            # Fallback: Try <h1> with class "article-title"
            h1 = soup.find("h1", class_="article-title")
            if h1:
                publication_name = h1.get_text(strip=True)
                self.logger.info(f"Paper name (from h1.article-title): {publication_name}")
                return publication_name

            self.logger.warning("Publication name not found in the page title or meta tags.")
            return "No title found"
        except Exception as e:
            self.logger.error(f"Error extracting publication title: {e}")
            return "No title found"

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
    Simplified class for fetching data from a DataFrame. 
    Now just a direct interface to the BackupDataStore.
    """
    def __init__(self, logger, raw_HTML_data_filepath=None):
        # Call parent with backup_file parameter
        super().__init__(logger, src='DatabaseFetcher', backup_file=raw_HTML_data_filepath)
        
        if not raw_HTML_data_filepath or not os.path.exists(raw_HTML_data_filepath):
            raise ValueError("DatabaseFetcher requires a valid raw_HTML_data_filepath.")
        
        if not self.backup_store:
            raise RuntimeError(f"Failed to initialize backup data store from {raw_HTML_data_filepath}")
        
        stats = self.backup_store.get_stats()
        self.logger.debug(f"DatabaseFetcher initialized with {stats['size']} publications (valid: {stats['valid']}).")

    def fetch_data(self, url_key, retries=3, delay=2, local_fetch_file=None, **kwargs):
        """
        Fetches data from the backup data store.

        :param url_key: The key to identify the data in the database.
        :returns: The raw HTML content of the page.
        """
        split_source_url = url_key.split('/')
        key = (split_source_url[-1] if len(split_source_url[-1]) > 0 else split_source_url[-2]).lower()
        
        self.logger.info(f"Fetching data for {key}")
        
        # Use the backup store directly
        data = self.backup_store.get_publication_data(key)
        if data:
            # Try to determine format - this is a simplification
            self.raw_data_format = 'HTML'  # Default assumption
            return data
        
        self.logger.warning(f"No data found for key: {key}")
        return None

    

# Implementation for fetching data from an API
class EntrezFetcher(DataFetcher):
    """
    Class for fetching data from an API using the requests library for ncbi e-utilities API.
    """
    def __init__(self, api_client, logger, backup_file='scripts/exp_input/Local_fulltext_pub_REV.parquet'):
        """
        Initializes the EntrezFetcher with the specified API client.

        :param api_client: The API client to use (e.g., requests).

        :param logger: The logger instance for logging messages.

        """
        super().__init__(logger, src='EntrezFetcher', backup_file=backup_file)
        self.api_client = api_client.Session()
        self.raw_data_format = 'XML'
        self.logger.info(f"Raw_data_format: {self.raw_data_format}")
        # Read the API key at runtime, fallback to empty string if not set
        NCBI_API_KEY = os.environ.get('NCBI_API_KEY', '')
        if not NCBI_API_KEY:
            self.base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=__PMCID__&retmode=xml'
            self.logger.warning("NCBI_API_KEY not set. Proceeding without an API key may lead to rate limiting. https://www.ncbi.nlm.nih.gov/books/NBK25497/")
        else:
            self.base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=__PMCID__&retmode=xml&api_key=' + NCBI_API_KEY
        self.publisher = 'PMC'
        self.logger.debug("EntrezFetcher initialized.")


    def fetch_data(self, article_id, retries=3, delay=2, **kwargs):
        """
        Fetches data from the API. First tries backup data (fast), then live API call if needed.

        :param article_id: The URL of the article to fetch data for.
        """

        self.raw_data_format = 'XML'

        try:
            # Extract the PMC ID from the article URL, ignore case
            pmcid = re.search(r'PMC\d+', article_id, re.IGNORECASE).group(0)
            self.article_id = pmcid
            
            if hasattr(self, 'backup_store') and self.backup_store is not None:
                backup_data = self.try_backup_fetch(pmcid)
                if backup_data:
                    self.logger.info(f"Found {pmcid} in local backup data (fast path, format: {self.raw_data_format})")
                    return backup_data

            else:
                self.logger.info(f"Local data not found, fetching live from API for {pmcid}")
            return self._fetch_live_api_data(pmcid, retries, delay)

        except Exception as e:
            # Log any exceptions and return None (backup already tried at start)
            self.logger.error(f"Error fetching data for {article_id}: {e}")
            return None

    def _fetch_live_api_data(self, pmcid, retries, delay):
        """Helper method to fetch data from live NCBI API."""
        api_call = re.sub('__PMCID__', pmcid, self.base)
        self.logger.info(f"API request: {api_call}")

        self.raw_data_format = 'XML'

        # Retry logic for API calls
        for attempt in range(retries):
            try:
                response = self.api_client.get(api_call)

                # Check if request was successful
                if response.status_code == 200:
                    self.logger.debug(f"Successfully fetched data for {pmcid}")
                    # Parse and return XML response
                    xml_content = response.content
                    root = ET.fromstring(xml_content)
                    return root  # Returning the parsed XML tree

                # Handle common issues
                elif response.status_code == 400:
                    if "API key invalid" in str(response.text):
                        self.logger.error(f"Invalid NCBI API key provided. https://support.nlm.nih.gov/kbArticle/?pn=KA-05317")
                        return None  # Stop retrying for API key errors
                    else:
                        self.logger.error(f"400 Bad Request for {pmcid}: {response.text}")
                        time.sleep(delay)

                # Log and retry for 5xx server-side errors or 429 (rate limit)
                elif response.status_code in [500, 502, 503, 504, 429]:
                    self.logger.warning(f"Server error {response.status_code} for {pmcid}, retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to fetch data for {pmcid}, Status code: {response.status_code}")
                    return None  # Stop retrying for other client errors

            except requests.exceptions.RequestException as req_err:
                self.logger.error(f"Network error on attempt {attempt + 1} for {pmcid}: {req_err}")
                if attempt < retries - 1:
                    time.sleep(delay)
                
        # If all retries exhausted
        self.logger.error(f"Live API fetch failed for {pmcid} after {retries} attempts")
        return None

    def download_xml(self, directory, api_data, pub_link):
        """
        Downloads the XML data to a specified directory.

        :param directory: The directory where the XML file will be saved.

        :param api_data: The XML data to be saved.
        """

        os.makedirs(directory, exist_ok=True)  # Ensure directory exists
        # Construct the file path
        pmcid = self.url_to_article_id(pub_link)
        title = self.extract_publication_title(api_data)
        title = re.sub(r'[\\/:*?"<>|]', '_', title)  # Replace invalid characters in filename

        fn = os.path.join(directory, f"{pmcid}__{title}.xml")

        # Check if the file already exists
        if os.path.exists(fn):
            self.logger.info(f"File already exists: {fn}. Skipping download.")
            return
        else:
            self.logger.info(f"Downloading XML data to {fn}")
            os.makedirs(os.path.dirname(fn), exist_ok=True)

        # Write the XML data to the file
        ET.ElementTree(api_data).write(fn, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        self.logger.info(f"Downloaded XML file: {fn}")

    def extract_publication_title(self, xml_data):
        """
        Extracts the publication title from the XML data.
        :param xml_data: The XML data as a string or ElementTree.
        :return: The publication title as a string.
        """
        # Parse if xml_data is a string
        if isinstance(xml_data, str):
            try:
                xml_data = ET.fromstring(xml_data)
            except Exception as e:
                self.logger.warning(f"Could not parse XML: {e}")
                return "No Title Found"
        
        else:
            self.logger.warning(f"xml_data is not a string. Type: {type(xml_data)}")

        # Now xml_data is an Element
        title_element = xml_data.find('.//article-title')
        if title_element is not None and title_element.text:
            return title_element.text.strip()
        else:
            self.logger.warning("No article title found in XML data.")
            return "No Title Found"

class PlaywrightFetcher(DataFetcher):
    """
    Class for fetching data from web pages using Playwright.
    Modern, faster alternative to Selenium with better JavaScript handling.
    Automatically detects and adapts to asyncio event loops (Jupyter notebooks).
    """
    def __init__(self, logger, browser_type='chromium', headless=True, use_async=False, backup_file='scripts/exp_input/Local_fulltext_pub_REV.parquet'):
        """
        Initializes the PlaywrightFetcher.
        
        :param logger: Logger instance for logging messages
        :param browser_type: Browser to use ('chromium', 'firefox', or 'webkit')
        :param headless: Whether to run in headless mode
        :param use_async: Whether to use async API (auto-detected for Jupyter)
        """
        super().__init__(logger, src='PlaywrightFetcher', browser=browser_type, headless=headless, backup_file=backup_file)
        self.browser_type = browser_type
        self.headless = headless
        self.use_async = use_async
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.raw_data_format = 'HTML'
        self.logger.debug(f"PlaywrightFetcher initialized with {browser_type} (headless={headless}, async={use_async})")
    
    def _ensure_browser_started(self):
        """Lazily start the browser when needed (sync version)."""
        if self.playwright is None and not self.use_async:
            from playwright.sync_api import sync_playwright
            
            self.logger.debug("Starting Playwright browser (sync API)")
            self.playwright = sync_playwright().start()
            
            # Select browser type
            if self.browser_type.lower() == 'firefox':
                browser_launcher = self.playwright.firefox
            elif self.browser_type.lower() == 'webkit':
                browser_launcher = self.playwright.webkit
            else:  # default to chromium
                browser_launcher = self.playwright.chromium
            
            self.browser = browser_launcher.launch(headless=self.headless)
            self.context = self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            self.page = self.context.new_page()
            self.logger.info(f"Playwright {self.browser_type} browser started (sync)")
    
    async def _ensure_browser_started_async(self):
        """Lazily start the browser when needed (async version for Jupyter)."""
        if self.playwright is None and self.use_async:
            from playwright.async_api import async_playwright
            
            self.logger.debug("Starting Playwright browser (async API)")
            self.playwright = await async_playwright().start()
            
            # Select browser type
            if self.browser_type.lower() == 'firefox':
                browser_launcher = self.playwright.firefox
            elif self.browser_type.lower() == 'webkit':
                browser_launcher = self.playwright.webkit
            else:  # default to chromium
                browser_launcher = self.playwright.chromium
            
            self.browser = await browser_launcher.launch(headless=self.headless)
            self.context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            self.page = await self.context.new_page()
            self.logger.info(f"Playwright {self.browser_type} browser started (async)")
    
    def fetch_data(self, url, retries=3, delay=2, wait_for_selector=None, wait_time=2000, **kwargs):
        """
        Fetches data from the given URL using Playwright.
        First tries backup data (fast), then live fetching if needed.
        Automatically uses async API if in event loop (Jupyter).
        
        :param url: The URL to fetch data from
        :param retries: Number of retries in case of failure
        :param delay: Delay time between retries
        :param wait_for_selector: Optional CSS selector to wait for before extracting HTML
        :param wait_time: Time to wait for page load (milliseconds)
        :return: The raw HTML content of the page
        """
        self.raw_data_format = 'HTML'
        
        # Try backup data FIRST (microsecond lookup)
        article_id = self.url_to_article_id(url)
        self.article_id = article_id
        if article_id and hasattr(self, 'backup_store') and self.backup_store is not None:
            backup_data = self.try_backup_fetch(article_id)
            if backup_data:
                self.logger.info(f"Found {article_id} in local backup data (fast path, format: {self.raw_data_format})")
                return backup_data
        
        # Detect asyncio loop at RUNTIME (not initialization)
        in_event_loop = False
        try:
            asyncio.get_running_loop()
            in_event_loop = True
            self.logger.debug("Asyncio event loop detected - will use workaround")
        except RuntimeError:
            self.logger.debug("No asyncio event loop - using sync API directly")
        
        # In Jupyter (event loop), use sync API but in a thread to avoid conflicts
        if in_event_loop:
            self.logger.info(f"Local data not found, fetching from {url} with Playwright (sync in thread)")
            return self._fetch_in_thread(url, retries, delay, wait_for_selector, wait_time)
        else:
            # Regular script - use sync API directly
            self.logger.info(f"Local data not found, fetching from {url} with Playwright (sync)")
            return self._fetch_data_sync(url, retries, delay, wait_for_selector, wait_time)
    
    def _fetch_in_thread(self, url, retries, delay, wait_for_selector, wait_time):
        """Run sync Playwright in a separate thread to avoid event loop conflicts."""
        import concurrent.futures
        self.logger.debug("Starting Playwright in separate thread")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                self._fetch_data_sync,
                url, retries, delay, wait_for_selector, wait_time
            )
            result = future.result()
            return result
    
    def _fetch_data_sync(self, url, retries, delay, wait_for_selector, wait_time):
        """Sync version of fetch_data for regular Python scripts."""
        # Ensure browser is started IN THIS THREAD (important for thread isolation)
        playwright_ctx = None
        browser = None
        page = None
        
        attempt = 0
        while attempt < retries:
            try:
                # Initialize Playwright in THIS thread if not already done
                if page is None:
                    from playwright.sync_api import sync_playwright
                    self.logger.debug("Initializing Playwright browser in current thread")
                    
                    playwright_ctx = sync_playwright().start()
                    
                    # Select browser type
                    if self.browser_type.lower() == 'firefox':
                        browser_launcher = playwright_ctx.firefox
                    elif self.browser_type.lower() == 'webkit':
                        browser_launcher = playwright_ctx.webkit
                    else:  # default to chromium
                        browser_launcher = playwright_ctx.chromium
                    
                    browser = browser_launcher.launch(headless=self.headless)
                    context = browser.new_context(
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    )
                    page = context.new_page()
                    self.logger.debug(f"Playwright {self.browser_type} browser started in thread")
                
                # Navigate to URL
                self.logger.debug(f"Playwright sync attempt {attempt + 1}: navigating to {url}")
                response = page.goto(url, wait_until='load', timeout=30000)
                self.logger.debug(f"Page loaded with status: {response.status if response else 'Unknown'}")
                
                # Wait for network to settle
                try:
                    page.wait_for_load_state('networkidle', timeout=10000)
                    self.logger.debug("Network is idle")
                except Exception as e:
                    self.logger.warning(f"Network idle timeout (expected for some pages): {e}")
                
                # Give extra time for JavaScript to finish
                self.logger.debug(f"Waiting additional {wait_time}ms for dynamic content")
                page.wait_for_timeout(wait_time)
                
                # Try to wait for Schema.org metadata to be present
                try:
                    self.logger.debug("Waiting for Schema.org metadata script...")
                    page.wait_for_selector('script[type="application/ld+json"]', timeout=5000)
                    self.logger.debug("✓ Schema.org metadata found")
                except Exception as e:
                    self.logger.warning(f"Schema.org metadata not found (may not be present): {e}")
                
                # Optional: Wait for specific selector
                if wait_for_selector:
                    self.logger.debug(f"Waiting for selector: {wait_for_selector}")
                    page.wait_for_selector(wait_for_selector, timeout=wait_time)
                
                # Extract page content
                html_content = page.content()
                self.logger.info(f"Successfully fetched {len(html_content)} bytes from {url}")
                
                # Debug: Log first 500 chars if suspiciously small
                if len(html_content) < 1000:
                    self.logger.warning(f"⚠️  Content is suspiciously small ({len(html_content)} bytes)")
                    self.logger.warning(f"First 500 chars: {html_content[:500]}")
                
                # Cleanup
                if browser:
                    browser.close()
                if playwright_ctx:
                    playwright_ctx.stop()
                
                return html_content
                
            except Exception as e:
                self.logger.warning(f"Playwright sync attempt {attempt + 1} failed: {e}")
                self.logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
                attempt += 1
                if attempt < retries:
                    time.sleep(delay)
        
        # Cleanup on failure
        if browser:
            try:
                browser.close()
            except:
                pass
        if playwright_ctx:
            try:
                playwright_ctx.stop()
            except:
                pass
        
        self.logger.error(f"Failed to fetch data from {url} after {retries} attempts (sync)")
        return None
    
    def _fetch_data_async_wrapper(self, url, retries, delay, wait_for_selector, wait_time):
        """Wrapper to run async fetch in Jupyter's event loop."""
        import asyncio
        
        # Check if we're already in an event loop (Jupyter)
        try:
            loop = asyncio.get_running_loop()
            # We're in Jupyter's event loop - schedule the coroutine in THIS loop
            self.logger.debug("In existing event loop (Jupyter), scheduling async task")
            
            # Create a task in the current loop and wait for it using run_until_complete
            # This is the key - we need to run it in the SAME loop where it was created
            try:
                import nest_asyncio
                nest_asyncio.apply()
                self.logger.debug("nest_asyncio applied, using asyncio.run()")
                # nest_asyncio allows nested loops
                return asyncio.run(self._fetch_data_async(url, retries, delay, wait_for_selector, wait_time))
            except ImportError:
                self.logger.error("nest_asyncio not installed - required for Playwright in Jupyter")
                self.logger.error("Install with: pip install nest-asyncio")
                raise ImportError("nest_asyncio required for Playwright in Jupyter. Install with: pip install nest-asyncio")
                    
        except RuntimeError:
            # No event loop, create one (normal Python script)
            self.logger.debug("No event loop, creating new one with asyncio.run()")
            return asyncio.run(self._fetch_data_async(url, retries, delay, wait_for_selector, wait_time))
            self.logger.debug("No event loop, creating new one")
            return asyncio.run(self._fetch_data_async(url, retries, delay, wait_for_selector, wait_time))
    
    async def _fetch_data_async(self, url, retries, delay, wait_for_selector, wait_time):
        """Async version of fetch_data for Jupyter notebooks."""
        attempt = 0
        while attempt < retries:
            try:
                await self._ensure_browser_started_async()
                
                # Navigate to URL
                self.logger.debug(f"Playwright async attempt {attempt + 1}: navigating to {url}")
                await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
                
                # Optional: Wait for specific selector
                if wait_for_selector:
                    self.logger.debug(f"Waiting for selector: {wait_for_selector}")
                    await self.page.wait_for_selector(wait_for_selector, timeout=wait_time)
                else:
                    # Default: wait for network to be idle
                    await self.page.wait_for_load_state('networkidle', timeout=wait_time)
                
                # Extract page content
                html_content = await self.page.content()
                self.logger.info(f"Successfully fetched {len(html_content)} bytes from {url}")
                
                return html_content
                
            except Exception as e:
                self.logger.warning(f"Playwright async attempt {attempt + 1} failed: {e}")
                attempt += 1
                if attempt < retries:
                    await asyncio.sleep(delay)
        
        self.logger.error(f"Failed to fetch data from {url} after {retries} attempts (async)")
        return None
    
    def simulate_user_scroll(self, scroll_pause=0.5, max_scrolls=10):
        """
        Simulates user scrolling to load dynamic content.
        
        :param scroll_pause: Pause between scrolls (seconds)
        :param max_scrolls: Maximum number of scroll attempts
        """
        if not self.page:
            self.logger.warning("No active page to scroll")
            return
        
        self.logger.debug("Simulating user scroll to load dynamic content")
        
        previous_height = self.page.evaluate("document.body.scrollHeight")
        scrolls = 0
        
        while scrolls < max_scrolls:
            # Scroll to bottom
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(scroll_pause)
            
            # Check if new content loaded
            new_height = self.page.evaluate("document.body.scrollHeight")
            if new_height == previous_height:
                self.logger.debug(f"No new content after {scrolls} scrolls")
                break
            
            previous_height = new_height
            scrolls += 1
        
        self.logger.debug(f"Completed {scrolls} scrolls")
    
    def extract_publication_title(self):
        """
        Extracts the publication title from the current page.
        
        :return: The publication title as a string
        """
        if not self.page:
            self.logger.warning("No active page to extract title from")
            return "No title found"
        
        try:
            # Try page title first
            title = self.page.title()
            if title and title.strip():
                self.logger.info(f"Paper title (from page.title()): {title}")
                return title.strip()
            
            # Fallback: Try citation_title meta tag
            meta_title = self.page.locator('meta[name="citation_title"]').get_attribute('content')
            if meta_title:
                self.logger.info(f"Paper title (from meta tag): {meta_title}")
                return meta_title.strip()
            
            # Fallback: Try h1.article-title
            h1_title = self.page.locator('h1.article-title').text_content()
            if h1_title:
                self.logger.info(f"Paper title (from h1): {h1_title}")
                return h1_title.strip()
            
            self.logger.warning("Could not extract publication title")
            return "No title found"
            
        except Exception as e:
            self.logger.error(f"Error extracting publication title: {e}")
            return "No title found"
    
    def html_page_source_download(self, directory, pub_link=None):
        """
        Downloads the HTML page source to a file.
        
        :param directory: Directory to save the HTML file
        :param pub_link: URL of the publication (optional)
        """
        if not self.page:
            self.logger.error("No active page to download")
            return
        
        os.makedirs(directory, exist_ok=True)
        
        # Get publication name
        pub_name = self.extract_publication_title()
        pub_name = re.sub(r'[\\/:*?"<>|]', '_', pub_name)
        pub_name = re.sub(r'[\s-]+PMC\s*$', '', pub_name)
        
        # Get PMCID
        pmcid = self.url_to_article_id(pub_link) if pub_link else "unknown"
        
        if directory[-1] != '/':
            directory += '/'
        
        fn = directory + f"{pmcid}__{pub_name}.html"
        self.logger.info(f"Downloading HTML page source to {fn}")
        
        try:
            html_content = self.page.content()
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(html_content)
            self.logger.info(f"Successfully saved HTML to {fn}")
        except Exception as e:
            self.logger.error(f"Error saving HTML: {e}")
    
    def screenshot(self, path, full_page=True):
        """
        Takes a screenshot of the current page.
        
        :param path: Path to save the screenshot
        :param full_page: Whether to capture the full page
        """
        if not self.page:
            self.logger.error("No active page for screenshot")
            return
        
        try:
            self.page.screenshot(path=path, full_page=full_page)
            self.logger.info(f"Screenshot saved to {path}")
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")
    
    def wait_for_element(self, selector, timeout=5000):
        """
        Wait for a specific element to appear.
        
        :param selector: CSS selector to wait for
        :param timeout: Timeout in milliseconds
        :return: True if element appears, False otherwise
        """
        if not self.page:
            return False
        
        try:
            self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception as e:
            self.logger.warning(f"Element {selector} not found: {e}")
            return False
    
    def click_element(self, selector):
        """
        Click an element on the page.
        
        :param selector: CSS selector of element to click
        """
        if not self.page:
            self.logger.error("No active page")
            return
        
        try:
            self.page.click(selector)
            self.logger.debug(f"Clicked element: {selector}")
        except Exception as e:
            self.logger.error(f"Error clicking element {selector}: {e}")
    
    def quit(self):
        """Close the browser and clean up resources."""
        try:
            if self.page:
                self.page.close()
                self.page = None
            if self.context:
                self.context.close()
                self.context = None
            if self.browser:
                self.browser.close()
                self.browser = None
            if self.playwright:
                self.playwright.stop()
                self.playwright = None
            self.logger.info("PlaywrightFetcher closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing PlaywrightFetcher: {e}")


class PdfFetcher(DataFetcher):
    """
    Class for fetching PDF files from URLs.
    """
    def __init__(self, logger, driver_path=None, browser='firefox', headless=True, backup_file='scripts/exp_input/Local_fulltext_pub_REV.parquet'):
        super().__init__(logger, src='PdfFetcher', driver_path=driver_path, browser=browser, headless=headless, backup_file=backup_file)
        self.logger.debug("PdfFetcher initialized.")

    def fetch_data(self, url, return_temp=True, retries=3, delay=2, **kwargs):
        """
        Fetches PDF data from the given URL with retry logic.

        :param url: The URL to fetch data from.
        :param return_temp: Whether to return a temporary file path or raw bytes.
        :param retries: Number of retry attempts for failed requests.
        :param delay: Delay between retries in seconds.
        :return: The raw content of the PDF file or temporary file path.
        """
        self.raw_data_format = 'PDF'
        self.logger.info(f"Fetching PDF data from {url}")

        if os.path.exists(url):
            self.logger.info(f"URL is a local file path. Reading PDF from {url}")
            return url

        # Retry logic for network errors
        for attempt in range(retries):
            try:
                self.logger.debug(f"PDF fetch attempt {attempt + 1}/{retries} for {url}")
                response = requests.get(
                    url, 
                    timeout=30,  # 30 second timeout
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                
                if response.status_code == 200:
                    if kwargs.get('write_raw_data', False):
                        return response.content
                    if return_temp:
                        # Write the PDF content to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                            temp_file.write(response.content)
                            self.logger.info(f"PDF data written to temporary file: {temp_file.name}")
                            return temp_file.name
                    else:
                        return response.content
                elif response.status_code == 403:
                    self.logger.warning(f"403 Forbidden for {url} - likely paywall or anti-bot protection")
                    return None
                else:
                    self.logger.warning(f"Attempt {attempt + 1}: Status code {response.status_code} for {url}")
                    if attempt < retries - 1:
                        time.sleep(delay)
                    
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"Attempt {attempt + 1}: Connection error for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay * 2)  # Longer delay for connection errors
                else:
                    self.logger.error(f"Failed to fetch PDF from {url} after {retries} attempts")
                    return None
                    
            except requests.exceptions.Timeout as e:
                self.logger.warning(f"Attempt {attempt + 1}: Timeout for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    self.logger.error(f"Timeout fetching PDF from {url} after {retries} attempts")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Unexpected error fetching PDF from {url}: {e}")
                return None
        
        self.logger.error(f"Failed to fetch PDF data from {url} after {retries} attempts")
        return None

    def download_pdf(self, directory, raw_data, src_url):
        """
        Downloads the PDF data to a specified directory.
        """
        # Validate that raw_data is bytes (PDF content)
        if not isinstance(raw_data, bytes):
            self.logger.error(f"PDF data from {src_url} is not bytes (got {type(raw_data).__name__}). Cannot save PDF.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        src_url = re.sub(r'\.pdf\s*$', '', src_url)
        
        fn = os.path.join(directory, f"{self.url_to_filename(src_url)}.pdf")

        self.logger.info(f"Downloading PDF from {src_url}")
        with open(fn, 'wb') as file:
            file.write(raw_data)
            self.logger.info(f"PDF data written to file: {file}")
        
        return fn

class DataCompletenessChecker:
    """
    Class to check the completeness of data sections in API responses.
    """
    def __init__(self, logger, publisher='PMC', retrieval_patterns_file='retrieval_patterns.json', raw_data_format='XML'):
        """
        Initializes the DataCompletenessChecker with the specified logger.

        :param logger: The logger instance for logging messages.

        :param publisher: The publisher to check for (default is 'PMC').

        """
        self.logger = logger
        if raw_data_format.upper() == 'XML':
            self.retriever = xmlRetriever(logger, publisher, retrieval_patterns_file)
        elif raw_data_format.upper() == 'HTML':
            self.retriever = htmlRetriever(logger, publisher, retrieval_patterns_file)
        else:
            raise ValueError(f"Unsupported raw data format: {raw_data_format}")
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
        self.retriever = xmlRetriever(self.logger)
        return self.retriever.is_xml_data_complete(raw_data, url, required_sections)

    def is_html_data_complete(self, raw_data, url,
                              required_sections=["data_availability_sections", "supplementary_data_sections"]) -> bool:
        """
        Check if required sections are present in the raw_data.
        Return True if all required sections are present.

        :param raw_data: Raw HTML data.

        :param url: The URL of the article.

        :param required_sections: List of required sections to check.

        :return: True if all required sections are present, False otherwise.
        """
        self.retriever = htmlRetriever(self.logger)
        return self.retriever.is_html_data_complete(raw_data, url, required_sections)

    def is_fulltext_complete(self, raw_data, url, raw_data_format=None,
                            required_sections=["data_availability_sections", "supplementary_data_sections"]) -> bool:
            """
            Check if required sections are present in the raw_data.
            Return True if all required sections are present.
    
            :param raw_data: Raw data (XML or HTML).
    
            :param url: The URL of the article.
    
            :param raw_data_format: Format of the raw data ('XML' or 'HTML').
    
            :param required_sections: List of required sections to check.
    
            :return: True if all required sections are present, False otherwise.
            """

            if raw_data is None:
                self.logger.error("Raw data is None, cannot check completeness.")
                return False

            if raw_data_format is None:
                if isinstance(raw_data, str):
                    self.logger.debug(f"Raw data is a string of length {len(raw_data)}")
                    raw_data_format = 'HTML'
                elif isinstance(raw_data, ET._Element) or isinstance(raw_data, ET._ElementTree):
                    self.logger.debug(f"Raw data is of type {type(raw_data)}")
                    raw_data_format = 'XML'
                else:
                    self.logger.error(f"Unsupported raw data type: {type(raw_data)}")
                    return False
            
            if required_sections is None:
                required_sections = ["data_availability_sections", "supplementary_data_sections"]

            if raw_data_format.upper() == 'XML':
                return self.is_xml_data_complete(raw_data, url, required_sections)
            elif raw_data_format.upper() == 'HTML':
                return self.is_html_data_complete(raw_data, url, required_sections)
            elif raw_data_format.upper() == 'PDF':
                return True # Assume PDFs are complete for this check
            else:
                self.logger.error(f"Unsupported raw data format: {raw_data_format}")
                return False