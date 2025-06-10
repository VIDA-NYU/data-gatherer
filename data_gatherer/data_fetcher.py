from abc import ABC, abstractmethod
import re
import logging
import numpy as np
from selenium.webdriver.common.by import By
import json
import os
import time
import requests
from lxml import etree as ET
from data_gatherer.selenium_setup import create_driver
from data_gatherer.logger_setup import setup_logging
import mimetypes
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse
import pandas as pd
from data_gatherer.resources_loader import load_config

# Abstract base class for fetching data
class DataFetcher(ABC):
    def __init__(self, logger, src='WebScraper', driver_path=None, browser='firefox', headless=True,
                 raw_HTML_data_filepath=None):
        self.dataframe_fetch = True  # Flag to indicate dataframe fetch supported or not
        self.raw_HTML_data_filepath = raw_HTML_data_filepath
        self.fetch_source = src
        self.logger = logger
        self.logger.debug("DataFetcher initialized.")
        self.driver_path = driver_path
        self.browser = browser
        self.headless = headless

    def url_to_publisher_domain(self, url):
        # Extract the domain name from the URL
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
            self.logger.info("Unknown publisher")
            return 'Unknown Publisher'

    def url_to_publisher_root(self, url):
        # Extract the root domain name from the URL
        match = re.match('(https?:\/\/[\w\.]+)\/', url)
        if match:
            root = match.group(1)
            self.logger.info(f"Root: {root}")
            return root
        else:
            return 'Unknown Publisher'

    def update_DataFetcher_settings(self, url, entire_doc_model, logger, HTML_fallback=False):
        """
        Sets up either a web scraper or API client based on the URL domain.
        Also used to avoid re_instantiating another selenium webdriver.

        :param url: The URL to fetch data from.

        :param entire_doc_model: Flag to indicate if the entire document model is being used.

        :param logger: The logger instance for logging messages.

        :param HTML_fallback: Flag to indicate if HTML fallback is needed.

        :return: An instance of the appropriate data fetcher (WebScraper or APIClient).
        """
        self.logger.debug(f"update_DataFetcher_settings for current URL")

        API = None

        if not HTML_fallback:
            # Check if the URL corresponds to PubMed Central (PMC)
            for ptr,src in API_supported_url_patterns.items():
                self.logger.debug(f"Checking {src} with pattern {ptr}")
                match = re.match(ptr, url)
                if match:
                    self.logger.debug(f"URL detected as {src}.")
                    API = f"{src}_API"
                    break

        if API is not None and not(entire_doc_model):
        # Initialize the corresponding API client, from API_supported_url_patterns
            self.logger.info(f"Initializing APIClient({'requests', API, 'self.config'})")
            return APIClient(requests, API, logger)

        if self.raw_HTML_data_filepath and self.dataframe_fetch and self.url_in_dataframe(url, self.raw_HTML_data_filepath):
            self.logger.info(f"URL {url} found in DataFrame. Using DatabaseFetcher.")
            return DatabaseFetcher(logger, self.raw_HTML_data_filepath)

        # Reuse existing driver if we already have one
        if isinstance(self, WebScraper) and self.scraper_tool is not None:
            self.logger.info(f"Reusing existing WebScraper driver: {self.scraper_tool}")
            return self  # Reuse current instance

        self.logger.info(f"WebScraper instance: {isinstance(self, WebScraper)}")
        self.logger.info(f"APIClient instance: {isinstance(self, APIClient)}")
        self.logger.info(f"scraper_cool attribute: {hasattr(self,'scraper_tool')}")

        self.logger.info("Initializing new selenium driver.")
        driver = create_driver(self.driver_path, self.browser, self.headless, self.logger)
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

        return True if pmcid.lower() in df_fetch['publication'].values else False

    def download_html(self, dir):
        """
        Downloads the HTML content to a specified directory.

        :param dir: The directory where the HTML file will be saved.

        """
        logging.info(f"Dir {dir} exists") if os.path.exists(dir) else os.mkdir(dir)

        pub_name = self.get_publication_name_from_driver()

        pub_name = re.sub(r'[\\/:*?"<>|]', '_', pub_name)  # Replace invalid characters in filename

        fn = dir + pub_name + '.html'

        with open(fn, 'w', encoding='utf-8') as f:
            f.write(self.scraper_tool.page_source)

    def is_url_API(self, url):
        return notImplementedError("This method has not been implemented yet.")

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
    def __init__(self, scraper_tool, logger, retrieval_patterns_file=None, driver_path=None, browser='firefox', headless=True):
        super().__init__(logger)
        self.scraper_tool = scraper_tool  # Inject your scraping tool (BeautifulSoup, Selenium, etc.)
        self.retrieval_patterns = load_config('retrieval_patterns.json')
        self.bad_patterns = self.retrieval_patterns['general']['bad_patterns']
        self.css_selectors = self.retrieval_patterns['general']['css_selectors']
        self.xpaths = self.retrieval_patterns['general']['xpaths']
        self.driver_path = driver_path
        self.browser = browser
        self.headless = headless

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
        self.scraper_tool.get(url)
        self.simulate_user_scroll(delay)
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

    def simulate_user_scroll(self, delay=2):
        np.random.random()*delay + 1
        last_height = self.scraper_tool.execute_script("return document.body.scrollHeight")
        while True:
            # Scroll down to bottom
            self.scraper_tool.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(np.random.random()*delay)

            # Calculate new height and compare with last height
            new_height = self.scraper_tool.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def update_class_patterns(self, publisher):
        patterns = self.retrieval_patterns[publisher]
        self.css_selectors.update(patterns['css_selectors'])
        self.xpaths.update(patterns['xpaths'])
        if 'bad_patterns' in patterns.keys():
            self.bad_patterns.extend(patterns['bad_patterns'])

    def normalize_links(self, links_raw_dict):
        """
        Convert Selenium WebElement objects to strings and keep existing string links unchanged.
        """
        normalized_links = {}
        for link, cls  in links_raw_dict.items():
            if isinstance(link, str):
                normalized_links[link] = cls
            else:
                try:
                    href = link.get_attribute('href')
                    if href:
                        normalized_links[href] = cls
                except Exception as e:
                    logging.error(f"Error getting href attribute from WebElement")
        return normalized_links

    def get_rule_based_matches(self, publisher):

        if publisher in self.retrieval_patterns:
            self.update_class_patterns(publisher)

        rule_based_matches = {}

        # Collect links using CSS selectors
        for css_selector in self.css_selectors:
            self.logger.debug(f"Parsing page with selector: {css_selector}")
            links = self.scraper_tool.find_elements(By.CSS_SELECTOR, css_selector)
            self.logger.debug(f"Found Links: {links}")
            for link in links:
                rule_based_matches[link] = self.css_selectors[css_selector]
        self.logger.info(f"Rule-based matches from css_selectors: {rule_based_matches}")

        # Collect links using XPath
        for xpath in self.xpaths:
            self.logger.info(f"Checking path: {xpath}")
            try:
                child_element = self.scraper_tool.find_element(By.XPATH, xpath)
                section_element = child_element.find_element(By.XPATH, "./ancestor::section")
                a_elements = section_element.find_elements(By.TAG_NAME, 'a')
                for a_element in a_elements:
                    rule_based_matches[a_element] = self.xpaths[xpath]
            except Exception as e:
                self.logger.error(f"Invalid xpath: {xpath}")

        return self.normalize_links(rule_based_matches)

    def normalize_HTML(self,html, keep_tags=None):
        """
        Normalize the HTML content by removing unnecessary tags and attributes.

        :param html: The raw HTML content to be normalized.

        :param keep_tags: List of tags to keep in the HTML content. May be useful for some servers that host target info inside specific tags (e.g., <form> or <script>) that are otherwise removed by the scraper.

        :return: The normalized HTML content.

        """
        try:
            # Parse the HTML content
            soup = BeautifulSoup(html, "html.parser")

            # 1. Remove script, style, and meta tags
            for tag in ["script", "style", 'img', 'noscript', 'svg', 'button', 'form', 'input']:
                if keep_tags and tag in keep_tags:
                    self.logger.info(f"Keeping tag: {tag}")
                    continue
                for element in soup.find_all(tag):
                    element.decompose()

            remove_meta_tags = True
            if remove_meta_tags:
                for meta in soup.find_all('meta'):
                    meta.decompose()

            # 2. Remove dynamic attributes
            for tag in soup.find_all(True):  # True matches all tags
                # Remove dynamic `id` attributes that match certain patterns (e.g., `tooltip-*`)
                if "id" in tag.attrs and re.match(r"tooltip-\d+", tag.attrs["id"]):
                    del tag.attrs["id"]

                # Remove dynamic `aria-describedby` attributes
                if "aria-describedby" in tag.attrs and re.match(r"tooltip-\d+", tag.attrs["aria-describedby"]):
                    del tag.attrs["aria-describedby"]

                # Remove inline styles
                if "style" in tag.attrs:
                    del tag.attrs["style"]

                # Remove all `data-*` attributes
                tag.attrs = {key: val for key, val in tag.attrs.items() if not key.startswith("data-")}

                # Remove `csrfmiddlewaretoken` inputs
                if tag.name == "input" and tag.get("name") == "csrfmiddlewaretoken":
                    tag.decompose()

            # 3. Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            # 4. Normalize whitespace
            normalized_html = re.sub(r"\s+", " ", soup.prettify())

            return normalized_html.strip()

        except Exception as e:
            self.logger.error(f"Error normalizing DOM: {e}")
            return ""

    def get_publication_name_from_driver(self):
        """
        Extracts the publication name from the WebDriver's current page title. **Remark**: this should be called after
        scraper_tool.get(url) to ensure the page is loaded.

        :return: The publication name as a string.

        """
        publication_name_pointer = self.scraper_tool.find_element(By.TAG_NAME, 'title')
        publication_name = re.sub("\n+", "", (publication_name_pointer.get_attribute("text")))
        publication_name = re.sub("^\s+", "", publication_name)
        self.logger.info(f"Paper name: {publication_name}")
        return publication_name

    def get_url_from_pubmed_id(self, pubmed_id):
        return f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"

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

    def get_filename_from_url(self,url):
        parsed_url = urlparse(self,url)
        return os.path.basename(parsed_url.path)

    def reconstruct_PMC_link(self, PMCID):
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{PMCID}"

    def get_opendata_from_pubmed_id(self, pmid):
        """
        Given a PubMed ID, fetches the corresponding PMC ID and DOI from PubMed.

        :param pmid: The PubMed ID to fetch data for.

        :return: A tuple containing the PMC ID and DOI.

        """
        url = self.get_url_from_pubmed_id(pmid)
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
        super().__init__(logger)
        self.data_file = raw_HTML_data_filepath
        self.dataframe = pd.read_parquet(self.data_file)

    def fetch_data(self, url_key, retries=3, delay=2):
        """
        Fetches data from a local file or database.

        :param url_key: The key to identify the data in the database.

        :returns: The raw HTML content of the page.
        """
        split_source_url = url_key.split('/')
        key = (split_source_url[-1] if len(split_source_url[-1]) > 0 else split_source_url[-2]).lower()
        self.logger.info(f"Fetching data for {key}")
        self.logger.info(f"Data file: {self.dataframe.columns}")
        self.logger.info(f"Data file: {self.dataframe[self.dataframe['publication'] == key]}")
        self.logger.info(f"Fetching data from {self.data_file}")
        self.fetch_source = 'Local_data'
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

    def get_rule_based_matches(self, publisher):

        if publisher in self.retrieval_patterns:
            self.update_class_patterns(publisher)

        rule_based_matches = {}

        # Collect links using CSS selectors
        for css_selector in self.css_selectors:
            self.logger.debug(f"Parsing page with selector: {css_selector}")
            links = self.scraper_tool.find_elements(By.CSS_SELECTOR, css_selector)
            self.logger.debug(f"Found Links: {links}")
            for link in links:
                rule_based_matches[link] = self.css_selectors[css_selector]
        self.logger.info(f"Rule-based matches from css_selectors: {rule_based_matches}")

        # Collect links using XPath
        for xpath in self.xpaths:
            self.logger.info(f"Checking path: {xpath}")
            try:
                child_element = self.scraper_tool.find_element(By.XPATH, xpath)
                section_element = child_element.find_element(By.XPATH, "./ancestor::section")
                a_elements = section_element.find_elements(By.TAG_NAME, 'a')
                for a_element in a_elements:
                    rule_based_matches[a_element] = self.xpaths[xpath]
            except Exception as e:
                self.logger.error(f"Invalid xpath: {xpath}")

        return self.normalize_links(rule_based_matches)

# Implementation for fetching data from an API
class APIClient(DataFetcher):
    """
    Class for fetching data from an API using the requests library.
    """
    def __init__(self, api_client, API, logger):
        """
        Initializes the APIClient with the specified API client.

        :param api_client: The API client to use (e.g., requests).

        :param API: The API to use (e.g., PMC).


        :param logger: The logger instance for logging messages.

        """
        super().__init__(logger, src=API)
        self.api_client = api_client.Session()
        self.base = self.config['API_base_url'][API]

    def fetch_data(self, article_url, retries=3, delay=2):
        """
        Fetches data from the API using the provided article URL.

        :param article_url: The URL of the article to fetch data for.

        """
        try:
            # Extract the PMC ID from the article URL
            PMCID = re.search(r'PMC\d+', article_url).group(0)
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
            self.logger.error(f"Network error fetching data for {article_url}: {req_err}")
            return None
        except Exception as e:
            # Log any other exceptions
            self.logger.error(f"Error fetching data for {article_url}: {e}")
            return None

    def download_xml(self, directory, api_data):
        """
        Downloads the XML data to a specified directory.

        :param directory: The directory where the XML file will be saved.

        :param api_data: The XML data to be saved.
        """
        # Construct the file path
        fn = os.path.join(directory, f"{self.extract_article_title()}.xml")

        # Check if the file already exists
        if os.path.exists(fn):
            self.logger.info(f"File already exists: {fn}. Skipping download.")
            return

        # Write the XML data to the file
        ET.ElementTree(api_data).write(fn, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        self.logger.info(f"Downloaded XML file: {fn}")

    from lxml import etree as ET

    def extract_article_title(self, api_data):
        """
        Extracts the article title and the surname of the first author from the XML content.

        :param xml_content: The XML content as a string.

        :return: A tuple containing the article title and the first author's surname.
        """
        try:
            # Extract the article title
            title = root.find(".//title-group/article-title")
            article_title = title.text.strip() if title is not None else None
            return article_title

        except ET.XMLSyntaxError as e:
            print(f"Error parsing XML: {e}")
            return None

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
        self.config = config
        self.logger = logger
        self.retrieval_patterns = load_config(retrieval_patterns_file)
        self.css_selectors = self.retrieval_patterns[publisher]['css_selectors']
        self.xpaths = self.retrieval_patterns[publisher]['xpaths']

    def extract_namespaces(self, xml_element):
        """Extract all namespaces in use, including xlink."""
        ns_map = {'xlink': 'http://www.w3.org/1999/xlink'}
        for elem in xml_element.iter():
            tag = getattr(elem, "tag", None)
            if isinstance(tag, str) and tag.startswith("{"):
                uri = tag[1:].split("}")[0]
                prefix = elem.prefix or 'ns0'
                ns_map[prefix] = uri
        return ns_map

    def is_xml_data_complete(self, raw_data, url, required_sections = ["data_availability", "supplementary_data"]) -> bool:
        """
        Check if required sections are present in the raw_data.
        Return True if all required sections are present.

        :param raw_data: Raw XML data.

        :param url: The URL of the article.

        :param required_sections: List of required sections to check.

        :return: True if all required sections are present, False otherwise.
        """
        self.logger.debug(f"Checking XML completeness for {url}")

        for section in required_sections:
            if not self.has_target_section(raw_data, section):
                self.logger.info(f"Missing section in XML: {section}")
                return False

        self.logger.info("XML data contains all required sections.")
        return True

    def has_target_section(self, raw_data, section_name: str) -> bool:
        """
        Check if the target section (data availability or supplementary data) exists in the raw data.

        :param raw_data: Raw XML data.

        :param section_name: Name of the section to check.

        :return: True if the section is found with relevant links, False otherwise.
        """

        if raw_data is None:
            self.logger.info("No raw data to check for sections.")
            return False

        self.logger.info(f"----Checking for {section_name} section in raw data.")
        section_patterns = self.config[section_name + "_sections"]
        namespaces = self.extract_namespaces(raw_data)

        for pattern in section_patterns:
            sections = raw_data.findall(pattern, namespaces=namespaces)
            if sections:
                for section in sections:
                    self.logger.info(f"----Found section: {ET.tostring(section, encoding='unicode')}")
                    if self.has_links_in_section(section, namespaces):
                        return True

        return False

    def has_links_in_section(self, section, namespaces: dict[str, str]) -> bool:
        """
        Check if the given section contains any external links.

        :param section: The section element to search for links.

        :param namespaces: Namespaces to use for XML parsing.

        :return: True if links are found, False otherwise.
        """
        ext_links = section.findall(".//ext-link", namespaces)
        #uris = section.findall(".//uri", namespaces)

        media_links = section.findall(".//media", namespaces)
        xlink_hrefs = [m.get('{http://www.w3.org/1999/xlink}href') for m in media_links if
                 m.get('{http://www.w3.org/1999/xlink}href')]

        self.logger.debug(f"Found {len(ext_links)} ext-links and {len(xlink_hrefs)} xlink:hrefs.")
        return bool(ext_links or xlink_hrefs)  #or uris)

    def url_to_publisher_domain(self, url):
        # Extract the domain name from the URL
        if re.match(r'^https?://www\.ncbi\.nlm\.nih\.gov/pmc', url) or re.match(r'^https?://pmc\.ncbi\.nlm\.nih\.gov/', url):
            return 'PMC'
        match = re.match(r'^https?://(?:\w+\.)?([\w\d\-]+)\.\w+', url)
        if match:
            domain = match.group(1)
            self.logger.info(f"Publisher: {domain}")
            return domain
        else:
            return 'Unknown Publisher'