from abc import ABC, abstractmethod
import re
import logging
from selenium.webdriver.common.by import By
import json
import os
import time
import requests
from lxml import etree as ET
from selenium_setup import create_driver
from logger_setup import setup_logging
import mimetypes
from bs4 import BeautifulSoup

# Abstract base class for fetching data
class DataFetcher(ABC):
    def __init__(self, config, logger, src='WebScraper'):
        self.config = config
        self.fetch_source = src
        self.logger = logger
        self.logger.debug("DataFetcher initialized.")

    def url_to_publisher_domain(self, url):
        # Extract the domain name from the URL
        if re.match(r'^https?://www\.ncbi\.nlm\.nih\.gov/pmc', url):
            return 'PMC'
        match = re.match(r'^https?://(?:\w+\.)?([\w\d\-]+)\.\w+\/', url)
        if match:
            domain = match.group(1)
            self.logger.info(f"Publisher: {domain}")
            return domain
        else:
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

    def update_DataFetcher_settings(self, url, entire_doc_model, logger):
        """Sets up either a web scraper or API client based on the URL domain."""
        self.logger.debug(f"update_DataFetcher_settings for current URL")

        API = None

        # Check if the URL corresponds to PubMed Central (PMC)
        for src,ptrn in self.config['API_supported_url_patterns'].items():
            #self.logger.info(f"Checking {src} with pattern {ptrn}")
            match = re.match(ptrn, url)
            if match:
                self.logger.debug(f"URL detected as {src}.")
                API = f"{src}_API"

        if API is not None and not(entire_doc_model):
        # Initialize the corresponding API client, from API_supported_url_patterns
            self.logger.debug(f"Initializing APIClient({'requests', API, 'self.config'})")
            return APIClient(requests, API, self.config, logger)

        else:
            self.logger.info("Non-API URL detected, or API unsupported. Webscraper update")
            self.fetch_source = 'WebScraper'
            driver = create_driver(self.config['DRIVER_PATH'], self.config['BROWSER'], self.config['HEADLESS'])
            return WebScraper(driver, self.config, logger)

        self.logger.info("Data fetcher setup completed.")

    def is_url_API(self, url):

        return True

    @abstractmethod
    def fetch_data(self, source):
        pass

# Implementation for fetching data via web scraping
class WebScraper(DataFetcher):
    def __init__(self, scraper_tool, config, logger):
        super().__init__(config, logger)
        self.scraper_tool = scraper_tool  # Inject your scraping tool (BeautifulSoup, Selenium, etc.)
        self.classification_patterns = json.load(open(self.config['classification_patterns']))
        self.bad_patterns = self.classification_patterns['general']['bad_patterns']
        self.css_selectors = self.classification_patterns['general']['css_selectors']
        self.xpaths = self.classification_patterns['general']['xpaths']


    def fetch_data(self, url):
        # Use the scraper tool to fetch raw HTML from the URL
        self.scraper_tool.get(url)
        self.simulate_user_scroll()
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

    def simulate_user_scroll(self):
        last_height = self.scraper_tool.execute_script("return document.body.scrollHeight")
        while True:
            # Scroll down to bottom
            self.scraper_tool.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(3)

            # Calculate new height and compare with last height
            new_height = self.scraper_tool.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def update_class_patterns(self, publisher):
        patterns = self.classification_patterns[publisher]
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

        if publisher in self.classification_patterns:
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

    def download_html(self, dir):
        logging.info(f"Dir {dir} exists") if os.path.exists(dir) else os.mkdir(dir)

        fn = dir + self.get_publication_name_from_driver() + '.html'

        with open(fn, 'w', encoding='utf-8') as f:
            f.write(self.scraper_tool.page_source)

    def get_publication_name_from_driver(self):
        publication_name_pointer = self.scraper_tool.find_element(By.TAG_NAME, 'title')
        publication_name = re.sub("\n+", "", (publication_name_pointer.get_attribute("text")))
        self.logger.info(f"Paper name: {publication_name}")
        return publication_name

    def quit(self):
        """Properly quits the underlying WebDriver."""
        if self.scraper_tool:
            self.scraper_tool.quit()
            self.logger.info("WebScraper driver quit.")


# Implementation for fetching data from an API
class APIClient(DataFetcher):
    def __init__(self, api_client, API, config, logger):
        super().__init__(config, logger, src=API)
        self.api_client = api_client.Session()
        self.base = self.config['API_base_url'][API]

    def fetch_data(self, article_url, retries=3, delay=2):
        try:
            # Extract the PMC ID from the article URL
            PMCID = re.search(r'PMC\d+', article_url).group(0)

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

    def download_xml(self, dir, api_data):

        ET.ElementTree(api_data).write(dir, pretty_print=True, xml_declaration=True,
                                          encoding='UTF-8')

class DataCompletenessChecker:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.safety_driver = create_driver(self.config['DRIVER_PATH'], self.config['BROWSER'], self.config['HEADLESS'])
        self.classification_patterns = json.load(open(self.config['classification_patterns']))
        self.css_selectors = self.classification_patterns['PMC']['css_selectors']
        self.xpaths = self.classification_patterns['PMC']['xpaths']

    def ensure_data_sections(self, raw_data, url):
        """
        Check if the data sections exist in the raw_data (support only for XML for now)
        :param raw_data:
        :param url:
        :return: additional data scraped from the web (list of dictionaries)
        """
        self.logger.debug(f"Function call ensure_data_sections({raw_data}, {url})")

        additional_data_ret = []
        required_sections = ["data_availability", "supplementary_data"]

        for section in required_sections:
            if not self.has_target_section(raw_data, section):
                self.logger.info(f"{section} section missing for {url}. Fetching from web...")
                additional_data_ret.extend(self.get_section_from_webpage(url, section))
            else:
                self.logger.info(f"{section} section found in raw data.")

        return additional_data_ret

    def has_target_section(self, raw_data, section_name: str) -> bool:
        """
        Check if the target section (data availability or supplementary data) exists in the raw data.
        :param raw_data: Parsed XML or HTML data.
        :param section_name: Name of the section to check.
        :return: True if the section is found with relevant links, False otherwise.
        """

        if raw_data is None:
            self.logger.info("No raw data to check for sections.")
            return False

        self.logger.debug(f"Checking for {section_name} section in raw data.")

        # Load section patterns from the configuration
        section_patterns = self.config[section_name + "_sections"]

        namespaces = {'ns0': 'http://www.w3.org/1999/xlink'}

        # Check if any of the patterns match sections in the raw data
        for pattern in section_patterns:
            sections = raw_data.findall(pattern)
            if sections:
                for section in sections:
                    # Check if the section contains relevant links
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
        self.logger.debug(f"Found {len(ext_links)} ext-links.")
        return bool(ext_links) #or uris)

    def get_section_from_webpage(self, url: str, section_name: str):
        """
        Scrape the webpage to extract the specified section (data availability or supplementary data).
        :param url: URL of the webpage.
        :param section_name: Name of the section to extract.
        :return: List of dictionaries with extracted data.
        """
        try:
            self.logger.info(f"Fetching {section_name} section from: {url}")
            self.safety_driver.get(url)
            time.sleep(1)  # Allow time for the page to load

            if self.config['write_htmls_xmls']:
                self.download_html(self.config['html_xml_dir'] + self.url_to_publisher_domain(url) + '/')


            # Use appropriate XPath patterns based on the section name
            xpaths = self.xpaths[section_name]
            self.logger.info(f"Using XPaths: {xpaths}")
            elements = self.extract_elements_by_xpath(xpaths, section_name)
            self.logger.info(f"Found {len(elements)} elements in {section_name} section.")

            # Format the extracted data
            items = []
            for el in elements:
                if el is None:
                    continue

                item = {
                    "link": el["link"],
                    "source_section": section_name + '_elements', # maybe change this to data_availability if text contains? Check later is better
                    "source_url": url,
                    "surrounding_text": el["text"] if "text" in el else "",
                    "download_link": self.reconstruct_download_link(el["link"], "local-data", url)
                }
                items.append(item)


            DAS_text_matches = self.extract_text_by_xpaths(xpaths)
            for elm in DAS_text_matches:
                self.logger.info(f"Found data-like text: {elm}")
                item = {
                    "link": "n/a",
                    "source_section": "data_availability",
                    "source_url": url,
                    "surrounding_text": elm,
                }
                if item not in items:
                    items.append(item)

            self.logger.info(f"items: {items}")

            return items

        except Exception as e:
            self.logger.error(f"Error fetching {section_name} section from {url}: {e}")


    def reconstruct_download_link(self, href, content_type, current_url_address):
        # https: // pmc.ncbi.nlm.nih.gov / articles / instance / 11252349 / bin / 41598_2024_67079_MOESM1_ESM.zip
        # https://pmc.ncbi.nlm.nih.gov/articles/instance/PMC11252349/bin/41598_2024_67079_MOESM1_ESM.zip
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11252349/bin/41598_2024_67079_MOESM1_ESM.zip
        download_link = None
        #repo = self.url_to_repo_domain(current_url_address)
        # match the digits of the PMC ID (after PMC) in the URL
        PMCID = re.search(r'PMC(\d+)', current_url_address).group(1)
        self.logger.debug(f"Inputs to reconstruct_download_link: {href}, {content_type}, {current_url_address}, {PMCID}")
        if content_type == 'local-data':
            download_link = "https://pmc.ncbi.nlm.nih.gov/articles/instance/" + PMCID + '/bin/' + href
        return download_link

    def extract_text_by_xpaths(self, xpaths):
        
        rule_based_matches = []

        for xpath in xpaths:
            self.logger.info(f"Checking path: {xpath}")
            try:
                child_element = self.safety_driver.find_element(By.XPATH, xpath)
                text = child_element.text
                if text:
                    self.logger.info(f"Found das-like text: {text}")
                    rule_based_matches.append(text)

            except Exception as e:
                self.logger.error(f"Invalid xpath: {xpath}")

        self.logger.info(f"Rule-based matches from xpaths: {len(rule_based_matches)} items.")
        self.logger.debug(f"Rule-based matches from xpaths: {rule_based_matches}.")
        return rule_based_matches

    def extract_elements_by_xpath(self, xpaths, section):
        """
        Extract elements from the webpage using the provided XPath patterns.
        :param xpaths: List of XPath patterns to use for extraction.
        :return: List of extracted element texts and links.
        """
        rule_based_matches = []

        for xpath in xpaths:
            self.logger.info(f"Checking path: {xpath}")
            try:
                child_element = self.safety_driver.find_element(By.XPATH, xpath)
                anchor = child_element.find_elements(By.TAG_NAME, 'a')
                # get href attribute from anchor elements if any links in anchor
                if anchor:
                    links = [a.get_attribute('href') for a in anchor]
                    # drop None values from link
                    links = [link for link in links if link]
                    self.logger.info(f"Found links: {links}")

                    text = child_element.text
                    rule_based_matches.extend([{"link": link, "surrounding_text": text} for link in links if link])

            except Exception as e:
                self.logger.error(f"Invalid xpath: {xpath}")

        self.logger.info(f"Rule-based matches from xpaths: {len(rule_based_matches)} items.")
        self.logger.debug(f"Rule-based matches from xpaths: {rule_based_matches}")
        return rule_based_matches

    def download_html(self, directory):
        logging.info(f"Dir {directory} exists") if os.path.exists(directory) else os.mkdir(directory)

        fn = directory + self.get_publication_name_from_driver() + '.html'

        with open(fn, 'w', encoding='utf-8') as f:
            f.write(self.safety_driver.page_source)

    def get_publication_name_from_driver(self):
        publication_name_pointer = self.safety_driver.find_element(By.TAG_NAME, 'title')
        publication_name = re.sub("\n+", "", (publication_name_pointer.get_attribute("text")))
        publication_name = re.sub("\s+", " ", (publication_name))
        self.logger.info(f"Paper name: {publication_name}")
        return publication_name

    def url_to_publisher_domain(self, url):
        # Extract the domain name from the URL
        if re.match(r'^https?://www\.ncbi\.nlm\.nih\.gov/pmc', url):
            return 'PMC'
        match = re.match(r'^https?://(?:\w+\.)?([\w\d\-]+)\.\w+\/', url)
        if match:
            domain = match.group(1)
            self.logger.info(f"Publisher: {domain}")
            return domain
        else:
            return 'Unknown Publisher'