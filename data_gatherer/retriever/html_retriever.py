from data_gatherer.retriever.base_retriever import BaseRetriever
from data_gatherer.resources_loader import load_config
from bs4 import BeautifulSoup
from lxml import html
import pandas as pd


class htmlRetriever(BaseRetriever):
    """
    HTML Retriever for extracting data availability elements from a webpage.
    This class is designed to extract specific data availability elements from a given webpage URL.
    It uses BeautifulSoup for parsing HTML and extracting relevant information.
    """
    def __init__(self, logger, publisher='PMC', retrieval_patterns_file='retrieval_patterns.json', headers=None):
        """
        Initialize the HTMLRetriever with a URL and optional headers.

        Args:
            url (str): The URL of the webpage to retrieve data from.
            headers (dict, optional): Optional HTTP headers to use for the request.
        """
        self.headers = headers if headers else {}
        self.logger = logger
        self.publisher = publisher
        self.retrieval_patterns = load_config(retrieval_patterns_file)
        self.css_selectors = self.retrieval_patterns[publisher]['css_selectors']
        self.xpaths = self.retrieval_patterns[publisher]['xpaths']

    def extract_href_from_supplementary_material(self, raw_html, current_url_address):
        """
        Extracts href links from supplementary material sections of the HTML.

        :param raw_html: str — raw HTML content.

        :param current_url_address: str — the current URL address being processed.

        :return: DataFrame containing extracted links and their context.

        """
        self.logger.info(f"Function_call: extract_href_from_supplementary_material(tree, {current_url_address})")

        tree = html.fromstring(raw_html)

        supplementary_links = []

        anchors = tree.xpath("//a[@data-ga-action='click_feat_suppl']")
        anchors.extend(tree.xpath("//a[@data-track-action='view supplementary info']"))
        self.logger.debug(f"Found {len(anchors)} anchors with data-ga-action='click_feat_suppl'.")

        for anchor in anchors:
            href = anchor.get("href")
            title = anchor.text_content().strip()

            # Extract ALL attributes from <a>
            anchor_attributes = anchor.attrib  # This gives you a dictionary of all attributes

            # Get <sup> sibling for file size/type info
            sup = anchor.getparent().xpath("./sup")
            file_info = sup[0].text_content().strip() if sup else "n/a"

            # Get <p> description if exists
            p_desc = anchor.getparent().xpath("./p")
            description = p_desc[0].text_content().strip() if p_desc else "n/a"
            self.logger.debug(f"Extracted link: {href}, title: {title}, file_info: {file_info}, description: {description}")

            # Extract attributes from parent <section> for context
            section = anchor.getparent().getparent()  # Assuming structure stays the same
            section_id = section.get('id', 'n/a')
            section_class = section.get('class', 'n/a')

            # Combine all extracted info
            link_data = {
                'link': href,
                'title': title,
                'file_info': file_info,
                'description': description,
                'source_section': section_id,
                'section_class': section_class,
            }

            if link_data['section_class'] == 'ref-list font-sm':
                self.logger.debug(
                    f"Skipping link with section_class 'ref-list font-sm', likely to be a reference list."
                )
                continue

            # if 'doi.org' in link_data['link'] or 'scholar.google.com' in link_data['link']: ############ Same as above
            #    continue

            link_data['download_link'] = self.reconstruct_download_link(href, link_data['section_class'],
                                                                        current_url_address)
            link_data['file_extension'] = self.extract_file_extension(
                link_data['download_link']) if link_data['download_link'] is not None else None

            # Merge anchor attributes (prefix keys to avoid collision)
            for attr_key, attr_value in anchor_attributes.items():
                link_data[f'a_attr_{attr_key}'] = attr_value

            supplementary_links.append(link_data)

        # Convert to DataFrame
        df_supp = pd.DataFrame(supplementary_links)

        # Drop duplicates based on link
        df_supp = df_supp.drop_duplicates(subset=['link'])
        self.logger.info(f"Extracted {len(df_supp)} unique supplementary material links from HTML.")

        return df_supp

    def get_data_availability_elements_from_webpage(self, preprocessed_html, publisher='PMC'):
        """
        Given the preprocessed HTML, extract paragraphs or links under data availability sections.
        """
        self.retrieval_patterns = load_config('retrieval_patterns.json')
        self.logger.info("Extracting data availability elements from HTML")

        # Merge general + publisher-specific selectors
        self.css_selectors = self.retrieval_patterns.get('general', {}).get('css_selectors', {})
        publisher_selectors = self.retrieval_patterns.get(publisher, {}).get('css_selectors', {})
        self.css_selectors.update(publisher_selectors)

        soup = BeautifulSoup(preprocessed_html, "html.parser")
        data_availability_elements = []

        for selector in self.css_selectors.get('data_availability_sections', []):
            self.logger.info(f"Using selector: {selector}")
            matches = soup.select(selector)
            for match in matches:
                if match.name in ['h2', 'h3']:  # headings usually don't contain content directly
                    container = match.find_parent('section') or match.find_next_sibling()
                    if container:
                        children = container.find_all(['p', 'li', 'a', 'div'], recursive=True)
                    else:
                        children = []
                else:
                    children = match.find_all(['p', 'li', 'a', 'div'], recursive=True)

                text_val, html_val = '', ''
                for child in children:
                    if not child.get_text(strip=True):
                        continue
                    text_val += "\n" + child.get_text(strip=True) + "\n"
                    html_val += "\n" + str(child) + "\n"

                element_info = {
                    'retrieval_pattern': selector,
                    'text': text_val,
                    'html': html_val
                }
                data_availability_elements.append(element_info)

                # fallback if nothing found
                if not children:
                    data_availability_elements.append({
                        'retrieval_pattern': selector,
                        'tag': match.name,
                        'text': match.get_text(strip=True),
                        'html': str(match)
                    })
                    data_availability_elements.append(element_info)

                self.logger.debug(f"Extracted data availability element: {element_info}")

        self.logger.info(f"Found {len(data_availability_elements)} data availability elements from HTML.")
        self.data_availability_elements = data_availability_elements
        return data_availability_elements

    def extract_publication_title(self, raw_data):
        """
        Extract the publication title from the HTML content.

        :return: str — the publication title.
        """
        self.logger.info("Extracting publication title from HTML")
        soup = BeautifulSoup(raw_data, "html.parser")
        title_tag = soup.find('article-title')

        h1 = soup.find("h1", class_="article-title")
        if h1:
            return h1.get_text(strip=True)
        elif title_tag:
            return title_tag.get_text(strip=True)
        else:
            self.logger.warning("No publication title found in the HTML data.")
            return "No title found"
    
    def load_target_sections_ptrs(self, section_name) -> dict:
        """
        Load the HTML selectors (CSS and XPath) for the specified section name. Publisher-specific.
        """
        self.logger.info(f"Loading target selectors for section name: {section_name}")
        selectors = {
            "css": self.css_selectors.get(section_name, []),
            "xpath": self.xpaths.get(section_name, [])
        }
        if not selectors["css"] and not selectors["xpath"]:
            self.logger.error(
                f"Invalid section name or no selectors found: {section_name}. Available: {list(self.css_selectors.keys()) + list(self.xpaths.keys())}")
            raise ValueError(f"Invalid section name: {section_name}")
        return selectors

    def has_target_section(self, raw_data, section_name: str) -> bool:
        """
        Check if the target section (data availability or supplementary data) exists in the raw HTML data.

        :param raw_data: Raw HTML data (str).
        :param section_name: Name of the section to check.
        :return: True if the section is found, False otherwise.
        """
        if raw_data is None:
            self.logger.info("No raw data to check for sections.")
            return False

        self.logger.info(f"Checking for {section_name} section in raw HTML data.")
        selectors = self.load_target_sections_ptrs(section_name)
        found = False

        # Check CSS selectors
        if selectors["css"]:
            soup = BeautifulSoup(raw_data, "html.parser")
            for selector in selectors["css"]:
                matches = soup.select(selector)
                if matches:
                    self.logger.info(f"Found section with CSS selector: {selector}")
                    found = True
                    break

        # Check XPaths if not found by CSS
        if not found and selectors["xpath"]:
            try:
                tree = html.fromstring(raw_data)
                for xpath_expr in selectors["xpath"]:
                    matches = tree.xpath(xpath_expr)
                    if matches:
                        self.logger.info(f"Found section with XPath: {xpath_expr}")
                        found = True
                        break
            except Exception as e:
                self.logger.warning(f"Error parsing HTML for XPath: {e}")

        if not found:
            self.logger.info(f"No section found for {section_name}.")
        return found

    def is_html_data_complete(self, raw_data, url,
                             required_sections=("data_availability_sections", "supplementary_data_sections")) -> bool:
        """
        Check if the HTML data is complete by looking for key sections.

        :param raw_data: str — raw HTML content.

        :return: bool — True if data is considered complete, False otherwise.

        """
        self.logger.info(f"Checking if HTML data is complete with required sections: {required_sections}")

        if isinstance(required_sections, list):
            for section in required_sections:
                if not self.has_target_section(raw_data, section):
                    self.logger.info(f"Missing section in HTML: {section}")
                    return False
        
        if isinstance(required_sections, int):
            soup = BeautifulSoup(raw_data, "html.parser")
            sec_elements = soup.find_all("section")
            n_sections_found = len(sec_elements)
            self.logger.info(f"Number of <section> elements found: {n_sections_found}")
            if n_sections_found < required_sections:
                self.logger.info(f"Number of sections {n_sections_found} is less than the required threshold of {required_sections}.")
                return False

        self.logger.info("HTML data contains all required sections.")
        return True