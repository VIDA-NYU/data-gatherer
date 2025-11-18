from data_gatherer.resources_loader import load_config
from data_gatherer.retriever.base_retriever import BaseRetriever
from lxml import etree as ET
import re
import os
import json


class xmlRetriever(BaseRetriever):
    """
    Class to check the completeness of data sections in API responses.
    """

    def __init__(self, logger, publisher='PMC', retrieval_patterns_file='retrieval_patterns.json'):
        """
        Initializes the DataCompletenessChecker with the specified logger.

        :param logger: The logger instance for logging messages.

        :param publisher: The publisher to check for (default is 'PMC').

        """
        super().__init__(publisher, retrieval_patterns_file)
        self.logger = logger
        self.publisher = publisher
        self.css_selectors = self.retrieval_patterns[publisher]['css_selectors']
        self.xpaths = self.retrieval_patterns[publisher]['xpaths']
        self.xml_tags = self.retrieval_patterns[publisher]['xml_tags']

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

    def is_xml_data_complete(self, raw_data, url,
                             required_sections=("data_availability_sections", "supplementary_data_sections")) -> bool:
        """
        Check if required sections are present in the raw_data.
        Check if number of sections is lower than min_threshold.
        Return True if all required sections are present.

        :param raw_data: Raw XML data.

        :param url: The URL of the article.

        :param required_sections: List of required sections to check.

        :return: True if all required sections are present, False otherwise.
        """
        self.logger.debug(f"Checking XML completeness with required sections: {required_sections}")

        if isinstance(required_sections, list):
            for section in required_sections:
                if not self.has_target_xml_tag(raw_data, section) and not self.has_target_xpath(raw_data, section):
                    self.logger.info(f"Missing section in XML: {section}")
                    return False
        
        if isinstance(required_sections, int):
            # find all sections and count them
            sec_elements = raw_data.findall(".//sec")
            n_sections_found = len(sec_elements)
            self.logger.info(f"Number of <sec> elements found: {n_sections_found}")
            if n_sections_found < required_sections:
                self.logger.info(f"Number of sections {n_sections_found} is less than the required threshold of {required_sections}.")
                return False

        self.logger.info("XML data contains all required sections.")
        return True

    def load_target_sections_ptrs(self, section_name) -> list:
        return self.load_target_sections_xml_tags(section_name)

    def load_target_sections_xml_tags(self, section_name) -> list:
        """
        Load the XML tags for the specified section name. Publisher-specific.
        """
        self.logger.info(f"Loading target sections for section name: {section_name}")
        target_sections = self.xml_tags
        if section_name not in target_sections:
            self.logger.error(
                f"Invalid section name: {section_name}. Available sections: {list(target_sections.keys())}")
            raise ValueError(f"Invalid section name: {section_name}")

        return target_sections[section_name]

    def load_target_sections_xpaths(self, section_name) -> dict:
        self.logger.info(f"Loading target sections for section name: {section_name}")
        target_sections = self.xpaths
        if section_name not in target_sections:
            self.logger.error(
                f"Invalid section name: {section_name}. Available sections: {list(target_sections.keys())}")
            raise ValueError(f"Invalid section name: {section_name}")

        return target_sections[section_name]

    def has_links_in_section(self, section, namespaces: dict[str, str]) -> bool:
        """
        Check if the given section contains any external links.

        :param section: The section element to search for links.

        :param namespaces: Namespaces to use for XML parsing.

        :return: True if links are found, False otherwise.
        """
        ext_links = section.findall(".//ext-link", namespaces)
        # uris = section.findall(".//uri", namespaces)

        media_links = section.findall(".//media", namespaces)
        xlink_hrefs = [m.get('{http://www.w3.org/1999/xlink}href') for m in media_links if
                       m.get('{http://www.w3.org/1999/xlink}href')]

        self.logger.debug(f"Found {len(ext_links)} ext-links and {len(xlink_hrefs)} xlink:hrefs.")
        return bool(ext_links or xlink_hrefs)  # or uris)

    def load_patterns_for_tgt_section(self, section_name, publisher='PMC'):
        """
        Load the XML tag patterns for the target section from the configuration.

        :param section_name: str — name of the section to load.

        :param publisher: str — name of the publisher to load patterns for (default is 'PMC').

        :return: str — XML tag patterns for the target section.
        """

        self.publisher = publisher
        self.logger.info(f"Loading patterns for section '{section_name}' for publisher '{self.publisher}'.")

        if self.publisher in self.retrieval_patterns:
            if 'xml_tags' not in self.retrieval_patterns[self.publisher]:
                self.logger.error(f"XML tags not set for publisher '{self.publisher}' in retrieval patterns.")
                return None
            else:
                section_xml_tags_patterns = self.retrieval_patterns[self.publisher]['xml_tags']
                self.logger.info(
                    f"Section pattern keys for publisher {self.publisher}: {section_xml_tags_patterns.keys()}")
                self.logger.info(f"Section name: {section_name}")
                if section_name in section_xml_tags_patterns.keys():
                    self.logger.info(f"Found section '{section_name}' in patterns for publisher '{self.publisher}'.")
                    ret = section_xml_tags_patterns[section_name]
                    self.logger.info(f"Loaded patterns for section '{section_name}': {ret}")
                    return ret

                else:
                    self.logger.error(f"Section name '{section_name}' not found in section patterns.")
                    return None

        else:
            self.logger.warning(
                f"Publisher '{self.publisher}' not found in retrieval patterns. Using default patterns.")

    def extract_href_from_data_availability(self, api_xml):
        """
        Extracts href links from data-availability sections of the XML.

        :param api_xml: lxml.etree.Element — parsed XML root.

        :return: List of dictionaries containing href links and their context.

        """
        # Namespace dictionary - adjust 'ns0' to match the XML if necessary
        self.logger.info(f"Function_call: extract_href_from_data_availability(api_xml)")
        namespaces = {'ns0': 'http://www.w3.org/1999/xlink'}

        title = self.extract_publication_title(api_xml)

        # Find all sections with "data-availability"
        data_availability_sections = []
        for ptr in self.load_patterns_for_tgt_section('data_availability_sections'):
            cont = api_xml.findall(ptr)
            if cont is not None:
                self.logger.info(f"Found {len(cont)} data availability sections. cont: {cont}")
                data_availability_sections.append({"ptr": ptr, "cont": cont})

        hrefs = []
        for das_element in data_availability_sections:
            sections = das_element['cont']
            pattern = das_element['ptr']
            # Find all <ext-link> elements in the section
            for section in sections:
                ext_links = section.findall(".//ext-link", namespaces)
                uris = section.findall(".//uris", namespaces)

                if uris is not None:
                    ext_links.extend(uris)

                self.logger.info(
                    f"Retrieved {len(ext_links)} ext-links in data availability section pattern {pattern}.")

                for link in ext_links:
                    # Extract href attribute
                    href = link.get('{http://www.w3.org/1999/xlink}href')  # Use correct namespace

                    # Extract the text within the ext-link tag
                    link_text = link.text.strip() if link.text else "No description"

                    # Extract surrounding text (parent and siblings)
                    surrounding_text = self.get_surrounding_text(link)

                    if href:
                        hrefs.append({
                            'href': href,
                            'title': title,
                            'link_text': link_text,
                            'surrounding_text': surrounding_text,
                            'source_section': 'data availability',
                            'retrieval_pattern': pattern
                        })
                        self.logger.info(f"Extracted item: {json.dumps(hrefs[-1], indent=4)}")

        return hrefs

    def extract_xrefs_from_data_availability(self, api_xml, current_url_address):
        """
        Extracts xrefs (cross-references) from data-availability sections of the XML.

        :param api_xml: lxml.etree.Element — parsed XML root.

        :param current_url_address: The current URL address being processed.

        :return: List of dictionaries containing xrefs and their context.

        """
        self.logger.info(f"Function_call: extract_xrefs_from_data_availability(api_xml, current_url_address)")

        # Find all sections with "data-availability"
        data_availability_sections = []
        for ptr in self.load_patterns_for_tgt_section('data_availability_sections'):
            self.logger.info(f"Searching for data availability sections using XPath: {ptr}")
            cont = api_xml.findall(ptr)
            if cont is not None:
                self.logger.info(f"Found {len(cont)} data availability sections. cont: {cont}")
                data_availability_sections.append({"ptr": ptr, "cont": cont})

        xrefs = []
        for das_element in data_availability_sections:
            sections = das_element['cont']
            pattern = das_element['ptr']
            for section in sections:
                # Find all <xref> elements in the section
                xref_elements = section.findall(".//xref")

                self.logger.info(f"Found {len(xref_elements)} xref elements in data availability section.")

                for xref in xref_elements:
                    # Extract cross-reference details
                    xref_text = xref.text.strip() if xref.text else "No xref description"
                    ref_type = xref.get('ref-type')
                    rid = xref.get('rid')
                    if ref_type == "bibr":
                        continue

                    # Extract surrounding text (parent and siblings)
                    surrounding_text = self.get_surrounding_text(xref)

                    xrefs.append({
                        'href': current_url_address + '#' + rid,
                        'link_text': xref_text,
                        'surrounding_text': surrounding_text,
                        'source_section': 'data availability',
                        'retrieval_pattern': pattern
                    })
                    self.logger.info(f"Extracted xref item: {json.dumps(xrefs[-1], indent=4)}")

        return xrefs

    def extract_href_from_supplementary_material(self, api_xml, current_url_address):
        """
        Extracts href links from supplementary material sections of the XML.

        :param api_xml: lxml.etree.Element — parsed XML root.

        :param current_url_address: The current URL address being processed.

        :return: List of dictionaries containing href links and their context.

        """

        self.logger.info(f"Function_call: extract_href_from_supplementary_material(api_xml, current_url_address)")

        # Namespace dictionary for xlink
        namespaces = {'xlink': 'http://www.w3.org/1999/xlink'}

        # Find all sections for "supplementary-material"
        supplementary_material_sections = []
        for ptr in self.load_patterns_for_tgt_section('supplementary_material_sections'):
            self.logger.debug(f"Searching for supplementary material sections using XPath: {ptr}")
            cont = api_xml.findall(ptr)
            if cont is not None and len(cont) != 0:
                self.logger.info(f"Found {len(cont)} supplementary material sections {ptr}. cont: {cont}")
                supplementary_material_sections.append({"ptr": ptr, "cont": cont})

        self.logger.debug(f"Found {len(supplementary_material_sections)} supplementary-material sections.")

        hrefs = []

        for section_element in supplementary_material_sections:
            self.logger.info(f"Processing section: {section_element}")
            sections = section_element['cont']
            pattern = section_element['ptr']
            for section in sections:
                # Find all <media> elements in the section (used to link to supplementary files)
                media_links = section.findall(".//media", namespaces)

                for media in media_links:
                    # Extract href attribute from the <media> tag
                    href = media.get('{http://www.w3.org/1999/xlink}href')  # Use correct namespace

                    # Get the parent <supplementary-material> to extract more info (like content-type, id, etc.)
                    supplementary_material_parent = media.getparent()

                    # Extract attributes from <supplementary-material>
                    content_type = supplementary_material_parent.get('content-type', 'Unknown content type')

                    download_link = self.reconstruct_download_link(href, content_type,
                                                                   current_url_address)
                    # this is not up to the retriever to reconstruct the download link, it will go inside parser

                    media_id = supplementary_material_parent.get('id', 'No ID')

                    # Extract the <title> within <caption> for the supplementary material title
                    title_element = supplementary_material_parent.find(".//caption/title")
                    title = title_element.text if title_element is not None else "No Title"

                    # Extract the surrounding text (e.g., description within <p> tag)
                    parent_p = media.getparent()  # Assuming the media element is within a <p> tag
                    if parent_p is not None:
                        surrounding_text = re.sub(r"[\s\n]+", "  ", " ".join(
                            parent_p.itertext()).strip())  # Gets all text within the <p> tag
                    else:
                        surrounding_text = "No surrounding text found"

                    # Extract the full description within the <p> tag if available
                    description_element = supplementary_material_parent.find(".//caption/p")
                    description = ". ".join(
                        description_element.itertext()).strip() if description_element is not None else "No description"

                    # Log media attributes and add to results
                    self.logger.debug(f"Extracted media item with href: {href}")
                    self.logger.debug(f"Source url: {current_url_address}")
                    self.logger.debug(f"Supplementary material title: {title}")
                    self.logger.debug(f"Content type: {content_type}, ID: {media_id}")
                    self.logger.debug(f"Surrounding text for media: {surrounding_text}")
                    self.logger.debug(f"Description: {description}")
                    self.logger.debug(f"Download_link: {download_link}")

                    if href and href not in [item['link'] for item in hrefs]:
                        hrefs.append({
                            'link': href,
                            'source_url': current_url_address,
                            'download_link': download_link,
                            'title': title,
                            'content_type': content_type,
                            'id': media_id,
                            'surrounding_text': surrounding_text,
                            'description': description,
                            'source_section': 'supplementary material',
                            "retrieval_pattern": pattern,
                        })
                        self.logger.debug(f"Extracted item: {json.dumps(hrefs[-1], indent=4)}")

                # Find all <inline-supplementary-material> elements in the section
                inline_supplementary_materials = section.findall(".//inline-supplementary-material")
                self.logger.debug(
                    f"Found {len(inline_supplementary_materials)} inline-supplementary-material elements.")

                for inline in inline_supplementary_materials:
                    # repeating steps like in media links above
                    hrefs.append({
                        "link": inline.get('{http://www.w3.org/1999/xlink}href'),
                        "content_type": inline.get('content-type', 'Unknown content type'),
                        "id": inline.get('id', 'No ID'),
                        "title": inline.get('title', 'No Title'),
                        "source_section": 'supplementary material inline',
                        "retrieval_pattern": ".//inline-supplementary-material",
                        "download_link": self.reconstruct_download_link(
                            inline.get('{http://www.w3.org/1999/xlink}href'),
                            inline.get('content-type', 'Unknown content type'),
                            current_url_address)
                    })

                self.logger.debug(f"Extracted supplementary material links:\n{hrefs}")
        return hrefs

    def get_surrounding_text(self, element):
        """
        Extracts text surrounding the element (including parent and siblings) for more context.
        It ensures that text around inline elements like <xref> and <ext-link> is properly captured.

        :param element: lxml.etree.Element — the element to extract text from.

        :return: str — concatenated text from the parent and siblings of the element.

        """
        # Get the parent element
        parent = element.getparent()

        if parent is None:
            return "No parent element found"

        # Collect all text within the parent element, including inline tags
        parent_text = []

        if parent.text:
            parent_text.append(parent.text.strip())  # Text before any inline elements

        # Traverse through all children (including inline elements) of the parent and capture their text
        for child in parent:
            if child.tag == 'ext-link':
                link_text = child.text if child.text else ''
                link_href = child.get('{http://www.w3.org/1999/xlink}href')
                parent_text.append(f"{link_text} ({link_href})")
            elif child.tag == 'xref':
                # Handle the case for cross-references
                xref_text = child.text if child.text else '[xref]'
                parent_text.append(xref_text)

            # Add the tail text (text after the inline element)
            if child.tail:
                parent_text.append(child.tail.strip())

        # Join the list into a single string for readability
        surrounding_text = " ".join(parent_text)

        return re.sub(r"[\s\n]+(\s+)", "\1", surrounding_text)

    def get_data_availability_sections(self, api_xml):
        data_availability_sections = []
        for ptr in self.load_patterns_for_tgt_section('data_availability_sections'):
            cont = api_xml.findall(ptr)
            if cont is not None:
                self.logger.debug(f"Found {len(cont)} data availability sections. cont: {cont}")
                data_availability_sections.extend(cont)

        self.logger.debug(f"Func get_data_availability_sections returns {len(data_availability_sections)} section")
        return data_availability_sections

    def has_target_xml_tag(self, raw_data, section_name: str) -> bool:
        """
        Check if the target section (data availability or supplementary data) exists in the raw data.

        :param raw_data: Raw XML data.

        :param section_name: Name of the section to check.

        :return: True if the section is found with relevant links, False otherwise.
        """

        if raw_data is None:
            self.logger.info("No raw data to check for sections.")
            return False

        self.logger.debug(f"type of raw_data: {type(raw_data)}, raw_data: {raw_data}")

        self.logger.info(f"----Checking for {section_name} section in raw data.")
        section_patterns = self.load_target_sections_ptrs(section_name)
        self.logger.debug(f"Section patterns: {section_patterns}")
        namespaces = self.extract_namespaces(raw_data)
        self.logger.debug(f"Namespaces: {namespaces}")

        for pattern in section_patterns:
            self.logger.debug(f"Checking pattern: {pattern}")
            sections = raw_data.findall(pattern, namespaces=namespaces)
            if sections:
                for section in sections:
                    self.logger.info(f"----Found section: {ET.tostring(section, encoding='unicode')[:100]}...")
                    if self.has_links_in_section(section, namespaces):
                        return True
                    else:
                        self.logger.warning("No links found in the section.")
                        return True

        return False

    def has_target_xpath(self, raw_data, section_name: str) -> bool:
        """
        Check if the target section (data availability or supplementary data) exists in the raw data.

        :param raw_data: Raw XML data.

        :param section_name: Name of the section to check.

        :return: True if the section is found with relevant links, False otherwise.
        """

        if raw_data is None:
            self.logger.info("No raw data to check for sections.")
            return False

        self.logger.debug(f"type of raw_data: {type(raw_data)}, raw_data: {raw_data}")

        self.logger.info(f"----Checking for {section_name} section in raw data.")
        section_patterns = self.load_target_sections_xpaths(section_name)
        self.logger.debug(f"Section patterns: {section_patterns}")
        namespaces = self.extract_namespaces(raw_data)
        self.logger.debug(f"Namespaces: {namespaces}")

        for pattern in section_patterns:
            self.logger.debug(f"Checking pattern: {pattern}")
            try:
                sections = raw_data.getroottree().xpath(pattern, namespaces=namespaces)
            except Exception as e:
                self.logger.error(f"XPath error for pattern {pattern}: {e}")
                continue

            if sections:
                for section in sections:
                    self.logger.info(f"----Found section: {ET.tostring(section, encoding='unicode')[:100]}...")
                    if self.has_links_in_section(section, namespaces):
                        return True
                    else:
                        self.logger.warning("No links found in the section.")
                        return True

        return False

    def extract_publication_title(self, xml_data):
        """
        Extracts the publication title from the XML data.

        :param xml_data: lxml.etree.Element — parsed XML root.

        :return: str — publication title or 'No title found'.
        """
        self.logger.info("Extracting publication title from XML data.")
        title_element = xml_data.find(".//article-title")
        if title_element is not None and title_element.text:
            title = title_element.text.strip()
            self.logger.info(f"Publication title found: {title}")
            return title
        else:
            self.logger.warning("No publication title found in the XML data.")
            return "No title found"

    def reconstruct_download_link(self, href, content_type, current_url_address):
        download_link = None
        # match the digits of the PMC ID (after PMC) in the URL
        self.logger.debug(f"Function_call: reconstruct_download_link({href}, {content_type}, {current_url_address})")
        if self.publisher == 'PMC':
            pmcid = re.search(r'PMC(\d+)', current_url_address, re.IGNORECASE).group(1)
            self.logger.debug(
                f"Inputs to reconstruct_download_link: {href}, {content_type}, {current_url_address}, {pmcid}")
            if content_type == 'local-data':
                download_link = "https://pmc.ncbi.nlm.nih.gov/articles/instance/" + pmcid + '/bin/' + href
            elif content_type == 'media p':
                file_name = os.path.basename(href)
                self.logger.debug(f"Extracted file name: {file_name} from href: {href}")
                download_link = "https://www.ncbi.nlm.nih.gov/pmc" + href
        return download_link
