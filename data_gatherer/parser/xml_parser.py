from data_gatherer.retriever.xml_retriever import xmlRetriever
from data_gatherer.parser.base_parser import *
from lxml import etree
import os
import pandas as pd
import json
import re

class XMLParser(LLMParser):
    def __init__(self, open_data_repos_ontology, logger, log_file_override=None, full_document_read=True,
                 prompt_dir="data_gatherer/prompts/prompt_templates", response_file="data_gatherer/prompts/LLMs_responses_cache.json",
                 llm_name=None, save_dynamic_prompts=False, save_responses_to_cache=False, use_cached_responses=False,
                 use_portkey_for_gemini=True):

        super().__init__(open_data_repos_ontology, logger, log_file_override=log_file_override,
                         full_document_read=full_document_read, prompt_dir=prompt_dir, response_file=response_file,
                         llm_name=llm_name, save_dynamic_prompts=save_dynamic_prompts,
                         save_responses_to_cache=save_responses_to_cache,
                         use_cached_responses=use_cached_responses, use_portkey_for_gemini=use_portkey_for_gemini
                         )

        self.logger = logger
        self.logger.info("Initializing xmlRetriever")
        self.retriever = xmlRetriever(self.logger, publisher='PMC')

    def extract_paragraphs_from_xml(self, xml_root) -> list[dict]:
        """
        Extract paragraphs and their section context from an XML document.

        Args:
            xml_root: lxml.etree.Element — parsed XML root.

        Returns:
            List of dicts with 'paragraph', 'section_title', and 'sec_type'.
        """
        paragraphs = []

        # Iterate over all section blocks
        for sec in xml_root.findall(".//sec"):
            sec_type = sec.get("sec-type", "unknown")
            title_elem = sec.find("title")
            section_title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No Title"

            for p in sec.findall(".//p"):
                itertext = " ".join(p.itertext()).strip()
                para_text = etree.tostring(p, encoding="unicode", method="xml").strip()
                if len(para_text) >= 5:  # avoid tiny/junk paragraphs
                    paragraphs.append({
                        "paragraph": para_text,
                        "section_title": section_title,
                        "sec_type": sec_type,
                        "text": itertext
                    })
                    # print(f"Extracted paragraph: {paragraphs[-1]}")

        return paragraphs

    def extract_sections_from_xml(self, xml_root) -> list[dict]:
        """
        Extract sections from an XML document.

        Args:
            xml_root: lxml.etree.Element — parsed XML root.

        Returns:
            List of dicts with 'section_title' and 'sec_type'.
        """
        sections = []

        if not isinstance(xml_root, etree._Element):
            raise TypeError(f"Invalid XML root type: {type(xml_root)}. Expected lxml.etree.Element.")

        # Iterate over all section blocks
        for sec in xml_root.findall(".//sec"):
            sec_type = sec.get("sec-type", "unknown")
            title_elem = sec.find("title")
            section_title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No Title"

            section_text_from_paragraphs = f'{section_title}\n'
            section_rawtxt_from_paragraphs = ''

            for p in sec.findall(".//p"):

                itertext = " ".join(p.itertext()).strip()

                if len(itertext) >= 5:
                    section_text_from_paragraphs += "\n" + itertext + "\n"

                para_text = etree.tostring(p, encoding="unicode", method="xml").strip()

                if len(para_text) >= 5:  # avoid tiny/junk paragraphs
                    section_rawtxt_from_paragraphs += "\n" + para_text + "\n"

            sections.append({
                "sec_txt": section_rawtxt_from_paragraphs,
                "section_title": section_title,
                "sec_type": sec_type,
                "sec_txt_clean": section_text_from_paragraphs
            })
        return sections

    def extract_publication_title(self, api_data):
        """
        Extracts the article title and the surname of the first author from the XML content.

        :param xml_content: The XML content as a string.

        :return: A tuple containing the article title and the first author's surname.
        """
        try:
            # Extract the article title
            title = api_data.find(".//title-group/article-title")
            pub_title = title.text.strip() if title is not None else None
            return pub_title

        except etree.XMLSyntaxError as e:
            self.logger.error(f"Error parsing XML: {e}")
            return None

    def parse_data(self, api_data, publisher, current_url_address, additional_data=None, raw_data_format='XML',
                   save_xml_output=False, html_xml_dir='html_xml_samples/', process_DAS_links_separately=False,
                   prompt_name='retrieve_datasets_simple_JSON', use_portkey_for_gemini=True, semantic_retrieval=False):
        """
        Parse the API data and extract relevant links and metadata.

        :param api_data: The raw API data (XML or HTML) to be parsed.

        :param publisher: The publisher name or identifier.

        :param current_url_address: The current URL address being processed.

        :param additional_data: Additional data to be processed (optional).

        :param raw_data_format: The format of the raw data ('XML' or 'HTML').

        :return: A DataFrame containing the extracted links and links to metadata - if repo is supported. Add support for unsupported repos in the ontology.

        """
        out_df = None
        # Check if api_data is a string, and convert to XML if needed
        self.logger.info(f"Function call: parse_data(api_data({type(api_data)}), {publisher}, {current_url_address}, "
                         f"additional_data, {raw_data_format})")
        self.publisher = publisher

        if isinstance(api_data, str):
            try:
                api_data = etree.fromstring(api_data)  # Convert string to lxml Element
                self.logger.info(f"api_data converted to lxml element")
            except Exception as e:
                self.logger.error(f"Error parsing API data: {e}")
                return None

        if isinstance(api_data, etree._Element):
            self.title = self.extract_publication_title(api_data)

            if not self.full_document_read:
                # Save XML content for debugging purposes
                if save_xml_output:
                    dir = html_xml_dir + publisher + '/' + self.title + '.xml'
                    # if directory does not exist, create it
                    self.logger.info(f"Saving XML content to: {dir}")
                    if not os.path.exists(os.path.dirname(dir)):
                        os.makedirs(os.path.dirname(dir))
                    etree.ElementTree(api_data).write(dir, pretty_print=True, xml_declaration=True, encoding='UTF-8')

                # supplementary_material_links
                supplementary_material_links = self.extract_href_from_supplementary_material(api_data, current_url_address)
                self.logger.debug(f"supplementary_material_links: {supplementary_material_links}")

                if process_DAS_links_separately:
                    # Extract dataset links
                    dataset_links = self.extract_href_from_data_availability(api_data)
                    dataset_links.extend(self.extract_xrefs_from_data_availability(api_data, current_url_address))
                    self.logger.info(f"dataset_links: {dataset_links}")
                    if len(dataset_links) == 0:
                        self.logger.info(
                            f"No dataset links in data-availability section from XML. Scraping {current_url_address}.")
                        #dataset_links = self.get_data_availability_section_from_webpage(current_url_address)
                    # Process dataset links to get more context
                    augmented_dataset_links = self.process_data_availability_links(dataset_links)
                    self.logger.info(f"Len of augmented_dataset_links: {len(augmented_dataset_links)}")

                    self.logger.debug(f"Additional data: {(additional_data)}")
                    if additional_data is not None and len(additional_data) > 0:
                        self.logger.info(f"Additional data ({type(additional_data), len(additional_data)} items) "
                                         f"and Parsed data ({type(augmented_dataset_links), len(augmented_dataset_links)} items).")
                        # extend the dataset links with additional data
                        augmented_dataset_links = augmented_dataset_links + self.process_additional_data(additional_data)
                        self.logger.debug(f"Type: {type(augmented_dataset_links)}")
                        self.logger.debug(f"Len of augmented_dataset_links: {len(augmented_dataset_links)}")

                    self.logger.debug(f"Content of augmented_dataset_links: {augmented_dataset_links}")

                else:
                    data_availability_cont = self.get_data_availability_text(api_data)

                    if semantic_retrieval:
                        corpus = self.extract_sections_from_xml(api_data)
                        top_k_sections = self.semantic_retrieve_from_corpus(corpus, topk_docs_to_retrieve=2)
                        top_k_sections_text = [item['text'] for item in top_k_sections]
                        data_availability_cont.extend(top_k_sections_text)

                    augmented_dataset_links = self.process_data_availability_text(data_availability_cont,
                                                                                  prompt_name=prompt_name)

                    if additional_data is not None and len(additional_data) > 0:
                        self.logger.info(f"Additional data ({type(additional_data), len(additional_data)} items) "
                                         f"and Parsed data ({type(augmented_dataset_links), len(augmented_dataset_links)} items).")
                        # extend the dataset links with additional data
                        augmented_dataset_links = augmented_dataset_links + self.process_additional_data(additional_data)
                        self.logger.debug(f"Type: {type(augmented_dataset_links)}, Len: {len(augmented_dataset_links)}")
                        self.logger.debug(f"Augmented_dataset_links: {augmented_dataset_links}")

                    self.logger.debug(f"Content of augmented_dataset_links: {augmented_dataset_links}")

                dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

                # Create a DataFrame from the dataset links union supplementary material links
                out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages).rename(
                    columns={'dataset_id': 'dataset_identifier', 'repository_reference': 'data_repository'}),
                                    pd.DataFrame(supplementary_material_links)])  # check index error here
                self.logger.info(f"Dataset Links type: {type(out_df)} of len {len(out_df)}, with cols: {out_df.columns}")
                self.logger.debug(f"Datasets: {out_df}")

                # Extract file extensions from download links if possible, and add to the dataframe out_df as column
                if 'download_link' in out_df.columns:
                    out_df['file_extension'] = out_df['download_link'].apply(lambda x: self.extract_file_extension(x))
                elif 'link' in out_df.columns:
                    out_df['file_extension'] = out_df['link'].apply(lambda x: self.extract_file_extension(x))

                # drop duplicates but keep nulls
                if 'dataset_identifier' in out_df.columns and 'download_link' in out_df.columns:
                    out_df = out_df.drop_duplicates(subset=['download_link', 'dataset_identifier'], keep='first')

                out_df['pub_title'] = self.title

                return out_df

            else:
                # Extract links from entire webpage
                if self.full_document_read:
                    self.logger.info(f"Extracting links from full XML content.")
                    # preprocess the content to get only elements that do not change over different sessions
                    supplementary_material_links = self.extract_href_from_supplementary_material(api_data, current_url_address)

                    preprocessed_data = self.normalize_XML(api_data)

                    self.logger.debug(f"Preprocessed data: {preprocessed_data}")

                    # Extract dataset links from the entire text
                    augmented_dataset_links = self.extract_datasets_info_from_content(preprocessed_data,
                                                                                  self.open_data_repos_ontology['repos'],
                                                                                  model=self.llm_name,
                                                                                  temperature=0,
                                                                                  prompt_name=prompt_name)

                    self.logger.info(f"Augmented dataset links: {augmented_dataset_links}")

                    dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

                    # Create a DataFrame from the dataset links union supplementary material links
                    out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages),
                                        pd.DataFrame(supplementary_material_links)])

                self.logger.info(f"Dataset Links type: {type(out_df)} of len {len(out_df)}, with cols: {out_df.columns}")

                # Extract file extensions from download links if possible, and add to the dataframe out_df as column
                if 'download_link' in out_df.columns:
                    out_df['file_extension'] = out_df['download_link'].apply(lambda x: self.extract_file_extension(x))
                elif 'link' in out_df.columns:
                    out_df['file_extension'] = out_df['link'].apply(lambda x: self.extract_file_extension(x))

                # drop duplicates but keep nulls
                if 'download_link' in out_df.columns and 'dataset_identifier' in out_df.columns:
                    out_df = out_df.drop_duplicates(subset=['download_link', 'dataset_identifier'], keep='first')
                elif 'download_link' in out_df.columns:
                    out_df = out_df.drop_duplicates(subset=['download_link'], keep='first')

                out_df['source_url'] = current_url_address
                out_df['pub_title'] = self.title

                return out_df


    def normalize_XML(self, xml_data):
        """
        Normalize XML data by removing unnecessary whitespace and ensuring proper structure.

        :param xml_data: The raw XML data to be normalized.

        :return: Normalized XML data as a string.
        """
        if isinstance(xml_data, str):
            try:
                xml_root = etree.fromstring(xml_data)
                return self.normalize_XML(xml_root)

            except etree.XMLSyntaxError as e:
                self.logger.error(f"Error parsing XML data for normalization: {e}")
                return None

        elif isinstance(xml_data, etree._Element):
            xml_root = xml_data
            # Remove unnecessary whitespace and normalize text
            for elem in xml_root.iter():
                if elem.text:
                    elem.text = elem.text.strip()
                if elem.tail:
                    elem.tail = elem.tail.strip()

            # Convert back to string with pretty print
            normalized_xml = etree.tostring(xml_root, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode('utf-8')

            return normalized_xml

    def extract_href_from_data_availability(self, api_xml):
        """
        Extracts href links from data-availability sections of the XML.

        :param api_xml: lxml.etree.Element — parsed XML root.

        :return: List of dictionaries containing href links and their context.

        """
        # Namespace dictionary - adjust 'ns0' to match the XML if necessary
        self.logger.info(f"Function_call: extract_href_from_data_availability(api_xml)")
        namespaces = {'ns0': 'http://www.w3.org/1999/xlink'}

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

                self.logger.info(f"Retrieved {len(ext_links)} ext-links in data availability section pattern {ptr}.")

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
                            'title': self.title,
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

                    download_link = self.reconstruct_download_link(href, content_type, current_url_address)

                    media_id = supplementary_material_parent.get('id', 'No ID')

                    # Extract the <title> within <caption> for the supplementary material title
                    title_element = supplementary_material_parent.find(".//caption/title")
                    title = title_element.text if title_element is not None else "No Title"

                    # Extract the surrounding text (e.g., description within <p> tag)
                    parent_p = media.getparent()  # Assuming the media element is within a <p> tag
                    if parent_p is not None:
                        surrounding_text = re.sub("[\s\n]+", "  ", " ".join(parent_p.itertext()).strip())  # Gets all text within the <p> tag
                    else:
                        surrounding_text = "No surrounding text found"

                    # Extract the full description within the <p> tag if available
                    description_element = supplementary_material_parent.find(".//caption/p")
                    description = " ".join(
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
                self.logger.debug(f"Found {len(inline_supplementary_materials)} inline-supplementary-material elements.")

                for inline in inline_supplementary_materials:
                    # repeating steps like in media links above
                    hrefs.append({
                        "link": inline.get('{http://www.w3.org/1999/xlink}href'),
                        "content_type": inline.get('content-type', 'Unknown content type'),
                        "id": inline.get('id', 'No ID'),
                        "title": inline.get('title', 'No Title'),
                        "source_section": 'supplementary material inline',
                        "retrieval_pattern": ".//inline-supplementary-material",
                        "download_link": self.reconstruct_download_link(inline.get('{http://www.w3.org/1999/xlink}href'),
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

        return re.sub("[\s\n]+(\s+)]", "\1", surrounding_text)

    def get_sibling_text(self, media_element):
        """
        Extracts text surrounding the <media> element including the parent and its siblings.
        This includes inline text and any <p> tags that may provide context for the media element.

        """
        sibling_text = []

        # Get the parent element's text (if any)
        parent = media_element.getparent()
        if parent is not None and parent.text:
            sibling_text.append(parent.text.strip())

        # Traverse through the following siblings of the media element
        for sibling in media_element.itersiblings():
            if sibling.tail:
                sibling_text.append(sibling.tail.strip())

        # Join all the sibling texts into a single string
        return " ".join(sibling_text)

    def get_data_availability_text(self, api_xml):
        """
        This function calls the retrieval step. Then it normalizes the results.

        :param api_xml: lxml.etree.Element — parsed XML root.

        :return: List of strings from sections that match the data availability section patterns.

        """

        data_availability_sections = self.retriever.get_data_availability_sections(api_xml)

        if data_availability_sections is None:
            if self.retriever is None:
                self.logger.error("self.retriever is None. Please check the initialization of xmlRetriever.")
            raise ValueError("self.retriever.get_data_availability_sections(api_xml) returned None. ")

        data_availability_cont = []

        # extract the text from the data availability section
        for sect in data_availability_sections:
            cont = ""
            for elem in sect.iter():
                if elem.text:
                    cont += ' '
                    cont += elem.text
                    cont += ' '
                if elem.tail:
                    cont += ' '
                    cont += elem.tail
                    cont += ' '
                # also include the links in the data availability section
                if elem.tag == 'ext-link':
                    cont += ' '
                    cont += elem.get('{http://www.w3.org/1999/xlink}href')
                    cont += ' '
                if elem.tag == 'xref':
                    cont += ' '
                    cont += elem.text
                    cont += ' '
            data_availability_cont.append(cont) if cont not in data_availability_cont else None

        supplementary_data_sections = []

        # find the data availability statement in other sections
        for ptr in self.load_patterns_for_tgt_section('supplementary_data_sections'):
            if ptr.startswith('.//'):
                supplementary_data_sections.extend(api_xml.findall(ptr))

        self.logger.info(f"Found {len(supplementary_data_sections)} supplementary data sections")

        for sect in supplementary_data_sections:
            # check if section contains data availability statement
            if sect.text is None:  # key resources table
                self.logger.debug(f"Section with no text: {sect}")
            elif 'data availability' in sect.text:
                data_availability_cont.append(sect.text) if sect.text not in data_availability_cont else None
            elif 'Deposited data' in sect.text:
                data_availability_cont.append(sect.text) if sect.text not in data_availability_cont else None
            elif 'Accession number' in sect.text:
                data_availability_cont.append(sect.text) if sect.text not in data_availability_cont else None

        key_resources_table = []

        for ptr in self.load_patterns_for_tgt_section('key_resources_table'):
            key_resources_table.extend(api_xml.xpath(ptr))

        for sect in key_resources_table:
            self.logger.info(f"Found key resources table: {sect}.")
            table_text = self.table_to_text(sect)
            self.logger.debug(f"Table text: {table_text}")
            data_availability_cont.append(table_text)

        self.logger.debug(f"Found data availability content: {data_availability_cont}")

        return data_availability_cont

    def table_to_text(self, table_wrap):
        """
        Convert the <table> inside a <table-wrap> element to plain text.

        :param table_wrap: The <table-wrap> element containing the table.

        :return: String representing the table as plain text.
        """
        table = table_wrap.find(".//table")  # Find the <table> element
        if table is None:
            return "No table found in the provided <table-wrap> element."

        rows = []
        for row in table.findall(".//tr"):  # Iterate over each table row
            cells = []
            for cell in row.findall(".//td") + row.findall(".//th"):  # Get all <td> and <th> cells
                # Extract text content from the cell
                cell_text = " ".join(cell.itertext()).strip()  # Include all nested text
                cells.append(cell_text)
            rows.append("\t".join(cells))  # Join cells with a tab for plain text formatting

        # Join rows with a newline to create the final table text
        return "\n".join(rows)
