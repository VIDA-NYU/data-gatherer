from abc import ABC, abstractmethod
from bs4 import BeautifulSoup, NavigableString, CData, Comment
import re
import logging
import pandas as pd
from lxml import etree
from lxml import html
from ollama import Client
from openai import OpenAI
import google.generativeai as genai
import typing_extensions as typing
from pydantic import BaseModel
import os
import json
import torch
from data_gatherer.prompts.prompt_manager import PromptManager
import tiktoken
from data_gatherer.resources_loader import load_config

dataset_response_schema_gpt = {
    "type": "json_schema",
        "json_schema": {
        "name": "GPT_response_schema",
        "schema": {
            "type": "object",  # Root must be an object
            "properties": {
                "datasets": {  # Use a property to hold the array
                "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dataset_id": {
                                "type": "string",
                                "description": "A unique identifier for the dataset."
                            },
                            "repository_reference": {
                                "type": "string",
                                "description": "A valid URI or string referring to the repository."
                            },
                            "decision_rationale": {
                                "type": "string",
                                "description": "Why did we select this dataset?"
                            }
                        },
                        "required": ["dataset_id", "repository_reference"]
                    },
                    "minItems": 1,
                    "uniqueItems": True
                }
            },
            "required": ["datasets"]
        }
    }
}

dataset_metadata_response_schema_gpt = {
    "type": "json_schema",
    "json_schema": {
        "name": "Dataset_metadata_response",
        "schema": {
            "type": "object",
            "properties": {
                "number_of_files": {
                    "type": "string",
                    "description": "Total number of files."
                },
                "sample_size": {
                    "type": "string",
                    "description": "How many samples are recorded in the dataset."
                },
                "file_size": {
                    "type": "string",
                    "description": "Cumulative file size or range."
                },
                "file_format": {
                    "type": "string",
                    "description": "Format of the file (e.g., CSV, FASTQ)."
                },
                "file_type": {
                    "type": "string",
                    "description": "Type or category of the file."
                },
                "dataset_description": {
                    "type": "string",
                    "description": "Short summary of the dataset contents, plus - if mentioned - the use in the research publication of interes."
                },
                "file_url": {
                    "type": "string",
                    "description": "Direct link to the file."
                },
                "file_name": {
                    "type": "string",
                    "description": "Filename or archive name."
                },
                "file_license": {
                    "type": "string",
                    "description": "License under which the file is distributed."
                },
                "request_access_needed": {
                    "type": "string",
                    "description": "[Yes or No] Whether access to the file requires a request."
                },
                "request_access_form_links": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "uri",
                        "description": "Links to forms or pages where access requests can be made."
                    },
                    "description": "Links to forms or pages where access requests can be made."
                },
                "dataset_id": {
                    "type": "string",
                    "description": "A unique identifier for the dataset."
                },
                "download_type": {
                    "type": "string",
                    "description": "Type of download (e.g., HTTP, FTP, API, ...)."
                }
            },
            "required": [
                "dataset_description",
                "request_access_needed"
            ]
        }
    }
}

# Abstract base class for parsing data
class Parser(ABC):
    def __init__(self, config_path, logger=None, log_file_override=None, full_document_read = True):
        self.config = self.config = load_config(config_path)
        self.logger = logger
        self.logger.info("Parser initialized.")
        self.full_DOM = full_document_read and self.config['llm_model'] in self.config['entire_document_models']

    @abstractmethod
    def parse_data(self, raw_data, publisher, current_url_address):
        pass

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
                "raw_sec_txt": section_rawtxt_from_paragraphs,
                "section_title": section_title,
                "sec_type": sec_type,
                "sec_txt": section_text_from_paragraphs
            })
        return sections

class Dataset(BaseModel):
    dataset_id: str
    repository_reference: str

class Dataset_w_Description(typing.TypedDict):
    dataset_id: str
    repository_reference: str
    rationale: str

class Dataset_metadata(BaseModel):
    number_of_files: int
    file_size: str
    file_format: str
    file_type: str
    dataset_description: str
    file_url: str
    file_name: str
    file_license: str
    request_access_needed: str
    dataset_id: str
    download_type: str


# Implementation for Rule-Based parsing of HTMLs (from web scraping)
class RuleBasedParser(Parser):

    def __init__(self, config, logger, log_file_override=None):
        super().__init__(config, logger)
        self.logger.info("RuleBasedParser initialized.")

    def parse_data(self, source_html, publisher, current_url_address, raw_data_format='HTML'):
        # initialize output
        links_on_webpage = []
        self.logger.info("Function call: RuleBasedParser.parse_data(source_html)")
        soup = BeautifulSoup(source_html, "html.parser")
        compressed_HTML = self.compress_HTML(source_html)
        count = 0
        for anchor in soup.find_all('a'):
            link = anchor.get('href')

            if link is not None and "/" in link and len(link) > 1:

                reconstructed_link = self.reconstruct_link(link, publisher)

                # match link and extract text around the link in the displayed page as description for future processing
                if link in compressed_HTML:
                    # extract raw link description
                    raw_link_description = compressed_HTML[
                                           compressed_HTML.index(link) - 200:compressed_HTML.index(link) + 200]
                    self.logger.debug(f"raw description: {raw_link_description}")
                else:
                    raw_link_description = 'raw description not available'
                    self.logger.debug(f"link {link} not found in compressed HTML")

                links_on_webpage.append(
                    {'source_url': current_url_address,
                     'link': link,
                     'reconstructed_link': reconstructed_link,
                     'element': str(anchor),  # Full HTML of the anchor element
                     'text': re.sub("\s+", " ", anchor.get_text(strip=True)),
                     'class': anchor.get('class'),
                     'id': anchor.get('id'),
                     'parent': str(anchor.parent),  # Full HTML of the parent element
                     'siblings': [str(sibling) for sibling in anchor.next_siblings if sibling.name is not None],
                     'raw_description': raw_link_description,
                     }
                )

                #print(f"found link: {link, anchor}")

                self.logger.debug(f"extracted element as: {links_on_webpage[-1]}")
                count += 1

        df_output = pd.DataFrame.from_dict(links_on_webpage)
        self.logger.info(f"Found {count} links on the webpage")
        return df_output

    def reconstruct_link(self, link, publisher):
        """
        Given a publisher name, return the root domain.
        """
        if not (link.startswith("http") or link.startswith("//") or link.startswith("ftp")):
            if (publisher.startswith("http") or publisher.startswith("//")):
                publisher_root = publisher
            else:
                publisher_root = "https://www." + publisher
            return publisher_root + link
        else:
            return link

    def compress_HTML(self, source_html):
        """
        This function should convert html to markdown and keep the links close to the text near them in the webpage GUI.
        """
        self.logger.debug(f"compress HTML. Original len: {len(source_html)}")
        # Parse the HTML content with BeautifulSoup
        soup = MyBeautifulSoup(source_html, "html.parser")
        text = re.sub("\s+", " ", soup.getText())
        self.logger.debug(f"compress HTML. Final len: {len(text)}")
        return text


class LLMParser(Parser):
    """
    This class is responsible for parsing data using LLMs. This will be done either:

    - Full Document Read (LLMs that can read the entire document)

    - Retrieve Then Read (LLMs will only read a target section retrieved from the document)
    """
    def __init__(self, config, logger, log_file_override=None, full_document_read=True):
        """
        Initialize the LLMParser with configuration, logger, and optional log file override.

        :param config: Configuration dictionary containing settings for the parser (llm_model, prompt_dir, etc.).

        :param logger: Logger instance for logging messages.

        :param log_file_override: Optional log file override.
        """
        super().__init__(config, logger, log_file_override, full_document_read)
        self.title = None
        self.prompt_manager = PromptManager(self.config['prompt_dir'], self.logger, self.config['response_file'])
        self.repo_names = self.get_repo_names()
        self.repo_domain_to_name_mapping = self.get_repo_domain_to_name_mapping()

        if self.config['llm_model'] == 'gemma2:9b':
            self.client = Client(host=os.environ['NYU_LLM_API'])  # env variable

        elif self.config['llm_model'] == 'gpt-4o-mini':
            self.client = OpenAI(api_key=os.environ['GPT_API_KEY'])

        elif self.config['llm_model'] == 'gpt-4o':
            self.client = OpenAI(api_key=os.environ['GPT_API_KEY'])

        elif self.config['llm_model'] == 'gemini-1.5-flash':
            genai.configure(api_key=os.environ['GEMINI_KEY'])
            self.client = genai.GenerativeModel('gemini-1.5-flash')

        elif self.config['llm_model'] == 'gemini-2.0-flash-exp':
            genai.configure(api_key=os.environ['GEMINI_KEY'])
            self.client = genai.GenerativeModel('gemini-2.0-flash-exp')

        elif self.config['llm_model'] == 'gemini-2.0-flash':
            genai.configure(api_key=os.environ['GEMINI_KEY'])
            self.client = genai.GenerativeModel('gemini-2.0-flash')

        elif self.config['llm_model'] == 'gemini-1.5-pro':
            genai.configure(api_key=os.environ['GEMINI_KEY'])
            self.client = genai.GenerativeModel('gemini-1.5-pro')

    def parse_data(self, api_data, publisher, current_url_address, additional_data=None, raw_data_format='XML'):
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
        if isinstance(api_data, str) and raw_data_format != 'full_HTML':
            try:
                api_data = etree.fromstring(api_data)  # Convert string to lxml Element
                self.logger.info(f"api_data converted to lxml element")
            except Exception as e:
                self.logger.error(f"Error parsing API data: {e}")
                return None

        if raw_data_format != 'full_HTML':
            # Extract title (adjust XPath to match the structure)
            title_element = api_data.find('.//title-group/article-title')  # XPath for article title
            title = title_element.text if title_element is not None else "No Title Found"
            self.logger.info(f"Extracted title:'{title}'")
            self.title = title

            # Save XML content for debugging purposes
            if self.config['save_xml_output']:
                dir = self.config['html_xml_dir'] + publisher + '/' + title + '.xml'
                # if directory does not exist, create it
                self.logger.info(f"Saving XML content to: {dir}")
                if not os.path.exists(os.path.dirname(dir)):
                    os.makedirs(os.path.dirname(dir))
                etree.ElementTree(api_data).write(dir, pretty_print=True, xml_declaration=True, encoding='UTF-8')

            # supplementary_material_links
            supplementary_material_links = self.extract_href_from_supplementary_material(api_data, current_url_address)
            self.logger.debug(f"supplementary_material_links: {supplementary_material_links}")

            if self.config['process_DAS_links_separately']:
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

                augmented_dataset_links = self.process_data_availability_text(data_availability_cont)

                if additional_data is not None and len(additional_data) > 0:
                    self.logger.info(f"Additional data ({type(additional_data), len(additional_data)} items) "
                                     f"and Parsed data ({type(augmented_dataset_links), len(augmented_dataset_links)} items).")
                    # extend the dataset links with additional data
                    augmented_dataset_links = augmented_dataset_links + self.process_additional_data(additional_data)
                    self.logger.debug(f"Type: {type(augmented_dataset_links)}, Len: {len(augmented_dataset_links)}")
                    self.logger.debug(f"Augmented_dataset_links: {augmented_dataset_links}")

                self.logger.debug(f"Content of augmented_dataset_links: {augmented_dataset_links}")

            dataset_links_w_target_pages = self.get_dataset_webpage(augmented_dataset_links)

            # Create a DataFrame from the dataset links union supplementary material links
            out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages),
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

            return out_df

        else:
            # Extract links from entire webpage
            if self.full_DOM:
                self.logger.info(f"Extracting links from full HTML content.")
                # preprocess the content to get only elements that do not change over different sessions
                supplementary_material_links = self.extract_href_from_html_supplementary_material(api_data, current_url_address)

                preprocessed_data = self.normalize_full_DOM(api_data)

                #self.logger.info(f"Preprocessed data: {preprocessed_data}")

                # Extract dataset links from the entire text
                augmented_dataset_links = self.retrieve_datasets_from_content(preprocessed_data, self.config['repos'],
                                                                              self.config['llm_model'],
                                                                              temperature=0)
                self.logger.info(f"Augmented dataset links: {augmented_dataset_links}")

                dataset_links_w_target_pages = self.get_dataset_webpage(augmented_dataset_links)

                # Create a DataFrame from the dataset links union supplementary material links
                out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages), supplementary_material_links])

            else:
                self.logger.info(f"Chunking the HTML content for the parsing step.")
                supplementary_material_links = self.extract_href_from_html_supplementary_material(api_data,
                                                                                                  current_url_address)
                preprocessed_data = self.normalize_full_DOM(api_data)

                # Extract dataset links from the entire text
                data_availability_elements = self.get_data_availability_elements_from_webpage(preprocessed_data)

                data_availability_str = "\n".join([item['html'] + "\n" for item in data_availability_elements])

                augmented_dataset_links = self.retrieve_datasets_from_content(data_availability_str, self.config['repos'],
                                                                              self.config['llm_model'])

                dataset_links_w_target_pages = self.get_dataset_webpage(augmented_dataset_links)

                out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages), supplementary_material_links])

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

            return out_df

    def normalize_full_DOM(self, api_data: str) -> str:
        """
        Normalize the full HTML DOM by removing dynamic elements and attributes
        that frequently change, such as random IDs, inline styles, analytics tags,
        and CSRF tokens.

        :param api_data: The raw HTML data to be normalized.

        :return: Normalized HTML string.

        """

        self.logger.info(f"Function_call: normalize_full_DOM(api_data). Length of raw api data: {self.count_tokens(api_data,self.config['llm_model'])} tokens")

        try:
            # Parse the HTML content
            soup = BeautifulSoup(api_data, "html.parser")

            # 1. Remove script, style, and meta tags
            for tag in ["script", "style", 'img', 'iframe', 'noscript', 'svg', 'button', 'form', 'input']:
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

    def extract_file_extension(self, download_link):
        self.logger.info(f"Function_call: extract_file_extension({download_link})")
        # Extract the file extension from the download link
        extension = None
        if type(download_link) == str:
            extension = download_link.split('.')[-1]
        if type(extension) == str and ("/" in extension):  # or "?" in extension
            return ""
        return extension

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
        for ptr in self.config['data_availability_sections']:
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
        for ptr in self.config['data_availability_sections']:
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

    def generate_dataset_description(self, data_file):
        # from data file
        # excel, csv, json, xml, etc.
        # autoDDG
        raise NotImplementedError("DDG not implemented yet")

    def extract_href_from_html_supplementary_material(self, raw_html, current_url_address):
        """
        Extracts href links from supplementary material sections of the HTML.

        :param raw_html: str — raw HTML content.

        :param current_url_address: str — the current URL address being processed.

        :return: DataFrame containing extracted links and their context.

        """
        self.logger.info(f"Function_call: extract_href_from_html_supplementary_material(tree, {current_url_address})")

        tree = html.fromstring(raw_html)

        supplementary_links = []

        anchors = tree.xpath("//a[@data-ga-action='click_feat_suppl']")
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
                self.logger.debug(f"Skipping link with section_class 'ref-list font-sm', likely to be a reference list.")
                continue

            #if 'doi.org' in link_data['link'] or 'scholar.google.com' in link_data['link']: ############ Same as above
            #    continue

            link_data['download_link'] = self.reconstruct_download_link(href, link_data['section_class'], current_url_address)
            link_data['file_extension'] = self.extract_file_extension(link_data['download_link']) if link_data['download_link'] is not None else None

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
        for ptr in self.config['supplementary_material_sections']:
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

                    if href:
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

    def reconstruct_download_link(self, href, content_type, current_url_address):
        # https: // pmc.ncbi.nlm.nih.gov / articles / instance / 11252349 / bin / 41598_2024_67079_MOESM1_ESM.zip
        # https://pmc.ncbi.nlm.nih.gov/articles/instance/PMC11252349/bin/41598_2024_67079_MOESM1_ESM.zip
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11252349/bin/41598_2024_67079_MOESM1_ESM.zip
        download_link = None
        #repo = self.url_to_repo_domain(current_url_address)
        # match the digits of the PMC ID (after PMC) in the URL
        self.logger.info(f"Function_call: reconstruct_download_link({href}, {content_type}, {current_url_address})")
        PMCID = re.search(r'PMC(\d+)', current_url_address, re.IGNORECASE).group(1)
        self.logger.debug(
            f"Inputs to reconstruct_download_link: {href}, {content_type}, {current_url_address}, {PMCID}")
        if content_type == 'local-data':
            download_link = "https://pmc.ncbi.nlm.nih.gov/articles/instance/" + PMCID + '/bin/' + href
        elif content_type == 'media p':
            file_name = os.path.basename(href)
            self.logger.debug(f"Extracted file name: {file_name} from href: {href}")
            download_link = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC" + PMCID + '/bin/' + file_name
        return download_link

    def get_sibling_text(self, media_element):
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

    def union_additional_data(self, parsed_data, additional_data):
        self.logger.info(f"Merging additional data ({type(additional_data)}) with parsed data({type(parsed_data)}).")
        self.logger.info(f"Additional data\n{additional_data}")
        return pd.concat([parsed_data, additional_data], ignore_index=True)

    def process_additional_data(self, additional_data):
        """
        Process the additional data from the webpage. This is the data matched from the HTML with the patterns in
        retrieval_patterns xpaths.

        :param additional_data: List of dictionaries containing additional data to be processed.

        :return: List of dictionaries containing processed data.

        """
        self.logger.info(f"Processing additional data: {len(additional_data)} items")
        repos_elements = []
        for repo, details in self.config['repos'].items():
            entry = repo
            if 'repo_name' in details:
                entry += f" ({details['repo_name']})"
            repos_elements.append(entry)

        # Join the elements into a properly formatted string
        repos = ', '.join(repos_elements)

        # Log for debugging
        self.logger.info(f"Repos elements: {repos_elements}")

        ret = []
        for element in additional_data:
            self.logger.info(f"Processing additional data element ({type(element)}): {element}")
            cont = element['surrounding_text']

            if 'Supplementary Material' in cont or 'supplementary material' in cont:
                continue

            if (element['source_section'] in ['data availability', 'data_availability', 'data_availability_elements']
                    or 'data availability' in cont) and len(cont) > 1:
                self.logger.info(f"Processing data availability text")
                # Call the generalized function
                datasets = self.retrieve_datasets_from_content(cont, repos_elements, model=self.config['llm_model'], temperature=0)

                for dt in datasets:
                    dt['source_section'] = element['source_section']
                    dt['retrieval_pattern'] = element['retrieval_pattern']

                ret.extend(datasets)
            else:
                self.logger.debug(f"Processing supplementary material element")
                ret.append(element)

        self.logger.info(f"Final ret additional data: {len(ret)} items")
        self.logger.debug(f"Final ret additional data: {ret}")
        return ret

    def process_data_availability_text(self, DAS_content):
        """
        Process the data availability section from the webpage.

        :param DAS_content: list of all text content matching the data availability section patterns.

        :return: List of dictionaries containing processed data.
        """
        self.logger.info(f"Processing DAS_content: {DAS_content}")
        repos_elements = []
        for repo, details in self.config['repos'].items():
            entry = repo
            if 'repo_name' in details:
                entry += f" ({details['repo_name']})"
            repos_elements.append(entry)

        # Join the elements into a properly formatted string
        repos = ', '.join(repos_elements)

        # Log for debugging
        self.logger.info(f"Repos elements: {repos_elements}")

        # Call the generalized function
        datasets = []
        for element in DAS_content:
            datasets.extend(self.retrieve_datasets_from_content(element, repos_elements,
                                                                model=self.config['llm_model'],
                                                                temperature=0))

        # Add source_section information and return
        ret = []
        self.logger.info(f"datasets ({type(datasets)}): {datasets}")
        for dataset in datasets:
            self.logger.info(f"iter dataset ({type(dataset)}): {dataset}")
            dataset['source_section'] = 'data_availability'
            dataset['retrieval_pattern'] = 'data availability'
            ret.append(dataset)

        self.logger.info(f"Final ret additional data: {len(ret)} items")
        self.logger.debug(f"Final ret additional data: {ret}")
        return ret

    def retrieve_datasets_from_content(self, content: str, repos: list, model: str, temperature: float = 0.0) -> list:
        """
        Retrieve datasets from the given content using a specified LLM model.
        Uses a static prompt template and dynamically injects the required content.

        :param content: The content to be processed.

        :param repos: List of repositories to be included in the prompt.

        :param model: The LLM model to be used for processing.

        :param temperature: The temperature setting for the model.

        :return: List of datasets retrieved from the content.
        """
        # Load static prompt template
        prompt_name = self.config['prompt_name']
        self.logger.info(f"Loading prompt: {prompt_name}")
        static_prompt = self.prompt_manager.load_prompt(prompt_name)
        n_tokens_static_prompt = self.count_tokens(static_prompt, model)

        if 'gpt-4o' in model:
            while self.tokens_over_limit(content, model,allowance_static_prompt=n_tokens_static_prompt):
                content = content[:-2000]
            self.logger.info(f"Content length: {len(content)}")

        self.logger.info(f"static_prompt: {static_prompt}")

        # Render the prompt with dynamic content
        messages = self.prompt_manager.render_prompt(
            static_prompt,
            entire_doc=self.full_DOM,
            content=content,
            repos=', '.join(repos)
        )
        self.logger.info(f"Prompt messages total length: {self.count_tokens(messages,model)} tokens")
        self.logger.debug(f"Prompt messages: {messages}")

        # Generate the checksum for the prompt content
        # Save the prompt and calculate checksum
        prompt_id = f"{model}-{temperature}-{self.prompt_manager._calculate_checksum(str(messages))}"
        self.logger.info(f"Prompt ID: {prompt_id}")
        # Save the prompt using the PromptManager
        if self.config['save_dynamic_prompts']:
            self.prompt_manager.save_prompt(prompt_id=prompt_id, prompt_content=messages)

        if self.config['use_cached_responses']:
            # Check if the response exists
            cached_response = self.prompt_manager.retrieve_response(prompt_id)

        if self.config['use_cached_responses'] and cached_response:
            self.logger.info(f"Using cached response {type(cached_response)} from model: {model}")
            if type(cached_response) == str and 'gpt-4o' in model:
                resps = [json.loads(cached_response)]
            if type(cached_response) == str:
                resps = cached_response.split("\n")
            elif type(cached_response) == list:
                resps = cached_response
        else:
            # Make the request to the model
            self.logger.info(
                f"Requesting datasets from content using model: {model}, temperature: {temperature}, messages: "
                f"{self.count_tokens(messages, model)} tokens")
            resps = []

            if self.config['llm_model'] == 'gemma2:9b':
                response = self.client.chat(model=model, options={"temperature": temperature}, messages=messages)
                self.logger.info(
                    f"Response received from model: {response.get('message', {}).get('content', 'No content')}")
                resps = response['message']['content'].split("\n")
                # Save the response
                self.prompt_manager.save_response(prompt_id, response['message']['content']) if self.config[
                    'save_responses_to_cache'] else None
                self.logger.info(f"Response saved to cache")

            elif self.config['llm_model'] == 'gpt-4o-mini' or self.config['llm_model'] == 'gpt-4o':
                response = None
                if self.config['process_entire_document']:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        response_format=dataset_response_schema_gpt
                    )
                else:
                    response = self.client.chat.completions.create(model=model, messages=messages,
                                                                   temperature=temperature)

                self.logger.info(f"GPT response: {response.choices[0].message.content}")

                if self.config['process_entire_document']:
                    resps = self.safe_parse_json(response.choices[0].message.content)  # 'datasets' keyError?
                    self.logger.info(f"Response is {type(resps)}: {resps}")
                    resps = resps.get("datasets", []) if resps is not None else []
                    self.logger.info(f"Response is {type(resps)}: {resps}")
                    self.prompt_manager.save_response(prompt_id, resps) if self.config['save_responses_to_cache'] else None
                else:
                    try:
                        resps = self.safe_parse_json(response.choices[0].message.content)  # Ensure it's properly parsed
                        self.logger.info(f"Response is {type(resps)}: {resps}")
                        if not isinstance(resps, list):  # Ensure it's a list
                            raise ValueError("Expected a list of datasets, but got something else.")

                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON decoding error: {e}")
                        resps = []

                    self.prompt_manager.save_response(prompt_id, resps) if self.config['save_responses_to_cache'] else None

                # Save the response
                self.logger.info(f"Response {type(resps)} saved to cache") if self.config['save_responses_to_cache'] else None

            elif 'gemini' in self.config['llm_model']:
                if self.config['llm_model'] == 'gemini-1.5-flash' or self.config['llm_model'] == 'gemini-2.0-flash-exp' or self.config[
                    'llm_model'] == 'gemini-2.0-flash':
                    response = self.client.generate_content(
                        messages,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=list[Dataset]
                        )
                    )
                    self.logger.debug(f"Gemini response: {response}")

                elif self.config['llm_model'] == 'gemini-1.5-pro':
                    response = self.client.generate_content(
                        messages,
                        request_options={"timeout": 1200},
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=list[Dataset]
                        )
                    )
                    self.logger.debug(f"Gemini Pro response: {response}")

                try:
                    candidates = response.candidates  # Get the list of candidates
                    if candidates:
                        self.logger.info(f"Found {len(candidates)} candidates in the response.")
                        response_text = candidates[0].content.parts[0].text  # Access the first part's text
                        self.logger.info(f"Gemini response text: {response_text}")
                        parsed_response = json.loads(response_text)  # Parse the JSON response
                        if self.config['save_responses_to_cache']:
                            self.prompt_manager.save_response(prompt_id, parsed_response)
                            self.logger.info(f"Response saved to cache")
                        parsed_response_dedup = self.deduplicate_response(parsed_response)
                        resps = parsed_response_dedup
                    else:
                        self.logger.error("No candidates found in the response.")
                except Exception as e:
                    self.logger.error(f"Error processing Gemini response: {e}")
                    return None

        if not self.full_DOM:
            return resps

        # Process the response content
        result = []
        for dataset in resps:
            self.logger.info(f"Processing dataset: {dataset}")
            if type(dataset) == str:
                self.logger.info(f"Dataset is a string")
                # Skip short or invalid responses
                if len(dataset) < 3 or dataset.split(",")[0].strip() == 'n/a' and dataset.split(",")[
                    1].strip() == 'n/a':
                    continue
                if len(dataset.split(",")) < 2:
                    continue
                if re.match(r'\*\s+\*\*[\s\w]+:\*\*', dataset):
                    dataset = re.sub(r'\*\s+\*\*[\s\w]+:\*\*', '', dataset)

                dataset_id, data_repository = [x.strip() for x in dataset.split(",")[:2]]

            elif type(dataset) == dict:
                self.logger.info(f"Dataset is a dictionary")
                dataset_id = 'n/a'
                if 'dataset_id' in dataset:
                    dataset_id = dataset['dataset_id']
                elif 'dataset_identifier' in dataset:
                    dataset_id = dataset['dataset_identifier']
                if 'data_repository' in dataset:
                    data_repository = dataset['data_repository']
                elif 'repository_reference' in dataset:
                    data_repository = dataset['repository_reference']

                if dataset_id == 'n/a' and data_repository in self.config['repos']:
                    self.logger.info(f"Dataset ID is 'n/a' and repository name from prompt")
                    continue

            result.append({
                "dataset_identifier": dataset_id,
                "data_repository": data_repository
            })

            if 'decision_rationale' in dataset:
                result[-1]['decision_rationale'] = dataset['decision_rationale']

            if 'dataset-publication_relationship' in dataset:
                result[-1]['dataset-publication_relationship'] = dataset['dataset-publication_relationship']

            self.logger.info(f"Extracted dataset: {result[-1]}")

        return result

    def deduplicate_response(self, response):
        """
        This function handles basic **postprocessing** of the LLM output.
        Normalize and deduplicate dataset responses by stripping DOI-style prefixes
        like '10.x/' from dataset IDs and keeping only one entry per PXD.

        :param response: List of dataset responses to be deduplicated (LLM Output).

        :return: List of deduplicated dataset responses.

        """
        seen = set()
        deduped = []

        for item in response:
            dataset_id = item.get("dataset_id", item.get("dataset_identifier", ""))
            if not dataset_id:
                continue

            # Normalize: remove DOI prefix if it matches '10.x/PXD123456'
            clean_id = re.sub(r'10\.\d+/(\bPXD\d+\b)', r'\1', dataset_id)

            if clean_id not in seen:
                # Update the dataset_id to the normalized version
                item["dataset_id"] = clean_id
                deduped.append(item)
                seen.add(clean_id)

        return deduped


    def safe_parse_json(self, response_text):
        """
        Cleans and safely parses JSON from an LLM response, fixing common issues.

        :param response_text: str — the JSON string to be parsed.

        :return: dict or None — parsed JSON object or None if parsing fails.
        """
        try:
            response_text = response_text.strip()  # Remove extra spaces/newlines

            # Fix common truncation issues by ensuring it ends properly
            # if not response_text.endswith("\"]}"):
            #     response_text += "\"]}"
            #
            # # Fix invalid JSON artifacts
            # response_text = response_text.replace("=>", ":")  # Convert invalid separators
            #response_text = re.sub(r',\s*}', '}', response_text)  # Remove trailing commas before closing braces
            #response_text = re.sub(r',\s*\]', ']', response_text)  # Remove trailing commas before closing brackets

            # process dict-like list
            if "{" not in response_text[1:-1] and "{" not in response_text[1:-1] and "[" in response_text[1:-1]:
                response_text = (response_text[:1] +
                                 response_text[1:-1].replace("[", "{").replace("]","}") + response_text[-1:])
            # Attempt JSON parsing
            return json.loads(response_text)

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print(f"Malformed JSON: {response_text[:500]}")  # Print first 500 chars for debugging
            return None  # Return None to indicate failure

    def get_data_availability_text(self, api_xml):
        """
        This function handles the retrieval step. Given the data availability statement, extract the dataset
        information from the text.

        :param api_xml: lxml.etree.Element — parsed XML root.

        :return: List of strings from sections that match the data availability section patterns.

        """
        # find the data availability section
        data_availability_sections = []
        for ptr in self.config['data_availability_sections']:
            data_availability_sections.extend(api_xml.findall(ptr))

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
        for ptr in self.config['supplementary_data_sections']:
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

        for ptr in self.config['key_resources_table']:
            key_resources_table.extend(api_xml.xpath(ptr))

        for sect in key_resources_table:
            self.logger.info(f"Found key resources table: {sect}")
            table_text = self.table_to_text(sect)
            self.logger.info(f"Table text: {table_text}")
            data_availability_cont.append(table_text)

        self.logger.info(f"Found data availability content: {data_availability_cont}")

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

        for selector in self.css_selectors.get('data_availability', []):
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
                    text_val += child.get_text(strip=True) + " \n"
                    html_val += str(child) + " \n"

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

                self.logger.info(f"Extracted data availability element: {element_info}")

        self.logger.info(f"Found {len(data_availability_elements)} data availability elements from HTML.")
        return data_availability_elements


    def process_data_availability_links(self, dataset_links):
        """
        Given the link, the article title, and the text around the link, create a column (identifier),
        and a column for the dataset.

        :param dataset_links: List of dictionaries containing href links and their context.

        :return: List of dictionaries containing processed dataset information.

        """
        self.logger.info(f"Analyzing data availability statement with {len(dataset_links)} links")
        self.logger.debug(f"Text from data-availability: {dataset_links}")

        model = self.config['llm_model']
        temperature = 0.3

        ret = []
        progress = 0

        for link in dataset_links:
            self.logger.info(f"Processing link: {link['href']}")

            if progress > len(dataset_links):
                break

            ret_element = {}

            # Detect if the link is already a dataset webpage
            detected = self.dataset_webpage_url_check(link['href'])
            if detected is not None:
                self.logger.info(f"Link {link['href']} points to dataset webpage")
                ret.append(detected)
                progress += 1
                continue

            # Prepare the dynamic prompt
            dynamic_content = {
                "href": link['href'],
                "surrounding_text": link['surrounding_text']
            }
            static_prompt = self.prompt_manager.load_prompt("retrieve_datasets_fromDAS")
            messages = self.prompt_manager.render_prompt(static_prompt, self.full_DOM, **dynamic_content)

            # Generate a unique checksum for the prompt
            prompt_id = f"{model}-{temperature}-{self.prompt_manager._calculate_checksum(str(messages))}"
            #prompt_id = self.prompt_manager._calculate_checksum(str(messages))
            self.logger.info(f"Prompt ID: {prompt_id}")

            # Check if the response exists
            cached_response = self.prompt_manager.retrieve_response(prompt_id)
            if cached_response:
                self.logger.info("Using cached response.")
                resp = cached_response.split("\n")
            else:
                # Make the request to the LLM model
                self.logger.info(f"Requesting datasets using model: {model}, messages: {messages}")
                resp = []

                if model == 'gemma2:9b':
                    response = self.client.chat(model='gemma2:9b', options={"temperature": 0.5}, messages=messages)
                    self.logger.info(
                        f"Response received from gemma2:9b: {response.get('message', {}).get('content', 'No content')}")
                    resp = response['message']['content'].split("\n")
                    self.prompt_manager.save_response(prompt_id, response['message']['content'])
                    self.logger.info(f"Response saved to cache")

                elif model == 'gpt-4o-mini':
                    response = self.client.chat.completions.create(model='gpt-4o-mini', messages=messages,
                                                                   temperature=0.5)
                    self.logger.info(f"GPT response: {response.choices[0].message.content}")
                    resp = response.choices[0].message.content.split("\n")
                    self.prompt_manager.save_response(prompt_id, response.choices[0].message.content)
                    self.logger.info(f"Response saved to cache")

            # Process the response
            ret_element['link'] = link['href']
            ret_element['content_type'] = 'data_link'

            if type(resp) == list:
                for r in resp:
                    append_item = ret_element.copy()
                    self.logger.info(f"Response: '{r}', len: {len(r)}")
                    # string that is less than 1 char + 1 comma + 1 char
                    if len(r) < 3:
                        continue
                    # skip strings that do not conform to expected output
                    if r.count(',') != 1:
                        continue
                    append_item['dataset_identifier'], append_item['data_repository'] = r.split(
                        ',')
                    append_item['dataset_identifier'] = append_item['dataset_identifier'].strip()
                    append_item['data_repository'] = append_item['data_repository'].strip()
                    append_item['source_section'] = link['source_section']
                    append_item['retrieval_pattern'] = link['retrieval_pattern'] if 'retrieval_pattern' in link.keys() else None
                    ret.append(append_item)
                    self.logger.info(f"Response appended to df {append_item}")
                    progress += 1
                self.logger.info(f"Updated ret: {ret}")
            else:
                self.logger.info(f"Response: '{resp}', len: {len(resp)}. Response appended to df")
                ret_element['dataset_identifier'], ret_element['data_repository'] = (
                    response['message']['content'].split(','))
                # trim leading and trailing whitespaces
                ret_element['dataset_identifier'] = ret_element['dataset_identifier'].strip()
                ret_element['data_repository'] = ret_element['data_repository'].strip()
                ret_element['source_section'] = link['source_section']
                ret_element['retrieval_pattern'] = link['retrieval_pattern'] if 'retrieval_pattern' in link.keys() else None
                ret.append(ret_element)
                progress += 1

        return ret

    def dataset_webpage_url_check(self, url):
        """
        Check if the URL directly points to a dataset webpage.

        :param url: str — the URL to be checked.

        :return: dict or None — dictionary with data repository information if one pattern from ontology matches that
        """
        ret = {}
        self.logger.info(f"Checking if link points to dataset webpage: {url}")
        domain = self.url_to_repo_domain(url)
        if domain in self.config['repos'].keys() and 'dataset_webpage_url_ptr' in self.config['repos'][domain].keys():
            self.logger.info(f"Link {url} could point to dataset webpage")
            pattern = self.config['repos'][domain]['dataset_webpage_url_ptr']
            self.logger.debug(f"Pattern: {pattern}")
            match = re.match(pattern, url)
            # if the link matches the pattern, extract the dataset identifier and the data repository
            if match:
                ret['data_repository'] = domain
                ret['dataset_identifier'] = match.group(1)
                ret['dataset_webpage'] = url
                ret['link'] = url
                return ret
            else:
                self.logger.info(f"Link does not match the pattern")
                return None

        return None

    def normalize_LLM_output(self, response):
        cont = response['message']['content']
        self.logger.info(f"Normalizing {type(cont)} LLM output: {cont}")
        output = cont.split(",")
        repo = re.sub("[\n\s]*", "", output.pop())
        self.logger.info(f"Repo: {repo}")
        ret = []
        for i in range(len(output)):
            ret.append(re.sub("\s*and\s+", " ", output[i]) + "," + repo)
        return ret

    def url_to_repo_domain(self, url):
        # Extract the domain name from the URL
        if url in self.config['repos'].keys():
            return url

        self.logger.info(f"Extracting repo domain from URL: {url}")
        match = re.match(r'^https?://([\.\w\-]+)\/', url)
        if match:
            domain = match.group(1)
            self.logger.debug(f"Repo Domain: {domain}")
            if domain in self.config['repos'].keys() and 'repo_mapping' in self.config['repos'][domain].keys():
                return self.config['repos'][domain]['repo_mapping']
            return domain
        elif '.' not in url:
            return url
        else:
            self.logger.error(f"Error extracting domain from URL: {url}")
            return 'Unknown_Publisher'

    def get_repo_names(self):
        # Get the all the repository names from the config file. (all the repos in ontology)
        repo_names = []
        for k, v in self.config['repos'].items():
            if 'repo_name' in v.keys():
                repo_names.append(v['repo_name'])
            else:
                repo_names.append(k)
        return repo_names

    def get_repo_domain_to_name_mapping(self):
        # Get the mapping of repository domains to names from ontology
        repo_mapping = {}
        for k, v in self.config['repos'].items():
            if 'repo_name' in v.keys():
                repo_mapping[k] = v['repo_name'].lower()
            else:
                repo_mapping[k] = k

        ret = {v:k for k,v in repo_mapping.items()}
        self.logger.debug(f"Repo mapping: {ret}")
        return ret

    def resolve_accession_id(self, dataset_identifier, data_repository):
        """
        This function resolves the accession ID for a given dataset identifier and data repository.
        It checks if the dataset identifier matches the expected pattern for the given repository (from ontology)
        """
        self.logger.info(f"Resolving accession ID for {dataset_identifier} in {data_repository}")
        if data_repository in self.config['repos']:
            repo_config = self.config['repos'][data_repository]
            pattern = repo_config.get('id_pattern')
            if pattern and not re.match(pattern, dataset_identifier):
                self.logger.warning(f"Identifier {dataset_identifier} does not match pattern for {data_repository}")
            if 'default_id_suffix' in repo_config:
                return dataset_identifier.lower() + repo_config['default_id_suffix']
        return dataset_identifier

    def resolve_data_repository(self, repo: str) -> str:
        """
        Normalize the repository domain from a URL or text reference using config mappings in ontology.

        :param repo: str — the repository name or URL to be normalized.

        :return: str — the normalized repository name.
        """
        self.logger.info(f"Resolving data repository for: {repo}")
        if ',' in repo:
            self.logger.warning(f"Repository contains a comma: {repo}. Same data may be in multiple repos.")
            ret = []
            for r in repo.split(','):
                r = r.strip()
                if r in self.config['repos']:
                    ret.append(self.resolve_data_repository(r))
            return ret

        for k, v in self.config["repos"].items():
            self.logger.debug(f"Checking if {repo} == {k}")
            repo = re.sub("\s+\(\w+\)\s*", "", repo)  # remove any text in parentheses
            # match where repo_link has been extracted
            if k == repo:
                self.logger.info(f"Exact match found for repo: {repo}")
                break

            elif 'repo_name' in v.keys():
                if repo.lower() == v['repo_name'].lower():
                    self.logger.info(f"Found repo_name match for {repo}")
                    repo = k
                    break

                elif v['repo_name'].lower() in repo.lower():
                    self.logger.info(f"Found partial match for {repo} in {v['repo_name']}")
                    repo = k
                    break

        return repo  # fallback

    def get_dataset_webpage(self, datasets):
        """
        Given a list of dataset dictionaries, fetch the webpage of the dataset, by using navigation patterns
        from the ontology. The function will add a new key to the dataset dictionary with the webpage URL.

        :param datasets: list of dictionaries containing dataset information.

        :return: list of dictionaries with updated dataset information including dataset webpage URL.
        """
        if datasets is None:
            return None

        self.logger.info(f"Fetching metadata for {len(datasets)} datasets")

        for i, item in enumerate(datasets):

            if type(item) != dict:
                self.logger.error(f"can't resolve dataset_webpage for non-dict item {1 + i}: {item}")
                continue

            self.logger.info(f"Processing dataset {1 + i} with keys: {item.keys()}")

            if 'data_repository' not in item.keys() and 'repository_reference' not in item.keys():
                self.logger.info(f"Skipping dataset {1 + i}: no data_repository for item")
                continue

            accession_id = item.get('dataset_identifier', item.get('dataset_id', 'n/a'))
            if accession_id == 'n/a':
                self.logger.info(f"Skipping dataset {1 + i}: no dataset_identifier for item")
                continue
            else:
                self.logger.info(f"Raw accession ID: {accession_id}")

            if ('dataset_webpage' in item.keys()):
                self.logger.debug(f"Skipping dataset {1 + i}: already has dataset_webpage")
                continue

            if 'data_repository' in item.keys():
                repo = self.resolve_data_repository(item['data_repository']).lower()
            elif 'repository_reference' in item.keys():
                repo = self.resolve_data_repository(item['repository_reference']).lower()
            else:
                self.logger.error(f"Error extracting data repository for item: {item}")
                continue

            accession_id = self.resolve_accession_id(accession_id, repo)

            self.logger.info(f"Processing dataset {1 + i} with repo: {repo} and accession_id: {accession_id}")
            self.logger.debug(f"Processing dataset {1 + i} with keys: {item.keys()}")

            updated_dt = False

            if repo in self.config['repos'].keys():

                if "dataset_webpage_url_ptr" in self.config['repos'][repo]:
                    dataset_webpage = re.sub('__ID__', accession_id, self.config['repos'][repo]['dataset_webpage_url_ptr'])

                elif 'url_concat_string' in self.config['repos'][repo]:
                    dataset_webpage = ('https://' + repo + re.sub('__ID__', accession_id,
                                                                   self.config['repos'][repo]['url_concat_string']))

                else:
                    self.logger.warning(f"No dataset_webpage_url_ptr or url_concat_string found for {repo}. Maybe lost in refactoring 21 April 2025")
                    dataset_webpage = 'na'

                self.logger.info(f"Dataset page: {dataset_webpage}")
                datasets[i]['dataset_webpage'] = dataset_webpage

                # add access mode
                if 'access_mode' in self.config['repos'][repo]:
                    self.logger.info(f"Adding access mode for dataset {1 + i}: {self.config['repos'][repo]['access_mode']}")
                    datasets[i]['access_mode'] = self.config['repos'][repo]['access_mode']

            else:
                self.logger.warning(f"Repository {repo} not supported in config. Skipping dataset {1 + i}.")
                continue

        self.logger.info(f"Updated datasets len: {len(datasets)}")
        return datasets

    def get_NuExtract_template(self):
        template = """{
            "Available Dataset": {
                "data repository name" = "",
                "data repository link" = "",
                "dataset identifier": "",
                "dataset webpage": "",
            }
        }"""

        return template

    def tokens_over_limit(self, html_cont : str, model="gpt-4", limit=128000, allowance_static_prompt=200):
        # Load the appropriate encoding for the model
        encoding = tiktoken.encoding_for_model(model)
        # Encode the prompt and count tokens
        tokens = encoding.encode(html_cont)
        self.logger.info(f"Number of tokens: {len(tokens)}")
        return len(tokens)+int(allowance_static_prompt*1.25)>limit

    def count_tokens(self, prompt, model="gpt-4") -> int:
        """
        Count the number of tokens in a given prompt for a specific model.

        :param prompt: str — the prompt to be tokenized.

        :param model: str — the model name (default: "gpt-4").

        :return: int — the number of tokens in the prompt.
        """
        n_tokens = 0

        # **Ensure `prompt` is a string**
        if isinstance(prompt, list):
            self.logger.info(f"Expected string but got list. Converting list to string.")
            prompt = " ".join([msg["content"] for msg in prompt if isinstance(msg, dict) and "content" in msg])

        elif not isinstance(prompt, str):
            self.logger.error(f"Unexpected type for prompt: {type(prompt)}. Converting to string.")
            prompt = str(prompt)

        # **Token count based on model**
        if 'gpt' in model:
            encoding = tiktoken.encoding_for_model(model)
            n_tokens = len(encoding.encode(prompt))

        elif 'gemini' in model:
            try:
                gemini_prompt = {"role": "user", "parts": [{"text": prompt}]}
                response = self.client.count_tokens([gemini_prompt])
                n_tokens = response.total_tokens
            except Exception as e:
                self.logger.error(f"Error counting tokens for Gemini: {e}")
                n_tokens = 0

        return n_tokens

    def predict_NuExtract(self, model, tokenizer, texts, template, batch_size=1, max_length=10_000,
                          max_new_tokens=4_000):
        template = json.dumps(json.loads(template), indent=4)
        prompts = [f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>""" for text in texts]

        outputs = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_encodings = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True,
                                            max_length=max_length).to(model.device)

                pred_ids = model.generate(**batch_encodings, max_new_tokens=max_new_tokens)
                outputs += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        return [output.split("<|output|>")[1] for output in outputs]

    def parse_metadata(self, metadata: str, model = 'gemini-2.0-flash') -> dict:
        """
        Given the metadata, extract the dataset information using the LLM.

        :param metadata: str — the metadata to be parsed.

        :param model: str — the model to be used for parsing (default: 'gemini-2.0-flash').

        :return: dict — the extracted metadata. This is sometimes project metadata, or study metadata, or dataset metadata. Ontology enhancement is needed to distinguish between these.
        """
        #metadata = self.normalize_full_DOM(metadata)
        self.logger.info(f"Parsing metadata len: {len(metadata)}")
        dataset_info = self.extract_dataset_info(metadata, subdir='metadata_prompts')
        return dataset_info

    def extract_dataset_info(self, metadata, subdir = ''):
        """
        Given the metadata, extract the dataset information using the LLM.

        :param metadata: str — the metadata to be parsed.

        :param subdir: str — the subdirectory for the prompt template (default: '').

        :return: dict — the extracted metadata. This is sometimes project metadata, or study metadata, or dataset
         metadata
        """
        self.logger.info(f"Extracting dataset information from metadata. Prompt from subdir: {subdir}")

        llm = LLMClient(
            model=self.config.get('llm_model', 'gemini-2.0-flash'),
            logger=self.logger,
            save_prompts=self.config.get('save_dynamic_prompts', False)
        )
        response = llm.api_call(metadata, subdir = subdir)

        # Post-process response into structured dict
        dataset_info = self.safe_parse_json(response)
        self.logger.info(f"Extracted dataset info: {dataset_info}")
        return dataset_info


class MyBeautifulSoup(BeautifulSoup):
    # this function will extract text from the HTML, by also keeping the links where they appear in the HTML
    def _all_strings(self, strip=False, types=(NavigableString, CData)):
        strings = []
        logging.info(f"descendants: {self.descendants}")
        # it should extract text from all elements in the HTML, but keep more than just text only with anchor elements
        for element in self.descendants:
            if element is None:
                continue
            logging.info(f"type of element: {type(element)}")
            if type(element) == 'bs4.element.Tag':
                logging.info(f"Tag attributes: {element.attrs}")
                print(f"Tag attributes: {element.attrs}")
            # element is Tag, we want to keep the anchor elements hrefs and the text in every Tag
            if element.name == 'a':  # or do the check of href in element.attrs
                logging.info("anchor element")
                newstring = re.sub("\s+", " ", element.getText())
                strings.append(newstring)
                if element.href is not None:
                    strings.append(element.href)
                    logging.info(f"link in 'a': {element.href}")
                    print(f"link in 'a': {element.href}")
            else:
                strings.append(re.sub("\s+", " ", element.getText()))
        #logging.info(f"strings: {strings}")
        return strings

    def get_text(self, separator="", strip=False,
                 types=(NavigableString, CData)):
        """Get all child strings of this PageElement, concatenated using the
        given separator.

        :param separator: Strings will be concatenated using this separator.

        :param strip: If True, strings will be stripped before being
            concatenated.

        :param types: A tuple of NavigableString subclasses. Any
            strings of a subclass not found in this list will be
            ignored. Although there are exceptions, the default
            behavior in most cases is to consider only NavigableString
            and CData objects. That means no comments, processing
            instructions, etc.

        :return: A string.
        """
        return separator.join([s for s in self._all_strings(
            strip, types=types)])

    getText = get_text
    text = property(get_text)

class LLMClient:
    def __init__(self, model:str, logger=None, save_prompts:bool=False):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Initializing LLMClient with model: {self.model}")
        self._initialize_client(model)
        self.save_prompts = save_prompts
        self.prompt_manager = PromptManager("data_gatherer/prompts/prompt_templates/metadata_prompts", self.logger)

    def _initialize_client(self, model):
        if model.startswith('gpt'):
            self.client = OpenAI(api_key=os.environ['GPT_API_KEY'])
        elif model.startswith('gemini'):
            genai.configure(api_key=os.environ['GEMINI_KEY'])
            self.client = genai.GenerativeModel(model)
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        self.logger.info(f"Client initialized: {self.client}")

    def api_call(self, content, subdir=''):
        self.logger.info(f"Calling {self.model} with prompt length {len(content)}, subdir: {subdir}")
        if self.model.startswith('gpt'):
            return self._call_openai(content, subdir=subdir)
        elif self.model.startswith('gemini'):
            return self._call_gemini(content, subdir=subdir)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _call_openai(self, content, temperature=0.0, subdir=''):
        self.logger.info(f"Calling OpenAI with content length {len(content)}, subdir: {subdir}")
        messages = self.prompt_manager.render_prompt(
            self.prompt_manager.load_prompt("gpt_metadata_extract",subdir=subdir),
            entire_doc=True,
            content=content,
        )
        # save prompt_eval
        if self.save_prompts:
            self.prompt_manager.save_prompt(prompt_id='abc', prompt_content=messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format=dataset_metadata_response_schema_gpt
        )
        return response['choices'][0]['message']['content']

    def _call_gemini(self, content, temperature=0.0, subdir=''):
        self.logger.info(f"Calling Gemini with content length {len(content)}, subdir: {subdir}")
        messages = self.prompt_manager.render_prompt(
            self.prompt_manager.load_prompt("gemini_metadata_extract",subdir=subdir),
            entire_doc=True,
            content=content,
        )

        # save prompt_eval
        if self.save_prompts:
            self.prompt_manager.save_prompt(prompt_id='abc', prompt_content=messages)

        response = self.client.generate_content(
            messages,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=temperature,
            )
        )
        return response.text