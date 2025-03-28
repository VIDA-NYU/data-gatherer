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
from pydantic import BaseModel, HttpUrl
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_manager import PromptManager
import tiktoken


# Abstract base class for parsing data
class Parser(ABC):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.logger.info("Parser initialized.")
        self.full_DOM = self.config['llm_model'] in self.config['entire_document_models'] and self.config['process_entire_document']

    @abstractmethod
    def parse_data(self, raw_data, publisher, current_url_address):
        pass


class Dataset(BaseModel):
    dataset_id: str
    repository_reference: str

class Dataset_w_Description(typing.TypedDict):
    dataset_id: str
    repository_reference: str
    rationale: str


# Implementation for Rule-Based parsing of HTMLs (from web scraping)
class RuleBasedParser(Parser):

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.logger.info("RuleBasedParser initialized.")

    def parse_data(self, source_html, publisher, current_url_address, raw_data_format='HTML'):
        # initialize output
        links_on_webpage = []
        self.logger.info("Function call: extract_links_data_from_source(source_html)")
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

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.title = None
        self.prompt_manager = PromptManager(self.config['prompt_dir'], self.logger, self.config['response_file'])

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
        out_df = None
        # Check if api_data is a string, and convert to XML if needed
        self.logger.info(f"Function call: parse_data(api_data({type(api_data)}), publisher, current_url_address, "
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
                preprocessed_data = self.normalize_full_DOM(api_data)

                #self.logger.info(f"Preprocessed data: {preprocessed_data}")

                # Extract dataset links from the entire text
                augmented_dataset_links = self.retrieve_datasets_from_content(preprocessed_data, self.config['repos'],
                                                                              self.config['llm_model'],
                                                                              temperature=0)
                self.logger.info(f"Augmented dataset links: {augmented_dataset_links}")

                dataset_links_w_target_pages = self.get_dataset_webpage(augmented_dataset_links)

                # Create a DataFrame from the dataset links union supplementary material links
                out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages)])  # check index error here

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

            return out_df

    def normalize_full_DOM(self, api_data: str) -> str:
        """
        Normalize the full HTML DOM by removing dynamic elements and attributes
        that frequently change, such as random IDs, inline styles, analytics tags,
        and CSRF tokens.

        Parameters:
        - api_data (str): The raw HTML content of the webpage.

        Returns:
        - str: The normalized HTML content.
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
        """
        Extracts the file extension from a download link.
        """
        self.logger.debug(f"Function_call: extract_file_extension(download_link)")
        # Extract the file extension from the download link
        extension = None
        if type(download_link) == str:
            extension = download_link.split('.')[-1]
        if type(extension) == str and ("/" in extension):  # or "?" in extension
            return ""
        return extension

    def extract_href_from_data_availability(self, api_xml):
        # Namespace dictionary - adjust 'ns0' to match the XML if necessary
        self.logger.info(f"Function_call: extract_href_from_data_availability(api_xml)")
        namespaces = {'ns0': 'http://www.w3.org/1999/xlink'}

        # Find all sections with "data-availability"
        data_availability_sections = []
        for ptr in self.config['data_availability_sections']:
            data_availability_sections.extend(api_xml.findall(ptr))

        hrefs = []
        for section in data_availability_sections:
            # Find all <ext-link> elements in the section
            ext_links = section.findall(".//ext-link", namespaces)
            uris = section.findall(".//uris", namespaces)

            if uris is not None:
                ext_links.extend(uris)

            self.logger.info(f"Found {len(ext_links)} ext-links in data availability section.")

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
                        'source_section': 'data availability'
                    })
                    self.logger.info(f"Extracted item: {json.dumps(hrefs[-1], indent=4)}")

        return hrefs

    def extract_xrefs_from_data_availability(self, api_xml, current_url_address):
        """
        Extracts xrefs (cross-references) from data-availability sections of the XML.
        """
        self.logger.info(f"Function_call: extract_xrefs_from_data_availability(api_xml, current_url_address)")

        # Find all sections with "data-availability"
        data_availability_sections = []
        for ptr in self.config['data_availability_sections']:
            self.logger.info(f"Searching for data availability sections using XPath: {ptr}")
            data_availability_sections.extend(api_xml.findall(ptr))

        xrefs = []
        for section in data_availability_sections:
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
                    'source_section': 'data availability'
                })
                self.logger.info(f"Extracted xref item: {json.dumps(xrefs[-1], indent=4)}")

        return xrefs

    def extract_href_from_supplementary_material(self, api_xml, current_url_address):

        self.logger.info(f"Function_call: extract_href_from_supplementary_material(api_xml, current_url_address)")

        # Namespace dictionary for xlink
        namespaces = {'xlink': 'http://www.w3.org/1999/xlink'}

        # Find all sections for "supplementary-material"
        supplementary_material_sections = []
        for ptr in self.config['supplementary_material_sections']:
            self.logger.debug(f"Searching for supplementary material sections using XPath: {ptr}")
            supplementary_material_sections.extend(api_xml.findall(ptr))

        self.logger.debug(f"Found {len(supplementary_material_sections)} supplementary-material sections.")

        hrefs = []

        for section in supplementary_material_sections:
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
                    surrounding_text = " ".join(parent_p.itertext()).strip()  # Gets all text within the <p> tag
                else:
                    surrounding_text = "No surrounding text found"

                # Extract the full description within the <p> tag if available
                description_element = supplementary_material_parent.find(".//caption/p")
                description = " ".join(
                    description_element.itertext()).strip() if description_element is not None else "No description"

                # Log media attributes and add to results
                self.logger.info(f"Extracted media item with href: {href}")
                self.logger.info(f"Source url: {current_url_address}")
                self.logger.info(f"Supplementary material title: {title}")
                self.logger.info(f"Content type: {content_type}, ID: {media_id}")
                self.logger.info(f"Surrounding text for media: {surrounding_text}")
                self.logger.info(f"Description: {description}")
                self.logger.info(f"Download_link: {download_link}")

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
                        'source_section': 'supplementary material'
                    })
                    self.logger.debug(f"Extracted item: {json.dumps(hrefs[-1], indent=4)}")

            # Find all <inline-supplementary-material> elements in the section
            inline_supplementary_materials = section.findall(".//inline-supplementary-material")
            self.logger.info(f"Found {len(inline_supplementary_materials)} inline-supplementary-material elements.")

            for inline in inline_supplementary_materials:
                # repeating steps like in media links above
                hrefs.append({
                    "link": inline.get('{http://www.w3.org/1999/xlink}href'),
                    "content_type": inline.get('content-type', 'Unknown content type'),
                    "id": inline.get('id', 'No ID'),
                    "title": inline.get('title', 'No Title'),
                    "source_section": 'supplementary material inline',
                    "download_link": self.reconstruct_download_link(inline.get('{http://www.w3.org/1999/xlink}href'),
                                                                    inline.get('content-type', 'Unknown content type'),
                                                                    current_url_address)
                })

            self.logger.info(f"Extracted supplementary material links:\n{hrefs}")

        return hrefs

    def reconstruct_download_link(self, href, content_type, current_url_address):
        # https: // pmc.ncbi.nlm.nih.gov / articles / instance / 11252349 / bin / 41598_2024_67079_MOESM1_ESM.zip
        # https://pmc.ncbi.nlm.nih.gov/articles/instance/PMC11252349/bin/41598_2024_67079_MOESM1_ESM.zip
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11252349/bin/41598_2024_67079_MOESM1_ESM.zip
        download_link = None
        #repo = self.url_to_repo_domain(current_url_address)
        # match the digits of the PMC ID (after PMC) in the URL
        PMCID = re.search(r'PMC(\d+)', current_url_address).group(1)
        self.logger.debug(
            f"Inputs to reconstruct_download_link: {href}, {content_type}, {current_url_address}, {PMCID}")
        if content_type == 'local-data':
            download_link = "https://pmc.ncbi.nlm.nih.gov/articles/instance/" + PMCID + '/bin/' + href
        return download_link

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

    def get_surrounding_text(self, element):
        """
        Extracts text surrounding the element (including parent and siblings) for more context.
        It ensures that text around inline elements like <xref> and <ext-link> is properly captured.
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

        return surrounding_text

    def union_additional_data(self, parsed_data, additional_data):
        """
        Merge the parsed data with additional data from the API.
        """
        self.logger.info(f"Merging additional data ({type(additional_data)}) with parsed data({type(parsed_data)}).")
        self.logger.info(f"Additional data\n{additional_data}")
        return pd.concat([parsed_data, additional_data], ignore_index=True)

    def process_additional_data(self, additional_data):
        """
        Process the additional data from the webpage. This is the data matched from the HTML with the patterns in
        retrieval_patterns xpaths.
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
            ret.append(dataset)

        self.logger.info(f"Final ret additional data: {len(ret)} items")
        self.logger.debug(f"Final ret additional data: {ret}")
        return ret

    def retrieve_datasets_from_content(self, content: str, repos: list, model: str, temperature: float = 0.3) -> list:
        """
        Retrieve datasets from the given content using a specified LLM model.
        Uses a static prompt template and dynamically injects the required content.
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
                f"Requesting datasets from content using model: {model}, temperature: {temperature}, messages: messages")
            resps = []

            if self.config['llm_model'] == 'gemma2:9b':
                response = self.client.chat(model=model, options={"temperature": temperature}, messages=messages)
                self.logger.info(
                    f"Response received from model: {response.get('message', {}).get('content', 'No content')}")
                resps = response['message']['content'].split("\n")
                # Save the response
                self.prompt_manager.save_response(prompt_id, response['message']['content'])
                self.logger.info(f"Response saved to cache")

            elif self.config['llm_model'] == 'gpt-4o-mini' or self.config['llm_model'] == 'gpt-4o':
                response = None
                if self.config['process_entire_document']:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        response_format={
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
                        resps = parsed_response
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

    def safe_parse_json(self, response_text):
        """ Cleans and safely parses JSON from a GPT response, fixing common issues. """
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
        Given the data availability statement, extract the dataset information from the text.
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

        self.logger.info(f"Found {supplementary_data_sections} supplementary data sections")

        for sect in supplementary_data_sections:
            # check if section contains data availability statement
            if sect.text is None:  # key resources table
                self.logger.info(f"Section with no text: {sect}")
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

    """   
    def get_data_availability_section_from_webpage(self, pmc_id):
        '''
        Given the pmc_id reconstruct the url
        :param pmc_id:
        :return: data availability section from the webpage
        '''
        self.logger.info(f"Fetching data availability from webpage for pmc_id: {pmc_id}")
        surrounding_text = "data availability section from the webpage"

        dataset_nonlink_refs = [{"surrounding_text": surrounding_text, "source_section": "data availability"}]

        return dataset_nonlink_refs
        """

    def process_data_availability_links(self, dataset_links):
        """
        Given the link, the article title, and the text around the link, create a column (identifier),
        and a column for the dataset.
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
                ret.append(ret_element)
                progress += 1

        return ret

    def dataset_webpage_url_check(self, url):
        """
        Check if the URL directly points to a dataset webpage.
        :param url:
        :return:
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

        return

    def normalize_LLM_output(self, response):
        """
        Given a response from the LLM API, normalize it to a list of strings.
        Also handle case when 1 repo has more than 1 identifier.
        """
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

    def get_dataset_webpage(self, datasets):
        """
        Given a list of dataset dictionaries, fetch the webpage of the dataset, by using navigation patterns
        """
        if datasets is None:
            return None

        self.logger.info(f"Fetching metadata for {len(datasets)} datasets")

        for i, item in enumerate(datasets):

            self.logger.info(f"Processing dataset {1 + i} with keys: {item.keys()}")

            if 'data_repository' not in item.keys() and 'repository_reference' not in item.keys():
                self.logger.info(f"Skipping dataset {1 + i}: no data_repository for item")
                continue

            if ('dataset_identifier' in item.keys() and item['dataset_identifier'] == 'n/a') or (
                    'dataset_id' in item.keys() and item['dataset_id'] == 'n/a'
            ):
                self.logger.info(f"Skipping dataset {1 + i}: no dataset_identifier for item")
                continue

            if ('dataset_webpage' in item.keys()):
                self.logger.debug(f"Skipping dataset {1 + i}: already has dataset_webpage")
                continue

            if 'data_repository' in item.keys():
                repo = self.url_to_repo_domain(item['data_repository'])
            elif 'repository_reference' in item.keys():
                repo = self.url_to_repo_domain(item['repository_reference'])
            else:
                self.logger.error(f"Error extracting data repository for item: {item}")
                continue

            self.logger.info(f"Processing dataset {1 + i} with repo: {repo}")

            updated_dt = False
            for k, v in self.config["repos"].items():
                self.logger.debug(f"Checking if {repo} == {k}")
                if k == repo and 'url_concat_string' in v.keys():
                    if 'repo_mapping' in v.keys():
                        repo_name = self.config['repos'][repo]['repo_mapping']
                    else:
                        repo_name = repo
                    self.logger.info(f"Found config options for {k}")

                    if 'dataset_identifier' in item.keys():
                        dataset_webpage = ('https://' + repo_name + re.sub('__ID__', item['dataset_identifier'],
                                                                           self.config['repos'][repo]['url_concat_string']))

                    elif 'dataset_id' in item.keys():
                        dataset_webpage = ('https://' + repo_name + re.sub('__ID__', item['dataset_id'],
                                                                           self.config['repos'][repo]['url_concat_string']))

                    else:
                        dataset_webpage = 'na'

                    datasets[i]['dataset_webpage'] = dataset_webpage
                    self.logger.info(f"Dataset page: {dataset_webpage}")
                    updated_dt = True
                    break

                elif not updated_dt:
                    datasets[i]['dataset_webpage'] = 'na'
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
        n_tokens = 0

        # **Ensure `prompt` is a string**
        if isinstance(prompt, list):
            self.logger.warning(f"Expected string but got list. Converting list to string.")
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
