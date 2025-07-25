from data_gatherer.parser.base_parser import *
from data_gatherer.retriever.html_retriever import htmlRetriever
import regex as re
import logging
import pandas as pd
from bs4 import BeautifulSoup, Comment, NavigableString, CData
from lxml import html


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


class HTMLParser(LLMParser):
    """
    Custom HTML parser that has only support for HTML or HTML-like input
    """

    def __init__(self, open_data_repos_ontology, logger, log_file_override=None, full_document_read=True,
                 prompt_dir="data_gatherer/prompts/prompt_templates",
                 llm_name=None, save_dynamic_prompts=False, save_responses_to_cache=False, use_cached_responses=False,
                 use_portkey_for_gemini=True):

        super().__init__(open_data_repos_ontology, logger, log_file_override=log_file_override,
                         full_document_read=full_document_read, prompt_dir=prompt_dir,
                         llm_name=llm_name, save_dynamic_prompts=save_dynamic_prompts,
                         save_responses_to_cache=save_responses_to_cache,
                         use_cached_responses=use_cached_responses, use_portkey_for_gemini=use_portkey_for_gemini
                         )

        self.logger = logger
        self.logger.info("Initializing htmlRetriever")
        self.retriever = htmlRetriever(logger, 'general', retrieval_patterns_file='retrieval_patterns.json',
                                       headers=None)

    def normalize_HTML(self, html, keep_tags=None):
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
            for tag in ["script", "style", 'img', 'iframe', 'noscript', 'svg', 'button', 'form', 'input']:
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

    def get_rule_based_matches(self, publisher, html_content):

        html_tree = html.fromstring(html_content)

        if publisher in self.retriever.retrieval_patterns:
            self.retriever.update_class_patterns(publisher)

        rule_based_matches = {}

        # Collect links using CSS selectors
        for css_selector in self.retriever.css_selectors:
            self.logger.debug(f"Parsing page with selector: {css_selector}")
            links = html_tree.cssselect(css_selector)
            self.logger.debug(f"Found Links: {links}")
            for link in links:
                rule_based_matches[link] = self.retriever.css_selectors[css_selector]
        self.logger.info(f"Rule-based matches from css_selectors: {rule_based_matches}")

        # Collect links using XPath
        for xpath in self.retriever.xpaths:
            self.logger.info(f"Checking path: {xpath}")
            try:
                child_element = html_tree.xpath(xpath)
                section_element = child_element.xpath("./ancestor::section")
                a_elements = section_element.xpath('a')
                for a_element in a_elements:
                    rule_based_matches[a_element] = self.retriever.xpaths[xpath]
            except Exception as e:
                self.logger.error(f"Invalid xpath: {xpath}")

        return self.normalize_links(rule_based_matches)

    def normalize_links(self, links_raw_dict):
        """
        Convert Selenium WebElement objects to strings and keep existing string links unchanged.
        """
        normalized_links = {}
        for link, cls in links_raw_dict.items():
            if isinstance(link, str):
                normalized_links[link] = cls
            else:
                try:
                    href = link.get_attribute('href')
                    if href:
                        normalized_links[href] = cls
                except Exception as e:
                    self.logger.error(f"Error getting href attribute from WebElement: {e}")
        return normalized_links

    def extract_paragraphs_from_html(self, html_content: str) -> list[dict]:
        """
        Extract paragraphs and their section context from an HTML document.

        Args:
            html_content: str — raw HTML content.

        Returns:
            List of dicts with 'paragraph', 'section_title', and 'sec_type'.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        paragraphs = []
        for section in soup.find_all(['section', 'div']):
            section_title = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            section_title_text = section_title.get_text(strip=True) if section_title else "No Title"
            sec_type = section.get('class', ['unknown'])[0] if section.has_attr('class') else "unknown"
            for p in section.find_all('p'):
                para_text = p.get_text(strip=True)
                if len(para_text) >= 5:
                    paragraphs.append({
                        "paragraph": para_text,
                        "section_title": section_title_text,
                        "sec_type": sec_type,
                        "text": para_text
                    })
        return paragraphs

    def extract_sections_from_html(self, html_content: str) -> list[dict]:
        """
        Extract sections from an HTML document.

        Args:
            html_content: str — raw HTML content.

        Returns:
            List of dicts with 'section_title' and 'sec_type'.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        sections = []
        for section in soup.find_all(['section']):  # 'div'
            if section.find(['section']):  # 'div'
                continue
            section_title = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            section_title_text = section_title.get_text(strip=True) if section_title else "No Title"
            sec_type = section.get('class', ['unknown'])[0] if section.has_attr('class') else "unknown"
            section_text = section.get_text(separator="\n", strip=True)
            sections.append({
                "section_title": section_title_text,
                "sec_type": sec_type,
                "sec_txt": section_text
            })
        return sections

    def extract_all_hrefs(self, source_html, publisher, current_url_address, raw_data_format='HTML'):
        # initialize output
        links_on_webpage = []
        self.logger.info("Function call: extract_all_hrefs")
        normalized_html = self.normalize_HTML(source_html)
        soup = BeautifulSoup(normalized_html, "html.parser")
        compressed_HTML = self.convert_HTML_to_text(normalized_html)
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

    def reconstruct_link(self, link, publisher, todo=True):
        """
        Given a publisher name, return the root domain.
        """
        if todo:
            raise NotImplementedError("This method should be implemented in subclasses.")
        if not (link.startswith("http") or link.startswith("//") or link.startswith("ftp")):
            if (publisher.startswith("http") or publisher.startswith("//")):
                publisher_root = publisher
            else:
                publisher_root = "https://www." + publisher
            return publisher_root + link
        else:
            return link

    def convert_HTML_to_text(self, source_html):
        """
        This function should convert html to markdown and keep the links close to the text near them in the webpage GUI.
        """
        self.logger.debug(f"compress HTML. Original len: {len(source_html)}")
        # Parse the HTML content with BeautifulSoup
        soup = MyBeautifulSoup(source_html, "html.parser")
        text = re.sub("\s+", " ", soup.getText())
        self.logger.debug(f"compress HTML. Final len: {len(text)}")
        return text

    def parse_data(self, html_str, publisher=None, current_url_address=None, additional_data=None,
                   raw_data_format='HTML',
                   article_file_dir='tmp/raw_files/', process_DAS_links_separately=False, section_filter=None,
                   prompt_name='retrieve_datasets_simple_JSON', use_portkey_for_gemini=True, semantic_retrieval=False):
        """
        Parse the API data and extract relevant links and metadata.

        Args:
            html_str (str): The raw HTML content to be parsed.
            publisher (str): The publisher of the content.
            current_url_address (str): The URL address of the content.
            additional_data (dict): Additional data to be used in the parsing.
            raw_data_format (str): The format of the raw data (default is 'HTML').
            article_file_dir (str): Directory to save article files (default is 'tmp/raw_files/').
            process_DAS_links_separately (bool): Whether to process Data Availability Statement links separately.
            section_filter (str): Filter for sections to be processed.
            prompt_name (str): Name of the prompt to be used for dataset extraction.
            use_portkey_for_gemini (bool): Whether to use Portkey for Gemini model.
            semantic_retrieval (bool): Whether to use semantic retrieval for extracting sections.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted dataset links and metadata (if repo is supported or info
            is available in paper). Feel free to add support for unsupported repos in the ontology!

        """
        out_df = None
        self.logger.info(f"Function call: parse_data(html_str, {publisher}, {current_url_address}, "
                         f"additional_data, {raw_data_format})")
        self.publisher = publisher

        filter_supp = section_filter == 'supplementary_material' or section_filter is None
        filter_das = section_filter == 'data_availability_statement' or section_filter is None

        if filter_supp or filter_supp is None:
            supplementary_material_links = self.extract_href_from_supplementary_material(html_str, current_url_address)
            supplementary_material_metadata = self.extract_supplementary_material_refs(html_str,
                                                                                       supplementary_material_links)
        else:
            supplementary_material_metadata = pd.DataFrame()

        preprocessed_data = self.normalize_HTML(html_str)
        self.logger.debug(f"Preprocessed data: {preprocessed_data}")

        if self.full_document_read and (filter_das is None or filter_das is True):
            self.logger.info(f"Extracting links from full HTML content.")

            # Extract dataset links from the entire text
            augmented_dataset_links = self.extract_datasets_info_from_content(preprocessed_data,
                                                                              self.open_data_repos_ontology['repos'],
                                                                              model=self.llm_name,
                                                                              temperature=0,
                                                                              prompt_name=prompt_name)

            self.logger.info(f"Augmented dataset links: {augmented_dataset_links}")

            dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

            # Create a DataFrame from the dataset links union supplementary material links
            out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages), supplementary_material_metadata])

        elif filter_das is None or filter_das is True:
            self.logger.info(f"Chunking the HTML content for the parsing step.")

            # Extract dataset links from the entire text
            data_availability_elements = self.retriever.get_data_availability_elements_from_webpage(preprocessed_data)

            data_availability_str = "\n".join([item['html'] + "\n" for item in data_availability_elements])

            if semantic_retrieval:
                corpus = self.extract_sections_from_html(preprocessed_data)
                top_k_sections = self.semantic_retrieve_from_corpus(corpus)
                top_k_sections_text = [item['text'] for item in top_k_sections]
                top_k_sections_str = "\n".join(top_k_sections_text)
                data_availability_str = top_k_sections_str + "\n" + data_availability_str

            augmented_dataset_links = self.extract_datasets_info_from_content(data_availability_str,
                                                                              self.open_data_repos_ontology['repos'],
                                                                              model=self.llm_name,
                                                                              temperature=0,
                                                                              prompt_name=prompt_name)

            dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

            out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages), supplementary_material_metadata])

        else:
            out_df = supplementary_material_metadata

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

        # Ensure out_df is a copy to avoid SettingWithCopyWarning
        out_df = out_df.copy()

        out_df.loc[:, 'source_url'] = current_url_address
        out_df.loc[:, 'pub_title'] = self.retriever.extract_publication_title(preprocessed_data)

        return out_df

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

            # Extract file description from the surrounding <p> or <div> element
            # Improved logic: look for a <p> in a preceding sibling <div> with class 'caption'
            description = "n/a"
            parent = anchor.getparent()
            # 1. Try direct child <p> of parent
            p_desc = parent.xpath("./p")
            if p_desc:
                description = p_desc[0].text_content().strip()
            else:
                # 2. Try preceding sibling <div> with class 'caption p' and get its <p>
                prev_caption_divs = parent.xpath("preceding-sibling::div[contains(@class, 'caption p')]")
                found = False
                for div in prev_caption_divs:
                    p_in_div = div.xpath(".//p")
                    if p_in_div:
                        description = p_in_div[0].text_content().strip()
                        found = True
                        break
                # 3. If not found, try to get <p> from parent's parent (for cases like <div class="media p"><div class="caption">...</div></div>)
                if not found:
                    grandparent = parent.getparent()
                    if grandparent is not None:
                        # Look for a preceding sibling <div> with class 'caption p' in grandparent
                        prev_caption_divs_gp = grandparent.xpath(
                            "preceding-sibling::div[contains(@class, 'caption p')]")
                        for div in prev_caption_divs_gp:
                            p_in_div = div.xpath(".//p")
                            if p_in_div:
                                description = p_in_div[0].text_content().strip()
                                found = True
                                break
                # 4. If still not found, try to get <p> from any ancestor <div class='caption p'>
                if not found:
                    ancestor_caption_divs = parent.xpath("ancestor::div[contains(@class, 'caption p')]")
                    for div in ancestor_caption_divs:
                        p_in_div = div.xpath(".//p")
                        if p_in_div:
                            description = p_in_div[0].text_content().strip()
                            break

            # Extract attributes from parent <section> for context
            section = anchor.getparent().getparent()  # Assuming structure stays the same
            section_id = section.get('id', 'n/a')
            section_class = section.get('class', 'n/a')
            ancestor_section = anchor.xpath("ancestor::section[1]")
            ancestor_section_id = ancestor_section[0].get('id') if ancestor_section and ancestor_section[0].get('id') else section_id

            # Combine all extracted info
            link_data = {
                'link': href,
                'title': title,
                'file_info': file_info,
                'description': description,
                'source_section': ancestor_section_id,
                'section_class': section_class,
            }

            if link_data['section_class'] == 'ref-list font-sm':
                self.logger.debug(
                    f"Skipping link with section_class 'ref-list font-sm', likely to be a reference list.")
                continue

            #if 'doi.org' in link_data['link'] or 'scholar.google.com' in link_data['link']: ############ Same as above
            #    continue

            link_data['download_link'] = self.reconstruct_download_link(href, link_data['section_class'],
                                                                        current_url_address)
            link_data['file_extension'] = self.extract_file_extension(link_data['download_link']) if link_data[
                                                                                                         'download_link'] is not None else None

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

    def extract_supplementary_material_refs(self, raw_html, supplementary_material_links):
        """
        Extract metadata from references to supplementary material ids in the HTML.
        For each supplementary material link, find <a href="#id"> and extract a short context sentence.
        Adds a 'context_description' column to the DataFrame.
        """
        self.logger.info("Function_call: extract_supplementary_material_refs(raw_html, supplementary_material_links)")
        tree = html.fromstring(raw_html)

        for i, row in supplementary_material_links.iterrows():
            context_descr = ""
            href_id = row.get('source_section', None)
            if not href_id or href_id == 'n/a':
                supplementary_material_links.at[i, 'context_description'] = ""
                continue

            a_elements = tree.xpath(f".//a[@href='#{href_id}']")
            self.logger.info(f"Found {len(a_elements)} <a> elements with href='#{href_id}'.")

            for ref in a_elements:
                surrounding_text = self.get_surrounding_text(ref)
                text_segment = self.get_sentence_segment(surrounding_text, href_id)
                if text_segment not in context_descr:
                    context_descr += text_segment + "\n"

            self.logger.info(f"Extracted context description (len: {len(context_descr)}) for {href_id}: "
                             f"{context_descr.strip()}")
            supplementary_material_links.at[i, 'context_description'] = context_descr.strip()

        return supplementary_material_links

    def naive_sentence_tokenizer(self, text):
        """
        Split text into sentences using a simple regex, avoiding common abbreviations.
        """
        pattern = r'(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|Inc|e\.g|i\.e|Fig|vs|et al))(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return sentences

    def get_sentence_segment(self, surrounding_text, ref_id):
        """
        Extract the shortest sentence(s) containing the reference id or anchor from the surrounding text.
        """
        ref_subst_text = re.sub(f'#{ref_id}', 'this file', surrounding_text)
        sentences = self.naive_sentence_tokenizer(ref_subst_text)
        ret = ""
        for sentence in sentences:
            if ref_id in sentence or 'this file' in sentence:
                if sentence not in ret:
                    ret += sentence.strip() + " "
        return ret.strip()

    def get_surrounding_text(self, element):
        """
        Extracts text surrounding the element (including parent and siblings) for more context.
        For HTML, this collects the parent text and inline elements.
        """
        parent = element.getparent()
        if parent is None:
            return "No parent element found"
        parent_text = []
        if parent.text:
            parent_text.append(parent.text.strip())
        for child in parent:
            if child.tag == 'a':
                link_text = child.text_content() if child.text_content() else ''
                href = child.get('href')
                if href:
                    parent_text.append(f"{link_text} [href={href}]")
                else:
                    parent_text.append(link_text)
            elif child.tag == 'xref':
                xref_text = child.text_content() if child.text_content() else '[xref]'
                rid = child.get('rid')
                if rid:
                    parent_text.append(f"{xref_text} [rid={rid}]")
                else:
                    parent_text.append(xref_text)
            if child.tail:
                parent_text.append(child.tail.strip())
        surrounding_text = " ".join(parent_text)
        return re.sub(r"[\s\n]+", " ", surrounding_text)

    def extract_publication_title(self, raw_data):
        """
        Extract the publication title from the HTML content.

        :return: str — the publication title.
        """
        self.logger.info("Extracting publication title from HTML")
        soup = BeautifulSoup(raw_data, "html.parser")
        title_tag = soup.find('title')

        h1 = soup.find("h1", class_="article-title")
        if h1:
            return h1.get_text(strip=True)
        elif title_tag:
            return title_tag.get_text(strip=True)
        else:
            self.logger.warning("No publication title found in the HTML data.")
            return "No title found"
