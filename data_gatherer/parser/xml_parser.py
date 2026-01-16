from data_gatherer.retriever.xml_retriever import xmlRetriever
from data_gatherer.parser.base_parser import *
from data_gatherer.retriever.embeddings_retriever import EmbeddingsRetriever
from lxml import etree
import os
import pandas as pd
import json
import regex as re


class XMLParser(LLMParser):

    def __init__(self, open_data_repos_ontology, logger, log_file_override=None, full_document_read=True,
                 prompt_dir="data_gatherer/prompts/prompt_templates",
                 llm_name=None, save_dynamic_prompts=False, save_responses_to_cache=False, use_cached_responses=False,
                 use_portkey=True, embeddings_model_name=None, embeds_cache_read=False, embeds_cache_write=False):

        super().__init__(open_data_repos_ontology, logger, log_file_override=log_file_override,
                         full_document_read=full_document_read, prompt_dir=prompt_dir,
                         llm_name=llm_name, save_dynamic_prompts=save_dynamic_prompts,
                         save_responses_to_cache=save_responses_to_cache,
                         use_cached_responses=use_cached_responses, use_portkey=use_portkey
                         )

        self.logger = logger
        self.logger.info(f"Initializing xmlRetriever with model: {embeddings_model_name}")
        self.retriever = xmlRetriever(self.logger, publisher='PMC')

        self.embeddings_retriever = EmbeddingsRetriever(
            model_name=embeddings_model_name,
            logger=self.logger,
            read_cache=embeds_cache_read,
            write_cache=embeds_cache_write
        )

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
                    # self.logger.info(f"Extracted paragraph: {paragraphs[-1]}")

        return paragraphs
    
    def extract_sections_from_text(self, xml_content: str) -> list[dict]:
        """
        alias for extract_sections_from_xml
        """
        if isinstance(xml_content, str):
            xml_content = etree.fromstring(xml_content.encode('utf-8'))
        
        return self.extract_sections_from_xml(xml_content)

    def from_section_to_text_content(self, sect_element) -> str:
        """
        Convert a section ET element to plain text content.

        Args:
            sect_element: lxml.etree.Element — section element from extract_sections_from_xml.

        Returns:
            str — plain text content of the section.
        """

        if not isinstance(sect_element, etree._Element):
            raise TypeError(f"Invalid section element type: {type(sect_element)}. Expected lxml.etree.Element.")

        str_cont_ret = ''

        # extract the text from the data availability section
        for elem in sect.iter():
            if elem.text and elem.text not in str_cont_ret:
                str_cont_ret += ' ' + elem.text + ' '
            if elem.tail and elem.tail not in str_cont_ret:
                str_cont_ret += ' ' + elem.tail + ' '
            if elem.tag == 'ext-link' and elem.get('{http://www.w3.org/1999/xlink}href') not in str_cont_ret:
                str_cont_ret += ' ' + elem.get('{http://www.w3.org/1999/xlink}href') + ' '
            if elem.tag == 'xref' and elem.text not in str_cont_ret:
                str_cont_ret += ' ' + elem.text + ' '
            
            # table elements 
            if elem.tag == 'table':
                table_text = self.table_to_text(elem)
                if table_text not in str_cont_ret:
                    str_cont_ret += ' ' + table_text + ' '

        return str_cont_ret        

    def extract_sections_from_xml(self, xml_root) -> list[dict]:
        """
        Extract sections from an XML document.

        Args:
            xml_root: lxml.etree.Element — parsed XML root.

        Returns:
            List of dicts with 'section_title' and 'sec_type'.
        """
        sections = []
        self.logger.info(f"Function_call: extract_sections_from_xml(xml_root) with type {type(xml_root)}")

        if not isinstance(xml_root, etree._Element):
            raise TypeError(f"Invalid XML root type: {type(xml_root)}. Expected lxml.etree.Element.")

        # Find all section-like elements (sec, notes, ack)
        sec_elements = xml_root.findall(".//sec")
        notes_elements = xml_root.findall(".//notes")
        
        all_sections = sec_elements + notes_elements
        self.logger.debug(f"Found {len(sec_elements)} <sec> blocks, {len(notes_elements)} <notes> blocks")
        self.logger.debug(f"Total {len(all_sections)} section-like blocks in XML")

        # Iterate over all section blocks
        for sec_idx, sec in enumerate(all_sections):
            self.logger.debug(f"Processing section {sec_idx + 1}/{len(all_sections)} (tag: {sec.tag})")
            
            # Handle different element types
            if sec.tag == "sec":
                sec_type = sec.get("sec-type", "unknown")
            elif sec.tag == "notes":
                sec_type = sec.get("notes-type", "notes")
            else:
                sec_type = sec.tag
                
            self.logger.debug(f"Section type: '{sec_type}'")
            
            title_elem = sec.find("title")
            section_title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No Title"
            self.logger.debug(f"Section title: '{section_title}'")
            parent_section_title = section_title
            
            section_text_from_paragraphs = f'{section_title}\n'
            section_rawtxt_from_paragraphs = ''

            # Find all paragraphs in this section
            paragraphs = sec.findall(".//p")
            self.logger.debug(f"Found {len(paragraphs)} paragraphs in section '{section_title}'")

            for p_idx, p in enumerate(paragraphs):
                self.logger.debug(f"Processing paragraph {p_idx + 1}/{len(paragraphs)} in section '{section_title}'")
                parent_section = p.getparent()
                grandparent_section = parent_section.getparent() if parent_section is not None else None
                if parent_section is not None and parent_section != sec:
                    self.logger.debug(f"We've entered a different section: {parent_section} != {sec}, so break out of the loop")
                    break
                elif grandparent_section is not None and grandparent_section.find("title") is not None:
                    title_elem = grandparent_section.find("title")
                    title_text = title_elem.text if title_elem is not None and title_elem.text is not None else ""
                    section_title = title_text + " > " + parent_section_title
                
                itertext = " ".join(p.itertext()).strip()
                self.logger.debug(f"Paragraph itertext length: {len(itertext)} chars")

                if len(itertext) >= 5:
                    section_text_from_paragraphs += "\n" + itertext + "\n"
                    self.logger.debug(f"Added itertext to section_text_from_paragraphs (total clean text length now: {len(section_text_from_paragraphs)})")
                else:
                    self.logger.debug(f"Skipped itertext (too short: {len(itertext)} chars)")

                para_text = etree.tostring(p, encoding="unicode", method="xml").strip()
                self.logger.debug(f"Paragraph XML length: {len(para_text)} chars")

                if len(para_text) >= 5:  # avoid tiny/junk paragraphs
                    section_rawtxt_from_paragraphs += "\n" + para_text + "\n"
                    self.logger.debug(f"Added XML to section_rawtxt_from_paragraphs (total raw text length now: {len(section_rawtxt_from_paragraphs)})")
                else:
                    self.logger.debug(f"Skipped XML paragraph (too short: {len(para_text)} chars)")
            
            if section_title.lower().strip() == section_text_from_paragraphs.lower().strip():
                self.logger.debug(f"Section skipped for not having valid paragraphs.")
                continue
                
            elif section_text_from_paragraphs in [sect['sec_txt_clean'] for sect in sections]:
                self.logger.debug(f"Section skipped for being a duplicate.")
                continue

            # Create section dictionary
            section_dict = {
                "sec_txt": section_rawtxt_from_paragraphs,
                "section_title": section_title,
                "sec_type": sec_type,
                "sec_txt_clean": section_text_from_paragraphs,
                "sec_txt_objs": paragraphs
            }
            
            sections.append(section_dict)
            self.logger.debug(f"Added section '{section_title}' (tag: {sec.tag}) to results. Final lengths - raw: {len(section_rawtxt_from_paragraphs)}, clean: {len(section_text_from_paragraphs)}")

        self.logger.info(f"Extracted {len(sections)} sections from XML.")
        self.logger.debug(f"Section titles extracted: {[s['section_title'] for s in sections]}")
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
            pub_title = title.text.strip() if title is not None and title.text is not None else None
            return pub_title

        except etree.XMLSyntaxError as e:
            self.logger.error(f"Error parsing XML: {e}")
            return None

    def from_sections_to_corpus(self, sections, max_tokens=None, skip_rule_based_retrieved_elm=False):
        """
        Convert structured XML sections to a flat corpus of documents for embeddings retrieval.
        This method takes the output from extract_sections_from_xml (list of dicts) and converts it
        to a list of strings suitable for embeddings, with intelligent token-aware processing.
        
        :param sections: list of dict — the sections from extract_sections_from_xml
        :return: list of dicts - same attributes as input but with 'sec_txt' chunked to fit token limits.
        """
        self.logger.info(f"Converting {len(sections)} XML sections to embeddings corpus")

        # Get model token limits from the initialized retriever
        if max_tokens is None:
            max_tokens = self.embeddings_retriever.model.get_max_seq_length()
            self.logger.debug(f"Using model max sequence length: {max_tokens} tokens")
        effective_max_tokens = int(max_tokens * 0.95)
        self.logger.debug(f"Effective max tokens per section: {effective_max_tokens}")
        
        # Reserve some tokens for the query and model overhead (typically 10-20% buffer)
        effective_max_tokens = int(max_tokens * 0.95)  # 95% of max to be safe
        self.logger.debug(f"Effective max tokens per section: {effective_max_tokens}")

        self.skip_text_matching = self.data_availability_cont_str if skip_rule_based_retrieved_elm else []
        self.logger.debug(f"Skipping rule-based retrieved elements: {len(self.skip_text_matching)}")
        
        corpus_documents = []
        for i, section_dict in enumerate(sections):
            if not section_dict or not isinstance(section_dict, dict):
                self.logger.info(f"Skipping invalid section at index {i}")
                continue
            section_title = section_dict.get('section_title', 'n.a.')
            section_paragraphs = section_dict.get('sec_txt_objs', [])
            self.logger.debug(f"Processing section '{section_title}' (i:{i}) with {len(section_paragraphs)} paragraphs")

            if not section_paragraphs:
                self.logger.debug(f"Skipping empty section '{section_title}' (i:{i})")
                continue

            if skip_rule_based_retrieved_elm:
                self.logger.debug(f"Skipping rule-based retrieved elements: {len(self.skip_text_matching)}")
                sec_cont = section_dict.get('sec_txt', '')
                if sec_cont in self.skip_text_matching:
                    self.logger.info(f"Skipping section at index {i} as it matches rule-based retrieved elements")
                    continue

            chunks_created = []
            self.logger.debug(f"Starting chunk creation for section '{section_title}'")
            for p_idx, paragraph in enumerate(section_paragraphs):
                self.logger.debug(f"Processing paragraph {p_idx + 1}/{len(section_paragraphs)} in section '{section_title}', type: {type(paragraph)}")
                
                try:
                    if hasattr(paragraph, 'itertext'):
                        para_text = " ".join(paragraph.itertext()).strip()
                    else:
                        para_text = str(paragraph).strip()

                except Exception as e:
                    self.logger.warning(f"Error extracting text from paragraph {p_idx}: {e}")
                    continue

                if len(para_text) < 5:
                    self.logger.debug(f"Skipping short paragraph {p_idx} in section '{section_title}'")
                    continue

                normalized_para = re.sub(r'\s+', ' ', para_text.strip())

                try:
                    para_tokens = self.embeddings_retriever.cnt_tokens(normalized_para)
                except Exception:
                    para_tokens = len(normalized_para) // 4
                    
                if para_tokens > effective_max_tokens:
                    self.logger.debug(f"Paragraph {p_idx} in section '{section_title}' exceeds token limit ({para_tokens} > {effective_max_tokens}), splitting...")
                    para_chunks = self._intelligent_chunk_section(normalized_para, effective_max_tokens)
                    self.logger.debug(f"para_chunks for paragraph {p_idx}: {para_chunks}")
                else:
                    para_chunks = [normalized_para]
                    self.logger.debug(f"para_chunks for paragraph {p_idx}: {para_chunks}")

                for chunk_text in para_chunks:
                    self.logger.debug(f"Creating chunk {len(chunks_created) + 1} for paragraph {p_idx}: '{chunk_text[:80]}...' (tokens: {self.embeddings_retriever.cnt_tokens(chunk_text) if hasattr(self.embeddings_retriever, 'cnt_tokens') else 'N/A'})")
                    chunk_doc = section_dict.copy()
                    chunk_doc['sec_txt'] = section_title + "\n" + chunk_text
                    chunk_doc['sec_txt_clean'] = section_title + "\n" + chunk_text
                    chunk_doc['text'] = section_title + "\n" + chunk_text
                    chunk_doc['chunk_id'] = len(chunks_created) + 1
                    chunk_doc['contains_id_pattern'] = any(re.search(pattern, chunk_text, re.IGNORECASE) for pattern in self.id_patterns)
                    chunks_created.append(chunk_doc)
                    self.logger.debug(f"chunks_created now has {len(chunks_created)} items")

            corpus_documents.extend(chunks_created)
            diff_chunk = len(chunks_created) != len(section_paragraphs)
            self.logger.info(f"Section '{section_title}' split into {len(chunks_created)} chunks from {len(section_paragraphs)} paragraphs") if diff_chunk else None
        
        # Remove duplicates based on normalized text content and merge section titles
        self.logger.info(f"Pre-deduplication: {len(corpus_documents)} corpus documents")
        
        unique_documents = []
        seen_texts = {}  # Changed to dict to track documents by text content
        
        for doc in corpus_documents:
            # Use normalized text as the deduplication key
            text_key = doc.get('text', '').strip().lower()
            current_section_title = doc.get('section_title', '').strip()
            
            # Check if we've seen this exact text before
            if text_key and text_key not in seen_texts:
                seen_texts[text_key] = doc
                unique_documents.append(doc)
                self.logger.debug(f"Added new document with section title: '{current_section_title}'")
            elif text_key:
                # Found duplicate content - check if section title is different
                existing_doc = seen_texts[text_key]
                existing_section_title = existing_doc.get('section_title', '').strip()
                
                if current_section_title and current_section_title != existing_section_title:
                    # Concatenate section titles if they are different
                    if current_section_title not in existing_section_title:
                        concatenated_title = f"{existing_section_title} | {current_section_title}"
                        existing_doc['section_title'] = concatenated_title
                        self.logger.debug(f"Merged section titles: '{existing_section_title}' + '{current_section_title}' → '{concatenated_title}'")
                    else:
                        self.logger.debug(f"Section title '{current_section_title}' already included in existing title")
                else:
                    self.logger.debug(f"Skipping duplicate content with same section title: '{text_key[:50]}...'")
        
        self.logger.info(f"XML sections converted: {len(sections)} sections → {len(unique_documents)} unique corpus documents (processed {len(corpus_documents) - len(unique_documents)} duplicates with title merging)")
        return unique_documents

    def _intelligent_chunk_section(self, section_text, max_tokens):
        """
        Intelligently chunk a paragraph of text into smaller parts that fit within the token limit.
        This method attempts to split by sentences or sub-paragraphs while respecting token limits.
        :param section_text: str — the full text of the paragraph to be chunked
        :param max_tokens: int — the maximum number of tokens allowed per chunk
        :return: list of str — list of text chunks fitting within token limits
        """
        self.logger.debug(f"Input section length: {len(section_text)} chars")
        # Simple sentence tokenizer (could be replaced with a more sophisticated one if needed)
        sentences = re.split(r'(?<=[.!?]) +', section_text)
        self.logger.debug(f"Split into {len(sentences)} sentences")
        chunks = []
        current_chunk = ""
        for i, sentence in enumerate(sentences):
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            try:
                test_tokens = self.embeddings_retriever.cnt_tokens(test_chunk)
            except Exception:
                test_tokens = len(test_chunk) // 4
            if test_tokens <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        self.logger.debug(f"Final result: {len(chunks)} chunks")
        return chunks

    def parse_data(self, api_data, publisher=None, current_url_address=None,
                   raw_data_format='XML', article_file_dir='tmp/raw_files/', section_filter=None,
                   prompt_name='GPT_FewShot', use_portkey=True, semantic_retrieval=False, top_k=2, 
                   response_format=dataset_response_schema_gpt, dedup=True, brute_force_RegEx_ID_ptrs=False,
                   article_id=None):
        """
        Parse the API data and extract relevant links and metadata.

        :param api_data: The raw API data (XML or HTML) to be parsed.

        :param publisher: The publisher name or identifier.

        :param current_url_address: The current URL address being processed.

        :param raw_data_format: The format of the raw data ('XML' or 'HTML').

        :param dedup: Whether to deduplicate the extracted snippets.

        :param brute_force_RegEx_ID_ptrs: Whether to use brute force regular expression matching for ID patterns.

        :return: A DataFrame containing the extracted links and links to metadata - if repo is supported. Add support for unsupported repos in the ontology.

        """
        out_df = None
        # Check if api_data is a string, and convert to XML if needed
        self.logger.info(f"Function call: parse_data(api_data({type(api_data)}), {publisher}, {current_url_address}, "
                         f"{raw_data_format})")
        self.publisher = publisher

        if isinstance(api_data, str):
            try:
                if os.path.exists(api_data):
                    with open(api_data, 'rb') as f:
                        api_data = f.read()
                    api_data = etree.fromstring(api_data)
                else:
                    if api_data.lstrip().startswith('<?xml'):
                        api_data = etree.fromstring(api_data.encode('utf-8'))
                    else:
                        api_data = etree.fromstring(api_data)
                self.logger.info("api_data converted to lxml element")
            except Exception as e:
                self.logger.error(f"Error parsing API data: {e}")
                return None

        filter_supp = section_filter == 'supplementary_material' or section_filter is None
        filter_das = section_filter == 'data_availability_statement' or section_filter is None

        if isinstance(api_data, etree._Element):
            self.title = self.extract_publication_title(api_data)

            if filter_supp is None or filter_supp:
                supplementary_material_links = self.extract_href_from_supplementary_material(api_data,
                                                                                             current_url_address)
                supplementary_material_metadata = self.extract_supplementary_material_refs(api_data,
                                                                                           supplementary_material_links)
            else:
                supplementary_material_metadata = pd.DataFrame()
            self.logger.debug(f"supplementary_material_metadata: {supplementary_material_metadata}")

            self.logger.warning(f"Semantic Retrieval Enabled, but not needed for full-document-read method") if semantic_retrieval and self.full_document_read else None

            if not self.full_document_read:
                if filter_das is None or filter_das:
                    output_fmt = 'list' if ('local' in self.llm_name.lower() or 'hf-' in self.llm_name.lower()) else 'text'
                    data_availability_cont = self.retrieve_relevant_content(
                                api_data,
                                semantic_retrieval=semantic_retrieval,
                                top_k=top_k,
                                article_id=article_id,
                                skip_rule_based_retrieved_elm=dedup,
                                include_snippets_with_ID_patterns=brute_force_RegEx_ID_ptrs,
                                output_format=output_fmt
                            )
                    
                    augmented_dataset_links = []
                    if isinstance(data_availability_cont, list):
                        for das_content in data_availability_cont:
                            augmented_dataset_links.extend(self.extract_datasets_info_from_content(
                                das_content, self.open_data_repos_ontology['repos'], model=self.llm_name,
                                temperature=0, prompt_name=prompt_name, response_format=response_format))

                    else:
                        augmented_dataset_links = self.process_data_availability_text(data_availability_cont,
                                                                                  prompt_name=prompt_name,
                                                                                  response_format=response_format)

                    self.logger.debug(f"Content of augmented_dataset_links: {augmented_dataset_links}")
                    dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

                else:
                    self.logger.info(
                        f"Skipping data availability statement extraction as per section_filter: {section_filter}")
                    dataset_links_w_target_pages = []

                available_data = pd.DataFrame(dataset_links_w_target_pages)
                # Create a DataFrame from the dataset links union supplementary material links
                out_df = pd.concat([available_data,supplementary_material_metadata], ignore_index=True)  # check index error here
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

                # Only set metadata if DataFrame is not empty
                if len(out_df) > 0:
                    out_df['pub_title'] = self.title
                    out_df['source_url'] = current_url_address
                    out_df['raw_data_format'] = raw_data_format
                else:
                    out_df = pd.DataFrame(columns=['pub_title', 'source_url', 'raw_data_format'])
                    self.logger.warning(f"No datasets found in the document")

                return out_df

            else:
                # Extract links from entire webpage
                if self.full_document_read and (filter_das is None or filter_das):
                    self.logger.info(f"Extracting links from full XML content.")

                    preprocessed_data = self.normalize_XML(api_data)

                    self.logger.debug(f"Preprocessed data: {preprocessed_data}")

                    # Extract dataset links from the entire text
                    augmented_dataset_links = self.extract_datasets_info_from_content(preprocessed_data,
                                                                                      self.open_data_repos_ontology[
                                                                                          'repos'],
                                                                                      model=self.llm_name,
                                                                                      temperature=0,
                                                                                      prompt_name=prompt_name,
                                                                                      response_format=response_format)

                    self.logger.info(f"Augmented dataset links: {augmented_dataset_links}")

                    dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

                    # Create a DataFrame from the dataset links union supplementary material links
                    out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages), supplementary_material_metadata])
                else:
                    out_df = supplementary_material_metadata

                self.logger.info(
                    f"Dataset Links type: {type(out_df)} of len {len(out_df)}, with cols: {out_df.columns}")

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

                # Only set metadata if DataFrame is not empty
                if len(out_df) > 0:
                    out_df['source_url'] = current_url_address
                    out_df['pub_title'] = self.title
                    out_df['raw_data_format'] = raw_data_format
                else:
                    out_df = pd.DataFrame(columns=['source_url', 'pub_title', 'raw_data_format'])
                    self.logger.warning(f"No datasets found in the document")

                return out_df
        else:
            raise TypeError(f"Invalid API data type: {type(api_data)}. Expected lxml.etree.Element.")

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
            normalized_xml = etree.tostring(xml_root, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode(
                'utf-8')

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

        :return: DataFrame containing href links and their context.

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
                        surrounding_text = re.sub(r"[\s\n]+", "  ", " ".join(
                            parent_p.itertext()).strip())  # Gets all text within the <p> tag
                    else:
                        surrounding_text = "No surrounding text found"

                    # Extract the full description within the <p> tag if available
                    caption_element = supplementary_material_parent.find(".//caption/p")
                    caption = " ".join(
                        caption_element.itertext()).strip() if caption_element is not None else "No description"

                    # Log media attributes and add to results
                    self.logger.debug(f"Extracted media item with href: {href}")
                    self.logger.debug(f"Source url: {current_url_address}")
                    self.logger.debug(f"Supplementary material title: {title}")
                    self.logger.debug(f"Content type: {content_type}, ID: {media_id}")
                    self.logger.debug(f"Surrounding text for media: {surrounding_text}")
                    self.logger.debug(f"Caption: {caption}")
                    self.logger.debug(f"Download_link: {download_link}")

                    if href and href not in [item['link'] for item in hrefs]:
                        hrefs.append({
                            'link': href,
                            'source_url': current_url_address,
                            'download_link': download_link,
                            'title': title,
                            'content_type': content_type,
                            'id': media_id,
                            'caption': caption,
                            'description': surrounding_text,
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
        return pd.DataFrame(hrefs)

    def extract_supplementary_material_refs(self, api_xml, supplementary_material_links):
        """
        Extract metadata from xrefs to supplementary material ids in the XML.
        """
        self.logger.info(f"Function_call: extract_supplementary_material_refs(api_xml, supplementary_material_links)")
        for i, row in supplementary_material_links.iterrows():
            # Find the <href> elements that reference the supplementary material <a href="#id">
            context_descr = ""
            href_id = row['id']
            self.logger.debug(f"Processing href_id: {href_id} for supplementary material links.")
            xref_elements = api_xml.xpath(f".//xref[@rid='{href_id}']")
            self.logger.debug(f"Found {len(xref_elements)} xref elements href_id: {href_id}.")
            # Iterate through each xref element:
            for xref in xref_elements:
                # Extract the sentence that contains the xref
                surrounding_text = self.get_surrounding_text(xref)
                text_segment = self.get_sentence_segment(surrounding_text, href_id)
                if text_segment not in context_descr:
                    context_descr += text_segment + "\n"
            # Add the context description to the supplementary material links DataFrame
            self.logger.info(f"Extracted context_descr for xref {href_id}: {context_descr}")
            supplementary_material_links.at[i, 'context_description'] = context_descr.strip()
        return supplementary_material_links

    def get_sentence_segment(self, surrounding_text, rid):
        """
        Extract inter-period sentence segments containing the xref from the XML content.
        """
        ref_subst_text = re.sub(f'rid={rid}', 'this file', surrounding_text)

        # Split the surrounding text into sentences based on periods that do not end with an abbreviation
        target_sentences = self.naive_sentence_tokenizer(ref_subst_text)

        ret = ""
        for sentence in target_sentences:
            if rid in sentence or 'this file' in sentence:
                # Return the first sentence that contains the xref
                if sentence not in ret:
                    ret += sentence.strip() + " "

        return ret

    def naive_sentence_tokenizer(self, text):
        # Initial split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Merge sentences ending with abbreviations
        abbreviations = ('Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'e.g.', 'i.e.', 'Fig.', 'vs.', 'et al.')
        merged = []
        for s in sentences:
            if merged and any(merged[-1].endswith(abbr) for abbr in abbreviations):
                merged[-1] += ' ' + s
            else:
                merged.append(s)
        return merged

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
                rid = child.get('rid')
                if rid:
                    parent_text.append(f"{xref_text} [rid={rid}]")
                else:
                    parent_text.append(xref_text)
            # Add the tail text (text after the inline element)
            if child.tail:
                parent_text.append(child.tail.strip())

        # Join the list into a single string for readability
        surrounding_text = " ".join(parent_text)

        return re.sub(r"[\s\n]+(\s+)", "\1", surrounding_text)

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
        self.logger.debug(f"Function call: get_data_availability_text(api_xml({type(api_xml)})")

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
                if elem.text and elem.text is not None:
                    cont += ' '
                    cont += elem.text
                    cont += ' '
                if elem.tail and elem.tail is not None:
                    cont += ' '
                    cont += elem.tail
                    cont += ' '
                # also include the links in the data availability section
                if elem.tag == 'ext-link' and elem.get('{http://www.w3.org/1999/xlink}href') is not None:
                    cont += ' '
                    cont += elem.get('{http://www.w3.org/1999/xlink}href')
                    cont += ' '
                if elem.tag == 'xref' and elem.text is not None:
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

        self.logger.info(f"Data Availability len: {len(data_availability_cont)}, type: {type(data_availability_cont)}")
        self.logger.debug(f"Found data availability content: {data_availability_cont}")

        self.data_availability_cont_str = ''.join(data_availability_cont)
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

    def regex_match_id_patterns(self, xml_element, id_patterns=None):
        """XML-specific version that preserves structure if needed"""
        # Extract specific XML sections first, then apply regex
        if type(xml_element) != str:
            text_content = etree.tostring(xml_element, encoding='unicode', method='text')
        else:
            text_content = xml_element
        return super().regex_match_id_patterns(text_content, id_patterns)

    def extract_citations(self, xml_root):
        """
        Extract citations from XML reference sections.
        Returns a list of dicts with label, text, and external links (DOI, PMID, PMCID, etc).
        """
        citations = []
        # Find all <ref> elements (references)
        for ref in xml_root.findall('.//ref'):
            label = None
            citation_text = None
            external_links = {}

            # Extract label (e.g., [1], [2])
            label_elem = ref.find('label')
            if label_elem is not None and label_elem.text:
                label = label_elem.text.strip()

            # Extract citation text from <mixed-citation> or <element-citation>
            mixed_cit = ref.find('.//mixed-citation')
            elem_cit = ref.find('.//element-citation')
            if mixed_cit is not None:
                citation_text = ' '.join(mixed_cit.itertext()).strip()
                # Extract external links from <pub-id> children
                for pub_id in mixed_cit.findall('pub-id'):
                    id_type = pub_id.get('pub-id-type')
                    if id_type and pub_id.text:
                        external_links[id_type] = pub_id.text.strip()
            elif elem_cit is not None:
                citation_text = ' '.join(elem_cit.itertext()).strip()
                for pub_id in elem_cit.findall('pub-id'):
                    id_type = pub_id.get('pub-id-type')
                    if id_type and pub_id.text:
                        external_links[id_type] = pub_id.text.strip()

            # Also check for <ext-link> elements for direct URLs
            for ext_link in ref.findall('.//ext-link'):
                href = ext_link.get('{http://www.w3.org/1999/xlink}href')
                if href:
                    ext_type = ext_link.get('ext-link-type', 'url')
                    external_links[ext_type] = href

            citations.append({
                'label': label,
                'citation_text': citation_text,
                'external_links': external_links
            })

        return citations
        

    @staticmethod
    def is_tei_xml_static(xml_root):
        """
        Static version for router use.
        """
        if not isinstance(xml_root, etree._Element):
            return False
        tei_ns = "http://www.tei-c.org/ns/1.0"

        def has_tei_ns(elem):
            if not isinstance(elem, etree._Element):
                return False
            try:
                ns = etree.QName(elem).namespace
            except Exception:
                return False

            if ns == tei_ns:
                return True
            for child in elem:
                if has_tei_ns(child):
                    return True
            return False

        return has_tei_ns(xml_root)


class TEI_XMLParser(XMLParser):
    """
    Parser for TEI XML documents (from Grobid or similar).
    Extend this class with TEI-specific extraction logic as needed.
    """

    def extract_sections_from_xml(self, tei_xml):
        self.logger.info(f"Extracting sections from TEI XML. Type: {type(tei_xml)}")
        # Accept both str and etree.Element
        if isinstance(tei_xml, str):
            root = etree.fromstring(tei_xml.encode('utf-8'))
        elif isinstance(tei_xml, etree._Element):
            root = tei_xml
        else:
            raise TypeError(f"Invalid TEI XML type: {type(tei_xml)}. Expected str or lxml.etree.Element.")

        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        sections = []

        # Find all <div> elements in TEI namespace
        divs = root.findall(".//{http://www.tei-c.org/ns/1.0}div")
        for div in divs:
            # Section title: text node of <div> before any children
            section_title = (div.text or "").strip() or "No Title"
            sec_type = "tei_div"
            section_text_from_paragraphs = section_title + "\n"
            section_rawtxt_from_paragraphs = ""
            # Find all <p> in this <div>
            for p in div.findall(".//{http://www.tei-c.org/ns/1.0}p"):
                itertext = " ".join(p.itertext()).strip()
                if len(itertext) >= 5:
                    section_text_from_paragraphs += "\n" + itertext + "\n"
                para_text = etree.tostring(p, encoding="unicode", method="xml").strip()
                if len(para_text) >= 5:
                    section_rawtxt_from_paragraphs += "\n" + para_text + "\n"
            # If no <p>, but <div> has text, still add the section
            if section_text_from_paragraphs.strip() or section_rawtxt_from_paragraphs.strip():
                sections.append({
                    "sec_txt": section_rawtxt_from_paragraphs,
                    "section_title": section_title,
                    "sec_type": sec_type,
                    "sec_txt_clean": section_text_from_paragraphs
                })
        self.logger.info(f"Extracted {len(sections)} sections from TEI XML.")
        return sections

    def extract_reference_content(self, ref_id, tei_xml):
        self.logger.debug(f"Extracting reference content for ID: {ref_id}")
        root = etree.fromstring(tei_xml.encode('utf-8'))
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        ref_id_clean = ref_id.lstrip('#')
        bibl = root.xpath(f".//tei:biblStruct[@xml:id='{ref_id_clean}']", namespaces=ns)
        if bibl:
            bibl = bibl[0]
            analytic_title = bibl.find('.//tei:analytic/tei:title', namespaces=ns)
            analytic_title_str = analytic_title.text.strip() if analytic_title is not None and analytic_title.text else ""
            authors = bibl.findall('.//tei:analytic/tei:author/tei:persName', namespaces=ns)
            author_names = []
            for a in authors:
                surname = a.find('tei:surname', namespaces=ns)
                forename = a.find('tei:forename', namespaces=ns)
                name = ""
                if forename is not None and forename.text:
                    name += forename.text + " "
                if surname is not None and surname.text:
                    name += surname.text
                if name:
                    author_names.append(name.strip())
            author_str = ", ".join(author_names)
            monogr_title = bibl.find('.//tei:monogr/tei:title', namespaces=ns)
            monogr_title_str = monogr_title.text.strip() if monogr_title is not None and monogr_title.text else ""
            date = bibl.find('.//tei:monogr/tei:imprint/tei:date', namespaces=ns)
            date_str = date.text.strip() if date is not None and date.text else ""
            idno = bibl.find('.//tei:idno', namespaces=ns)
            idno_str = idno.text.strip() if idno is not None and idno.text else ""
            parts = []
            if author_str:
                parts.append(author_str)
            if analytic_title_str:
                parts.append(analytic_title_str)
            if monogr_title_str:
                parts.append(monogr_title_str)
            if date_str:
                parts.append(date_str)
            if idno_str:
                parts.append(idno_str)
            ref_content = ", ".join(parts)
            self.logger.debug(f"Extracted reference: {ref_content}")
            return ref_content if ref_content else ref_id
        return ref_id

    def extract_paragraphs(self, tei_xml, ref_substitutions=False):
        self.logger.info(f"Extracting paragraphs from TEI XML. Type: {type(tei_xml)}")
        root = etree.fromstring(tei_xml.encode('utf-8'))
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        paragraphs = []
        for p in root.xpath('.//tei:p', namespaces=ns):
            section_title = "No Title"
            parent = p.getparent()
            while parent is not None:
                if parent.tag.endswith('div'):
                    head = parent.find('tei:head', namespaces=ns)
                    if head is not None and head.text:
                        section_title = head.text.strip()
                        break
                parent = parent.getparent()
            if ref_substitutions:
                para_fragments = []
                for node in p.iter():
                    self.logger.debug(f"Processing node: {node.tag} with text: {node}")
                    if node.tag.endswith('ref') and node.get('target'):
                        ref_id = node.get('target')
                        ref_content = self.extract_reference_content(ref_id, tei_xml)
                        para_fragments.append(ref_content)
                        if node.tail:
                            para_fragments.append(node.tail.strip())
                    elif node is p:
                        if node.text:
                            para_fragments.append(node.text.strip())
                    elif node.tail and node.getparent() is p:
                        para_fragments.append(node.tail.strip())
                para_text = " ".join(para_fragments).strip()
            else:
                para_text = etree.tostring(p, encoding="unicode", method="text").strip()
            itertext = " ".join(p.itertext()).strip()
            if len(para_text) >= 5:
                paragraphs.append({
                    "section_title": section_title,
                    "text": para_text,
                })
        self.logger.info(f"Extracted {len(paragraphs)} paragraphs from TEI XML.")
        return paragraphs

    def extract_text(self, tei_xml):
        self.logger.info(f"Extracting text from TEI XML. Type: {type(tei_xml)}")
        paragraphs = self.extract_paragraphs(tei_xml, ref_substitutions=True)
        return "\n".join(paragraphs['text'] for paragraphs in paragraphs if 'text' in paragraphs)

    def extract_publication_title(self, root):
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        title_el = root.find('.//tei:analytic/tei:title', namespaces=ns)
        self.logger.info(f"Extracting publication title. Found element: {title_el}")
        if title_el is not None and title_el.text:
            return title_el.text.strip()
        title_el = root.find('.//tei:monogr/tei:title', namespaces=ns)
        self.logger.info(f"Extracting publication title. Found element: {title_el}")
        if title_el is not None and title_el.text:
            return title_el.text.strip()
        return ""

    def parse_data(self, api_data, publisher=None, current_url_address=None,
                   raw_data_format='XML', article_file_dir='tmp/raw_files/', section_filter=None,
                   prompt_name='GPT_FewShot', use_portkey=True, semantic_retrieval=False, top_k=2, 
                   response_format=dataset_response_schema_gpt, dedup=True, brute_force_RegEx_ID_ptrs=False,
                   article_id=None):
        """
        Parse the API data and extract relevant links and metadata.

        :param api_data: The raw API data (XML or HTML) to be parsed.

        :param publisher: The publisher name or identifier.

        :param current_url_address: The current URL address being processed.

        :param raw_data_format: The format of the raw data ('XML' or 'HTML').

        :param dedup: Whether to deduplicate the extracted snippets.

        :param brute_force_RegEx_ID_ptrs: Whether to use brute force regular expression matching for ID patterns.

        :param article_id: PMCID/DOI for tracing / cache save.

        :return: A DataFrame containing the extracted links and links to metadata - if repo is supported. Add support for unsupported repos in the ontology.

        """
        out_df = None
        # Check if api_data is a string, and convert to XML if needed
        self.logger.info(f"Function call: parse_data(api_data({type(api_data)}), {publisher}, {current_url_address}, "
                         f"{raw_data_format})")
        self.publisher = publisher

        if isinstance(api_data, str):
            try:
                if os.path.exists(api_data):
                    with open(api_data, 'rb') as f:
                        api_data = f.read()
                    api_data = etree.fromstring(api_data)
                else:
                    if api_data.lstrip().startswith('<?xml'):
                        api_data = etree.fromstring(api_data.encode('utf-8'))
                    else:
                        api_data = etree.fromstring(api_data)
                self.logger.info("api_data converted to lxml element")
            except Exception as e:
                self.logger.error(f"Error parsing API data: {e}")
                return None

        filter_supp = section_filter == 'supplementary_material' or section_filter is None
        filter_das = section_filter == 'data_availability_statement' or section_filter is None

        if isinstance(api_data, etree._Element):
            self.title = self.extract_publication_title(api_data)

            if filter_supp is None or filter_supp:
                supplementary_material_links = self.extract_href_from_supplementary_material(api_data,
                                                                                             current_url_address)
                supplementary_material_metadata = self.extract_supplementary_material_refs(api_data,
                                                                                           supplementary_material_links)
            else:
                supplementary_material_metadata = pd.DataFrame()
            self.logger.debug(f"supplementary_material_metadata: {supplementary_material_metadata}")

            self.logger.warning(f"Semantic Retrieval Enabled, but not needed for full-document-read method") if semantic_retrieval and self.full_document_read else None

            if not self.full_document_read:
                if filter_das is None or filter_das:
                    output_fmt = 'list' if ('local' in self.llm_name.lower() or 'hf-' in self.llm_name.lower()) else 'text'
                    data_availability_cont = self.retrieve_relevant_content(
                                api_data,
                                semantic_retrieval=semantic_retrieval,
                                top_k=top_k,
                                article_id=article_id,
                                skip_rule_based_retrieved_elm=dedup,
                                include_snippets_with_ID_patterns=brute_force_RegEx_ID_ptrs,
                                output_format=output_fmt
                            )
                    
                    augmented_dataset_links = []
                    if isinstance(data_availability_cont, list):
                        for das_content in data_availability_cont:
                            augmented_dataset_links.extend(self.extract_datasets_info_from_content(
                                das_content, self.open_data_repos_ontology['repos'], model=self.llm_name,
                                temperature=0, prompt_name=prompt_name, response_format=response_format))

                    else:
                        augmented_dataset_links = self.process_data_availability_text(data_availability_cont,
                                                                                  prompt_name=prompt_name,
                                                                                  response_format=response_format)

                    self.logger.debug(f"Content of augmented_dataset_links: {augmented_dataset_links}")
                    dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

                else:
                    self.logger.info(
                        f"Skipping data availability statement extraction as per section_filter: {section_filter}")
                    dataset_links_w_target_pages = []

                # Create a DataFrame from the dataset links union supplementary material links
                out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages), supplementary_material_metadata], ignore_index=True)  # check index error here
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
                if self.full_document_read and (filter_das is None or filter_das):
                    self.logger.info(f"Extracting links from full XML content.")

                    preprocessed_data = self.normalize_XML(api_data)

                    self.logger.debug(f"Preprocessed data: {preprocessed_data}")

                    # Extract dataset links from the entire text
                    augmented_dataset_links = self.extract_datasets_info_from_content(preprocessed_data,
                                                                                      self.open_data_repos_ontology[
                                                                                          'repos'],
                                                                                      model=self.llm_name,
                                                                                      temperature=0,
                                                                                      prompt_name=prompt_name,
                                                                                      response_format=response_format)

                    self.logger.info(f"Augmented dataset links: {augmented_dataset_links}")

                    dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

                    # Create a DataFrame from the dataset links union supplementary material links
                    out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages), supplementary_material_metadata])
                else:
                    out_df = supplementary_material_metadata

                self.logger.info(
                    f"Dataset Links type: {type(out_df)} of len {len(out_df)}, with cols: {out_df.columns}")

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

                # Only set metadata if DataFrame is not empty
                if len(out_df) > 0:
                    out_df['source_url'] = current_url_address
                    out_df['pub_title'] = self.title
                else:
                    out_df = pd.DataFrame(columns=['source_url', 'pub_title'])
                    self.logger.warning(f"No datasets found in the document")

                return out_df
        else:
            raise TypeError(f"Invalid API data type: {type(api_data)}. Expected lxml.etree.Element.")

class XMLRouter:
    """
    Routes XML parsing to the appropriate parser class based on XML type (TEI or not).
    """

    def __init__(self, open_data_repos_ontology, logger, **parser_kwargs):
        self.open_data_repos_ontology = open_data_repos_ontology
        self.logger = logger
        self.parser_kwargs = parser_kwargs

    def get_parser(self, xml_root):
        """
        Returns an instance of the appropriate parser (TEI_XMLParser or XMLParser)
        based on the XML content.
        """
        self.logger.info(f"Function_call: get_parser(xml_root) with type {type(xml_root)}")
        if not isinstance(xml_root, etree._Element) and isinstance(xml_root, str):
            if os.path.exists(xml_root):
                self.logger.info(f"Loading XML from file: {xml_root}")
                with open(xml_root, 'r', encoding='utf-8') as f:
                    xml_root = f.read()
            else:
                self.logger.info(f"Parsing XML string: {xml_root[:100]}...")
            try:
                xml_root = etree.fromstring(xml_root.encode('utf-8'))
            except etree.XMLSyntaxError as e:
                self.logger.error(f"Failed to parse XML root: {e}")
                raise ValueError("Invalid XML root provided.")
        self.logger.info(f"Function_call: is_tei_xml_static(xml_root) with type {type(xml_root)}")
        if XMLParser.is_tei_xml_static(xml_root):
            self.logger.info("Detected TEI XML. Using TEI_XMLParser.")
            return TEI_XMLParser(self.open_data_repos_ontology, self.logger, **self.parser_kwargs)
        else:
            self.logger.info("Detected non-TEI XML. Using XMLParser.")
            return XMLParser(self.open_data_repos_ontology, self.logger, **self.parser_kwargs)
