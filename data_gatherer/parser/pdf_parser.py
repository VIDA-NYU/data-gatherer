from data_gatherer.parser.base_parser import *
import re
import pandas as pd
import pymupdf, fitz
import unicodedata
from collections import Counter
from data_gatherer.llm.llm_client import LLMClient_dev

class PDFParser(LLMParser):
    """
    Custom PDF parser that has only support for PDF or HTML-like input
    """

    def __init__(self, open_data_repos_ontology, logger, log_file_override=None, full_document_read=True,
                 prompt_dir="data_gatherer/prompts/prompt_templates",
                 llm_name=None, save_dynamic_prompts=False, save_responses_to_cache=False, use_cached_responses=False,
                 use_portkey=True):

        super().__init__(open_data_repos_ontology, logger, log_file_override=log_file_override,
                         full_document_read=full_document_read, prompt_dir=prompt_dir,
                         llm_name=llm_name, save_dynamic_prompts=save_dynamic_prompts,
                         save_responses_to_cache=save_responses_to_cache,
                         use_cached_responses=use_cached_responses, use_portkey=use_portkey
                         )

        self.logger = logger

    def remove_temp_file(self, file_path):
        """
        Remove temporary file if it exists.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"Temporary file {file_path} removed successfully.")
            else:
                self.logger.warning(f"Temporary file {file_path} does not exist.")
        except Exception as e:
            self.logger.error(f"Error removing temporary file {file_path}: {e}")

    def extract_sections_from_text(self, pdf_content: str) -> list[dict]:
        """
        Heuristically extract section titles and texts from normalized PDF content.
        Treat the 'References' section separately as a vocabulary, not as a regular section.

        Args:
            pdf_content: str — normalized plain-text PDF content.

        Returns:
            List[dict] — list of sections with 'section_title' and 'text'.
            If a 'References' section is found, it is returned as a separate dict with key 'references'.
        """
        self.logger.info("Extracting sections using heuristics.")

        lines = pdf_content.splitlines()
        sections = []
        current_section = {"section_title": "Start", "text": ""}
        candidate_pattern = re.compile(r"^\s*(\d{1,2}(\.\d{1,2})*\.?\s*)?[A-Z][\w\s,\-:]{3,80}$")
        references_section = None
        in_references = False

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Detect start of References section
            if re.match(r"^references$", line_stripped, re.IGNORECASE):
                # Save previous section
                if current_section["text"].strip():
                    sections.append(current_section)
                # Start collecting references
                in_references = True
                references_section = {"section_title": "references", "text": ""}
                continue

            if in_references:
                references_section["text"] += line + "\n"
                continue

            # Check if line is a candidate section heading
            if candidate_pattern.match(line_stripped) and not line_stripped.endswith('.'):
                # Start new section
                if current_section["text"].strip():
                    sections.append(current_section)
                current_section = {
                    "section_title": line_stripped,
                    "text": ""
                }
            else:
                current_section["text"] += line + "\n"

        if not in_references and current_section["text"].strip():
            sections.append(current_section)

        if references_section:
            sections.extend(self.split_references(references_section['text']))
        else:
            self.logger.info("No References section found in the document.")

        self.logger.info(f"Extracted {len(sections)} sections (including references if present).")
        self.logger.info(f"sections: {sections}")
        return sections

    def split_references(self, references_text):
        """
        Split a references section into individual references, robust to different bibstyles.
        Uses heuristics: a new reference likely starts with a capitalized word (author) and a year (possibly with a/b/c),
        and may span multiple lines.
        Returns a list of dicts with 'section_title' and 'text'.
        """
        self.logger.info("Splitting references section into individual references (robust heuristic).")
        # Pattern: line starts with capitalized word(s), then year (possibly in parentheses), then period or colon
        # e.g. "Smith J. 2020.", "Smith J, Doe A. (2020).", "Smith J. 2020a:"
        # Accepts: "Ahnesjö I. 1992a. ...", "Alonso-Alvarez C, Velando A. 2012. ..."
        # Also allow for lines that start with a number (for numbered bibs)
        ref_start = re.compile(r"^(\d+\.\s*)?([A-ZÅÄÖ][\w\-']+(\s+[A-Z][\w\-']+)*[\s,\.]+)+((\(|\s)?\d{4}[a-zA-Z]?\)?)[\.:\s]")
        lines = references_text.strip().splitlines()
        refs = []
        current_ref = ''
        for line in lines:
            if ref_start.match(line.strip()):
                if current_ref.strip():
                    refs.append(current_ref.strip())
                current_ref = line.strip()
            else:
                # Continuation of previous reference
                current_ref += ' ' + line.strip()
        if current_ref.strip():
            refs.append(current_ref.strip())
        # Save as dicts
        refs = [{"section_title": f"Ref_{i + 1}", "text": ref} for i, ref in enumerate(refs) if ref]
        self.logger.info(f"Split into {len(refs)} individual references.")
        return refs

    def extract_text_from_pdf(self, file_path, start_page=0, end_page=None):
        """
        Extracts plain text from a PDF file using PyMuPDF.

        Parameters:
            file_path (str): Path to the PDF file.
            start_page (int): Page number to start from (0-indexed).
            end_page (int or None): Page number to end at (exclusive). If None, reads till the end.

        Returns:
            str: Extracted and normalized text content.
        """

        self.logger.info(f"Extracting text from PDF using PyMuPDF: {file_path}")
        try:
            doc = fitz.open(file_path)
            num_pages = len(doc)
            end_page = end_page or num_pages

            text_chunks = []
            for page_num in range(start_page, min(end_page, num_pages)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text:
                    # Normalize unicode (e.g., diacritics, smart quotes)
                    text = unicodedata.normalize("NFKC", text)
                    text_chunks.append(text)

            return ("\nNewPage\n").join(text_chunks)

        except Exception as e:
            self.logger.error(f"Failed to read PDF {file_path}: {e}")
            return ""
        finally:
            if 'doc' in locals():
                doc.close()

    def pdf_to_markdown_with_links(self, file_path):
        doc = fitz.open(file_path)
        markdown = ""

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            links = page.get_links()
            link_map = {}  # bbox -> uri

            for link in links:
                if link.get("uri") and link.get("from"):
                    link_map[tuple(link["from"])] = link["uri"]

            for block in blocks:
                if block["type"] != 0:
                    continue  # Skip images for now

                for line in block.get("lines", []):
                    line_md = ""
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue

                        bbox = tuple(span["bbox"])
                        uri = next((u for b, u in link_map.items() if fitz.Rect(b).intersects(fitz.Rect(bbox))), None)

                        # Basic formatting (bold for large font size)
                        font_size = span.get("size", 10)
                        if font_size > 16:
                            text = f"**{text}**"

                        # Wrap with hyperlink if it's linked
                        if uri:
                            text = f"[{text}]({uri})"

                        line_md += text + " "

                    markdown += line_md.strip() + "\n"

            markdown += "\n---\n"  # Page separator

        doc.close()
        return markdown

    def normalize_extracted_text(self, text, top_n=10, bottom_n=10, repeat_thresh=0.5):
        self.logger.info("Normalizing extracted text. Initial length: " f"{len(text)} characters")

        # More flexible page splitting
        pages = re.split(r'\n{2,}', text)
        self.logger.info(f"Number of pages detected: {len(pages)}")

        header_footer_lines = []

        for page in pages:
            lines = page.strip().splitlines()
            header_footer_lines.extend([line.strip() for line in lines[:top_n]])
            header_footer_lines.extend([line.strip() for line in lines[-bottom_n:]])

        counts = Counter(header_footer_lines)
        page_count = len(pages)
        threshold = repeat_thresh * page_count

        lines_to_remove = {line for line, count in counts.items() if count >= threshold}
        for line in lines_to_remove:
            self.logger.info(f"Removing frequent header/footer: '{line}'")

        cleaned_lines = []
        for line in text.splitlines():
            if line.strip() not in lines_to_remove:
                cleaned_lines.append(line)

        normalized_text = "\n".join(cleaned_lines)

        normalized_text = re.sub(r'\n?Page\s+\d+\s*\n?', '\n', normalized_text, flags=re.IGNORECASE)

        ret_text = self.merge_pdf_line_breaks(normalized_text)

        self.logger.info(f"Normalized text length: {len(ret_text)}")
        self.logger.debug(f"Normalized text: {ret_text}")
        return ret_text

    def merge_pdf_line_breaks(self, text):
        """
        Merge lines that are broken by hard PDF line breaks but do not end with sentence-ending punctuation.
        Also fix broken words split by hyphen and newline.
        Additionally, merge line breaks that occur within URLs (e.g., after 'http', 'https', 'www', or inside a domain).
        Keep newlines if the next character is a capital letter (likely a heading or new section).
        """
        # Fix broken words: e.g. "wo-\nrd" -> "word"
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

        # Remove line breaks within URLs (e.g., after http, https, www, or inside a domain)
        # This will join lines that are split in the middle of a URL
        text = re.sub(r'(https?://[^\s\n]+)\n([^\s\n]+)', r'\1\2', text)
        text = re.sub(r'(www\.[^\s\n]+)\n([^\s\n]+)', r'\1\2', text)
        # Also handle line breaks after domain dots (e.g., dx.doi.\norg)
        text = re.sub(r'(\.[a-z]{2,6})\n([a-zA-Z0-9])', r'\1\2', text)

        # Protect double newlines (paragraph breaks)
        text = text.replace('\n\n', '<PARA>')

        # Merge single newlines not after . ! ? and not followed by a capital letter
        text = re.sub(r'(?<![.!?])\n(?![A-Z])', ' ', text)

        # Restore paragraph breaks
        text = text.replace('<PARA>', '\n\n')
        return text

    def parse_data(self, file_path, publisher=None, current_url_address=None, raw_data_format='PDF',
                   file_path_is_temp=False, article_file_dir='tmp/raw_files/', prompt_name='GPT_FewShot', use_portkey=True, 
                   semantic_retrieval=False, top_k=2, section_filter=None, response_format=dataset_response_schema_gpt, 
                   dedup=True, brute_force_RegEx_ID_ptrs=False, article_id=None):
        """
        Parse the PDF file and extract metadata of the relevant datasets.

        :param file_path: The file_path to the PDF.

        :param current_url_address: The current URL address being processed.


        :param raw_data_format: The format of the raw data ('XML' or 'HTML').

        :param file_path_is_temp: Boolean indicating if the file_path is a temporary file.

        :return: A DataFrame containing the extracted links and links to metadata - if repo is supported. Add support for unsupported repos in the ontology.

        """
        out_df = None
        # Check if api_data is a string, and convert to XML if needed
        self.logger.info(f"Function call: parse_data({file_path}, {current_url_address}, "
                         f"{raw_data_format})")

        text_from_pdf = self.extract_text_from_pdf(file_path)
        preprocessed_data = self.normalize_extracted_text(text_from_pdf)

        self.logger.debug(f"Preprocessed data: {preprocessed_data}")

        self.logger.warning(f"Semantic Retrieval Enabled, but not needed for full-document-read method") if semantic_retrieval and self.full_document_read else None

        if self.full_document_read:
            self.logger.info(f"Extracting links from full content.")

            # Extract dataset links from the entire text
            augmented_dataset_links = self.extract_datasets_info_from_content(preprocessed_data,
                                                                              self.open_data_repos_ontology['repos'],
                                                                              model=self.llm_name,
                                                                              temperature=0,
                                                                              prompt_name=prompt_name,
                                                                              response_format=response_format)

            self.logger.info(f"Augmented dataset links: {augmented_dataset_links}")

            # dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

            # Create a DataFrame from the dataset links union supplementary material links
            out_df = pd.concat([pd.DataFrame(augmented_dataset_links)])

        else:
            self.logger.info(f"Chunking the content for the parsing step.")

            if semantic_retrieval:
                self.logger.info("Semantic retrieval is enabled, extracting sections from the preprocessed data.")
                corpus = self.extract_sections_from_text(preprocessed_data)
                top_k_sections = self.semantic_retrieve_from_corpus(corpus, topk_docs_to_retrieve=top_k)
                top_k_sections_text = [item['section_title'] + '\n' + item['text'] for item in top_k_sections]
                data_availability_str = "\n".join(top_k_sections_text)
            else:
                self.logger.warning("Semantic retrieval is not enabled, set it to True for better results.")

            augmented_dataset_links = self.extract_datasets_info_from_content(data_availability_str,
                                                                              self.open_data_repos_ontology['repos'],
                                                                              model=self.llm_name,
                                                                              temperature=0,
                                                                              prompt_name=prompt_name,
                                                                              response_format=response_format)

            # dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

            out_df = pd.concat([pd.DataFrame(augmented_dataset_links)])

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

        # Only set metadata if DataFrame is not empty
        if len(out_df) > 0:
            out_df['source_url'] = current_url_address if current_url_address else ''
            out_df['source_file_path'] = file_path
            out_df['pub_title'] = self.extract_publication_title(preprocessed_data)
            out_df['raw_data_format'] = raw_data_format
        else:
            # Create empty DataFrame with proper columns
            out_df = pd.DataFrame(columns=['source_url', 'source_file_path', 'pub_title', 'raw_data_format'])
            self.logger.warning(f"No datasets found in the document")

        self.remove_temp_file(file_path) if os.path.exists(file_path) and file_path_is_temp else None

        return out_df
    def extract_datasets_info_from_content(self, content: str, repos: list, model: str = 'gpt-4o-mini',
                                           temperature: float = 0.0,
                                           prompt_name: str = 'GPT_FewShot',
                                           full_document_read=True,
                                           response_format=dataset_response_schema_gpt) -> list:
        """
        Extract datasets from the given content using a specified LLM model.
        Uses a static prompt template and dynamically injects the required content.
        It also performs token counting and llm response normalization.

        :param content: The content to be processed.

        :param repos: List of repositories to be included in the prompt.

        :param model: The LLM model to be used for processing.

        :param temperature: The temperature setting for the model.

        :return: List of datasets retrieved from the content.
        """
        self.logger.info(f"Function_call: extract_datasets_info_from_content(...)")
        self.logger.debug(f"Loading prompt: {prompt_name} for model {model}")
        static_prompt = self.prompt_manager.load_prompt(prompt_name)
        n_tokens_static_prompt = self.count_tokens(static_prompt, model)

        if 'gpt' in model:
            while self.tokens_over_limit(content, model, allowance_static_prompt=n_tokens_static_prompt):
                content = content[:-2000]
        self.logger.info(f"Content length: {len(content)}")

        self.logger.debug(f"static_prompt: {static_prompt}")

        # Render the prompt with dynamic content
        messages = self.prompt_manager.render_prompt(
            static_prompt,
            entire_doc=self.full_document_read,
            content=content,
            repos=', '.join(repos)
        )
        self.logger.info(f"Prompt messages total length: {self.count_tokens(messages, model)} tokens")
        self.logger.debug(f"Prompt messages: {messages}")

        # Generate the checksum for the prompt content
        # Save the prompt and calculate checksum
        prompt_id = f"{model}-{temperature}-{self.prompt_manager._calculate_checksum(str(messages))}"
        self.logger.info(f"Prompt ID: {prompt_id}")
        # Save the prompt using the PromptManager
        if self.save_dynamic_prompts:
            self.prompt_manager.save_prompt(prompt_id=prompt_id, prompt_content=messages)

        if self.use_cached_responses:
            # Check if the response exists
            cached_response = self.prompt_manager.retrieve_response(prompt_id)

        if self.use_cached_responses and cached_response:
            self.logger.info(f"Using cached response {type(cached_response)} from model: {model}")
            if type(cached_response) == str and 'gpt' in model:
                resps = [json.loads(cached_response)]
            if type(cached_response) == str:
                resps = cached_response.split("\n")
            elif type(cached_response) == list:
                resps = cached_response
        else:
            # Make the request using the unified LLM client
            self.logger.info(
                f"Requesting datasets from content using model: {model}, temperature: {temperature}, "
                f"messages length: {self.count_tokens(messages, model)} tokens, schema: {response_format}")
            
            # Use the generic make_llm_call method
            raw_response = self.llm_client.make_llm_call(
                messages=messages, 
                temperature=temperature, 
                response_format=response_format,
                full_document_read=self.full_document_read
            )
            
            # Use the unified response processing method
            self.logger.debug(f"Calling process_llm_response with raw_response type: {type(raw_response)}")
            resps = self.llm_client.process_llm_response(
                raw_response=raw_response,
                response_format=response_format,
                expected_key="datasets"
            )
            self.logger.debug(f"process_llm_response returned: {resps} (type: {type(resps)})")
            
            # Apply task-specific deduplication
            self.logger.debug(f"Applying normalize_response_type to: {resps}")
            resps = self.normalize_response_type(resps)
            self.logger.debug(f"normalize_response_type returned: {resps} (type: {type(resps)})")
            
            # Save the processed response to cache
            if self.save_responses_to_cache:
                self.prompt_manager.save_response(prompt_id, resps)
                self.logger.info(f"Response {type(resps)} saved to cache")
            else:
                self.logger.info(f"Response {type(resps)} not saved (caching disabled)")

        #if not self.full_document_read:
        #    return resps

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

                # Use base parser's schema_validation method for consistency
                dataset_id, data_repository, dataset_webpage = self.schema_validation(dataset)

                if (dataset_id is None or data_repository is None) and dataset_webpage is None:
                    self.logger.info(f"Skipping dataset due to missing ID, repository, dataset page: {dataset}")
                    continue

                if (dataset_id == 'n/a' or data_repository == 'n/a') and dataset_webpage == 'n/a':
                    self.logger.info(f"Skipping dataset due to missing ID, repository, dataset page: {dataset}")
                    continue

            result.append({
                "dataset_identifier": dataset_id,
                "data_repository": data_repository,
                "dataset_webpage": dataset_webpage if dataset_webpage is not None else 'n/a',
                "citation_type": dataset.get('citation_type', 'n/a') if isinstance(dataset, dict) else 'n/a'
            })

            if isinstance(dataset, dict):
                if 'decision_rationale' in dataset:
                    result[-1]['decision_rationale'] = dataset['decision_rationale']

                if 'dataset-publication_relationship' in dataset:
                    result[-1]['dataset-publication_relationship'] = dataset['dataset-publication_relationship']

                # Preserve dataset_context_from_paper field if present (for PaperMiner enhanced schema)
                if 'dataset_context_from_paper' in dataset:
                    result[-1]['dataset_context_from_paper'] = dataset['dataset_context_from_paper']

            self.logger.debug(f"Extracted dataset: {result[-1]}")

        self.logger.info(f"Final result: {result}")

        return result

    def extract_publication_title(self, raw_data):
        """
        Extract the publication title from the HTML content.

        :return: str — the publication title.
        """
        self.logger.info("Extracting publication title from PDF")

        # simple heuristic to extract title or GROBID

        return ' '