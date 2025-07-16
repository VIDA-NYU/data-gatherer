from data_gatherer.parser.base_parser import *
import re
import pandas as pd
import pymupdf, fitz
import pypdf
import unicodedata
from collections import Counter

class PDFParser(LLMParser):
    """
    Custom PDF parser that has only support for PDF or HTML-like input
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

        Args:
            pdf_content: str — normalized plain-text PDF content.

        Returns:
            List[dict] — list of sections with 'section_title' and 'text'
        """
        self.logger.info("Extracting sections using heuristics.")

        lines = pdf_content.splitlines()
        sections = []
        current_section = {"section_title": "Start", "text": ""}
        candidate_pattern = re.compile(r"^\s*(\d{1,2}(\.\d{1,2})*\.?\s*)?[A-Z][\w\s,\-:]{3,80}$")

        for i, line in enumerate(lines):
            line_stripped = line.strip()

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

        # Append last section
        if current_section["text"].strip():
            sections.append(current_section)

        self.logger.info(f"Extracted {len(sections)} sections.")
        return sections


    def extract_text_from_pdf(self, file_path, pdf_reader='pymupdf', start_page=0, end_page=None):
        """
        Extracts plain text from a PDF file using PyMuPDF.

        Parameters:
            file_path (str): Path to the PDF file.
            pdf_reader (str): Reader type; currently supports only 'pymupdf'.
            start_page (int): Page number to start from (0-indexed).
            end_page (int or None): Page number to end at (exclusive). If None, reads till the end.

        Returns:
            str: Extracted and normalized text content.
        """
        if pdf_reader != 'pymupdf':
            raise ValueError(f"Unsupported PDF reader: {pdf_reader}")

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
        self.logger.info("Normalizing extracted text.")

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

        self.logger.info(f"Normalized text length: {len(normalized_text)}")
        return normalized_text

    def parse_data(self, file_path, publisher=None, current_url_address=None, additional_data=None, raw_data_format='PDF',
                   file_path_is_temp=False, article_file_dir='tmp/raw_files/', process_DAS_links_separately=False,
                   prompt_name='retrieve_datasets_simple_JSON', use_portkey_for_gemini=True, semantic_retrieval=False,
                   section_filter=None):
        """
        Parse the PDF file and extract metadata of the relevant datasets.

        :param file_path: The file_path to the PDF.

        :param current_url_address: The current URL address being processed.

        :param additional_data: Additional data to be processed (optional).

        :param raw_data_format: The format of the raw data ('XML' or 'HTML').

        :param file_path_is_temp: Boolean indicating if the file_path is a temporary file.

        :return: A DataFrame containing the extracted links and links to metadata - if repo is supported. Add support for unsupported repos in the ontology.

        """
        out_df = None
        # Check if api_data is a string, and convert to XML if needed
        self.logger.info(f"Function call: parse_data({file_path}, {current_url_address}, "
                         f"additional_data, {raw_data_format})")

        text_from_pdf = self.extract_text_from_pdf(file_path)
        preprocessed_data = self.normalize_extracted_text(text_from_pdf)

        self.logger.debug(f"Preprocessed data: {preprocessed_data}")

        if self.full_document_read:
            self.logger.info(f"Extracting links from full content.")

            # Extract dataset links from the entire text
            augmented_dataset_links = self.extract_datasets_info_from_content(preprocessed_data,
                                                                              self.open_data_repos_ontology['repos'],
                                                                              model=self.llm_name,
                                                                              temperature=0,
                                                                              prompt_name=prompt_name)

            self.logger.info(f"Augmented dataset links: {augmented_dataset_links}")

            dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

            # Create a DataFrame from the dataset links union supplementary material links
            out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages)])

        else:
            self.logger.info(f"Chunking the content for the parsing step.")

            if semantic_retrieval:
                self.logger.info("Semantic retrieval is enabled, extracting sections from the preprocessed data.")
                corpus = self.extract_sections_from_text(preprocessed_data)
                top_k_sections = self.semantic_retrieve_from_corpus(corpus)
                top_k_sections_text = [item['section_title'] + '\n' + item['text'] for item in top_k_sections]
                data_availability_str = "\n".join(top_k_sections_text)
            else:
                self.logger.warning("Semantic retrieval is not enabled, set it to True for better results.")

            augmented_dataset_links = self.extract_datasets_info_from_content(data_availability_str,
                                                                              self.open_data_repos_ontology['repos'],
                                                                              model=self.llm_name,
                                                                              temperature=0,
                                                                              prompt_name=prompt_name)

            dataset_links_w_target_pages = self.get_dataset_page(augmented_dataset_links)

            out_df = pd.concat([pd.DataFrame(dataset_links_w_target_pages)])

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

        out_df['source_url'] = current_url_address if current_url_address else ''
        out_df['source_file_path'] = file_path
        out_df['pub_title'] = self.extract_publication_title(preprocessed_data)

        self.remove_temp_file(file_path) if os.path.exists(file_path) and file_path_is_temp else None

        return out_df

    def extract_publication_title(self, raw_data):
        """
        Extract the publication title from the HTML content.

        :return: str — the publication title.
        """
        self.logger.info("Extracting publication title from PDF")

        # simple heuristic to extract title or GROBID

        return ' '