import os
import subprocess
import time
import pandas as pd
import requests
import shutil
from lxml import etree
from .pdf_parser import PDFParser

class GrobidPDFParser(PDFParser):
    """
    PDFParser subclass that uses a local GROBID server to extract structured text and metadata from PDFs.
    """
    def __init__(self, open_data_repos_ontology, logger, log_file_override=None, full_document_read=True,
                 prompt_dir="data_gatherer/prompts/prompt_templates",
                 llm_name=None, save_dynamic_prompts=False, save_responses_to_cache=False, use_cached_responses=False,
                 use_portkey_for_gemini=True, grobid_home=None, grobid_port=8070):

        super().__init__(open_data_repos_ontology, logger, log_file_override=log_file_override,
                         full_document_read=full_document_read, prompt_dir=prompt_dir,
                         llm_name=llm_name, save_dynamic_prompts=save_dynamic_prompts,
                         save_responses_to_cache=save_responses_to_cache,
                         use_cached_responses=use_cached_responses, use_portkey_for_gemini=use_portkey_for_gemini
                         )

        self.logger = logger

        if grobid_home is None:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            grobid_home = os.path.join(BASE_DIR, "grobid-0.8.2")
        self.grobid_home = grobid_home

        if not os.path.isdir(self.grobid_home):
            raise FileNotFoundError(
                f"GROBID home directory '{self.grobid_home}' does not exist.\n"
                f"Please install GROBID or update the path <grobid_home> parameter."
            )

        self.logger.info(f"Using GROBID home directory: {self.grobid_home}")
        self.grobid_port = grobid_port
        self.grobid_process = None
        self._check_prerequisites()
        self._start_grobid_server()

    def _check_prerequisites(self):
        # Check Java
        if shutil.which('java') is None:
            raise EnvironmentError("Java is required for GROBID but was not found in PATH.")
        # Check GROBID directory
        if not os.path.isdir(self.grobid_home):
            raise FileNotFoundError(f"GROBID directory '{self.grobid_home}' not found.")
        # Check gradlew
        gradlew_path = os.path.join(self.grobid_home, 'gradlew')
        if not os.path.isfile(gradlew_path):
            raise FileNotFoundError(f"GROBID gradlew not found at '{gradlew_path}'.")

    def _start_grobid_server(self):
        try:
            self.logger.info(f"Checking if GROBID server is already running on port {self.grobid_port}...")
            r = requests.get(f"http://localhost:{self.grobid_port}/api/isalive", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        self.logger.info("Starting GROBID server...")
        gradlew_path = os.path.join(self.grobid_home, 'gradlew')
        cmd = [gradlew_path, 'run']
        self.grobid_process = subprocess.Popen(cmd, cwd=self.grobid_home, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for server to be up
        for _ in range(30):
            try:
                r = requests.get(f"http://localhost:{self.grobid_port}/api/isalive", timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                time.sleep(1)
        raise RuntimeError("GROBID server did not start within timeout.")

    def extract_full_text_xml(self, pdf_path):
        grobid_url = f"http://localhost:{self.grobid_port}/api/processFulltextDocument"
        with open(pdf_path, 'rb') as f:
            files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
            response = requests.post(grobid_url, files=files)
        if response.status_code != 200:
            raise RuntimeError(f"GROBID failed: {response.status_code} {response.text}")
        self.logger.debug(f"Extracted full text XML {response.text}.")
        return response.text

    def extract_refs_xml(self, pdf_path):
        grobid_url = f"http://localhost:{self.grobid_port}/api/processReferences"
        with open(pdf_path, 'rb') as f:
            files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
            response = requests.post(grobid_url, files=files)
        if response.status_code != 200:
            raise RuntimeError(f"GROBID failed: {response.status_code} {response.text}")
        return response.text

    def extract_sections(self, tei_xml):
        """
        Extract section titles and paragraphs from a TEI XML string.
        Returns a list of dicts: [{section_title, text}]
        """
        root = etree.fromstring(tei_xml.encode('utf-8'))
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        sections = []
        # Find all section divs in the body, or fallback to divs without type if none found
        divs = root.xpath('.//tei:text/tei:body//tei:div[@type="section"]', namespaces=ns)
        if not divs:
            # fallback: get all divs under body (may be untyped)
            divs = root.xpath('.//tei:text/tei:body//tei:div', namespaces=ns)
        for div in divs:
            title_el = div.find('tei:head', namespaces=ns)
            section_title = title_el.text if title_el is not None else ''
            paragraphs = [etree.tostring(p, method="text", encoding="unicode").strip() for p in div.findall('tei:p', namespaces=ns)]
            text = '\n'.join(paragraphs)
            # Only add if there is any text
            if section_title or text:
                sections.append({'section_title': section_title, 'text': text})
        return sections

    def extract_reference_content(self, ref_id, tei_xml):
        """
        Given a reference id (e.g., '#b61'), extract the referenced content from the TEI XML.
        Returns a string with the reference content, or the id if not found.
        """
        self.logger.debug(f"Extracting reference content for ID: {ref_id}")
        root = etree.fromstring(tei_xml.encode('utf-8'))
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        ref_id_clean = ref_id.lstrip('#')
        bibl = root.xpath(f".//tei:biblStruct[@xml:id='{ref_id_clean}']", namespaces=ns)
        if bibl:
            bibl = bibl[0]
            # Analytic title
            analytic_title = bibl.find('.//tei:analytic/tei:title', namespaces=ns)
            analytic_title_str = analytic_title.text.strip() if analytic_title is not None and analytic_title.text else ""
            # Authors
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
            # Monograph title
            monogr_title = bibl.find('.//tei:monogr/tei:title', namespaces=ns)
            monogr_title_str = monogr_title.text.strip() if monogr_title is not None and monogr_title.text else ""
            # Date
            date = bibl.find('.//tei:monogr/tei:imprint/tei:date', namespaces=ns)
            date_str = date.text.strip() if date is not None and date.text else ""
            # idno (e.g., DOI)
            idno = bibl.find('.//tei:idno', namespaces=ns)
            idno_str = idno.text.strip() if idno is not None and idno.text else ""
            # Compose reference string
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
        """
        Extract paragraphs and their section context from a TEI XML document.

        Args:
            tei_xml: str — TEI XML as string.
            ref_substitutions: bool — if True, substitute <ref> elements with their referenced content.

        Returns:
            List of dicts with 'paragraph', 'section_title', and 'text'.
        """
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
        """
        Extract all text from a TEI XML string.
        Returns a single string with all text content.
        """
        self.logger.info(f"Extracting text from TEI XML. Type: {type(tei_xml)}")
        paragraphs = self.extract_paragraphs(tei_xml, ref_substitutions=True)
        return "\n".join(paragraphs['text'] for paragraphs in paragraphs if 'text' in paragraphs)

    def __del__(self):
        if self.grobid_process:
            self.grobid_process.terminate()

    def parse_data(self, file_path, publisher=None, current_url_address=None, additional_data=None, raw_data_format='PDF',
                   file_path_is_temp=False, article_file_dir='tmp/raw_files/', process_DAS_links_separately=False,
                   prompt_name='retrieve_datasets_simple_JSON', use_portkey_for_gemini=True, semantic_retrieval=False,
                   top_k=2, section_filter=None):
        """
        Parse the PDF file and extract metadata of the relevant datasets.
        """
        out_df = None
        self.logger.info(f"Function call: parse_data({file_path}, {current_url_address}, "
                         f"additional_data, {raw_data_format})")

        try:
            full_cont_xml = self.extract_full_text_xml(file_path)
            self.logger.info(f"Parsing full text TEI XML from GROBID response.")

            if self.full_document_read:
                cont = self.extract_text(full_cont_xml)
            else:
                if semantic_retrieval:
                    paragraphs = self.extract_paragraphs(full_cont_xml, True)
                    top_k_sections = self.semantic_retrieve_from_corpus(paragraphs, topk_docs_to_retrieve=top_k)
                    top_k_sections_text = [item['text'] for item in top_k_sections]
                    cont = "\n".join(top_k_sections_text)
                    self.logger.info(f"Extracted top sections content for semantic retrieval.\n {cont}")

                else:
                    raise ValueError("Set semantic_retrieval to True.")

            datasets = []
            datasets.extend(self.extract_datasets_info_from_content(cont, self.repo_names, model=self.llm_name,
                                                                    temperature=0, prompt_name=prompt_name))

            out_df = pd.DataFrame(datasets)

            out_df['source_file name'] = os.path.basename(file_path)
            out_df['source_file_path'] = file_path
            out_df['pub_title'] = self.extract_publication_title(full_cont_xml)

            return out_df


        except Exception as e:

            self.logger.error(f"GROBID failed on {file_path}: {e}")

            self.logger.info("Attempting fallback with PyMuPDF parser...")

            try:
                from .pdf_parser import PDFParser
                fallback_parser = PDFParser(self.repo_ontology, self.logger, log_file_override=self.log_file_override,
                                            full_document_read=self.full_document_read, prompt_dir=self.prompt_dir,
                                            llm_name=self.llm_name, save_dynamic_prompts=self.save_dynamic_prompts,
                                            save_responses_to_cache=self.save_responses_to_cache,
                                            use_cached_responses=self.use_cached_responses,
                                            use_portkey_for_gemini=self.use_portkey_for_gemini)

                return fallback_parser.parse_data(file_path, publisher=publisher,
                                                  current_url_address=current_url_address,
                                                  additional_data=additional_data, raw_data_format=raw_data_format,
                                                  file_path_is_temp=file_path_is_temp,
                                                  article_file_dir=article_file_dir,
                                                  process_DAS_links_separately=process_DAS_links_separately,
                                                  prompt_name=prompt_name,
                                                  use_portkey_for_gemini=use_portkey_for_gemini,
                                                  semantic_retrieval=semantic_retrieval, top_k=top_k,
                                                  section_filter=section_filter)

            except Exception as fallback_error:
                self.logger.error(f"Fallback parser also failed: {fallback_error}")
                return pd.DataFrame()

    def extract_publication_title(self, tei_xml):
        """
        Extract the publication title from the TEI XML.
        Returns the title as a string, or an empty string if not found.
        """
        root = etree.fromstring(tei_xml.encode('utf-8'))
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        # Prefer analytic title (article title) over monograph title (journal/repository)
        title_el = root.find('.//tei:analytic/tei:title', namespaces=ns)
        if title_el is not None and title_el.text:
            return title_el.text.strip()
        title_el = root.find('.//tei:monogr/tei:title', namespaces=ns)
        if title_el is not None and title_el.text:
            return title_el.text.strip()
        return ""
