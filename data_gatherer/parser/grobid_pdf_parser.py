import os
import subprocess
import time
import pandas as pd
import requests
import shutil
from lxml import etree
from .pdf_parser import PDFParser
from .xml_parser import XMLRouter, TEI_XMLParser

class GrobidPDFParser(PDFParser):
    """
    PDFParser subclass that uses a local GROBID server to extract structured text and metadata from PDFs.
    Uses a cached TEI_XMLParser instance for XML normalization and processing.
    """
    def __init__(self, open_data_repos_ontology, logger, log_file_override=None, full_document_read=True,
                 prompt_dir="data_gatherer/prompts/prompt_templates", write_XML=False,
                 llm_name=None, save_dynamic_prompts=False, save_responses_to_cache=False, use_cached_responses=False,
                 use_portkey=True, grobid_home=None, grobid_port=8070):

        super().__init__(open_data_repos_ontology, logger, log_file_override=log_file_override,
                         full_document_read=full_document_read, prompt_dir=prompt_dir,
                         llm_name=llm_name, save_dynamic_prompts=save_dynamic_prompts,
                         save_responses_to_cache=save_responses_to_cache,
                         use_cached_responses=use_cached_responses, use_portkey=use_portkey
                         )

        self.logger = logger
        self.write_XML = write_XML
        
        # Initialize TEI XML parser as a helper (not replacing self as parser)
        self._tei_parser = TEI_XMLParser(
            open_data_repos_ontology=open_data_repos_ontology,
            logger=logger,
            log_file_override=log_file_override,
            full_document_read=full_document_read,
            prompt_dir=prompt_dir,
            llm_name=llm_name,
            save_dynamic_prompts=save_dynamic_prompts,
            save_responses_to_cache=save_responses_to_cache,
            use_cached_responses=use_cached_responses,
            use_portkey=use_portkey
        )
        self.logger.debug("Initialized TEI_XMLParser helper for GROBID XML processing")

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
                self.logger.info("GROBID server is already running.")
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
        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
                response = requests.post(grobid_url, files=files, timeout=60)
            
            if response.status_code != 200:
                error_msg = f"GROBID failed: {response.status_code} {response.text}"
                self.logger.warning(error_msg)
                raise RuntimeError(error_msg)
            
            self.logger.debug(f"Successfully extracted TEI XML from {pdf_path}")
            return response.text
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"GROBID request timed out for {pdf_path}")
        except Exception as e:
            raise RuntimeError(f"GROBID processing failed for {pdf_path}: {str(e)}")

    def extract_refs_xml(self, pdf_path):
        grobid_url = f"http://localhost:{self.grobid_port}/api/processReferences"
        with open(pdf_path, 'rb') as f:
            files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
            response = requests.post(grobid_url, files=files)
        if response.status_code != 200:
            raise RuntimeError(f"GROBID failed: {response.status_code} {response.text}")
        return response.text

    def __del__(self):
        if self.grobid_process:
            self.grobid_process.terminate()

    def normalize_XML(self, xml_data):
        """
        Normalize TEI XML using the helper TEI_XMLParser.
        This delegates to the cached TEI parser instance for modular XML processing.
        
        Args:
            xml_data: XML data (string, bytes, or Element)
            
        Returns:
            Normalized XML string
        """
        return self._tei_parser.normalize_XML(xml_data)

    def parse_data(
        self, 
        file_path, 
        publisher=None, 
        current_url_address=None, 
        raw_data_format='PDF',
        file_path_is_temp=False, 
        article_file_dir='tmp/raw_files/',
        prompt_name='GPT_FewShot', 
        use_portkey=True, 
        semantic_retrieval=False, 
        top_k=2, 
        section_filter=None, 
        response_format=None, 
        dedup=True, 
        brute_force_RegEx_ID_ptrs=False,
        article_id=None
        ):
        """
        Parse the PDF file and extract metadata of the relevant datasets.
        """
        self.logger.info(f"Function call: parse_data({file_path}, {current_url_address}, {raw_data_format})")

        try:
            # 1. Convert PDF to XML using the dedicated method
            xml_root = self.pdf_to_xml(file_path, current_url_address, article_file_dir)
            
            if xml_root is None:
                raise RuntimeError("PDF to XML conversion failed")
            
            # 2. Create XML parser using XMLRouter
            router = XMLRouter(
                self.open_data_repos_ontology, 
                self.logger, 
                llm_name=self.llm_name, 
                full_document_read=self.full_document_read,
                use_portkey=use_portkey,
                save_dynamic_prompts=self.save_dynamic_prompts,
                save_responses_to_cache=self.save_responses_to_cache,
                use_cached_responses=self.use_cached_responses
            )
            xml_parser = router.get_parser(xml_root)

            # 3. Parse XML to extract dataset metadata
            out_df = xml_parser.parse_data(
                xml_root,
                publisher=publisher,
                current_url_address=current_url_address,
                raw_data_format='XML',
                article_file_dir=article_file_dir,
                section_filter=section_filter,
                prompt_name=prompt_name,
                use_portkey=use_portkey,
                semantic_retrieval=semantic_retrieval,
                top_k=top_k,
                response_format=response_format,
                dedup=dedup,
                brute_force_RegEx_ID_ptrs=brute_force_RegEx_ID_ptrs
            )

            return out_df

        except Exception as e:
            self.logger.error(f"GROBID failed on {file_path}: {e}")
            
            # Validate PDF file before fallback
            if not os.path.exists(file_path):
                self.logger.error(f"PDF file does not exist: {file_path}")
                return pd.DataFrame()
            
            file_size = os.path.getsize(file_path)
            self.logger.info(f"PDF file size: {file_size} bytes")
            
            if file_size == 0:
                self.logger.error(f"PDF file is empty: {file_path}")
                return pd.DataFrame()
            
            # Check if file is a valid PDF
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(5)
                    if header != b'%PDF-':
                        self.logger.warning(f"File does not appear to be a valid PDF (header: {header})")
            except Exception as check_error:
                self.logger.warning(f"Could not verify PDF header: {check_error}")
            
            # Fallback to PyMuPDF parser
            self.logger.info("Attempting fallback with PyMuPDF parser...")

            try:
                from .pdf_parser import PDFParser
                fallback_parser = PDFParser(
                    self.open_data_repos_ontology, 
                    self.logger, 
                    full_document_read=self.full_document_read, 
                    prompt_dir=self.prompt_manager.prompt_dir,
                    llm_name=self.llm_name, 
                    save_dynamic_prompts=self.save_dynamic_prompts,
                    save_responses_to_cache=self.save_responses_to_cache,
                    use_cached_responses=self.use_cached_responses,
                    use_portkey=self.use_portkey
                )

                return fallback_parser.parse_data(
                    file_path, 
                    publisher=publisher,
                    current_url_address=current_url_address,
                    raw_data_format=raw_data_format,
                    file_path_is_temp=file_path_is_temp,
                    article_file_dir=article_file_dir,
                    prompt_name=prompt_name,
                    use_portkey=use_portkey,
                    semantic_retrieval=semantic_retrieval,
                    section_filter=section_filter,
                    response_format=response_format,
                    dedup=dedup, 
                    brute_force_RegEx_ID_ptrs=brute_force_RegEx_ID_ptrs)

            except Exception as fallback_error:
                self.logger.error(f"Fallback parser also failed: {fallback_error}")
                return pd.DataFrame()

    def pdf_to_xml(self, pdf_path, current_url_address=None, article_file_dir=None):
        """
        Convert a PDF file to TEI XML using GROBID and save to output directory.

        :param pdf_path: Path to the PDF file
        :param current_url_address: URL or identifier for the document
        :param article_file_dir: Directory to save the XML file
        :return: lxml Element tree of the TEI XML, or None on failure
        """
        self.logger.info(f"Function call: pdf_to_xml({pdf_path}, {current_url_address})")

        try:
            # 1. Extract TEI XML from PDF
            full_cont_xml = self.extract_full_text_xml(pdf_path)
            self.logger.info(f"Parsing full text TEI XML from GROBID response.")

            # 2. Parse TEI XML into lxml Element tree
            xml_root = etree.fromstring(full_cont_xml.encode('utf-8'))

            # 3. Optionally save XML to file
            if self.write_XML and article_file_dir is not None:
                xml_output_path = os.path.join(article_file_dir, os.path.basename(pdf_path) + '.xml')
                os.makedirs(article_file_dir, exist_ok=True)
                with open(xml_output_path, 'wb') as xml_file:
                    xml_file.write(full_cont_xml.encode('utf-8'))
                self.logger.info(f"Saved TEI XML to {xml_output_path}")

            return xml_root
            
        except Exception as e:
            self.logger.error(f"Failed to convert PDF to XML for {pdf_path}: {e}")
            return None

    def extract_publication_title(self, root):
        return self._tei_parser.extract_publication_title(root)

    def extract_href_from_supplementary_material(self, root, current_url_address):
        if isinstance(root, str):
            root = etree.fromstring(root.encode('utf-8'))
        return self._tei_parser.extract_href_from_supplementary_material(root, current_url_address)

    def extract_supplementary_material_refs(self, root, supplementary_material_links):
        if isinstance(root, str):
            root = etree.fromstring(root.encode('utf-8'))
        return self._tei_parser.extract_supplementary_material_refs(root, supplementary_material_links)