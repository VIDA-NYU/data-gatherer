import pandas as pd
from typing import Optional, List, Dict, Any
from data_gatherer.data_gatherer import DataGatherer
from data_gatherer.llm.response_schema import dataset_response_schema_gpt, dataset_response_schema_with_use_description
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
server = FastMCP(
    name="DataGatherer",
    instructions="This MCP server exposes DataGatherer utilities for datasets information extraction from scientific articles."
)

response_schema_options = {
    "dataset_response_schema_gpt": dataset_response_schema_gpt,
    "dataset_response_schema_with_use_description": dataset_response_schema_with_use_description
}
dg = DataGatherer(log_file_override="/tmp/data_gatherer_orchestrator.log", log_level="INFO")

@server.tool()
async def process_url_mcp(
    url: str,
    save_staging_table: bool = False,
    article_file_dir: str = 'tmp/raw_files/',
    use_portkey: bool = True,
    driver_path: Optional[str] = None,
    browser: str = 'Firefox',
    headless: bool = True,
    prompt_name: str = 'GPT_FewShot',
    semantic_retrieval: bool = False,
    section_filter: Optional[List[str]] = None,
    response_format: Any = None,
    HTML_fallback: bool = False,
    grobid_for_pdf: bool = False,
    write_htmls_xmls: bool = False,
    full_document_read: bool = False
) -> Dict:
    """
    Orchestrates the process for a single given source URL (publication).
    1. Fetches raw data using the data fetcher (WebScraper or EntrezFetcher).
    2. Parses the raw data using the parser (LLMParser).
    3. Collects Metadata.

    :param url: The URL to process. REQUIRED!
    :param save_staging_table: Flag to save the staging table. OPTIONAL.
    :param article_file_dir: Directory to save the raw HTML/XML/PDF files. OPTIONAL.
    :param use_portkey: Flag to use Portkey for Gemini LLM. OPTIONAL.
    :param driver_path: Path to your local WebDriver executable (if applicable). When set to None, Webdriver manager will be used. OPTIONAL.
    :param browser: Browser to use for scraping (if applicable). Supported values are 'Firefox', 'Chrome'. OPTIONAL.
    :param headless: Whether to run the browser in headless mode (if applicable). OPTIONAL.
    :param prompt_name: Name of the prompt to use for LLM parsing (Depending on this we will extract more or less information - Change dataset schema accordingly) --> possible values are {GPT_FDR_FewShot_Descr,GPT_FDR_FewShot, GPT_FewShot}. REQUIRED!
    :param semantic_retrieval: Flag to indicate if semantic retrieval should be used. REQUIRED!
    :param section_filter: Optional filter to apply to the sections (supplementary_material', 'data_availability_statement'). OPTIONAL.
    :param response_format: The response schema to use for parsing the data. Supported values are the types contained in the dict response_schema_options with the keys: `dataset_response_schema_gpt`, `dataset_response_schema_with_use_description`. REQUIRED!
    :param HTML_fallback: Flag to indicate if HTML fallback should be used when fetching data. This will override any other fetching resource (i.e. API). OPTIONAL.
    :param grobid_for_pdf: Flag to indicate if GROBID should be used for PDF processing. OPTIONAL.
    :param write_htmls_xmls: Flag to indicate if raw HTML/XML files should be saved. Overwrites the default setting. OPTIONAL.
    :param full_document_read: Flag to indicate if the entire document should be read (useful for description generation). Overwrites the default setting. REQUIRED!

    :return: DataFrame of classified links or None if an error occurs.
    """

    if type(response_format) == str:
        if response_format in response_schema_options:
            response_format = response_schema_options[response_format]
        else:
            raise ValueError(f"Unsupported response_format string: {response_format}. Supported values are: {list(response_schema_options.keys())}")

    df = dg.process_url(
        url,
        save_staging_table=save_staging_table,
        article_file_dir=article_file_dir,
        use_portkey=use_portkey,
        driver_path=driver_path,
        browser=browser,
        headless=headless,
        prompt_name=prompt_name,
        semantic_retrieval=semantic_retrieval,
        section_filter=section_filter,
        response_format=response_format,
        HTML_fallback=HTML_fallback,
        grobid_for_pdf=grobid_for_pdf,
        write_htmls_xmls=write_htmls_xmls,
        full_document_read=full_document_read
    )
    return {"result": df.to_dict(orient="records")} if isinstance(df, pd.DataFrame) else {"result": df}

@server.tool()
async def process_articles_mcp(
    url_list: List[str],
    log_modulo: int = 10,
    save_staging_table: bool = False,
    article_file_dir: str = 'tmp/raw_files/',
    driver_path: Optional[str] = None,
    browser: str = 'Firefox',
    headless: bool = True,
    use_portkey: bool = True,
    response_format: Any = None,
    prompt_name: str = 'GPT_FewShot',
    semantic_retrieval: bool = False,
    section_filter: Optional[List[str]] = None,
    grobid_for_pdf: bool = False,
    write_htmls_xmls: bool = False
) -> Dict[str, Any]:
    """
    Processes a list of article URLs and returns parsed data.

    :param url_list: List of URLs/PMCIDs to process. REQUIRED!
    :param log_modulo: Frequency of logging progress (useful when url_list is long). OPTIONAL.
    :param save_staging_table: Flag to save the staging table. OPTIONAL.
    :param article_file_dir: Directory to save the raw HTML/XML/PDF files. OPTIONAL.
    :param driver_path: Path to your local WebDriver executable (if applicable). When set to None, Webdriver manager will be used. OPTIONAL.
    :param browser: Browser to use for scraping (if applicable). Supported values are 'Firefox', 'Chrome'. OPTIONAL.
    :param headless: Whether to run the browser in headless mode (if applicable). OPTIONAL.
    :param use_portkey: Flag to use Portkey for Gemini LLM. OPTIONAL.
    :param response_format: The response schema to use for parsing the data. REQUIRED!
    :param prompt_name: Name of the prompt to use for LLM parsing. REQUIRED!
    :param semantic_retrieval: Flag to indicate if semantic retrieval should be used. REQUIRED!
    :param section_filter: Optional filter to apply to the sections (supplementary_material', 'data_availability_statement').  OPTIONAL.
    :param grobid_for_pdf: Flag to indicate if GROBID should be used for PDF processing. OPTIONAL.
    :return: Dictionary with URLs as keys and DataFrames of classified data as values. OPTIONAL.
    """
    results = dg.process_articles(
        url_list,
        log_modulo=log_modulo,
        save_staging_table=save_staging_table,
        article_file_dir=article_file_dir,
        driver_path=driver_path,
        browser=browser,
        headless=headless,
        use_portkey=use_portkey,
        response_format=response_format,
        prompt_name=prompt_name,
        semantic_retrieval=semantic_retrieval,
        section_filter=section_filter,
        grobid_for_pdf=grobid_for_pdf,
        write_htmls_xmls=write_htmls_xmls
    )
    # Convert each DataFrame to dict
    return {url: df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else df for url, df in results.items()}

def run_mcp_server():
    print("Starting MCP server for DataGatherer...")
    server.run(transport="stdio")

if __name__ == "__main__":
    run_mcp_server()