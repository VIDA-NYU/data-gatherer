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
    write_htmls_xmls: bool = False
) -> Dict:
    """
    Orchestrates the process for a single given source URL (publication).
    1. Fetches raw data using the data fetcher (WebScraper or EntrezFetcher).
    2. Parses the raw data using the parser (LLMParser).
    3. Collects Metadata.
    4. Classifies the parsed data using the classifier (LLMClassifier).

    :param url: The URL to process.
    :param save_staging_table: Flag to save the staging table.
    :param article_file_dir: Directory to save the raw HTML/XML/PDF files.
    :param use_portkey: Flag to use Portkey for Gemini LLM.
    :param driver_path: Path to your local WebDriver executable (if applicable). When set to None, Webdriver manager will be used.
    :param browser: Browser to use for scraping (if applicable). Supported values are 'Firefox', 'Chrome'.
    :param headless: Whether to run the browser in headless mode (if applicable).
    :param prompt_name: Name of the prompt to use for LLM parsing (Depending on this we will extract more or less information - Change dataset schema accordingly) --> possible values are {GPT_FDR_FewShot_Descr,GPT_FDR_FewShot, GPT_FewShot}.
    :param semantic_retrieval: Flag to indicate if semantic retrieval should be used.
    :param section_filter: Optional filter to apply to the sections (supplementary_material', 'data_availability_statement').
    :param response_format: The response schema to use for parsing the data. Supported values are the tytpes imported from data_gatherer.llm.response_schema.
    :param HTML_fallback: Flag to indicate if HTML fallback should be used when fetching data. This will override any other fetching resource (i.e. API).
    :param grobid_for_pdf: Flag to indicate if GROBID should be used for PDF processing.
    :param write_htmls_xmls: Flag to indicate if raw HTML/XML files should be saved. Overwrites the default setting.

    :return: DataFrame of classified links or None if an error occurs.
    """
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
        write_htmls_xmls=write_htmls_xmls
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

    :param url_list: List of URLs/PMCIDs to process.
    :param log_modulo: Frequency of logging progress (useful when url_list is long).
    :param save_staging_table: Flag to save the staging table.
    :param article_file_dir: Directory to save the raw HTML/XML/PDF files.
    :param driver_path: Path to your local WebDriver executable (if applicable). When set to None, Webdriver manager will be used.
    :param browser: Browser to use for scraping (if applicable). Supported values are 'Firefox', 'Chrome'.
    :param headless: Whether to run the browser in headless mode (if applicable).
    :param use_portkey: Flag to use Portkey for Gemini LLM.
    :param response_format: The response schema to use for parsing the data.
    :param prompt_name: Name of the prompt to use for LLM parsing.
    :param semantic_retrieval: Flag to indicate if semantic retrieval should be used.
    :param section_filter: Optional filter to apply to the sections (supplementary_material', 'data_availability_statement').
    :param grobid_for_pdf: Flag to indicate if GROBID should be used for PDF processing.
    :return: Dictionary with URLs as keys and DataFrames of classified data as values.
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