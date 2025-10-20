
import pandas as pd
from typing import Optional, List, Dict, Any
from data_gatherer.data_gatherer import DataGatherer
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
server = FastMCP(
    name="DataGatherer",
    instructions="This MCP server exposes DataGatherer utilities for datasets information extraction from scientific articles."
)

dg = DataGatherer()

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
    MCP tool: Run DataGatherer.process_url and return results as dict.
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
    return df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else df

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
    MCP tool: Run DataGatherer.process_articles and return results as dict.
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
    server.run(transport="stdio")

if __name__ == "__main__":
    run_mcp_server()