from data_gatherer.orchestrator import Orchestrator
from data_gatherer.parser import LLMParser
from data_gatherer.logger_setup import setup_logging
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def test_orchestrator_data_preview():
    # Check if the DataFrame is not empty
    combined_df = pd.read_csv('test_data/test_combined_data.csv')
    assert not combined_df.empty, "The DataFrame is empty."
    orchestrator = Orchestrator('gemini-2.0-flash', log_file_override=None)
    orchestrator.write_raw_metadata = False
    orchestrator.parser = LLMParser('open_bio_data_repos.json', orchestrator.logger,
                                    log_file_override=None, llm_name='gemini-2.0-flash')
    metadata_list = orchestrator.get_data_preview(combined_df, interactive=False, return_metadata=True)
    assert isinstance(metadata_list, list), "Metadata list is not a list."
    assert len(metadata_list) > 0, "Metadata list is empty."
    assert all(isinstance(metadata, dict) for metadata in metadata_list), "Not all items in metadata list are dictionaries."