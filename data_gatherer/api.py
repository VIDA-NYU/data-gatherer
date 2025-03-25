import os
import json
from typing import List, Dict, Optional, Union
import pandas as pd

from data_gatherer.core.orchestrator import Orchestrator


class DataGatherer:
    """
    Main API class for the data-gatherer package.
    Provides an interface to extract dataset links from scientific articles.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 custom_config: Optional[Dict] = None,
                 headless: bool = True):
        """
        Initialize the DataGatherer with either a config file path or a custom config dictionary.
        
        Args:
            config_path: Path to a JSON configuration file (optional)
            custom_config: Custom configuration dictionary (optional)
            headless: Whether to run browser in headless mode (default: True)
        """
        if custom_config is not None:
            self.config = custom_config
        elif config_path is not None:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Use default config
            default_config_path = os.path.join(os.path.dirname(__file__), 'config', 'default_config.json')
            with open(default_config_path, 'r') as f:
                self.config = json.load(f)
                
        # Override browser headless setting if specified
        self.config['HEADLESS'] = headless
        
        # Initialize the orchestrator
        self.orchestrator = Orchestrator(self.config)
    
    def process_urls(self, urls: List[str], output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process a list of URLs to extract dataset links.
        
        Args:
            urls: List of URLs to process
            output_file: Path to save the results (optional)
            
        Returns:
            DataFrame containing the extracted dataset links
        """
        # TEMPORARY TESTING CODE - REMOVE FOR PRODUCTION
        # This is just for testing the API structure
        import pandas as pd
        
        print(f"Would process these URLs: {urls}")
        
        # Create a dummy DataFrame for testing
        data = {
            'source_url': urls,
            'link': [f"{url}/dataset" for url in urls],
            'text': ["Sample dataset link" for _ in urls],
            'classification': ["dataset" for _ in urls]
        }
        results = pd.DataFrame(data)
        
        # Simulate writing to file if specified
        if output_file:
            print(f"Would write results to: {output_file}")
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results.to_csv(output_file, index=False)
            
        return results
        
        # REAL IMPLEMENTATION - UNCOMMENT FOR PRODUCTION
        """
        # Create temporary input file with the URLs
        temp_input_file = os.path.join(os.path.dirname(__file__), 'config', 'temp_input.txt')
        with open(temp_input_file, 'w') as f:
            for url in urls:
                f.write(f"{url}\n")
                
        # Update config to use the temporary input file
        self.config['input_urls_filepath'] = temp_input_file
        
        # Set output file if provided
        if output_file:
            self.config['full_output_file'] = output_file
            
        # Process URLs and return results
        results = self.orchestrator.run()
        
        # Clean up temporary file
        os.remove(temp_input_file)
        
        return results
        """
    
    def process_file(self, input_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process URLs from an input file.
        
        Args:
            input_file: Path to the input file containing URLs (one per line)
            output_file: Path to save the results (optional)
            
        Returns:
            DataFrame containing the extracted dataset links
        """
        # TEMPORARY TESTING CODE - REMOVE FOR PRODUCTION
        # Read the first few URLs from the file
        with open(input_file, 'r') as f:
            urls = [line.strip() for line in f.readlines()[:5]]
        
        print(f"Would process these URLs from file {input_file}: {urls}")
        
        # Create a dummy DataFrame for testing
        data = {
            'source_url': urls,
            'link': [f"{url}/dataset" for url in urls],
            'text': ["Sample dataset link from file" for _ in urls],
            'classification': ["dataset" for _ in urls]
        }
        results = pd.DataFrame(data)
        
        # Simulate writing to file if specified
        if output_file:
            print(f"Would write results to: {output_file}")
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results.to_csv(output_file, index=False)
            
        return results
        
        # REAL IMPLEMENTATION - UNCOMMENT FOR PRODUCTION
        """
        # Update config to use the provided input file
        self.config['input_urls_filepath'] = input_file
        
        # Set output file if provided
        if output_file:
            self.config['full_output_file'] = output_file
            
        # Process URLs and return results
        results = self.orchestrator.run()
        
        return results
        """
    
    def configure(self, config_updates: Dict) -> None:
        """
        Update the configuration with new values.
        
        Args:
            config_updates: Dictionary with configuration updates
        """
        self.config.update(config_updates)
        self.orchestrator = Orchestrator(self.config)
