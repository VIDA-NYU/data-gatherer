#!/usr/bin/env python3
"""
Example script demonstrating how to use the Data Gatherer package.
"""

from data_gatherer import DataGatherer

def main():
    # Initialize data gatherer with default configuration
    gatherer = DataGatherer()
    
    # Example 1: Process a list of URLs
    print("Example 1: Processing a list of URLs")
    urls = [
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10113009",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8608617"
    ]
    results = gatherer.process_urls(urls, output_file="output/example_results.csv")
    print(f"Found {len(results)} data links")
    print(results.head())
    
    # Example 2: Process URLs from a file
    print("\nExample 2: Processing URLs from a file")
    results = gatherer.process_file("input/test_input.txt", output_file="output/example_file_results.csv")
    print(f"Found {len(results)} data links")
    print(results.head())

if __name__ == "__main__":
    main()