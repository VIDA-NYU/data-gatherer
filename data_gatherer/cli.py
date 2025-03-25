import argparse
import sys
import os
from data_gatherer.api import DataGatherer

def main():
    """
    Command-line interface for the data-gatherer package.
    """
    parser = argparse.ArgumentParser(description='Extract dataset links from scientific articles')
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--input-file', '-i', type=str, help='Path to input file with URLs (one per line)')
    input_group.add_argument('--urls', '-u', nargs='+', help='One or more URLs to process')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-file', '-o', type=str, help='Path to output CSV file')
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument('--config', '-c', type=str, help='Path to custom configuration file')
    config_group.add_argument('--headless', action='store_true', default=True, help='Run browser in headless mode')
    config_group.add_argument('--no-headless', action='store_false', dest='headless', help='Run browser in visible mode')
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not args.input_file and not args.urls:
        parser.error('No input provided. Please specify either --input-file or --urls')
    
    # Initialize data gatherer
    gatherer = DataGatherer(config_path=args.config, headless=args.headless)
    
    # Process input and display results
    if args.input_file:
        results = gatherer.process_file(args.input_file, args.output_file)
    else:
        results = gatherer.process_urls(args.urls, args.output_file)
    
    # Display summary of results
    print(f"Processed {len(results)} items")
    print(f"Results saved to {args.output_file or gatherer.config['full_output_file']}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
