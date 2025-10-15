import re
from ollama import Client
import os
from data_gatherer.prompts.prompt_manager import PromptManager
from data_gatherer.resources_loader import load_config
import pandas as pd

class LLMClassifier:
    def __init__(self, logger, retrieval_patterns_file='retrieval_patterns.json'):
        self.retrieval_patterns = load_config(retrieval_patterns_file)
        self.logger = logger
        self.setup_client()
        self.show_classify_stats = True
        self.prompt_manager = PromptManager("data_gatherer/prompts/prompt_templates", self.logger)

    def setup_client(self, llm_name='put your model'):
        """
        Initialize the client for LLM classification.
        """
        if llm_name == 'gemma2:9b':
            self.llm_client = Client(host=os.environ['OLLAMA_CLIENT'])  # env variable

    def classify_element(self, element):
        """
        Classify a single anchor element based on its content, text, and context.
        The element is a dictionary containing attributes from parsed_data DataFrame.
        """
        skip_patterns = self.retrieval_patterns['general']['skip_llm_classification_patterns']
        for pattern,classified in skip_patterns.items():
            if re.search(pattern, element['download_link']):
                self.logger.info(f"Skipping LLM classification for {element['reconstructed_link']}")
                return classified

        labels_used = ["Tabular Data", "Supplementary Material", "External Navigation Link", "Related Works",
                       "Non relevant"]

        rule_based_class = ""
        if element['rule_based_classification'] != "n/a":
            rule_based_class = f" - Rule-based model classification - maybe incorrect - output: {element['rule_based_classification']}"

        # classify link based on its attributes
        response = self.llm_client.chat(model='gemma2:9b', messages=[
            {
                "role": "system",
                 "content": """You are an intelligent system that classifies HTML anchor elements with links. You will 
                 classify the link using one of the predefined categories, with a focus on relevance to experimental data, 
                 supplementary material, or related external resources. Output must be no longer than 40 characters."""
            },
            {
                "role": "user",
                "content": f"""For the link and element classification you will consider the following details:
                                - The description of the link: {element['raw_description']}
                                - The text of the anchor element: {element['text']}
                                - The link itself: {element['reconstructed_link']}
                                {rule_based_class}
                                You can use one of the labels you have already used for past elements: {", ".join(labels_used)}.
                                Do not provide any other output text or explanation.""",
            },
        ])

        system_message = response['message']['content']
        # self.logger.info(system_message)
        output_class = " ".join(system_message.split())

        if output_class == 'Non relevant':
            return 'Non relevant'

        else:
            msg = f"""response: {output_class}
                      - description:  {element['raw_description']}
                      - text:  {element['text']}
                      - link:  {element['reconstructed_link']}
                      - past labels:  {labels_used}
                      """
            self.logger.info(msg)

        return output_class

    def classify_anchor_elements_links(self, parsed_data):
        """
        Apply classification to each element (row) in the parsed_data DataFrame.
        """
        parsed_data['classification'] = parsed_data.apply(self.classify_element, axis=1)

        if self.show_classify_stats:
            self.logger.info(f"Classification Stats: {parsed_data['classification'].value_counts()}")

        return parsed_data

    def get_raw_data_files(self, data_resources_dfs):
        """
        Get the raw data files from the data resources.

        :param data_resources_dfs: Dictionary of data resources dataframes

        :return: Dataframe of raw data file URLs
        """
        self.logger.debug(f"Input type: {type(data_resources_dfs)}")
        resources = []

        for publication_url, df in data_resources_dfs.items():
            if not isinstance(df, pd.DataFrame):
                self.logger.warning(f"Skipping non-DataFrame entry for URL: {publication_url}")
                continue

            self.logger.info(f"Processing DataFrame for URL: {publication_url}")
            df['publication_url'] = publication_url  # Assign publication_url to all rows
            resources.append(df)

        if resources:
            resources_df = pd.concat(resources, ignore_index=True)
            self.logger.debug(f"Resources type: {type(resources_df)}")
            self.logger.debug(f"Resources columns: {resources_df.columns}")
            resources_df = resources_df.dropna(subset=['dataset_identifier'])
            resources_df = resources_df[[
                'publication_url', 'dataset_identifier', 'data_repository', 'dataset_webpage',]]
            return resources_df
        else:
            self.logger.warning("No valid DataFrames found in input.")
            return pd.DataFrame()  # Return an empty DataFrame if no valid input

    @staticmethod
    def get_domain_from_href(href):
        """Extract domain from the URL for domain-specific rules."""
        return href.split("/")[2].replace("www.", "")