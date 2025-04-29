import json
import hashlib
import os
from data_gatherer.resources_loader import load_prompt

class PromptManager:
    def __init__(self, prompt_dir, logger, response_file="LLMs_responses_cache.json", save_dir="prompt_evals"):
        self.prompt_dir = prompt_dir
        self.prompt_save_dir = save_dir
        self.response_file = response_file
        self.logger = logger
        os.makedirs(self.prompt_dir, exist_ok=True)
        if not os.path.exists(self.response_file):
            with open(self.response_file, 'w') as f:
                json.dump({}, f)

    def save_prompt(self, prompt_id, prompt_content):
        """Save the static prompt content if it does not already exist."""
        prompt_file = os.path.join(self.prompt_save_dir, f"{prompt_id}.json")
        if os.path.exists(prompt_file):
            pass
        with open(prompt_file, 'w') as f:
            json.dump(prompt_content, f)

    def load_prompt(self, prompt_name, user_prompt_dir=None, subdir=""):
        """Load a static prompt template."""
        self.logger.info(f"Loading prompt: {prompt_name} from user_prompt_dir: {user_prompt_dir}, subdir: {subdir}")
        return load_prompt(prompt_name, user_prompt_dir=user_prompt_dir, subdir=subdir)

    def render_prompt(self, static_prompt, entire_doc, **dynamic_parts):
        """Render a dynamic prompt by replacing placeholders."""
#        if entire_doc:
        if entire_doc or "parts" in static_prompt[0]:
            # Handle the "parts" elements in the prompt
            for item in static_prompt:
                if "parts" in item:
                    item["parts"] = [
                        {
                            "text": part["text"].format(**dynamic_parts)
                            if "{" in part["text"] and "}" in part["text"]
                            else part["text"]
                        }
                        for part in item["parts"]
                    ]
                elif "content" in item:
                    if "{" in item["content"] and "}" in item["content"]:
                        item["content"] = item["content"].format(**dynamic_parts)
                    else:
                        item["content"]
            return static_prompt
        else:
            self.logger.info(f"Rendering prompt with dynamic parts({type(dynamic_parts)}): {dynamic_parts}, and items: {static_prompt}")
            return [
                {**item, "content": item["content"].format(**dynamic_parts)}
                for item in static_prompt
            ]

    def save_response(self, prompt_id, response):
        """Save the response with prompt_id as the key. Skip if prompt_id is already responded."""
        self.logger.info(f"Saving response for prompt_id: {prompt_id}")

        # Load existing responses safely
        with open(self.response_file, 'r') as f:
            responses = json.load(f)  # The file is assumed to be well-formed JSON

        # Skip if the prompt_id already has a response
        if prompt_id in responses:
            self.logger.info(f"Prompt already responded: {prompt_id}")
            return

        # Add new response
        responses[prompt_id] = response

        # Write back safely, ensuring no leftover data
        with open(self.response_file, 'w') as f:
            json.dump(responses, f, indent=2)
            f.truncate()  # Ensure no extra data remains

    def retrieve_response(self, prompt_id):
        """Retrieve a saved response based on the prompt_id."""
        if not os.path.exists(self.response_file):
            return None
        with open(self.response_file, 'r') as f:
            self.logger.debug(f"Retrieving response for prompt_id: {prompt_id}")
            responses = json.load(f)
            if prompt_id not in responses:
                return None
            return responses.get(prompt_id)

    def _calculate_checksum(self, prompt):
        """Calculate checksum for a given content."""
        self.logger.debug(f"Calculating checksum for content: prompt")
        return hashlib.sha256(prompt.encode()).hexdigest()