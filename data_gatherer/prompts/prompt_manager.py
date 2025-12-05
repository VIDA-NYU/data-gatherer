import json
import hashlib
import os
from data_gatherer.resources_loader import load_prompt
from data_gatherer.env import CACHE_BASE_DIR

class PromptManager:
    def __init__(self, prompt_dir, logger,
                 save_dir="data_gatherer/prompts/prompt_evals", save_dynamic_prompts=False, log_file_override=None,
                 save_responses_to_cache=False, use_cached_responses=False):
        self.prompt_dir = prompt_dir
        self.prompt_save_dir = save_dir
        self.response_file = os.path.join(CACHE_BASE_DIR + "process_url_cache.json")
        self.logger = logger
        if not os.path.exists(self.response_file) and save_responses_to_cache and use_cached_responses:
            with open(self.response_file, 'w') as f:
                json.dump({}, f)

    def save_prompt(self, prompt_id, prompt_content):
        """Save the static prompt content if it does not already exist."""
        prompt_file = os.path.join(self.prompt_save_dir, f"{prompt_id}.json")

        # Ensure the directory exists
        os.makedirs(self.prompt_save_dir, exist_ok=True)

        with open(prompt_file, 'w') as f:
            json.dump(prompt_content, f)

        self.logger.info(f"Prompt saved to {prompt_file}")

    def load_prompt(self, prompt_name, user_prompt_dir=None, subdir=""):
        """Load a static prompt template."""
        self.logger.info(f"Loading prompt: {prompt_name} from user_prompt_dir: {user_prompt_dir}, subdir: {subdir}")
        return load_prompt(prompt_name, user_prompt_dir=user_prompt_dir, subdir=subdir)

    def render_prompt(self, static_prompt, entire_doc, **dynamic_parts):
        """Render a dynamic prompt by replacing placeholders."""
        self.logger.debug(f"Rendering prompt with static_prompt:{static_prompt}, entire_doc: {entire_doc}, dynamic parts({type(dynamic_parts)}): {dynamic_parts}")
        
        # Escape curly braces in dynamic_parts to prevent .format() from interpreting them as placeholders
        escaped_dynamic_parts = {}
        for key, value in dynamic_parts.items():
            if isinstance(value, str):
                # Replace { with {{ and } with }} to escape them for .format()
                escaped_dynamic_parts[key] = value.replace("{", "{{").replace("}", "}}")
            else:
                escaped_dynamic_parts[key] = value
        
        if entire_doc or "parts" in static_prompt[0]:
            # Handle the "parts" elements in the prompt
            for item in static_prompt:
                if "parts" in item:
                    item["parts"] = [
                        {
                            "text": part["text"].format(**escaped_dynamic_parts)
                            if "{" in part["text"] and "}" in part["text"]
                            else part["text"]
                        }
                        for part in item["parts"]
                    ]
                elif "content" in item:
                    if "{" in item["content"] and "}" in item["content"]:
                        item["content"] = item["content"].format(**escaped_dynamic_parts)
                    else:
                        item["content"]
            return static_prompt
        else:
            self.logger.debug(f"Rendering prompt with dynamic parts({type(dynamic_parts)}): {dynamic_parts}, and items: {static_prompt}")
            
            # Escape curly braces in dynamic_parts to prevent .format() from interpreting them as placeholders
            escaped_dynamic_parts = {}
            for key, value in dynamic_parts.items():
                if isinstance(value, str):
                    # Replace { with {{ and } with }} to escape them for .format()
                    escaped_dynamic_parts[key] = value.replace("{", "{{").replace("}", "}}")
                else:
                    escaped_dynamic_parts[key] = value
            
            ret = [
                {**item, "content": item["content"].format(**escaped_dynamic_parts)}
                for item in static_prompt
            ]
            return ret

    # In PromptManager

    def render_prompts_for_chunks(self, static_prompt, content, token_limit=2046, tokenizer=None, entire_doc=False,
                                  **dynamic_parts):
        """
        Split content into chunks fitting the token limit, render prompts for each chunk.
        Returns a list of rendered prompts.
        """

        # Helper to count tokens
        def count_tokens(text):
            return len(tokenizer.encode(text)) if tokenizer else len(text.split())//3.2

        # Split content into chunks
        chunks = []
        current = ""
        for paragraph in content.split('\n'):
            if count_tokens(current + paragraph) > token_limit and current:
                chunks.append(current)
                current = paragraph
            else:
                current += "\n" + paragraph if current else paragraph
        if current:
            chunks.append(current)

        # Render prompts for each chunk
        prompts = []
        for chunk in chunks:
            dynamic_parts_chunk = {**dynamic_parts, "content": chunk}
            rendered = self.render_prompt(static_prompt, entire_doc, **dynamic_parts_chunk)
            prompts.append(rendered)
        return prompts

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
            self.logger.warning(f"Response file does not exist: {self.response_file}")
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
