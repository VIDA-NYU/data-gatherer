import json
import hashlib
import os


class PromptManager:
    def __init__(self, prompt_dir, response_file="LLMs_responses.json", save_dir="prompts/prompt_evals"):
        self.prompt_dir = prompt_dir
        self.prompt_save_dir = save_dir
        self.response_file = response_file
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

    def load_prompt(self, prompt_id):
        """Load a static prompt template."""
        prompt_file = os.path.join(self.prompt_dir, f"{prompt_id}.json")
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with open(prompt_file, 'r') as f:
            return json.load(f)

    def render_prompt(self, static_prompt, **dynamic_parts):
        """Render a dynamic prompt by replacing placeholders."""
        return [
            {**item, "content": item["content"].format(**dynamic_parts)}
            for item in static_prompt
        ]

    def save_response(self, prompt_id, response):
        """Save the response with prompt_id as the key. skip if prompt_id is already responded."""
        with open(self.response_file, 'r+') as f:
            responses = json.load(f)
            if prompt_id in responses:
                return
            responses[prompt_id] = response
            f.seek(0)
            json.dump(responses, f, indent=2)

    def retrieve_response(self, prompt_id):
        """Retrieve a saved response based on the prompt_id."""
        if not os.path.exists(self.response_file):
            return None
        with open(self.response_file, 'r') as f:
            responses = json.load(f)
            return responses.get(prompt_id)

    def _calculate_checksum(self, content):
        """Calculate checksum for a given content."""
        return hashlib.sha256(content.encode()).hexdigest()