import logging
from ollama import Client
from openai import OpenAI
import google.generativeai as genai
from portkey_ai import Portkey
from data_gatherer.prompts.prompt_manager import PromptManager
from data_gatherer.env import PORTKEY_GATEWAY_URL, PORTKEY_API_KEY, PORTKEY_ROUTE, PORTKEY_CONFIG, NYU_LLM_API, GPT_API_KEY, GEMINI_KEY, DATA_GATHERER_USER_NAME
from data_gatherer.llm.response_schema import *

class LLMClient_dev:
    def __init__(self, model: str, logger=None, save_prompts: bool = False, use_portkey_for_gemini: bool = True):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Initializing LLMClient with model: {self.model}")
        self.use_portkey_for_gemini = use_portkey_for_gemini
        self._initialize_client(model)
        self.save_prompts = save_prompts
        self.prompt_manager = PromptManager("data_gatherer/prompts/prompt_templates/metadata_prompts", self.logger)

    def _initialize_client(self, model):
        if self.use_portkey_for_gemini and 'gemini' in model:
            self.portkey = Portkey(
                api_key=PORTKEY_API_KEY,
                virtual_key=PORTKEY_ROUTE,
                base_url=PORTKEY_GATEWAY_URL,
                config=PORTKEY_CONFIG,
                metadata={"_user": DATA_GATHERER_USER_NAME}
            )

        elif model.startswith('gemma3'):
            self.client = Client(host="http://localhost:11434")

        elif model == 'gemma2:9b':
            self.client = Client(host=NYU_LLM_API)  # env variable

        elif model == 'gpt-4o-mini':
            self.client = OpenAI(api_key=GPT_API_KEY)

        elif model == 'gpt-4o':
            self.client = OpenAI(api_key=GPT_API_KEY)

        elif model == 'gemini-1.5-flash':
            if not self.use_portkey_for_gemini:
                genai.configure(api_key=GEMINI_KEY)
                self.client = genai.GenerativeModel('gemini-1.5-flash')
            else:
                self.client = None

        elif model == 'gemini-2.0-flash-exp':
            if not self.use_portkey_for_gemini:
                genai.configure(api_key=GEMINI_KEY)
                self.client = genai.GenerativeModel('gemini-2.0-flash-exp')
            else:
                self.client = None

        elif model == 'gemini-2.0-flash':
            if not self.use_portkey_for_gemini:
                genai.configure(api_key=GEMINI_KEY)
                self.client = genai.GenerativeModel('gemini-2.0-flash')
            else:
                self.client = None

        elif model == 'gemini-1.5-pro':
            if not self.use_portkey_for_gemini:
                genai.configure(api_key=GEMINI_KEY)
                self.client = genai.GenerativeModel('gemini-1.5-pro')
            else:
                self.client = None
        else:
            raise ValueError(f"Unsupported LLM name: {model}.")

        self.logger.info(f"Client initialized: {self.client}")

    def api_call(self, content, response_format, temperature=0.0, **kwargs):
        self.logger.info(f"Calling {self.model} with prompt length {len(content)}")
        if self.model.startswith('gpt'):
            return self._call_openai(content, **kwargs)
        elif self.model.startswith('gemini'):
            if self.use_portkey_for_gemini:
                return self._call_portkey_gemini(content, **kwargs)
            else:
                return self._call_gemini(content, **kwargs)
        elif self.model.startswith('gemma'):
            return self._call_ollama(content, response_format, temperature=temperature)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _call_openai(self, messages, temperature=0.0, **kwargs):
        self.logger.info(f"Calling OpenAI")
        if self.save_prompts:
            self.prompt_manager.save_prompt(prompt_id='abc', prompt_content=messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format=kwargs.get('response_format', {"type": "json_object"})
        )
        return response.choices[0].message.content

    def _call_gemini(self, messages, temperature=0.0, **kwargs):
        self.logger.info(f"Calling Gemini")
        if self.save_prompts:
            self.prompt_manager.save_prompt(prompt_id='abc', prompt_content=messages)
        response = self.client.generate_content(
            messages,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=temperature,
            )
        )
        return response.text

    def _call_ollama(self, messages, response_format, temperature=0.0):
        self.logger.info(f"Calling Ollama with messages: {messages}")
        if self.save_prompts:
            self.prompt_manager.save_prompt(prompt_id='abc', prompt_content=messages)
        response = self.client.chat(model=self.model, options={"temperature": temperature}, messages=messages,
                                    format=response_format)
        self.logger.info(f"Ollama response: {response['message']['content']}")
        return response['message']['content']

    def _call_portkey_gemini(self, messages, temperature=0.0, **kwargs):
        self.logger.info(f"Calling Gemini via Portkey")
        if self.save_prompts:
            self.prompt_manager.save_prompt(prompt_id='abc', prompt_content=messages)
        portkey_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        self.logger.debug(f"Portkey payload: {portkey_payload}")
        try:
            response = self.portkey.chat.completions.create(
                api_key=PORTKEY_API_KEY,
                route=PORTKEY_ROUTE,
                headers={"Content-Type": "application/json"},
                **portkey_payload
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Portkey API call failed: {e}")

    def get_datasets_from_content(self, messages, model=None, temperature=0.0, full_document_read=True):
        """
        Handles the LLM API call for dataset extraction, given prepared messages.
        Returns the raw response from the LLM.
        """
        model = model or self.model
        if model == 'gemma2:9b':
            response = self.client.chat(model=model, options={"temperature": temperature}, messages=messages)
            return response['message']['content']
        elif model in ['gpt-4o-mini', 'gpt-4o']:
            if full_document_read:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_schema"}
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
            return response.choices[0].message.content
        elif model.startswith('gemini'):
            if self.use_portkey_for_gemini:
                portkey_payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                response = self.portkey.chat.completions.create(
                    api_key=PORTKEY_API_KEY,
                    route=PORTKEY_ROUTE,
                    **portkey_payload
                )
                return response
            else:
                response = self.client.generate_content(
                    messages,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=temperature,
                    )
                )
                return response.text
        else:
            raise ValueError(f"Unsupported model: {model}")
