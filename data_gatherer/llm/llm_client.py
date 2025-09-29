import logging
import json
import re
from ollama import Client
from openai import OpenAI
import google.generativeai as genai
from portkey_ai import Portkey
from json_repair import repair_json
from data_gatherer.prompts.prompt_manager import PromptManager
from data_gatherer.env import PORTKEY_GATEWAY_URL, PORTKEY_API_KEY, PORTKEY_ROUTE, PORTKEY_CONFIG, OLLAMA_CLIENT, GPT_API_KEY, GEMINI_KEY, DATA_GATHERER_USER_NAME
from data_gatherer.llm.response_schema import *

class LLMClient_dev:
    def __init__(self, model: str, logger=None, save_prompts: bool = False, use_portkey: bool = True, 
                 save_dynamic_prompts: bool = False, save_responses_to_cache: bool = False, 
                 use_cached_responses: bool = False, prompt_dir: str = "data_gatherer/prompts/prompt_templates"):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        print(f"Initializing LLMClient with model: {self.model}")
        self.use_portkey = use_portkey
        self.save_prompts = save_prompts
        self.save_dynamic_prompts = save_dynamic_prompts
        self.save_responses_to_cache = save_responses_to_cache
        self.use_cached_responses = use_cached_responses
        
        # Determine full document read capability
        entire_document_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-2.0-flash",
                                  "gemini-2.5-flash", "gpt-4o", "gpt-4o-mini", "gpt-5-nano", "gpt-5-mini", "gpt-5"]
        self.full_document_read = model in entire_document_models
        
        self._initialize_client(model)
        self.prompt_manager = PromptManager(prompt_dir, self.logger,
                                            save_dynamic_prompts=save_dynamic_prompts,
                                            save_responses_to_cache=save_responses_to_cache,
                                            use_cached_responses=use_cached_responses)

    def _initialize_client(self, model):
        print(f"[DEBUG] _initialize_client called with model={model}, use_portkey={self.use_portkey}")
        
        if self.use_portkey and 'gemini' in model:
            print(f"[DEBUG] Initializing Portkey client for Gemini model: {model}")
            self.portkey = Portkey(
                api_key=PORTKEY_API_KEY,
                virtual_key=PORTKEY_ROUTE,
                base_url=PORTKEY_GATEWAY_URL,
                config=PORTKEY_CONFIG,
                metadata={"_user": DATA_GATHERER_USER_NAME}
            )
            self.client = None  # For Portkey, we use self.portkey instead of self.client
            print(f"[DEBUG] Portkey client initialized: {self.portkey}")

        elif model.startswith('gemma3') or model.startswith('qwen'):
            print(f"[DEBUG] Initializing Ollama client for model: {model}")
            self.client = Client(host="http://localhost:11434")

        elif model == 'gemma2:9b':
            print(f"[DEBUG] Initializing Ollama client for gemma2:9b")
            self.client = Client(host=OLLAMA_CLIENT)  # env variable

        elif model.startswith('gpt'):
            print(f"[DEBUG] Initializing OpenAI client for model: {model}")
            self.client = OpenAI(api_key=GPT_API_KEY)

        elif model.startswith('gemini') and not self.use_portkey:
            print(f"[DEBUG] Initializing direct Gemini client for model: {model}")
            genai.configure(api_key=GEMINI_KEY)
            self.client = genai.GenerativeModel(model)
            
        else:
            print(f"[DEBUG] Unsupported model: {model}")
            raise ValueError(f"Unsupported LLM name: {model}.")

        print(f"[DEBUG] Client initialization complete. self.client: {self.client}, self.portkey: {getattr(self, 'portkey', 'Not set')}")

    def api_call(self, content, response_format, temperature=0.0, **kwargs):
        print(f"Calling {self.model} with prompt length {len(content)}")
        if self.model.startswith('gpt'):
            return self._call_openai(content, **kwargs)
        elif self.model.startswith('gemini'):
            if self.use_portkey:
                return self._call_portkey_gemini(content, **kwargs)
            else:
                return self._call_gemini(content, **kwargs)
        elif self.model.startswith('gemma') or "qwen" in self.model:
            return self._call_ollama(content, response_format, temperature=temperature)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _call_openai(self, messages, temperature=0.0, **kwargs):
        print(f"Calling OpenAI")
        if self.save_prompts:
            self.prompt_manager.save_prompt(prompt_id='abc', prompt_content=messages)
        if 'gpt-5' in self.model:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                text={
                    "format": kwargs.get('response_format', {"type": "json_object"})
                }
            )
        elif 'gpt-4' in self.model:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                text={
                "format": kwargs.get('response_format', {"type": "json_object"})
            }
        )
        return response.output

    def _call_gemini(self, messages, temperature=0.0, **kwargs):
        print(f"Calling Gemini")
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
        print(f"Calling Ollama with messages: {messages}")
        if self.save_prompts:
            self.prompt_manager.save_prompt(prompt_id='abc', prompt_content=messages)
        response = self.client.chat(model=self.model, options={"temperature": temperature}, messages=messages,
                                    format=response_format)
        print(f"Ollama response: {response['message']['content']}")
        return response['message']['content']

    def _call_portkey_gemini(self, messages, temperature=0.0, **kwargs):
        print(f"Calling Gemini via Portkey")
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

    def make_llm_call(self, messages, temperature: float = 0.0, response_format=None, 
                      full_document_read: bool = None) -> str:
        """
        Generic method to make LLM API calls. This is task-agnostic and only handles the raw API interaction.
        
        :param messages: The messages/prompt to send to the LLM
        :param temperature: Temperature setting for the model
        :param response_format: Optional response format schema
        :param full_document_read: Override for full document read capability
        :return: Raw response from the LLM as string
        """
        if full_document_read is None:
            full_document_read = self.full_document_read
            
        print(f"Making LLM call with model: {self.model}, temperature: {temperature}")
        
        if self.model == 'gemma2:9b':
            response = self.client.chat(model=self.model, options={"temperature": temperature}, messages=messages)
            print(f"Response received from model: {response.get('message', {}).get('content', 'No content')}")
            return response['message']['content']
            
        elif self.model in ['gemma3:1b', 'gemma3:4b', 'qwen:4b']:
            if response_format:
                return self.api_call(messages, response_format=response_format.model_json_schema(), temperature=temperature)
            else:
                return self.api_call(messages, response_format={}, temperature=temperature)
                
        elif 'gpt' in self.model:
            response = None
            if 'gpt-5' in self.model:
                if full_document_read and response_format:
                    response = self.client.responses.create(
                        model=self.model,
                        input=messages,
                        text={"format": response_format}
                    )
                else:
                    response = self.client.responses.create(
                        model=self.model,
                        input=messages
                    )
            elif 'gpt-4o' in self.model:
                if full_document_read and response_format:
                    response = self.client.responses.create(
                        model=self.model,
                        input=messages,
                        temperature=temperature,
                        text={"format": response_format}
                    )
                else:
                    response = self.client.responses.create(
                        model=self.model,
                        input=messages,
                        temperature=temperature
                    )
            
            print(f"GPT response: {response.output}")
            return response.output
                
        elif 'gemini' in self.model:
            if self.use_portkey:
                # Portkey Gemini call
                portkey_payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                try:
                    response = self.portkey.chat.completions.create(**portkey_payload)
                    print(f"Portkey Gemini response: {response}")
                    return response
                except Exception as e:
                    self.logger.error(f"Portkey Gemini call failed: {e}")
                    raise RuntimeError(f"Portkey API call failed: {e}")
            else:
                # Direct Gemini call
                if 'gemini' in self.model and 'flash' in self.model:
                    response = self.client.generate_content(
                        messages,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=list[Dataset] if response_format else None
                        )
                    )
                elif self.model == 'gemini-1.5-pro':
                    response = self.client.generate_content(
                        messages,
                        request_options={"timeout": 1200},
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=list[Dataset] if response_format else None
                        )
                    )
                
                try:
                    candidates = response.candidates
                    if candidates:
                        print(f"Found {len(candidates)} candidates in the response.")
                        response_text = candidates[0].content.parts[0].text
                        print(f"Gemini response text: {response_text}")
                        return response_text
                    else:
                        self.logger.error("No candidates found in the response.")
                        return ""
                except Exception as e:
                    self.logger.error(f"Error processing Gemini response: {e}")
                    raise RuntimeError(f"Gemini response processing failed: {e}")
        else:
            raise ValueError(f"Unsupported model: {self.model}. Please use a supported LLM model.")
    
    def process_llm_response(self, raw_response, response_format=None, expected_key=None):
        """
        Task-agnostic method to process LLM responses based on model type.
        Handles parsing, normalization, and basic post-processing.
        
        :param raw_response: Raw response from the LLM
        :param response_format: Expected response format schema
        :param expected_key: Expected key in JSON response (e.g., 'datasets')
        :return: Processed and normalized response
        """
        print(f"[DEBUG] process_llm_response called with model: {self.model}")
        print(f"[DEBUG] raw_response type: {type(raw_response)}, length: {len(str(raw_response))}")
        print(f"[DEBUG] response_format: {response_format}")
        print(f"[DEBUG] expected_key: {expected_key}")
        print(f"[DEBUG] raw_response content (first 500 chars): {str(raw_response)[:500]}")
        
        if self.model == 'gemma2:9b':
            print(f"[DEBUG] Processing gemma2:9b response")
            # Split by newlines for this model
            result = raw_response.split("\n")
            print(f"[DEBUG] gemma2:9b result: {result}")
            return result
            
        elif self.model in ['gemma3:1b', 'gemma3:4b', 'qwen:4b']:
            print(f"[DEBUG] Processing {self.model} response")
            parsed_resp = self.safe_parse_json(raw_response)
            print(f"[DEBUG] Parsed JSON response: {parsed_resp}")
            if isinstance(parsed_resp, dict) and expected_key and expected_key in parsed_resp:
                print(f"[DEBUG] Found expected key '{expected_key}' in parsed response")
                result = self.normalize_response_format(parsed_resp[expected_key])
            else:
                print(f"[DEBUG] Expected key '{expected_key}' not found or not dict, using full response")
                result = self.normalize_response_format(parsed_resp)
            print(f"[DEBUG] Final result for {self.model}: {result}")
            return result
                
        elif 'gpt' in self.model:
            print(f"[DEBUG] Processing GPT model response")
            parsed_response = self.safe_parse_json(raw_response)
            print(f"[DEBUG] GPT parsed response: {parsed_response}, type: {type(parsed_response)}")
            if self.full_document_read and isinstance(parsed_response, dict):
                result = parsed_response.get(expected_key, []) if expected_key else parsed_response
                print(f"[DEBUG] GPT full_document_read=True, extracted result: {result}")
            else:
                result = parsed_response or []
                print(f"[DEBUG] GPT full_document_read=False, result: {result}")
            final_result = self.normalize_response_format(result)
            print(f"[DEBUG] GPT final normalized result: {final_result}")
            return final_result
            
        elif 'gemini' in self.model:
            print(f"[DEBUG] Processing Gemini model response, use_portkey: {self.use_portkey}")
            if self.use_portkey:
                # For Portkey, raw_response is already the response object
                parsed_response = self.safe_parse_json(raw_response)
                print(f"[DEBUG] Gemini Portkey parsed response: {parsed_response}")
                if self.full_document_read and isinstance(parsed_response, dict):
                    result = parsed_response.get(expected_key, []) if expected_key else parsed_response
                else:
                    result = parsed_response if isinstance(parsed_response, list) else []
                print(f"[DEBUG] Gemini Portkey result: {result}")
                return self.normalize_response_format(result)
            else:
                # For direct Gemini, raw_response is the text content
                print(f"[DEBUG] Gemini direct response processing")
                try:
                    parsed_response = json.loads(raw_response)
                    print(f"[DEBUG] Gemini direct parsed response: {parsed_response}")
                    result = self.normalize_response_format(parsed_response)
                    print(f"[DEBUG] Gemini direct final result: {result}")
                    return result
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] Gemini JSON decoding error: {e}")
                    self.logger.error(f"JSON decoding error: {e}")
                    return []
        else:
            print(f"[DEBUG] Unsupported model: {self.model}")
            raise ValueError(f"Unsupported model: {self.model}. Please use a supported LLM model.")
    
    def normalize_response_format(self, response):
        """
        Task-agnostic response normalization and deduplication.
        Handles basic post-processing of LLM responses.
        """
        print(f"[DEBUG] normalize_response_format called with response type: {type(response)}")
        print(f"[DEBUG] normalize_response_format input: {response}")
        
        if not response:
            print(f"[DEBUG] Empty response, returning empty list")
            return []
            
        if not isinstance(response, list):
            if isinstance(response, dict):
                print(f"[DEBUG] Converting dict to list: {response}")
                response = [response]
            else:
                print(f"[DEBUG] Non-list, non-dict response, returning as-is: {response}")
                return response
                
        # Basic normalization - remove empty or invalid items
        normalized = []
        for i, item in enumerate(response):
            print(f"[DEBUG] Processing item {i}: {item} (type: {type(item)})")
            if isinstance(item, str) and len(item.strip()) < 3:
                print(f"[DEBUG] Skipping short string: {item}")
                continue
            if isinstance(item, dict) and not any(item.values()):
                print(f"[DEBUG] Skipping empty dict: {item}")
                continue
            print(f"[DEBUG] Adding item to normalized: {item}")
            normalized.append(item)
            
        print(f"[DEBUG] normalize_response_format final result: {normalized}")
        return normalized
    
    def safe_parse_json(self, response_text):
        """
        Task-agnostic JSON parsing with error handling.
        This can be used by any task that needs to parse JSON from LLM responses.
        """
        print(f"[DEBUG] safe_parse_json called with type: {type(response_text)}")
        print(f"[DEBUG] safe_parse_json input (first 300 chars): {str(response_text)[:300]}")
        result = self._safe_parse_json_internal(response_text)
        print(f"[DEBUG] safe_parse_json result: {result}")
        return result
    
    def generate_prompt_id(self, messages, temperature: float = 0.0):
        """Generate a unique prompt ID for caching."""
        return f"{self.model}-{temperature}-{self.prompt_manager._calculate_checksum(str(messages))}"
    

    

    
    def _safe_parse_json_internal(self, response_text):
        """Internal JSON parsing with error handling."""
        print(f"Function_call: _safe_parse_json_internal(response_text {type(response_text)})")
        
        # Handle different response object types
        if hasattr(response_text, "choices"):
            try:
                response_text = response_text.choices[0].message.content
                self.logger.debug(f"Extracted content from response object, type: {type(response_text)}")
            except Exception as e:
                self.logger.warning(f"Could not extract content from response object: {e}")
                return None
        elif isinstance(response_text, list):
            print(f"Response is a list of length: {len(response_text)}")
            for response_item in response_text:
                print(f"List item type: {type(response_item)}")
                if hasattr(response_item, "content"):
                    print(f"Item has content attribute, of type: {type(response_item.content)}")
                    if isinstance(response_item.content, list) and len(response_item.content) > 0:
                        response_text = response_item.content[0].text
                else:
                    response_text = str(response_item)
        elif isinstance(response_text, dict):
            try:
                response_text = response_text["choices"][0]["message"]["content"]
            except Exception as e:
                self.logger.warning(f"Could not extract content from response dict: {e}")
                return None
        
        if not isinstance(response_text, str):
            return response_text
        
        response_text = response_text.strip()
        
        # Remove markdown formatting
        if response_text.startswith("```"):
            response_text = re.sub(r"^```[a-zA-Z]*\n?", "", response_text)
            response_text = re.sub(r"\n?```$", "", response_text)
        
        self.logger.debug(f"Cleaned response text for JSON parsing: {response_text[:500]}")
        
        try:
            # First try standard parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            self.logger.warning("Initial JSON parsing failed. Attempting json_repair...")
            try:
                repaired = repair_json(response_text)
                repaired = re.sub(r',\s*\{\}\]', ']', repaired)  # Remove trailing empty objects in lists
                return json.loads(repaired)
            except Exception as e:
                self.logger.warning(f"json_repair failed: {e}")
                return None


