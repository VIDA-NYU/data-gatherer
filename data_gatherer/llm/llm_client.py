import logging
import json
import re
import time
from typing import Dict, List, Any, Optional, Union
from ollama import Client
from openai import OpenAI
import google.generativeai as genai
from portkey_ai import Portkey
from json_repair import repair_json
from data_gatherer.prompts.prompt_manager import PromptManager
from data_gatherer.env import PORTKEY_GATEWAY_URL, PORTKEY_API_KEY, PORTKEY_ROUTE, PORTKEY_CONFIG, OLLAMA_CLIENT, GPT_API_KEY, GEMINI_KEY, DATA_GATHERER_USER_NAME
from data_gatherer.llm.response_schema import *
from data_gatherer.llm.batch_storage import BatchStorageManager, BatchRequestBuilder

class LLMClient_dev:
    def __init__(self, model: str, logger=None, save_prompts: bool = False, use_portkey: bool = True, 
                 save_dynamic_prompts: bool = False, save_responses_to_cache: bool = False, 
                 use_cached_responses: bool = False, prompt_dir: str = "data_gatherer/prompts/prompt_templates"):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Initializing LLMClient with model: {self.model}")
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
        
        # Initialize batch processing utilities
        self.batch_storage = BatchStorageManager(self.logger)
        self.batch_builder = BatchRequestBuilder(self.logger)

    def _initialize_client(self, model):
        self.logger.debug(f"_initialize_client called with model={model}, use_portkey={self.use_portkey}")
        
        if self.use_portkey and 'gemini' in model:
            self.logger.debug(f"Initializing Portkey client for Gemini model: {model}")
            self.llm_client = Portkey(
                api_key=PORTKEY_API_KEY,
                virtual_key=PORTKEY_ROUTE,
                base_url=PORTKEY_GATEWAY_URL,
                config=PORTKEY_CONFIG,
                metadata={"_user": DATA_GATHERER_USER_NAME}
            )
            self.portkey = self.llm_client  # Keep backward compatibility
            self.logger.debug(f"Portkey client initialized: {self.llm_client}")

        elif model.startswith('gemma3') or model.startswith('qwen'):
            self.logger.debug(f"Initializing Ollama client for model: {model}")
            self.llm_client = Client(host="http://localhost:11434")

        elif model == 'gemma2:9b':
            self.logger.debug(f"Initializing Ollama client for gemma2:9b")
            self.llm_client = Client(host=OLLAMA_CLIENT)  # env variable

        elif model.startswith('gpt'):
            self.logger.debug(f"Initializing OpenAI client for model: {model}")
            self.llm_client = OpenAI(api_key=GPT_API_KEY)

        elif model.startswith('gemini') and not self.use_portkey:
            self.logger.debug(f"Initializing direct Gemini client for model: {model}")
            genai.configure(api_key=GEMINI_KEY)
            self.llm_client = genai.GenerativeModel(model)
        
        elif model.startswith('local-flan-t5'):
            self.logger.debug(f"Initializing local Flan-T5 model: {model}")
            model_path = self._resolve_local_model_path(model)
            from data_gatherer.llm.local_model_client import LocalModelClient
            self.llm_client = LocalModelClient(model_path, logger=self.logger)
            self.llm_client.load_model()
        
        elif model.startswith('hf-'):
            self.logger.debug(f"Initializing Hugging Face model client for model: {model}")
            hf_model_name = model[len('hf-'):]
            from data_gatherer.llm.hf_model_client import HFModelClient
            self.llm_client = HFModelClient(hf_model_name, logger=self.logger)
            self.llm_client.load_model()

        else:
            self.logger.debug(f"Unsupported model: {model}")
            raise ValueError(f"Unsupported LLM name: {model}.")

        self.logger.debug(f"Client initialization complete. self.llm_client: {self.llm_client}, self.portkey: {getattr(self, 'portkey', 'Not set')}")

    def _resolve_local_model_path(self, model_name):
        """
        Resolve the local model path from the model name.
        Supports environment variables and relative paths.
        
        :param model_name: Model name (e.g., 'local-flan-t5-dataset-extraction')
        :return: Absolute path to the model directory
        """
        import os
        from pathlib import Path
        
        # Check if an environment variable is set for local models
        env_var_name = "DATA_GATHERER_LOCAL_MODELS_PATH"
        if env_var_name in os.environ:
            base_path = Path(os.environ[env_var_name])
            self.logger.info(f"Using local models base path from environment: {base_path}")
        else:
            # Default to scripts/Local_model_finetuning/flan-t5-models/
            base_path = Path(__file__).parent.parent.parent / "scripts" / "Local_model_finetuning" / "flan-t5-models"
            self.logger.info(f"Using default local models base path: {base_path}")
        
        # Extract the specific model directory name from the model_name
        # e.g., 'local-flan-t5-dataset-extraction' -> look for 'final_model' or specific checkpoint
        if "local-flan-t5" in model_name:
            # Default to final_model directory
            model_path = base_path / "final_model"
            
            # Allow specifying checkpoint in model name, e.g., 'local-flan-t5-checkpoint-1530'
            if "checkpoint" in model_name:
                checkpoint_num = model_name.split("checkpoint-")[-1]
                model_path = base_path / f"checkpoint-{checkpoint_num}"
        else:
            # Fallback: use the model name as-is
            model_path = base_path / model_name.replace("local-", "")
        
        if not model_path.exists():
            self.logger.error(f"Model path does not exist: {model_path}")
            raise FileNotFoundError(f"Local model not found at {model_path}")
        
        self.logger.info(f"Resolved local model path: {model_path}")
        return str(model_path)

    def _call_ft_model(self, messages, temperature=0.0):
        # Extract content from messages format
        if isinstance(messages, list):
            content = messages[-1].get('content', messages[-1])
        else:
            content = messages
        
        return self.llm_client.generate(content, temperature=temperature)
    
    def api_call(self, content, response_format, temperature=0.0, **kwargs):
        self.logger.info(f"Calling {self.model} with prompt length {len(content)}")
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
        self.logger.info(f"Calling OpenAI")
        if self.save_prompts:
            self.prompt_manager.save_prompt(prompt_id='abc', prompt_content=messages)
        if 'gpt-5' in self.model:
            response = self.llm_client.responses.create(
                model=self.model,
                input=messages,
                text={
                    "format": kwargs.get('response_format', {"type": "json_object"})
                }
            )
        elif 'gpt-4' in self.model:
            response = self.llm_client.responses.create(
                model=self.model,
                input=messages,
                text={
                "format": kwargs.get('response_format', {"type": "json_object"})
            }
        )
        return response.output

    def _call_gemini(self, messages, temperature=0.0, **kwargs):
        self.logger.info(f"Calling Gemini")
        if self.save_prompts:
            self.prompt_manager.save_prompt(prompt_id='abc', prompt_content=messages)
        response = self.llm_client.generate_content(
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
        response = self.llm_client.chat(model=self.model, options={"temperature": temperature}, messages=messages,
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

    def make_llm_call(self, messages, temperature: float = 0.0, response_format=None, 
                      full_document_read: bool = None, batch_mode: bool = False, 
                      batch_requests: Optional[List[Dict]] = None,
                      batch_file_path: Optional[str] = None,
                      api_provider: str = 'openai') -> Union[str, Dict[str, Any]]:
        """
        Generic method to make LLM API calls. This is task-agnostic and only handles the raw API interaction.
        
        :param messages: The messages/prompt to send to the LLM (ignored in batch mode)
        :param temperature: Temperature setting for the model
        :param response_format: Optional response format schema
        :param full_document_read: Override for full document read capability
        :param batch_mode: If True, creates batch requests instead of immediate call
        :param batch_requests: List of batch request data (required for batch mode)
        :param batch_file_path: Path for saving batch JSONL file (required for batch mode)
        :param api_provider: API provider for batch processing ('openai' or 'portkey')
        :return: Raw response from the LLM as string, or batch info dict in batch mode
        """
        if full_document_read is None:
            full_document_read = self.full_document_read
        
        # Handle batch mode
        if batch_mode:
            return self._handle_batch_mode(
                batch_requests=batch_requests,
                batch_file_path=batch_file_path,
                temperature=temperature,
                response_format=response_format,
                api_provider=api_provider
            )
            
        self.logger.info(f"Making LLM call with model: {self.model}, temperature: {temperature}")
        
        if self.model == 'gemma2:9b':
            response = self.llm_client.chat(model=self.model, options={"temperature": temperature}, messages=messages)
            self.logger.info(f"Response received from model: {response.get('message', {}).get('content', 'No content')}")
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
                    response = self.llm_client.responses.create(
                        model=self.model,
                        input=messages,
                        text={"format": response_format}
                    )
                else:
                    response = self.llm_client.responses.create(
                        model=self.model,
                        input=messages
                    )
            elif 'gpt-4o' in self.model:
                if full_document_read and response_format:
                    response = self.llm_client.responses.create(
                        model=self.model,
                        input=messages,
                        temperature=temperature,
                        text={"format": response_format}
                    )
                else:
                    response = self.llm_client.responses.create(
                        model=self.model,
                        input=messages,
                        temperature=temperature
                    )
            
            self.logger.info(f"GPT response: {response.output}")
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
                    response = self.llm_client.chat.completions.create(**portkey_payload)
                    self.logger.info(f"Portkey Gemini response: {response}")
                    return response
                except Exception as e:
                    self.logger.error(f"Portkey Gemini call failed: {e}")
                    raise RuntimeError(f"Portkey API call failed: {e}")
            else:
                # Direct Gemini call
                if 'gemini' in self.model and 'flash' in self.model:
                    response = self.llm_client.generate_content(
                        messages,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=list[Dataset] if response_format else None
                        )
                    )
                elif self.model == 'gemini-1.5-pro':
                    response = self.llm_client.generate_content(
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
                        self.logger.info(f"Found {len(candidates)} candidates in the response.")
                        response_text = candidates[0].content.parts[0].text
                        self.logger.info(f"Gemini response text: {response_text}")
                        return response_text
                    else:
                        self.logger.error("No candidates found in the response.")
                        return ""
                except Exception as e:
                    self.logger.error(f"Error processing Gemini response: {e}")
                    raise RuntimeError(f"Gemini response processing failed: {e}")
        
        elif self.model.startswith('local-flan-t5'):
            return self._call_ft_model(messages, temperature=temperature)

        elif self.model.startswith('hf-'):
            return self._call_ft_model(messages, temperature=temperature)

        else:
            raise ValueError(f"Unsupported model: {self.model}. Please use a supported LLM model.")
    
    def process_llm_response(self, raw_response, response_format=None, expected_key=None, from_batch_mode=False):
        """
        Task-agnostic method to process LLM responses based on model type.
        Handles parsing, normalization, and basic post-processing.
        
        :param raw_response: Raw response from the LLM
        :param response_format: Expected response format schema
        :param expected_key: Expected key in JSON response (e.g., 'datasets')
        :return: Processed and normalized response
        """
        self.logger.info(f"process_llm_response called with model: {self.model}")
        #self.logger.debug(f"raw_response type: {type(raw_response)}, length: {len(str(raw_response))}")
        self.logger.debug(f"response_format: {response_format}")
        self.logger.debug(f"expected_key: {expected_key}")
        #self.logger.debug(f"raw_response content (first 500 chars): {str(raw_response)[:500]}")
        
        if self.model == 'gemma2:9b':
            self.logger.debug(f"Processing gemma2:9b response")
            # Split by newlines for this model
            result = raw_response.split("\n")
            self.logger.debug(f"gemma2:9b result: {result}")
            return result
            
        elif self.model in ['gemma3:1b', 'gemma3:4b', 'qwen:4b']:
            self.logger.debug(f"Processing {self.model} response")
            parsed_resp = self.safe_parse_json(raw_response)
            self.logger.debug(f"Parsed JSON response: {parsed_resp}")
            if isinstance(parsed_resp, dict) and expected_key and expected_key in parsed_resp:
                self.logger.debug(f"Found expected key '{expected_key}' in parsed response")
                result = self.normalize_response_format(parsed_resp[expected_key])
            else:
                self.logger.debug(f"Expected key '{expected_key}' not found or not dict, using full response")
                result = self.normalize_response_format(parsed_resp)
            self.logger.debug(f"Final result for {self.model}: {result}")
            return result
                
        elif 'gpt' in self.model:
            self.logger.debug(f"Processing GPT model response")
            self.logger.debug(f"raw_response type: {type(raw_response)}, length: {len(str(raw_response))}, response: {raw_response}")
            if from_batch_mode:
                self.logger.debug(f"From batch mode, raw_response type: {type(raw_response)}, length: {len(raw_response)}")
                for i, item in enumerate(raw_response):
                    self.logger.debug(f"Batch item type: {type(item)}, content (first 100 chars): {str(item)[:100]}")
                    if item['type'] == 'reasoning':
                        continue
                    elif item['type'] == 'message':
                        if len(item['content']) == 1:
                            raw_response = item['content'][0]['text']
                            self.logger.debug(f"Using single message content for processing: {raw_response[:100]}")
                            break
                        elif len(item['content'][i]['text']) == 0:
                            self.logger.debug(f"Empty content in batch item {i}, skipping")
                            continue
                        raw_response = item['content'][i]['text']
                        self.logger.debug(f"Using message content for processing: {raw_response[:100]}")
                        break

            parsed_response = self.safe_parse_json(raw_response)
            self.logger.info(f"GPT parsed response: {parsed_response}, type: {type(parsed_response)}")
            if self.full_document_read and isinstance(parsed_response, dict) and expected_key in parsed_response:
                result = parsed_response.get(expected_key, []) if expected_key else parsed_response
                self.logger.debug(f"GPT full_document_read=True, extracted result: {result}")
            else:
                result = parsed_response or []
                self.logger.debug(f"GPT full_document_read=False, result: {result}")
            final_result = self.normalize_response_format(result)
            self.logger.debug(f"GPT final normalized result: {final_result}")
            return final_result
            
        elif 'gemini' in self.model:
            self.logger.debug(f"Processing Gemini model response, use_portkey: {self.use_portkey}")
            if self.use_portkey:
                # For Portkey, raw_response is already the response object
                parsed_response = self.safe_parse_json(raw_response)
                self.logger.debug(f"Gemini Portkey parsed response: {parsed_response}")
                if self.full_document_read and isinstance(parsed_response, dict):
                    result = parsed_response.get(expected_key, []) if expected_key else parsed_response
                else:
                    result = parsed_response if isinstance(parsed_response, list) else []
                self.logger.debug(f"Gemini Portkey result: {result}")
                return self.normalize_response_format(result)
            else:
                # For direct Gemini, raw_response is the text content
                self.logger.debug(f"Gemini direct response processing")
                try:
                    parsed_response = json.loads(raw_response)
                    self.logger.debug(f"Gemini direct parsed response: {parsed_response}")
                    result = self.normalize_response_format(parsed_response)
                    self.logger.debug(f"Gemini direct final result: {result}")
                    return result
                except json.JSONDecodeError as e:
                    self.logger.debug(f"Gemini JSON decoding error: {e}")
                    self.logger.error(f"JSON decoding error: {e}")
                    return []
        
        elif self.model.startswith('local-flan-t5'):
            parsed_response = self.safe_parse_json(raw_response)
            self.logger.debug(f"Processing local Flan-T5 model response: {parsed_response}")
            return parsed_response

        elif self.model.startswith('hf-'):
            parsed_response = self.safe_parse_json(raw_response)
            self.logger.debug(f"Processing Hugging Face model response: {parsed_response}")
            return parsed_response

        else:
            self.logger.debug(f"Unsupported model: {self.model}")
            raise ValueError(f"Unsupported model: {self.model}. Please use a supported LLM model.")
    
    def normalize_response_format(self, response):
        """
        Task-agnostic response normalization and deduplication.
        Handles basic post-processing of LLM responses.
        """
        self.logger.debug(f"normalize_response_format called with response type: {type(response)}")
        self.logger.debug(f"normalize_response_format input: {response}")
        
        if not response:
            self.logger.debug(f"Empty response, returning empty list")
            return []
            
        if not isinstance(response, list):
            if isinstance(response, dict):
                self.logger.debug(f"Converting dict to list: {response}")
                response = [response]
            else:
                self.logger.debug(f"Non-list, non-dict response, returning as-is: {response}")
                return response
                
        # Basic normalization - remove empty or invalid items
        normalized = []
        for i, item in enumerate(response):
            self.logger.debug(f"Processing item {i}: {item} (type: {type(item)})")
            if isinstance(item, str) and len(item.strip()) < 3:
                self.logger.debug(f"Skipping short string: {item}")
                continue
            if isinstance(item, dict) and not any(item.values()):
                self.logger.debug(f"Skipping empty dict: {item}")
                continue
            self.logger.debug(f"Adding item to normalized: {item}")
            normalized.append(item)
            
        self.logger.debug(f"normalize_response_format final result: {normalized}")
        return normalized
    
    def safe_parse_json(self, response_text):
        """
        Task-agnostic JSON parsing with error handling.
        This can be used by any task that needs to parse JSON from LLM responses.
        """
        self.logger.debug(f"safe_parse_json called with type: {type(response_text)}")
        #self.logger.debug(f"safe_parse_json input (first 300 chars): {str(response_text)[:300]}")
        result = self._safe_parse_json_internal(response_text)
        self.logger.debug(f"safe_parse_json result: {result}")
        return result
    
    def generate_prompt_id(self, messages, temperature: float = 0.0):
        """Generate a unique prompt ID for caching."""
        return f"{self.model}-{temperature}-{self.prompt_manager._calculate_checksum(str(messages))}"
    
    def _safe_parse_json_internal(self, response_text):
        """Internal JSON parsing with error handling."""
        self.logger.info(f"Function_call: _safe_parse_json_internal(response_text {type(response_text)})")
        
        # Handle different response object types
        if hasattr(response_text, "choices"):
            try:
                response_text = response_text.choices[0].message.content
                self.logger.debug(f"Extracted content from response object, type: {type(response_text)}")
            except Exception as e:
                self.logger.warning(f"Could not extract content from response object: {e}")
                return None
        elif isinstance(response_text, list):
            self.logger.info(f"Response is a list of length: {len(response_text)}")
            for response_item in response_text:
                self.logger.info(f"List item type: {type(response_item)}")
                if hasattr(response_item, "content"):
                    self.logger.info(f"Item has content attribute, of type: {type(response_item.content)}")
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
            self.logger.warning(f"Initial JSON parsing failed. Attempting json_repair on {response_text[:100]}...")
            try:
                fixed = False
                # Pre-process common malformed patterns before json_repair
                repaired = response_text

                # Check if we have the T5 duplicate key pattern
                if '"dataset_identifier"' in repaired and repaired.count('"dataset_identifier"') > 1 and 't5' in self.model:
                    self.logger.info("Detected T5 duplicate key pattern, attempting to split into array of objects...")
                    
                    pairs_pattern = r'"(dataset_identifier|repository_reference|data_repository|dataset_webpage)"\s*:\s*"([^"]*)"'
                    matches = re.findall(pairs_pattern, repaired)
                    
                    if matches:
                        # Group pairs into objects (every 2-4 pairs = 1 object)
                        objects = []
                        current_obj = {}
                        
                        for key, value in matches:
                            if key == 'dataset_identifier' and 'dataset_identifier' in current_obj:
                                objects.append(current_obj)
                                current_obj = {}
                            current_obj[key] = value
                        
                        if current_obj:
                            objects.append(current_obj)
                        
                        if objects:
                            # Ensure proper JSON list format with outer braces
                            repaired = json.dumps(objects)
                            self.logger.info(f"Reconstructed {len(objects)} objects from T5 output: {repaired[:200]}")
                            fixed = True
                
                if not fixed:
                    # Replace array brackets with object braces when they contain key-value pairs
                    def fix_malformed_objects(match):
                        content = match.group(1)
                        # If content has key-value pairs, wrap in object braces
                        if '"' in content and ':' in content:
                            return '{' + content + '}'
                        return '[' + content + ']'  # Keep as array if no key-value pairs
                    
                    # Find nested arrays that should be objects
                    repaired = re.sub(r'\[\s*([^[\]]*"[^"]*"\s*:\s*"[^"]*"[^[\]]*)\s*\]', fix_malformed_objects, repaired)
                    
                    self.logger.info(f"After malformed pattern fix: {repaired[:200]}...")
                    
                    # Use json_repair
                    repaired = repair_json(repaired)
                    
                    # Clean up artifacts
                    repaired = re.sub(r',\s*\{\}\]', ']', repaired)  # Remove trailing empty objects in lists
                
                parsed = json.loads(repaired)
                self.logger.info(f"Successfully parsed after repair: {type(parsed)} with {len(parsed) if hasattr(parsed, '__len__') else 'N/A'} items")
                
                # If we get a nested structure, try to flatten it
                if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list):
                    self.logger.info("Detected nested list structure, attempting to flatten...")
                    # Extract valid dictionaries from nested structure
                    flattened = []
                    for item in parsed[0]:
                        if isinstance(item, dict):
                            flattened.append(item)
                    if flattened:
                        return flattened
                
                return parsed
            except Exception as e:
                self.logger.warning(f"json_repair failed: {e}")
                # Last resort: try to extract individual JSON objects using regex
                try:
                    self.logger.info("Attempting regex extraction of JSON objects...")
                    
                    # Look for the specific malformed pattern in the raw response
                    malformed_pattern = r'\[\s*"dataset_identifier"\s*:\s*"([^"]+)"\s*,\s*"data_repository"\s*:\s*"([^"]+)"\s*(?:,\s*"dataset_webpage"\s*:\s*"([^"]+)")?\s*\]'
                    matches = re.findall(malformed_pattern, response_text)
                    
                    if matches:
                        objects = []
                        for match in matches:
                            obj = {
                                "dataset_identifier": match[0],
                                "data_repository": match[1]
                            }
                            if match[2]:  # dataset_webpage is optional
                                obj["dataset_webpage"] = match[2]
                            objects.append(obj)
                        self.logger.info(f"Extracted {len(objects)} objects using regex pattern matching")
                        return objects
                    
                    # Fallback: look for any objects with dataset_identifier
                    pattern = r'\{[^{}]*"dataset_identifier"[^{}]*\}'
                    matches = re.findall(pattern, response_text)
                    if matches:
                        objects = []
                        for match in matches:
                            try:
                                obj = json.loads(match)
                                objects.append(obj)
                            except:
                                continue
                        if objects:
                            return objects
                except Exception:
                    pass
                return None
    
    def _handle_batch_mode(self,
                          batch_requests: List[Dict],
                          batch_file_path: str,
                          temperature: float = 0.0,
                          response_format: Optional[Dict] = None,
                          api_provider: str = 'openai') -> Dict[str, Any]:
        """
        Handle batch mode processing by creating JSONL file with properly formatted requests.
        
        :param batch_requests: List of batch request data
        :param batch_file_path: Path for saving batch JSONL file
        :param temperature: Temperature setting
        :param response_format: Response format schema
        :param api_provider: API provider ('openai' or 'portkey')
        :return: Dictionary with batch file info and statistics
        """
        try:
            self.logger.info(f"Processing {len(batch_requests)} requests for batch mode")
            
            # Convert batch requests to proper API format
            formatted_requests = []
            
            for request_data in batch_requests:
                custom_id = request_data.get('custom_id')
                messages = request_data.get('messages')
                metadata = request_data.get('metadata')
                
                if not custom_id or not messages:
                    self.logger.warning(f"Skipping invalid batch request: missing custom_id or messages")
                    continue
                
                # Create properly formatted request based on API provider
                if api_provider.lower() == 'openai':
                    # Use a compatible OpenAI model for batch processing
                    batch_model = self.model
                    if 'gemini' in self.model.lower():
                        batch_model = 'gpt-4o-mini'  # Use OpenAI model for batch processing
                        self.logger.info(f"Using OpenAI model {batch_model} for batch processing instead of {self.model}")
                    
                    formatted_request = self.batch_builder.create_openai_request(
                        custom_id=custom_id,
                        messages=messages,
                        model=batch_model,
                        temperature=temperature,
                        response_format=response_format,
                        metadata=metadata
                    )
                elif api_provider.lower() == 'portkey':
                    formatted_request = self.batch_builder.create_portkey_request(
                        custom_id=custom_id,
                        messages=messages,
                        model=self.model,
                        temperature=temperature,
                        response_format=response_format
                    )
                else:
                    raise ValueError(f"Unsupported API provider: {api_provider}")
                
                formatted_requests.append(formatted_request)
            
            # Create JSONL file
            file_stats = self.batch_storage.create_jsonl_batch_file(
                requests=formatted_requests,
                output_path=batch_file_path
            )
            
            # Validate the created file
            validation_result = self.batch_storage.validate_jsonl_format(batch_file_path)
            
            # Determine the actual model used for batch processing
            actual_model = self.model
            if api_provider.lower() == 'openai' and 'gemini' in self.model.lower():
                actual_model = 'gpt-4o-mini'
            
            result = {
                'batch_file_path': batch_file_path,
                'total_requests': len(formatted_requests),
                'skipped_requests': len(batch_requests) - len(formatted_requests),
                'api_provider': api_provider,
                'model': actual_model,
                'file_stats': file_stats,
                'validation': validation_result,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"Batch file created successfully: {batch_file_path} with {len(formatted_requests)} requests")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in batch mode processing: {e}")
            raise
    
    def submit_batch_job(self, 
                        batch_file_path: str, 
                        api_provider: str = 'openai',
                        batch_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit a batch job to the specified API provider.
        
        :param batch_file_path: Path to the JSONL batch file
        :param api_provider: API provider ('openai' or 'portkey')
        :param batch_description: Optional description for the batch job
        :return: Batch job information
        """
        try:
            if api_provider.lower() == 'openai':
                return self._submit_openai_batch(batch_file_path, batch_description)
            elif api_provider.lower() == 'portkey':
                return self._submit_portkey_batch(batch_file_path, batch_description)
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
                
        except Exception as e:
            self.logger.error(f"Error submitting batch job: {e}")
            raise
    
    def _submit_openai_batch(self, batch_file_path: str, batch_description: Optional[str]) -> Dict[str, Any]:
        """Submit batch to OpenAI Batch API."""
        # Create a dedicated OpenAI client for batch operations
        # This ensures we use the direct OpenAI API even if the main client uses Portkey
        openai_client = OpenAI(api_key=GPT_API_KEY)

        # Filter out invalid keys (i.e. 'metadata') from batch request JSONL file
        # OpenAI Batch API doesn't accept 'metadata' field in request body
        cleaned_file_path = batch_file_path.replace('.jsonl', '_cleaned.jsonl')
        self.logger.info(f"Filtering invalid keys from batch file: {batch_file_path}")
        
        with open(batch_file_path, 'r', encoding='utf-8') as infile, \
             open(cleaned_file_path, 'w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, 1):
                if not line.strip():
                    continue
                try:
                    request = json.loads(line)
                    if 'metadata' in request:
                        self.logger.debug(f"Removing 'metadata' field from request {line_num}")
                        del request['metadata']
                    outfile.write(json.dumps(request) + '\n')
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON at line {line_num}: {e}")
                    raise
        
        self.logger.info(f"Created cleaned batch file: {cleaned_file_path}")
        
        # Upload the cleaned batch file
        self.logger.info(f"Uploading cleaned batch file to OpenAI: {cleaned_file_path}")
        with open(cleaned_file_path, 'rb') as file:
            batch_input_file = openai_client.files.create(
                file=file,
                purpose="batch"
            )
        
        # Create the batch job
        self.logger.info(f"Creating batch job with file ID: {batch_input_file.id}")
        batch_job = openai_client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "description": batch_description or f"LLMClient batch job - {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "model": self.model,
                "created_by": "llm_client"
            }
        )
        
        self.logger.info(f"OpenAI batch job created. ID: {batch_job.id}, Status: {batch_job.status}")
        
        return {
            'batch_id': batch_job.id,
            'status': batch_job.status,
            'input_file_id': batch_input_file.id,
            'created_at': batch_job.created_at,
            'api_provider': 'openai',
            'endpoint': batch_job.endpoint,
            'completion_window': batch_job.completion_window
        }
    
    def _submit_portkey_batch(self, batch_file_path: str, batch_description: Optional[str]) -> Dict[str, Any]:
        raise NotImplementedError ("This method hasn't been implemented yet")
    
    def check_batch_status(self, batch_id: str, api_provider: str = 'portkey') -> Dict[str, Any]:
        """
        Check the status of a batch job.
        
        :param batch_id: Batch job ID
        :param api_provider: API provider ('openai' or 'portkey')
        :return: Batch status information
        """
        try:
            if api_provider.lower() == 'openai':
                # Create a dedicated OpenAI client for batch operations
                openai_client = OpenAI(api_key=GPT_API_KEY)
                batch_job = openai_client.batches.retrieve(batch_id)
                
                return {
                    'batch_id': batch_job.id,
                    'status': batch_job.status,
                    'output_file_id': getattr(batch_job, 'output_file_id', None),
                    'error_file_id': getattr(batch_job, 'error_file_id', None),
                    'created_at': batch_job.created_at,
                    'completed_at': getattr(batch_job, 'completed_at', None),
                    'failed_at': getattr(batch_job, 'failed_at', None),
                    'request_counts': getattr(batch_job, 'request_counts', {}),
                    'api_provider': api_provider
                }
                
            elif api_provider.lower() == 'portkey':
                raise NotImplementedError("This provider isn't supported yet")

            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
                
        except Exception as e:
            self.logger.error(f"Error checking batch status: {e}")
            raise
    
    def download_batch_results(self, 
                              batch_id: str, 
                              output_file_path: str,
                              api_provider: str = 'openai') -> Dict[str, Any]:
        """
        Download batch results when completed.
        
        :param batch_id: Batch job ID
        :param output_file_path: Path to save the results
        :param api_provider: API provider ('openai' or 'portkey')
        :return: Download information and statistics
        """
        try:
            # Check batch status first
            status_info = self.check_batch_status(batch_id, api_provider)
            
            if status_info['status'] not in ['completed', 'cancelled']:
                raise ValueError(f"Batch {batch_id} is not completed. Status: {status_info['status']}")
            
            if not status_info.get('output_file_id'):
                raise ValueError(f"No output file available for batch {batch_id}")
            
            # Download the results file
            if api_provider.lower() == 'openai':
                # Create a dedicated OpenAI client for batch operations
                openai_client = OpenAI(api_key=GPT_API_KEY)
                file_response = openai_client.files.content(status_info['output_file_id'])
                
                with open(output_file_path, 'wb') as f:
                    f.write(file_response.content)
                    
            elif api_provider.lower() == 'portkey':
                raise NotImplementedError ("This provider isn't supported yet")
            
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
            
            # Validate and get statistics from the downloaded file
            validation_result = self.batch_storage.validate_jsonl_format(output_file_path)
            file_info = self.batch_storage.get_file_info(output_file_path)
            
            result = {
                'output_file_path': output_file_path,
                'batch_id': batch_id,
                'api_provider': api_provider,
                'file_info': file_info,
                'validation': validation_result,
                'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"Batch results downloaded successfully: {output_file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error downloading batch results: {e}")
            raise
    
    def process_batch_responses(self, 
                              batch_results_file: str,
                              response_format: Optional[Dict] = None,
                              expected_key: Optional[str] = None):

        """
        Process batch API results using existing response processing logic.
        Minimal processing since results will be passed to LLMParser for further processing.
        
        :param batch_results_file: Path to the batch results JSONL file
        :param response_format: Expected response format schema
        :param expected_key: Expected key in JSON responses (e.g., 'datasets')
        :return: Processing results and statistics
        """
        try:
            self.logger.info(f"Processing batch responses from: {batch_results_file}")
            
            # Read batch results using BatchStorageManager
            batch_responses = self.batch_storage.read_and_parse_batch_results(batch_results_file)
            
            processed_results = []
            successful_count = 0
            error_count = 0
            
            for batch_response in batch_responses:
                try:
                    self.logger.debug(f"Processing batch response of type: {type(batch_response)}, object: {batch_response}")
                    custom_id = batch_response.get('custom_id', 'unknown')
                    
                    # Handle different batch response formats
                    if 'response' in batch_response:
                        # OpenAI batch format
                        llm_responses = batch_response['response']['body']['output']
                    elif 'body' in batch_response and 'output' in batch_response['body']:
                        # Direct format
                        llm_responses = batch_response['body']['output']
                    else:
                        # Fallback - assume the response is the batch_response itself
                        llm_responses = batch_response
                    
                    processed_response = self.process_llm_response(
                        raw_response=llm_responses,
                        response_format=response_format,
                        expected_key=expected_key,
                        from_batch_mode=True
                        )
                                
                    processed_results.append({
                        'custom_id': custom_id,
                        'processed_response': processed_response,
                        'status': 'success'
                        })
                    successful_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing batch response {custom_id}: {e}")
                    processed_results.append({
                        'custom_id': custom_id,
                        'error': str(e),
                        'status': 'error'
                    })
                    error_count += 1
            
            # Prepare processing summary
            processing_summary = {
                'total_responses': len(batch_responses),
                'successful_processed': successful_count,
                'errors': error_count,
                'processed_results': processed_results,
                'batch_results_file': batch_results_file,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"Batch processing complete: {successful_count} successful, {error_count} errors")
            return processing_summary
            
        except Exception as e:
            self.logger.error(f"Error processing batch responses: {e}")
            raise


