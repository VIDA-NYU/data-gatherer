from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
import os

class HFModelClient:
    def __init__(self, model_id, device='auto', logger=None, use_auth_token=None):
        """
        Initialize HuggingFace model client for loading models from HuggingFace Hub.
        
        :param model_id: HuggingFace model ID (e.g., 'vida-nyu/flan-t5-base-dataref-info-extract')
        :param device: Device specification ('auto', 'mps', 'cuda', 'cpu')
        :param logger: Logger instance
        :param use_auth_token: HuggingFace auth token for private models (or True to use stored token)
        """
        self.model_id = model_id
        self.logger = logger or logging.getLogger(__name__)
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.use_auth_token = use_auth_token
        
        self.logger.info(f"HFModelClient initialized with model_id: {model_id}, device: {self.device}")
    
    def _setup_device(self, device):
        """
        Set up the appropriate device for model inference.
        Auto-detects MPS (M1/M2 Mac), CUDA (GPU), or falls back to CPU.
        
        :param device: Device specification ('auto', 'mps', 'cuda', 'cpu')
        :return: torch.device object
        """
        if device == 'auto':
            # Auto-detect best available device
            if torch.cuda.is_available():
                self.logger.info("CUDA available - using GPU")
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.logger.info("MPS (Metal Performance Shaders) available - using MPS device")
                return torch.device("mps")
            else:
                self.logger.info("No GPU available - using CPU")
                return torch.device("cpu")
        else:
            # Use specified device
            self.logger.info(f"Using specified device: {device}")
            return torch.device(device)
        
    def load_model(self):
        """Load the model and tokenizer from HuggingFace Hub."""
        try:
            self.logger.info(f"Loading tokenizer from HuggingFace Hub: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.use_auth_token
            )
            
            self.logger.info(f"Loading model from HuggingFace Hub: {self.model_id}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                token=self.use_auth_token
            )
            
            self.logger.info(f"Moving model to device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully: {type(self.model).__name__}")
        except Exception as e:
            self.logger.error(f"Error loading model {self.model_id} from HuggingFace Hub: {e}")
            raise
        
    def batch_generate(self, input_texts, max_output_length=512, temperature=0.0):
        """
        Generate outputs for a list of input strings in a single GPU pass.

        :param input_texts: List of plain content strings (already extracted from rendered prompts)
        :param max_output_length: Max tokens to generate per output
        :param temperature: 0.0 = greedy decoding (fastest); >0 enables sampling
        :return: List of generated JSON strings, one per input
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # T5 input prefix used during fine-tuning — must match training convention
        formatted = [f"Extract dataset information: {t}" for t in input_texts]
        self.logger.info(f"batch_generate: {len(formatted)} inputs")
        for i, t in enumerate(input_texts):
            self.logger.info(f"batch_generate T5 input [{i}] (first 300 chars): {t[:300]!r}")

        # max_length here caps INPUT tokens (T5's context window is 1024)
        inputs = self.tokenizer(
            formatted, return_tensors="pt",
            max_length=1024, truncation=True, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # num_beams=1 → greedy decoding: fastest, no beam search overhead
        generation_kwargs = {"max_length": max_output_length, "num_beams": 1}
        if temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.95})

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        results = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        for i, r in enumerate(results):
            self.logger.info(f"batch_generate T5 output [{i}]: {r!r}")
        return results

    def generate(self, input_text, max_length=512, temperature=0.0):
        """
        Generate output for a single input text.
        
        :param input_text: Input text to process
        :param max_length: Maximum length of generated output
        :param temperature: Sampling temperature (0.0 = greedy decoding)
        :return: Generated text output (JSON string)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Add "Extract dataset information: " prefix (same as training)
        formatted_input = f"Extract dataset information: {input_text}"

        self.logger.info(f"generate T5 input (first 300 chars): {input_text[:300]!r}")
        
        try:
            inputs = self.tokenizer(formatted_input, return_tensors="pt", 
                                   max_length=1024, truncation=True)
            # Move inputs to device (need to move tensor dict items)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generation_kwargs = {
                    "max_length": max_length,
                    "num_beams": 1,  # Greedy decoding by default
                }
                
                if temperature > 0:
                    generation_kwargs.update({
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": 0.95,
                    })
                
                outputs = self.model.generate(**inputs, **generation_kwargs)
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clear MPS cache if using MPS to prevent memory buildup
            if self.device.type == "mps":
                torch.mps.empty_cache()

            self.logger.info(f"generate T5 output: {result!r}")
            return result  # Should be JSON string
            
        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise
