from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
import os

class LocalModelClient:
    def __init__(self, model_path, device='auto', logger=None):
        self.model_path = model_path
        self.logger = logger or logging.getLogger(__name__)
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        
        self.logger.info(f"LocalModelClient initialized with model_path: {model_path}, device: {self.device}")
    
    def _setup_device(self, device):
        """
        Set up the appropriate device for model inference.
        Auto-detects MPS (M1/M2 Mac), CUDA (GPU), or falls back to CPU.
        
        :param device: Device specification ('auto', 'mps', 'cuda', 'cpu')
        :return: torch.device object
        """
        if device == 'auto':
            # Auto-detect best available device
            if torch.backends.mps.is_available():
                self.logger.info("MPS (Metal Performance Shaders) available - using MPS device")
                return torch.device("mps")
            elif torch.cuda.is_available():
                self.logger.info("CUDA available - using GPU")
                return torch.device("cuda")
            else:
                self.logger.info("No GPU available - using CPU")
                return torch.device("cpu")
        else:
            # Use specified device
            self.logger.info(f"Using specified device: {device}")
            return torch.device(device)
        
    def load_model(self):
        """Load the model and tokenizer from the local path."""
        try:
            self.logger.info(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            
            self.logger.info(f"Moving model to device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully: {type(self.model).__name__}")
        except Exception as e:
            self.logger.error(f"Error loading model from {self.model_path}: {e}")
            raise
        
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
        
        self.logger.debug(f"Generating output for input length: {len(input_text)} characters")
        
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
            
            self.logger.debug(f"Generated output length: {len(result)} characters")
            return result  # Should be JSON string
            
        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise