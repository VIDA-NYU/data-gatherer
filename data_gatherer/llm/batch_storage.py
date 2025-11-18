"""
Batch Storage Utility for handling file operations related to batch API processing.
This utility handles JSONL file creation, upload/download operations, and metadata management.
"""

import json
import os
import time
import tempfile
from typing import Dict, List, Any, Optional, Union
from data_gatherer.llm.response_schema import *
import logging
import math


class BatchStorageManager:
    """
    Manages file storage operations for batch API processing.
    Handles JSONL creation, file upload/download, and metadata tracking.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def create_jsonl_batch_file(self, requests: List[Dict[str, Any]], output_path: str) -> Dict[str, Any]:
        """
        Create a JSONL file from batch requests.
        
        :param requests: List of batch request dictionaries
        :param output_path: Path where the JSONL file will be saved
        :return: Dictionary with file information and statistics
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for request in requests:
                    f.write(json.dumps(request, ensure_ascii=False) + '\n')
            
            file_stats = {
                'file_path': output_path,
                'total_requests': len(requests),
                'file_size_bytes': os.path.getsize(output_path),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"Created JSONL batch file: {output_path} with {len(requests)} requests")
            return file_stats
            
        except Exception as e:
            self.logger.error(f"Error creating JSONL batch file: {e}")
            raise
    
    def read_jsonl_batch_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read and parse a JSONL batch file.
        
        :param file_path: Path to the JSONL file
        :return: List of parsed JSON objects
        """
        try:
            requests = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line.strip():  # Skip empty lines
                            request = json.loads(line.strip())
                            requests.append(request)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
            
            self.logger.info(f"Read {len(requests)} requests from {file_path}")
            return requests
            
        except Exception as e:
            self.logger.error(f"Error reading JSONL batch file: {e}")
            raise
    
    def save_batch_metadata(self, metadata: Dict[str, Any], metadata_path: str) -> None:
        """
        Save batch metadata to a JSON file.
        
        :param metadata: Metadata dictionary to save
        :param metadata_path: Path where metadata will be saved
        """
        try:
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved batch metadata to: {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving batch metadata: {e}")
            raise
    
    def load_batch_metadata(self, metadata_path: str) -> Optional[Dict[str, Any]]:
        """
        Load batch metadata from a JSON file.
        
        :param metadata_path: Path to the metadata file
        :return: Metadata dictionary or None if file doesn't exist
        """
        try:
            if not os.path.exists(metadata_path):
                return None
                
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.logger.info(f"Loaded batch metadata from: {metadata_path}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error loading batch metadata: {e}")
            return None
    
    def create_temp_file(self, suffix: str = '.jsonl', prefix: str = 'batch_') -> str:
        """
        Create a temporary file for batch operations.
        
        :param suffix: File suffix/extension
        :param prefix: File prefix
        :return: Path to the temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=suffix, 
            prefix=prefix, 
            delete=False
        )
        temp_file.close()
        
        self.logger.info(f"Created temporary file: {temp_file.name}")
        return temp_file.name
    
    def validate_jsonl_format(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a JSONL file format and return statistics.
        
        :param file_path: Path to the JSONL file to validate
        :return: Validation results and statistics
        """
        try:
            valid_lines = 0
            invalid_lines = 0
            total_lines = 0
            errors = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    
                    if not line.strip():  # Skip empty lines
                        continue
                        
                    try:
                        json.loads(line.strip())
                        valid_lines += 1
                    except json.JSONDecodeError as e:
                        invalid_lines += 1
                        errors.append(f"Line {line_num}: {str(e)}")
            
            validation_result = {
                'is_valid': invalid_lines == 0,
                'total_lines': total_lines,
                'valid_lines': valid_lines,
                'invalid_lines': invalid_lines,
                'errors': errors,
                'file_size_bytes': os.path.getsize(file_path)
            }
            
            if validation_result['is_valid']:
                self.logger.info(f"JSONL file validation passed: {file_path}")
            else:
                self.logger.warning(f"JSONL file validation failed: {invalid_lines} invalid lines")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating JSONL file: {e}")
            raise
    
    def cleanup_temp_files(self, file_paths: List[str]) -> None:
        """
        Clean up temporary files.
        
        :param file_paths: List of file paths to remove
        """
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Error cleaning up file {file_path}: {e}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file.
        
        :param file_path: Path to the file
        :return: File information dictionary
        """
        try:
            if not os.path.exists(file_path):
                return {'exists': False}
            
            stat = os.stat(file_path)
            
            return {
                'exists': True,
                'path': file_path,
                'size_bytes': stat.st_size,
                'created_at': time.ctime(stat.st_ctime),
                'modified_at': time.ctime(stat.st_mtime),
                'is_file': os.path.isfile(file_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting file info for {file_path}: {e}")
            return {'exists': False, 'error': str(e)}
    
    def read_and_parse_batch_results(self, results_file_path: str) -> List[Dict[str, Any]]:
        """
        Read batch results JSONL file and extract individual responses.
        
        :param results_file_path: Path to the batch results JSONL file
        :return: List of parsed batch response objects
        """
        try:
            batch_responses = []
            with open(results_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line.strip():  # Skip empty lines
                            batch_response = json.loads(line.strip())
                            batch_responses.append(batch_response)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_num} in batch results: {e}")
                        continue
            
            self.logger.info(f"Read {len(batch_responses)} batch responses from {results_file_path}")
            return batch_responses
            
        except Exception as e:
            self.logger.error(f"Error reading batch results file: {e}")
            raise
    
    def chunk_batch_file(self, large_batch_file_path: str, max_file_size_mb: float = 200.0) -> List[Dict[str, Any]]:
        """
        Chunk a large JSONL batch file into smaller files under the size limit and submit them sequentially.
        
        :param large_batch_file_path: Path to the large JSONL file to chunk
        :param max_file_size_mb: Maximum size per chunk file in MB (default: 200MB for OpenAI limit)
        :param wait_between_submissions: Seconds to wait between submissions
        :param submit_kwargs: Additional keyword arguments to pass to the submit function
        :return: List of submission results for each chunk
        """
        try:
            max_file_size_bytes = max_file_size_mb * 1024 * 1024
            
            # Check if the file needs chunking
            original_size = os.path.getsize(large_batch_file_path)
            self.logger.info(f"Original batch file size: {original_size} bytes")
            
            if original_size <= max_file_size_bytes:
                self.logger.info("File is already under the size limit, no chunking needed")
                return [{'chunk_info': {
                    'chunk_number': 1,
                    'total_chunks': 1,
                    'chunk_file_path': large_batch_file_path,
                    'requests_in_chunk': self.read_jsonl_batch_file(large_batch_file_path).__len__(),
                    'chunk_size_mb': original_size / 1024 / 1024}}]
            
            # Read all requests from the original file
            requests = self.read_jsonl_batch_file(large_batch_file_path)
            self.logger.info(f"Read {len(requests)} requests from original file")
            
            # Calculate approximate size per request for chunking
            n_requests = math.ceil(original_size * 1.1 / max_file_size_bytes)
            requests_per_chunk = math.floor(len(requests) / n_requests)
            
            # Create chunks
            chunks = []
            for i in range(0, len(requests), requests_per_chunk):
                chunk_requests = requests[i:i + requests_per_chunk]
                chunks.append(chunk_requests)
            
            self.logger.info(f"Split into {len(chunks)} chunks with ~{requests_per_chunk} requests each")
            
            # Create chunk files
            base_path = os.path.splitext(large_batch_file_path)[0]
            base_dir = os.path.dirname(base_path)
            base_name = os.path.basename(base_path)
            
            chunked_batches = []
            
            for chunk_idx, chunk_requests in enumerate(chunks):
                chunk_file_path = os.path.join(base_dir, f"{base_name}_chunk_{chunk_idx + 1:03d}.jsonl")
                
                # Create chunk file
                chunk_stats = self.create_jsonl_batch_file(chunk_requests, chunk_file_path)
                chunk_size_mb = chunk_stats['file_size_bytes'] / 1024 / 1024
                
                self.logger.info(f"Created chunk {chunk_idx + 1}/{len(chunks)}: {chunk_file_path} "
                               f"({chunk_stats['total_requests']} requests, {chunk_size_mb:.2f} MB)")
                
                # Just return file info if no submit function
                chunked_batches.append({
                    'chunk_info': {
                    'chunk_number': chunk_idx + 1,
                    'total_chunks': len(chunks),
                    'chunk_file_path': chunk_file_path,
                    'requests_in_chunk': len(chunk_requests),
                    'chunk_size_mb': chunk_size_mb
                    }
                })

            self.logger.info(f"Chunking completed. Created {len(chunks)} chunks, "
                           f"submitted {len([r for r in chunked_batches if 'batch_id' in r])} successfully")
            
            return chunked_batches
            
        except Exception as e:
            self.logger.error(f"Error in chunking and submitting batch file: {e}")
            raise

class BatchRequestBuilder:
    """
    Helper class to build batch requests in the correct format for different API providers.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def create_openai_request(self, 
                             custom_id: str, 
                             messages: List[Dict[str, str]], 
                             model: str,
                             temperature: float = 0.0,
                             response_format: Optional[Dict] = dataset_response_schema_with_use_description,
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a batch request in OpenAI format for the responses endpoint.
        
        :param custom_id: Unique identifier for the request
        :param messages: List of message dictionaries with roles (system, user, assistant)
        :param model: Model name
        :param temperature: Temperature setting
        :param response_format: Optional response format schema
        :return: Formatted batch request
        """

        self.logger.info(f"Creating OpenAI request with custom_id: {custom_id}, model: {model}, format: {response_format}")
        
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "input": messages,
                "text": { "format": response_format }
            }
        }
        
        if metadata:
            # Add only serializable metadata values
            serializable_metadata = {}
            for key, value in metadata.items():
                try:
                    # Test if the value is JSON serializable
                    json.dumps(value)
                    serializable_metadata[key] = value
                except (TypeError, ValueError) as e:
                    self.logger.warning(f"Skipping non-serializable metadata key '{key}': {e}")
            
            if serializable_metadata:
                request["metadata"] = serializable_metadata
            
        return request
    
    def create_portkey_request(self,
                              custom_id: str,
                              messages: List[Dict[str, str]],
                              model: str,
                              temperature: float = 0.0,
                              response_format: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create a batch request in Portkey format.
        
        :param custom_id: Unique identifier for the request
        :param messages: List of message dictionaries
        :param model: Model name
        :param temperature: Temperature setting
        :param response_format: Optional response format schema
        :return: Formatted batch request
        """
    
        raise NotImplementedError("This method hasn't been implemented yet.")
    
    def generate_custom_id(self, 
                          model: str, 
                          identifier: str, 
                          timestamp: Optional[int] = None) -> str:
        """
        Generate a unique custom ID for batch requests.
        
        :param model: Model name
        :param identifier: Unique identifier (e.g., article_id)
        :param timestamp: Optional timestamp (current time if None)
        :return: Generated custom ID
        """
        import re
        
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        
        # Clean the identifier to ensure it's valid
        clean_identifier = re.sub(r'[^a-zA-Z0-9_-]', '_', str(identifier))
        custom_id = f"{model}_{clean_identifier}_{timestamp}"
        
        # Ensure it's within the 64 character limit
        if len(custom_id) > 64:
            # Truncate the identifier part to fit
            max_identifier_len = 64 - len(model) - len(str(timestamp)) - 2  # 2 underscores
            clean_identifier = clean_identifier[:max_identifier_len]
            custom_id = f"{model}_{clean_identifier}_{timestamp}"
        
        return custom_id