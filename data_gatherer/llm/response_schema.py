import typing_extensions as typing
from pydantic import BaseModel

dataset_response_schema_gpt_completions = {
    "type": "json_schema",
        "json_schema": {
        "name": "GPT_chat_completions_schema",
        "schema": {
            "type": "object",  # Root must be an object
            "properties": {
                "datasets": {  # Use a property to hold the array
                "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dataset_identifier": {
                                "type": "string",
                                "description": "A unique identifier for the dataset."
                            },
                            "repository_reference": {
                                "type": "string",
                                "description": "A valid URI or string referring to the repository."
                            },
                            "decision_rationale": {
                                "type": "string",
                                "description": "Why did we select this dataset?"
                            }
                        },
                        "required": ["dataset_identifier", "repository_reference"]
                    },
                    "minItems": 1,
                    "uniqueItems": True
                }
            },
            "required": ["datasets"]
        }
    }
}

dataset_response_schema_gpt = {
    "type": "json_schema",
    "name": "GPT_responses_schema",
    "schema": {
        "type": "object",  # Root must be an object
        "properties": {
            "datasets": {  # Use a property to hold the array
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dataset_identifier": {
                            "type": "string",
                            "description": "A unique identifier or accession code for the dataset."
                        },
                        "repository_reference": {
                            "type": "string",
                            "description": "A valid URI or string referring to the repository."
                        }
                    },
                    "additionalProperties": False,
                    "required": ["dataset_identifier", "repository_reference"]
                },
                "minItems": 1,
                "additionalProperties": False
            }
        },
        "additionalProperties": False,
        "required": ["datasets"],
    }
}

dataset_metadata_response_schema_gpt = {
    "type": "json_schema",
    "json_schema": {
        "name": "Dataset_metadata_response",
        "schema": {
            "type": "object",
            "properties": {
                "number_of_files": {
                    "type": "string",
                    "description": "Total number of files."
                },
                "sample_size": {
                    "type": "string",
                    "description": "How many samples are recorded in the dataset."
                },
                "file_size": {
                    "type": "string",
                    "description": "Cumulative file size or range."
                },
                "file_format": {
                    "type": "string",
                    "description": "Format of the file (e.g., CSV, FASTQ)."
                },
                "file_type": {
                    "type": "string",
                    "description": "Type or category of the file."
                },
                "dataset_description": {
                    "type": "string",
                    "description": "Short summary of the dataset contents, plus - if mentioned - the use in the research publication of interes."
                },
                "file_url": {
                    "type": "string",
                    "description": "Direct link to the file."
                },
                "file_name": {
                    "type": "string",
                    "description": "Filename or archive name."
                },
                "file_license": {
                    "type": "string",
                    "description": "License under which the file is distributed."
                },
                "request_access_needed": {
                    "type": "string",
                    "description": "[Yes or No] Whether access to the file requires a request."
                },
                "request_access_form_links": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "uri",
                        "description": "Links to forms or pages where access requests can be made."
                    },
                    "description": "Links to forms or pages where access requests can be made."
                },
                "dataset_identifier": {
                    "type": "string",
                    "description": "A unique identifier for the dataset."
                },
                "download_type": {
                    "type": "string",
                    "description": "Type of download (e.g., HTTP, FTP, API, ...)."
                }
            },
            "required": [
                "dataset_description"
            ]
        }
    }
}

class Dataset(BaseModel):
    dataset_identifier: str
    repository_reference: str

class Dataset_w_Page(BaseModel):
    dataset_identifier: str
    repository_reference: str
    dataset_webpage: str

class Dataset_w_CitationType(BaseModel):
    dataset_identifier: str
    repository_reference: str
    citation_type: str

class Array_Dataset_w_CitationType(BaseModel):
    datasets: list[Dataset_w_CitationType]

class Dataset_w_Description(typing.TypedDict):
    dataset_identifier: str
    repository_reference: str
    rationale: str

class Dataset_metadata(BaseModel):
    number_of_files: int
    file_size: str
    file_format: str
    file_type: str
    dataset_description: str
    file_url: str
    file_name: str
    file_license: str
    request_access_needed: str
    dataset_identifier: str
    download_type: str
