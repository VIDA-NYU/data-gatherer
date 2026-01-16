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
                            "data_repository": {
                                "type": "string",
                                "description": "A valid URI or string referring to the repository."
                            },
                            "decision_rationale": {
                                "type": "string",
                                "description": "Why did we select this dataset?"
                            }
                        },
                        "required": ["dataset_identifier", "data_repository"]
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
                            "description": "A unique identifier or accession code for the dataset.",
                            "maxLength": 64
                        },
                        "data_repository": {
                            "type": "string",
                            "description": "A valid URI or string referring to the repository.",
                            "maxLength": 128
                        },
                        "dataset_webpage": {
                            "type": "string",
                            "description": "If present the URL of the dataset page, otherwise return 'n/a'.",
                            "maxLength": 128
                        }
                    },
                    "additionalProperties": False,
                    "required": ["dataset_identifier", "data_repository", "dataset_webpage"]
                },
                "minItems": 1
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

# Simplified schema focused on dataset provenance and reuse enablement
dataset_response_schema_with_use_description_and_short = {
    "type": "json_schema",
    "name": "PaperMiner_dataset_provenance_schema",
    "schema": {
        "type": "object",
        "properties": {
            "datasets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dataset_identifier": {
                            "type": "string",
                            "description": "A unique identifier or accession code for the dataset.",
                            "maxLength": 64
                        },
                        "data_repository": {
                            "type": "string",
                            "description": "A valid URI or string referring to the repository where the dataset can be found.",
                            "maxLength": 128
                        },
                        "dataset_context_from_paper": {
                            "type": "string",
                            "description": "Relevant text passages from the paper that either describe this dataset and provide context of its use or refer to it more implicitly.",
                            "maxLength": 1024
                        },
                        "dataset_keywords": {
                            "type": "string",
                            "description": "Two or three keywords to help user understand if they want to reuse this dataset (about content and scope).",
                            "maxLength": 128
                        },
                        "citation_type": {
                            "type": "string",
                            "description": "Type of citation used for this dataset. It can be either Primary (firsthand information collected by the researcher for a specific purpose) or Secondary (pre-existing information collected by someone else and then used by another researcher).",
                            "maxLength": 16
                        }
                    },
                    "additionalProperties": False,
                    "required": ["dataset_identifier", "data_repository", "dataset_context_from_paper", "dataset_keywords", "citation_type"]
                },
                "minItems": 1,
                "additionalProperties": False
            }
        },
        "additionalProperties": False,
        "required": ["datasets"]
    }
}

autoDDG_from_context_schema = {
    "type": "json_schema",
    "name": "autoDDG_from_context_schema",
    "schema": {
        "type": "object",
        "properties": {
            "dataset_description": {
                "type": "string",
                "description": "Description of the dataset extracted from context."
            },
            "citation_type": {
                "type": "string",
                "description": "Type of citation used for this dataset.",
                "enum": ["Primary", "Secondary"]
            },
            "reuse_potential": {
                "type": "string",
                "description": "A brief explanation of the potential for reusing this dataset in future research."
            }
        },
        "additionalProperties": False,
        "required": ["dataset_description", "citation_type", "reuse_potential"]
    }
}


dataset_response_schema_with_use_description = {
    "type": "json_schema",
    "name": "PaperMiner_dataset_provenance_schema",
    "schema": {
        "type": "object",
        "properties": {
            "datasets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dataset_identifier": {
                            "type": "string",
                            "description": "A unique identifier or accession code for the dataset."
                        },
                        "data_repository": {
                            "type": "string",
                            "description": "A valid URI or string referring to the repository where the dataset can be found."
                        },
                        "dataset_context_from_paper": {
                            "type": "string",
                            "description": "Relevant text passages from the paper that either describe this dataset and provide context of its use or refer to it more implicitly."
                        },
                        "citation_type": {
                            "type": "string",
                            "description": "Type of citation used for this dataset. It can be either Primary (firsthand information collected by the researcher for a specific purpose) or Secondary (pre-existing information collected by someone else and then used by another researcher)."
                        }
                    },
                    "additionalProperties": False,
                    "required": ["dataset_identifier", "data_repository", "dataset_context_from_paper", "citation_type"]
                },
                "minItems": 1,
                "additionalProperties": False
            }
        },
        "additionalProperties": False,
        "required": ["datasets"]
    }
}

# Enhanced JSON schema for GPT responses with context
dataset_response_schema_with_context = {
    "type": "json_schema",
    "name": "GPT_dataset_context_schema",
    "schema": {
        "type": "object",
        "properties": {
            "datasets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dataset_identifier": {
                            "type": "string",
                            "description": "A unique identifier or accession code for the dataset."
                        },
                        "data_repository": {
                            "type": "string",
                            "description": "A valid URI or string referring to the repository."
                        },
                        "dataset_usage_role": {
                            "type": "string",
                            "enum": ["training_data", "validation_data", "comparison_baseline", "reference_standard", "supplementary_data", "replication_data", "meta_analysis_source", "other"],
                            "description": "The primary role of this dataset in the research study."
                        },
                        "usage_description": {
                            "type": "string",
                            "description": "A brief, specific description of how this dataset was used in the study."
                        },
                        "results_relationship": {
                            "type": "string",
                            "enum": ["supports_main_findings", "contradicts_previous_work", "provides_context", "enables_methodology", "validates_approach", "other"],
                            "description": "How this dataset relates to the study's findings and conclusions."
                        },
                        "decision_rationale": {
                            "type": "string",
                            "description": "Chain of thought reasoning explaining why this dataset was selected and its significance."
                        },
                        "dataset_scope": {
                            "type": "string",
                            "enum": ["primary_analysis", "secondary_analysis", "background_context", "methodology_development", "comparative_study", "other"],
                            "description": "The analytical scope where this dataset fits in the research."
                        }
                    },
                    "additionalProperties": False,
                    "required": ["dataset_identifier", "data_repository", "dataset_usage_role", "usage_description", "results_relationship", "decision_rationale", "dataset_scope"]
                },
                "minItems": 1,
                "additionalProperties": False
            }
        },
        "additionalProperties": False,
        "required": ["datasets"]
    }
}


class Dataset(BaseModel):
    dataset_identifier: str
    data_repository: str

class Dataset_w_Page(BaseModel):
    dataset_identifier: str
    data_repository: str
    dataset_webpage: str

class Dataset_w_CitationType(BaseModel):
    dataset_identifier: str
    data_repository: str
    citation_type: str

class Array_Dataset_w_CitationType(BaseModel):
    datasets: list[Dataset_w_CitationType]

class Dataset_w_Description(typing.TypedDict):
    dataset_identifier: str
    data_repository: str
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

class Dataset_w_Use_Description(BaseModel):
    dataset_identifier: str
    data_repository: str
    dataset_context_from_paper: str  # Rich description of how this dataset was used in the paper - enables data reuse

class Single_Target_Dataset_Context(BaseModel):
    dataset_identifier: str
    data_repository: str
    dataset_context_from_paper: str  # Relevant text passages describing how this specific dataset was used
    citation_type: str  # "Primary" or "Secondary"

class Array_Single_Target_Dataset_Context(BaseModel):
    datasets: list[Single_Target_Dataset_Context]



class Dataset_w_Context(BaseModel):
    dataset_identifier: str
    data_repository: str
    dataset_usage_role: str  # How the dataset is used: "training_data", "validation_data", "comparison_baseline", "reference_standard", "supplementary_data", "replication_data", "meta_analysis_source", "other"
    usage_description: str  # Brief description of how this dataset was used in the study
    results_relationship: str  # How it relates to findings: "supports_main_findings", "contradicts_previous_work", "provides_context", "enables_methodology", "validates_approach", "other"
    decision_rationale: str  # Why this dataset was selected and its significance
    dataset_scope: str  # Scope of the dataset: "primary_analysis", "secondary_analysis", "background_context", "methodology_development", "comparative_study", "other"


# Response schema for supplementary files keywords extraction
supplementary_files_keywords_schema = {
    "type": "json_schema",
    "name": "Supplementary_files_keywords_schema",
    "schema": {
        "type": "object",
        "properties": {
            "supplementary_file_keywords": {
                "type": "array",
                "description": "Array of keyword strings, one for each supplementary file in the same order as input",
                "items": {
                    "type": "string",
                    "description": "3-5 keywords separated by commas describing the file content, data type, and purpose",
                    "maxLength": 128
                },
                "minItems": 1
            }
        },
        "required": ["supplementary_file_keywords"],
        "additionalProperties": False
    }
}

class SupplementaryFileKeywords(BaseModel):
    supplementary_file_keywords: list[str]

