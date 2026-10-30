from data_gatherer.llm.llm_client import LLMClient_dev


def test_vllm_response_unwraps_expected_key_in_retrieval_mode():
    client = LLMClient_dev.__new__(LLMClient_dev)
    client.model = "vllm-openai/gpt-oss-20b"
    client.full_document_read = False
    client.logger = None
    client.normalize_response_format = lambda value: value

    result = client.process_llm_response(
        '{"datasets": [{"name": "example"}]}', expected_key="datasets"
    )

    assert result == [{"name": "example"}]


def test_vllm_response_keeps_payload_without_expected_key():
    client = LLMClient_dev.__new__(LLMClient_dev)
    client.model = "vllm-openai/gpt-oss-20b"
    client.full_document_read = False
    client.logger = None
    client.normalize_response_format = lambda value: value

    result = client.process_llm_response(
        '{"other": [{"name": "example"}]}', expected_key="datasets"
    )

    assert result == {"other": [{"name": "example"}]}
