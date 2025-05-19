import pytest
import httpx
from fastapi.testclient import TestClient
import os
import json
import asyncio # Added import
from unittest.mock import patch, MagicMock, AsyncMock

# Adjust the import path based on how you run pytest
# If running pytest from the project root (langchain-openai-proxy/):
from app.api import app  # Assuming your FastAPI app instance is named 'app' in app/api.py
from app.models import ChatCompletionRequest, ChatMessage, ChatCompletionResponse, ErrorResponse
from app.utils import load_config # To potentially load test-specific configs or inspect main config

# If .env file is in the project root, load it for tests that might depend on env vars
# from dotenv import load_dotenv
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


@pytest.fixture(scope="module")
def client():
    """
    Test client for the FastAPI application.
    """
    # Ensure that the main config is loaded, or a test-specific one.
    # Environment variables for API keys should be set for tests that hit real LLMs (if any),
    # or LLM calls should be mocked.
    # For most tests, we'll mock LLM calls.
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="module")
def mock_env_vars():
    """
    Mocks essential environment variables if not already set.
    This is crucial if config.yml relies on them and they aren't in the test environment.
    """
    # Example:
    # os.environ["PRIMARY_LLM_API_KEY"] = "test_primary_key"
    # os.environ["TARGET_LLM_API_KEY"] = "test_target_key"
    # Ensure these are cleaned up or only set if undefined.
    # A better way for tests is to have a test-specific config or mock load_config.
    pass # Placeholder

# --- Test Data ---
SAMPLE_CHAT_REQUEST_NON_STREAM = {
    "model": "gpt-3.5-turbo", # This will be overridden by target LLM config
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"}
    ],
    "stream": False,
    "temperature": 0.7
}

SAMPLE_CHAT_REQUEST_STREAM = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": "Tell me a joke."}
    ],
    "stream": True
}

# --- Mocked LLM Responses ---
# These would be more complex to simulate actual LLM behavior.
MOCK_PRIMARY_LLM_RESPONSE_CONTENT = "This is the answer from the primary LLM."
MOCK_TARGET_LLM_RESPONSE_CONTENT_NON_STREAM = "This is the final answer from the target LLM."
MOCK_TARGET_LLM_RESPONSE_STREAM_CHUNKS = [
    {"id": "chatcmpl-test123", "object": "chat.completion.chunk", "created": 12345, "model": "mock-target-model", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
    {"id": "chatcmpl-test123", "object": "chat.completion.chunk", "created": 12345, "model": "mock-target-model", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
    {"id": "chatcmpl-test123", "object": "chat.completion.chunk", "created": 12345, "model": "mock-target-model", "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]},
    {"id": "chatcmpl-test123", "object": "chat.completion.chunk", "created": 12345, "model": "mock-target-model", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
]


# --- Tests ---

def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Service is healthy"}

@patch("app.core.get_llm_client") # Patch where get_llm_client is *used*
def test_chat_completion_non_stream_success(mock_get_llm_client, client: TestClient):
    # Mock the primary LLM client
    mock_primary_llm = AsyncMock()
    # Configure the mock's agenerate_stream to simulate OpenAI-like stream chunks
    async def primary_stream_gen(*args, **kwargs):
        yield {"choices": [{"delta": {"content": "Primary "}}]}
        yield {"choices": [{"delta": {"content": "answer"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    mock_primary_llm.agenerate_stream = primary_stream_gen
    
    # Mock the target LLM client
    mock_target_llm = AsyncMock()
    mock_target_llm.agenerate.return_value = {
        "id": "chatcmpl-mocktarget",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "mock-target-model", # Should match what the target LLM would be
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": MOCK_TARGET_LLM_RESPONSE_CONTENT_NON_STREAM,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    # Configure mock_get_llm_client to return the correct mock based on type
    def side_effect_get_llm_client(llm_type: str):
        if llm_type == "primary":
            return mock_primary_llm
        elif llm_type == "target":
            return mock_target_llm
        raise ValueError(f"Unexpected llm_type: {llm_type}")
    
    mock_get_llm_client.side_effect = side_effect_get_llm_client

    response = client.post("/v1/chat/completions", json=SAMPLE_CHAT_REQUEST_NON_STREAM)

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["choices"][0]["message"]["content"] == MOCK_TARGET_LLM_RESPONSE_CONTENT_NON_STREAM
    assert json_response["model"] == "mock-target-model" # Check if model name is from target
    assert "usage" in json_response
    
    # Verify primary LLM was called with reconstructed prompt (harder to check exact prompt here without more mocking)
    # For now, check it was called.
    # mock_primary_llm.agenerate_stream.assert_called_once() # Or called if it's used
    
    # Verify target LLM was called with messages including primary's answer
    # The actual messages sent to target_llm.agenerate would be original + "\n" + "Primary answer"
    # This requires inspecting the call arguments.
    # For example:
    # called_args, called_kwargs = mock_target_llm.agenerate.call_args
    # messages_to_target = called_kwargs['messages'] # or called_args[0] depending on signature
    # assert messages_to_target[-1]['content'].endswith("Primary answer")


@patch("app.core.get_llm_client")
def test_chat_completion_stream_success(mock_get_llm_client, client: TestClient):
    # Mock primary LLM (similar to non-stream, its full response is aggregated internally)
    mock_primary_llm = AsyncMock()
    async def primary_stream_gen_stream_test(*args, **kwargs):
        yield {"choices": [{"delta": {"content": "Thoughtful primary response. "}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    mock_primary_llm.agenerate_stream = primary_stream_gen_stream_test

    # Mock target LLM to produce a stream
    mock_target_llm = AsyncMock()
    async def target_stream_gen(*args, **kwargs):
        for chunk in MOCK_TARGET_LLM_RESPONSE_STREAM_CHUNKS:
            yield chunk
            await asyncio.sleep(0.01) # Simulate network delay
    mock_target_llm.agenerate_stream = target_stream_gen
    
    def side_effect_get_llm_client_stream(llm_type: str):
        if llm_type == "primary":
            return mock_primary_llm
        elif llm_type == "target":
            return mock_target_llm
        raise ValueError(f"Unexpected llm_type for stream: {llm_type}")
        
    mock_get_llm_client.side_effect = side_effect_get_llm_client_stream

    response = client.post("/v1/chat/completions", json=SAMPLE_CHAT_REQUEST_STREAM)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Process the stream
    received_chunks = []
    full_content = ""
    for line in response.iter_lines():
        if line.startswith("data: "):
            data_str = line[len("data: "):]
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            received_chunks.append(chunk)
            if chunk["choices"][0]["delta"].get("content"):
                full_content += chunk["choices"][0]["delta"]["content"]
    
    assert len(received_chunks) == len(MOCK_TARGET_LLM_RESPONSE_STREAM_CHUNKS)
    assert full_content == "Hello world" # From MOCK_TARGET_LLM_RESPONSE_STREAM_CHUNKS
    assert received_chunks[-1]["choices"][0]["finish_reason"] == "stop"


# TODO: Add more tests:
# - Test with different configurations in config.yml (mock load_config)
# - Test error handling:
#   - Primary LLM fails
#   - Target LLM fails (non-stream and stream)
#   - Invalid request format (Pydantic validation should catch this, FastAPI returns 422)
#   - Authentication/Authorization errors (if implemented)
# - Test token counting (requires tiktoken and more detailed mocking of usage)
# - Test `tool_calls` and other OpenAI features if they are to be supported.
# - Test the actual prompt reconstruction for primary LLM.
# - Test the message update logic (Step 7).

# Example of how to mock load_config if needed for specific test configurations:
# @patch("app.utils.load_config")
# def test_with_custom_config(mock_load_config_custom, client: TestClient):
#     mock_load_config_custom.return_value = {
#         "server": {"host": "127.0.0.1", "port": 8080},
#         "llms": {
#             "primary": {"provider": "mock", "model": "mock-primary"},
#             "target": {"provider": "mock", "model": "mock-target"}
#         },
#         "prompts": {"system": "Test system prompt"}
#     }
#     # ... your test logic that depends on this mocked config ...