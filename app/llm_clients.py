import json
import httpx
from openai import AsyncOpenAI, OpenAI # For OpenAI and compatible APIs like DeepSeek
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI as LangchainChatOpenAI
from langchain_deepseek import ChatDeepSeek # Added import for DeepSeek

from typing import Dict, Any, AsyncGenerator, List, Optional, Union
from abc import ABC, abstractmethod
from langsmith import traceable # Import traceable
import time # Added import
import uuid # Added import
import os # Added import

from .utils import load_config, logger, create_openai_error_response
from .models import ChatMessage, PrimaryLLMStreamResponse, Thought # Pydantic models

# --- Configuration Loading ---
try:
    config = load_config()
    llm_configs = config.get("llms", {})
    PRIMARY_LLM_CONFIG = llm_configs.get("primary", {})
    TARGET_LLM_CONFIG = llm_configs.get("target", {})
except Exception as e:
    logger.error("llm_clients.py: Failed to load LLM configurations", error=str(e))
    PRIMARY_LLM_CONFIG = {}
    TARGET_LLM_CONFIG = {}

# --- Helper function to map Langchain messages to OpenAI-like dicts ---
def _to_openai_format(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            # Handle AIMessage content (could be str or list of parts)
            content = msg.content
            # if isinstance(content, list): # For multimodal, not directly used here yet
            #     content = [part.dict() if hasattr(part, 'dict') else part for part in content]
            formatted_messages.append({"role": "assistant", "content": content})
        elif isinstance(msg, SystemMessage):
            formatted_messages.append({"role": "system", "content": msg.content})
        # Add other message types if necessary (ToolMessage, etc.)
    return formatted_messages

def _to_langchain_format(messages: List[Dict[str, Any]]) -> List[BaseMessage]:
    lc_messages = []
    for msg_dict in messages:
        role = msg_dict.get("role")
        content = msg_dict.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "system":
            lc_messages.append(SystemMessage(content=content))
        # Add other roles if necessary
    return lc_messages

# --- Abstract Base LLM Client ---
class AbstractLLMClient(ABC):
    def __init__(self, client_config: Dict[str, Any]):
        self.client_config = client_config
        self.model_name = client_config.get("model", "unknown-model")
        self.api_key = client_config.get("api_key")
        self.base_url = client_config.get("base_url") # Used for OpenAI and potentially DeepSeek custom endpoint
        
        if not self.api_key and not (self.base_url and "localhost" in self.base_url):
             # Warning if API key is missing, unless it's a local deployment that might not need one.
             # Specific clients (OpenAI, Deepseek) will log more detailed warnings if needed for their specific auth.
            logger.warning(f"API key not configured for LLM client with model {self.model_name} at {self.base_url or 'default endpoint'}")
        
        self.http_client = httpx.AsyncClient(timeout=client_config.get("timeout", 60.0)) # Use timeout from config or default

    @abstractmethod
    async def agenerate_stream(
        self, messages: List[Dict[str, Any]], stream_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generates a stream of responses.
        For OpenAI compatible, this would yield ChatCompletionStreamResponse like chunks.
        For PrimaryLLM, this would yield PrimaryLLMStreamResponse like chunks.
        """
        pass # yield {} # Must be implemented by subclasses

    @abstractmethod
    async def agenerate(
        self, messages: List[Dict[str, Any]], request_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates a single, non-streamed response.
        For OpenAI compatible, this would return a ChatCompletionResponse like dict.
        """
        pass

    async def close(self):
        await self.http_client.aclose()

# --- OpenAI Compatible Client (for both primary and target if they use OpenAI-like APIs or Langchain integrations) ---
class OpenAICompatibleClient(AbstractLLMClient):
    def __init__(self, client_config: Dict[str, Any]):
        super().__init__(client_config)
        self.provider = client_config.get("provider", "").lower()
        self.openai_client: Optional[AsyncOpenAI] = None
        self.deepseek_client: Optional[ChatDeepSeek] = None

        self.default_temperature = client_config.get("temperature", 0.7)
        self.default_max_tokens = client_config.get("max_tokens")

        if self.provider == "deepseek":
            if not self.api_key and not (self.base_url and "localhost" in self.base_url):
                logger.warning(f"DeepSeek API key not configured for model {self.model_name}. Langchain client might rely on environment variable DEEPSEEK_API_KEY or local setup.")
            
            chat_deepseek_params = {
                "model_name": self.model_name,
                "api_key": self.api_key if self.api_key else os.getenv("DEEPSEEK_API_KEY"), # Changed parameter name
                "temperature": self.default_temperature,
            }
            if self.default_max_tokens is not None:
                chat_deepseek_params["max_tokens"] = self.default_max_tokens
            
            if self.base_url: # Pass base_url if provided in config
                chat_deepseek_params["base_url"] = self.base_url # Changed parameter name
                logger.info(f"Using custom base_url for DeepSeek ({self.model_name}): {self.base_url}")
            
            request_timeout_val = client_config.get("request_timeout", client_config.get("timeout"))
            if request_timeout_val is not None:
                 chat_deepseek_params["request_timeout"] = request_timeout_val

            if "max_retries" in client_config:
                 chat_deepseek_params["max_retries"] = client_config.get("max_retries")

            try:
                self.deepseek_client = ChatDeepSeek(**chat_deepseek_params)
                logger.info(f"Initialized Langchain ChatDeepSeek client for model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize ChatDeepSeek client for model {self.model_name}: {e}", exc_info=True)
                # Potentially raise this or handle it so agenerate/astream fail gracefully
                raise ValueError(f"ChatDeepSeek client initialization failed for {self.model_name}") from e
        
        elif self.provider in ["openai", "azure_openai"]: 
            # Existing OpenAI/AzureOpenAI client initialization logic
            if not self.api_key and self.provider != "azure_openai" and not (self.base_url and "localhost" in self.base_url):
                 logger.warning(f"API key not configured for LLM client with model {self.model_name} at {self.base_url} for provider {self.provider}")

            self.openai_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=self.http_client 
            )
            logger.info(f"Initialized OpenAI AsyncOpenAI client for model: {self.model_name} at {self.base_url or 'default OpenAI endpoint'} (provider: {self.provider})")
        else:
            raise ValueError(f"OpenAICompatibleClient received an unsupported provider: '{self.provider}'. Supported: 'openai', 'deepseek', 'azure_openai'.")

    @traceable(run_type="llm")
    async def agenerate_stream(
        self,
        messages: List[Dict[str, Any]],
        stream_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if stream_config is None:
            stream_config = {}

        if self.deepseek_client:
            lc_messages = _to_langchain_format(messages)
            lc_call_config = {}
            configurable_params = {}
            
            temp_to_use = stream_config.get("temperature", self.default_temperature)
            max_tokens_to_use = stream_config.get("max_tokens", self.default_max_tokens)

            configurable_params["temperature"] = temp_to_use
            if max_tokens_to_use is not None:
                configurable_params["max_tokens"] = max_tokens_to_use
            
            # top_p for DeepSeek via configurable if supported by its Langchain implementation
            if "top_p" in stream_config:
                # Assuming ChatDeepSeek might support top_p in .configurable()
                # configurable_params["top_p"] = stream_config["top_p"]
                logger.debug("top_p in stream_config for DeepSeek. Ensure ChatDeepSeek supports this via 'configurable'.")

            if configurable_params:
                lc_call_config["configurable"] = configurable_params

            try:
                async for lc_chunk in self.deepseek_client.astream(lc_messages, config=lc_call_config):
                    chunk_content = lc_chunk.content
                    # Langchain AIMessageChunk might have response_metadata with finish_reason
                    finish_reason = None
                    if lc_chunk.response_metadata and "finish_reason" in lc_chunk.response_metadata:
                        finish_reason = lc_chunk.response_metadata["finish_reason"]
                    
                    chunk_id = lc_chunk.id if isinstance(lc_chunk.id, str) and lc_chunk.id else f"chatcmpl-ds-{uuid.uuid4().hex}"

                    openai_chunk = {
                        "id": chunk_id,
                        "choices": [{
                            "delta": {"role": "assistant", "content": chunk_content if chunk_content is not None else ""},
                            "index": 0,
                        }],
                        "created": int(time.time()),
                        "model": self.model_name, # Or a more specific model name if available from lc_chunk
                        "object": "chat.completion.chunk",
                    }
                    if finish_reason:
                        openai_chunk["choices"][0]["finish_reason"] = finish_reason
                    
                    yield openai_chunk
            except Exception as e:
                logger.error(f"Error during DeepSeek stream for model {self.model_name}: {e}", exc_info=True)
                yield create_openai_error_response(str(e), model_name=self.model_name, request_id=f"dse-stream-{uuid.uuid4().hex}")

        elif self.openai_client:
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "temperature": stream_config.get("temperature", self.default_temperature),
                "top_p": stream_config.get("top_p", self.client_config.get("top_p", 1.0)),
                "max_tokens": stream_config.get("max_tokens", self.default_max_tokens),
            }
            api_params = {k: v for k, v in api_params.items() if v is not None}

            try:
                stream = await self.openai_client.chat.completions.create(**api_params)
                async for chunk in stream:
                    yield chunk.model_dump()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTPStatusError during OpenAI stream for model {self.model_name}: {e.response.text}", exc_info=True)
                yield create_openai_error_response(str(e), model_name=self.model_name, request_id=e.response.headers.get("x-request-id"))
            except httpx.RequestError as e:
                logger.error(f"RequestError during OpenAI stream for model {self.model_name}: {e}", exc_info=True)
                yield create_openai_error_response(str(e), model_name=self.model_name)
            except Exception as e:
                logger.error(f"Generic error during OpenAI stream for model {self.model_name}: {e}", exc_info=True)
                yield create_openai_error_response(str(e), model_name=self.model_name)
        else:
            logger.error(f"agenerate_stream called on {self.model_name} with no valid LLM client configured (provider: {self.provider}).")
            yield create_openai_error_response("Client not properly initialized or unsupported provider.", model_name=self.model_name)

    @traceable(run_type="llm")
    async def agenerate(
        self,
        messages: List[Dict[str, Any]],
        request_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if request_config is None:
            request_config = {}

        if self.deepseek_client:
            lc_messages = _to_langchain_format(messages)
            lc_call_config = {}
            configurable_params = {}

            temp_to_use = request_config.get("temperature", self.default_temperature)
            max_tokens_to_use = request_config.get("max_tokens", self.default_max_tokens)

            configurable_params["temperature"] = temp_to_use
            if max_tokens_to_use is not None:
                configurable_params["max_tokens"] = max_tokens_to_use

            if "top_p" in request_config:
                # configurable_params["top_p"] = request_config["top_p"]
                logger.debug("top_p in request_config for DeepSeek. Ensure ChatDeepSeek supports this via 'configurable'.")

            if configurable_params:
                lc_call_config["configurable"] = configurable_params
            
            try:
                lc_response_msg = await self.deepseek_client.ainvoke(lc_messages, config=lc_call_config)
                
                finish_reason = "stop" # Default
                if lc_response_msg.response_metadata and "finish_reason" in lc_response_msg.response_metadata:
                    finish_reason = lc_response_msg.response_metadata["finish_reason"]
                
                usage_data = None
                if lc_response_msg.usage_metadata:
                    usage_data = {
                        "prompt_tokens": lc_response_msg.usage_metadata.get("input_tokens", 0),
                        "completion_tokens": lc_response_msg.usage_metadata.get("output_tokens", 0),
                        "total_tokens": lc_response_msg.usage_metadata.get("total_tokens", 0),
                    }
                
                response_id = lc_response_msg.id if isinstance(lc_response_msg.id, str) and lc_response_msg.id else f"chatcmpl-ds-{uuid.uuid4().hex}"

                openai_response = {
                    "id": response_id,
                    "choices": [{
                        "message": {"role": "assistant", "content": lc_response_msg.content},
                        "finish_reason": finish_reason,
                        "index": 0,
                    }],
                    "created": int(time.time()),
                    "model": self.model_name, # Or a more specific model name from lc_response_msg if available
                    "object": "chat.completion",
                    "usage": usage_data if usage_data else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                return openai_response
            except Exception as e:
                logger.error(f"Error during DeepSeek non-stream for model {self.model_name}: {e}", exc_info=True)
                return create_openai_error_response(str(e), model_name=self.model_name, request_id=f"dse-{uuid.uuid4().hex}")

        elif self.openai_client:
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "temperature": request_config.get("temperature", self.default_temperature),
                "top_p": request_config.get("top_p", self.client_config.get("top_p", 1.0)),
                "max_tokens": request_config.get("max_tokens", self.default_max_tokens),
            }
            api_params = {k: v for k, v in api_params.items() if v is not None}

            try:
                response = await self.openai_client.chat.completions.create(**api_params)
                return response.model_dump()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTPStatusError during OpenAI non-stream for model {self.model_name}: {e.response.text}", exc_info=True)
                return create_openai_error_response(str(e), model_name=self.model_name, request_id=e.response.headers.get("x-request-id"))
            except httpx.RequestError as e:
                logger.error(f"RequestError during OpenAI non-stream for model {self.model_name}: {e}", exc_info=True)
                return create_openai_error_response(str(e), model_name=self.model_name)
            except Exception as e:
                logger.error(f"Generic error during OpenAI non-stream for model {self.model_name}: {e}", exc_info=True)
                return create_openai_error_response(str(e), model_name=self.model_name)
        else:
            logger.error(f"agenerate called on {self.model_name} with no valid LLM client configured (provider: {self.provider}).")
            return create_openai_error_response("Client not properly initialized or unsupported provider.", model_name=self.model_name)

    async def close(self):
        # The shared self.http_client is used by AsyncOpenAI.
        # Langchain clients like ChatDeepSeek manage their own HTTP clients internally.
        if self.openai_client: # Implying self.http_client was potentially used by AsyncOpenAI
            await self.http_client.aclose()
            logger.info(f"Closed shared http_client for {self.provider} model {self.model_name} (if it was OpenAI-based).")
        elif self.provider == "deepseek":
            logger.info(f"No explicit close needed for Langchain ChatDeepSeek client ({self.model_name}). It manages its own resources.")
        else:
            # If http_client was initialized but no openai_client (e.g. only deepseek or other future langchain client)
            # it might still need closing if it was intended for general use by AbstractLLMClient subclasses.
            # However, current design ties its use to AsyncOpenAI.
            # For safety, if it exists and wasn't closed via openai_client path:
            if hasattr(self, 'http_client') and not self.http_client.is_closed:
                 await self.http_client.aclose()
                 logger.info(f"Closed http_client for {self.model_name} as a fallback.")

# --- Specialized Primary LLM Client (Step 5-6) ---
# This client needs to handle the "thoughts" and "answer" structure.
# For simplicity, we'll assume it's also OpenAI-compatible but we will parse its output differently.
# If it had a truly custom API, this class would be significantly different.

class PrimaryLLMClient(OpenAICompatibleClient): # Inherits for OpenAI compatibility
    def __init__(self, client_config: Dict[str, Any]):
        super().__init__(client_config)
        logger.info(f"PrimaryLLMClient initialized for model: {self.model_name}")

    async def agenerate_thoughts_and_answer_stream(
        self, 
        messages: List[Dict[str, Any]], 
        stream_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[PrimaryLLMStreamResponse, None]:
        """
        Streams response from the primary LLM.
        It expects the LLM to first stream a complete JSON object for 'thoughts',
        and then stream the 'answer' text.
        This method buffers incoming tokens (delta_content) and attempts to parse
        the buffer as JSON for thoughts. Once thoughts are parsed, subsequent tokens
        are treated as answer_delta.
        """
        state = "PARSING_THOUGHTS"  # States: PARSING_THOUGHTS, STREAMING_ANSWER
        thoughts_buffer = ""
        # This client is responsible for parsing the raw OpenAI stream (or compatible)
        # into PrimaryLLMStreamResponse objects.
        # Errors from the underlying stream (super().agenerate_stream) should ideally be
        # OpenAI-formatted error chunks if they are API/HTTP errors.

        async for chunk in super().agenerate_stream(messages, stream_config):
            # If the chunk from super().agenerate_stream() is an error structure
            if chunk.get("error"):
                # core.py expects to see this error if primary_llm.agenerate_stream yields it.
                # However, this method's type hint is PrimaryLLMStreamResponse.
                # This indicates a mismatch in error handling contracts if errors are not
                # wrapped in PrimaryLLMStreamResponse.
                # For now, we log it. A more robust solution would be to define how
                # upstream errors are propagated through this layer.
                # If core.py checks for chunk.get("error") on the output of this method,
                # then yielding the raw error chunk might be intended, but violates type hint.
                logger.error(f"PrimaryLLMClient received an error chunk from underlying stream: {chunk}")
                # Option 1: Re-yield the error chunk if caller handles it (violates type hint)
                # yield chunk 
                # Option 2: Try to wrap in PrimaryLLMStreamResponse (if model supports error fields)
                # yield PrimaryLLMStreamResponse(error=chunk.get("error")) # Requires model change
                # Option 3: Raise an exception that core.py can catch
                # raise _PrimaryLLMInternalError(chunk) # core.py uses this for its own errors
                # For now, let's assume if an error chunk comes, content processing stops for this chunk.
                # If it's a fatal error, the stream might end with a finish_reason.
                # If not, and the stream continues, subsequent non-error chunks will be processed.
                # This part needs a clear error contract.
                # Let's assume for now that if an error chunk is received, we might not get content.
                # If a finish_reason is also present, it will be handled.
                pass # Current pass-through means error might be silently ignored if no content/finish_reason

            delta_content: Optional[str] = None
            finish_reason: Optional[str] = None
            
            choices = chunk.get("choices")
            if choices and isinstance(choices, list) and len(choices) > 0:
                choice = choices[0]
                if choice.get("delta"):
                    delta_content = choice["delta"].get("content")
                finish_reason = choice.get("finish_reason")
            # Some LLM providers might send finish_reason at the top level of the chunk
            elif "finish_reason" in chunk and not finish_reason: 
                 finish_reason = chunk.get("finish_reason")


            if state == "PARSING_THOUGHTS":
                if delta_content:
                    thoughts_buffer += delta_content
                
                # Attempt to parse the buffer if it looks like it might be complete JSON,
                # or if the stream is about to finish.
                # Heuristic: starts with '{'. A more robust check would be `thoughts_buffer.strip().endswith("}")`
                # but `delta_content` might only contain a part of the closing brace.
                # We rely on json.loads to validate full structure.
                if thoughts_buffer.strip().startswith("{"):
                    try:
                        # Try to parse the entire accumulated buffer.
                        # This assumes the "thoughts" JSON is sent as one contiguous block of text.
                        parsed_json = json.loads(thoughts_buffer)
                        
                        if isinstance(parsed_json, dict) and "thoughts" in parsed_json:
                            yield PrimaryLLMStreamResponse(thoughts=parsed_json["thoughts"])
                            
                            # If the same JSON object also contained an initial answer part
                            if "answer_delta" in parsed_json and parsed_json["answer_delta"]:
                                yield PrimaryLLMStreamResponse(answer_delta=parsed_json["answer_delta"])
                            
                            state = "STREAMING_ANSWER"
                            thoughts_buffer = ""  # Clear buffer as thoughts are processed
                        else:
                            # Parsed to JSON, but not the expected "thoughts" structure.
                            # If the stream is ending, this is all we got. Treat as answer.
                            if finish_reason:
                                logger.warning(
                                    f"Primary LLM stream ended. Parsed JSON from buffer was not 'thoughts'. "
                                    f"Buffer: {thoughts_buffer[:200]}"
                                )
                                if thoughts_buffer: # Ensure there's content to yield
                                    yield PrimaryLLMStreamResponse(answer_delta=thoughts_buffer)
                                state = "STREAMING_ANSWER" # Transition to allow finish_reason to be processed
                                thoughts_buffer = ""
                            # else: if not finish_reason, it might be an incomplete JSON or a different JSON.
                            # We continue accumulating, hoping it resolves into the thoughts JSON.
                            # If it never does, the finish_reason block below will handle the leftover buffer.
                    except json.JSONDecodeError:
                        # Not a complete JSON object yet. Continue accumulating.
                        # If the stream is ending and it's still not valid JSON, then it's an error or malformed.
                        if finish_reason:
                            logger.warning(
                                f"Primary LLM stream ended. Accumulated buffer was not valid JSON for 'thoughts'. "
                                f"Buffer: {thoughts_buffer[:500]}"
                            )
                            if thoughts_buffer: # Ensure there's content
                                 yield PrimaryLLMStreamResponse(answer_delta=thoughts_buffer)
                            state = "STREAMING_ANSWER" # Transition
                            thoughts_buffer = ""
                
                # If stream is ending (finish_reason is present) and we are still in PARSING_THOUGHTS:
                # This means either parsing failed, or buffer was empty, or it parsed to non-thoughts JSON.
                # The try-except block above handles cases where parsing was attempted due to finish_reason.
                # This is an additional check if, for some reason, the buffer still has content
                # and thoughts were not successfully yielded.
                if finish_reason and state == "PARSING_THOUGHTS":
                    if thoughts_buffer: # If anything is left in the buffer unprocessed
                        logger.info(
                            f"Stream finished in PARSING_THOUGHTS. Yielding remaining buffer as answer: "
                            f"{thoughts_buffer[:100]}"
                        )
                        yield PrimaryLLMStreamResponse(answer_delta=thoughts_buffer)
                        thoughts_buffer = "" # Clear it as it's now processed
                    # The finish_reason itself will be yielded by the common block below.
                    # Ensure state transition if not already, so we don't re-process buffer.
                    state = "STREAMING_ANSWER"


            elif state == "STREAMING_ANSWER":
                if delta_content:
                    yield PrimaryLLMStreamResponse(answer_delta=delta_content)

            if finish_reason:
                # If there was a leftover thoughts_buffer when finish_reason arrived and thoughts were not parsed,
                # it should have been handled above and yielded as answer_delta.
                # This final yield ensures the finish_reason is always sent.
                yield PrimaryLLMStreamResponse(is_final_answer=True, finish_reason=finish_reason)
                return # End generation after finish_reason
        
        # Fallback: If the stream ends without a finish_reason in the last content chunk
        # (e.g., connection drops or LLM stops abruptly). This is less common for well-behaved APIs.
        # If thoughts_buffer still has content and we never switched to STREAMING_ANSWER.
        if state == "PARSING_THOUGHTS" and thoughts_buffer:
            logger.warning(
                f"Stream ended unexpectedly (no finish_reason in last chunk). "
                f"Yielding leftover thoughts_buffer as answer_delta: {thoughts_buffer[:200]}"
            )
            yield PrimaryLLMStreamResponse(answer_delta=thoughts_buffer)
            # Optionally, yield a synthetic finish_reason to signal completion to downstream.
            yield PrimaryLLMStreamResponse(is_final_answer=True, finish_reason="unknown_end_of_stream")


# --- Factory function to get LLM clients ---
async_clients_cache: Dict[str, AbstractLLMClient] = {}

async def get_llm_client(llm_type: str = "target") -> AbstractLLMClient:
    """
    Gets an LLM client based on the type ("primary" or "target").
    Uses cached instances.
    """
    global async_clients_cache

    if llm_type in async_clients_cache:
        return async_clients_cache[llm_type]

    if llm_type == "primary":
        config_to_use = PRIMARY_LLM_CONFIG
        # The spec implies PrimaryLLMClient has special streaming.
        # If its provider is 'openai' or 'deepseek', it will use OpenAICompatibleClient's stream
        # and the parsing of thoughts/answer needs to be extremely robust or done by the LLM outputting structured JSONs.
        # For now, let's assume PrimaryLLMClient is for an LLM that streams the PrimaryLLMStreamResponse structure.
        # If the provider is OpenAI/Deepseek, this client will try to parse its output as such.
        if not config_to_use:
            raise ValueError("Primary LLM configuration is missing.")
        provider = config_to_use.get("provider", "").lower()
        if provider in ["openai", "deepseek", "azure_openai"]: # Known OpenAI-compatible
            # This is where the difficulty lies: an OpenAI client won't natively stream PrimaryLLMStreamResponse.
            # The PrimaryLLMClient's agenerate_thoughts_and_answer_stream needs to be very smart
            # or the LLM needs to be prompted to output JSON that fits PrimaryLLMStreamResponse *per chunk*.
            client = PrimaryLLMClient(config_to_use) # Uses special streaming method
        else:
            # Potentially a custom client for other providers
            logger.warning(f"Primary LLM provider '{provider}' is not explicitly OpenAI-compatible. Falling back to generic OpenAICompatibleClient logic for it, but it might not support thoughts/answer streaming correctly.")
            client = OpenAICompatibleClient(config_to_use) # Fallback, may not work as primary
            # raise NotImplementedError(f"LLM provider '{provider}' for primary LLM is not yet supported with specialized streaming.")
        async_clients_cache[llm_type] = client
        return client
        
    elif llm_type == "target":
        config_to_use = TARGET_LLM_CONFIG
        if not config_to_use:
            raise ValueError("Target LLM configuration is missing.")
        provider = config_to_use.get("provider", "").lower()
        if provider in ["openai", "deepseek", "azure_openai"]:
            client = OpenAICompatibleClient(config_to_use)
        else:
            raise NotImplementedError(f"LLM provider '{provider}' for target LLM is not yet supported.")
        async_clients_cache[llm_type] = client
        return client
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}. Must be 'primary' or 'target'.")

async def close_llm_clients():
    """Closes all cached LLM clients."""
    global async_clients_cache
    for client in async_clients_cache.values():
        await client.close()
    async_clients_cache.clear()
    logger.info("All LLM clients closed.")


if __name__ == "__main__":
    # Example Usage (requires .env file with API keys and config.yml)
    # Ensure your config.yml has 'primary' and 'target' LLMs defined.
    # And OPENAI_API_KEY, DEEPSEEK_API_KEY (if used) in .env

    async def test_clients():
        # Test Primary LLM (assuming it's OpenAI compatible and prompted to stream JSONs)
        try:
            print("\n--- Testing Primary LLM Client ---")
            primary_llm = await get_llm_client("primary")
            # This test assumes the primary LLM is prompted to output JSON that fits PrimaryLLMStreamResponse
            # For a standard OpenAI model, this won't happen unless the messages make it do so.
            # The current PrimaryLLMClient.agenerate_thoughts_and_answer_stream tries to parse JSON from OpenAI stream.
            
            # Let's test the underlying OpenAICompatibleClient's stream method for primary,
            # as the thoughts/answer parsing is complex and LLM-prompt-dependent.
            if isinstance(primary_llm, OpenAICompatibleClient): # True for PrimaryLLMClient too
                print(f"Primary LLM ({primary_llm.model_name}) is OpenAI compatible. Testing its raw stream.")
                test_messages_primary = [
                    {"role": "system", "content": "You are a helpful assistant. First, provide thoughts as a JSON object like {'thoughts': [{'step_name':'thinking', 'details':{}}]}, then provide the answer directly."},
                    {"role": "user", "content": "What is 2+2? Explain your thought process first in JSON, then the answer."}
                ]
                
                # If we call primary_llm.agenerate_thoughts_and_answer_stream:
                # It will try to parse the output of the above prompt.
                # This is highly dependent on the LLM faithfully producing streamable JSON chunks.
                
                # Let's test the direct stream and see what it gives:
                # async for chunk in primary_llm.agenerate_stream(test_messages_primary): # The raw OpenAI stream
                #     print(f"Primary LLM Raw Chunk: {chunk}")

                # Now testing the specialized stream (if it's PrimaryLLMClient)
                if hasattr(primary_llm, 'agenerate_thoughts_and_answer_stream'):
                    print(f"\nTesting Primary LLM ({primary_llm.model_name}) with agenerate_thoughts_and_answer_stream:")
                    async for response_part in primary_llm.agenerate_thoughts_and_answer_stream(test_messages_primary):
                        print(f"Primary LLM Parsed Part: {response_part.model_dump_json(indent=2)}")
                else: # Should not happen if get_llm_client works as intended for "primary"
                    print("Primary LLM does not have agenerate_thoughts_and_answer_stream method.")

        except Exception as e:
            print(f"Error testing primary LLM: {e}")
            logger.error("Error in primary LLM test", exc_info=True)

        # Test Target LLM
        try:
            print("\n--- Testing Target LLM Client ---")
            target_llm = await get_llm_client("target")
            print(f"Target LLM client type: {type(target_llm)}, Model: {target_llm.model_name}")
            
            test_messages_target = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! What's the weather like today?"}
            ]

            print("\nTarget LLM - Non-streaming:")
            response = await target_llm.agenerate(test_messages_target)
            print(json.dumps(response, indent=2))

            print("\nTarget LLM - Streaming:")
            async for chunk in target_llm.agenerate_stream(test_messages_target):
                print(f"Target LLM Chunk: {chunk}")

        except Exception as e:
            print(f"Error testing target LLM: {e}")
            logger.error("Error in target LLM test", exc_info=True)
        finally:
            await close_llm_clients()

    # Setup logging for the test
    from .utils import setup_logging as util_setup_logging
    util_setup_logging()
    
    # Create a dummy .env if it doesn't exist for local testing
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n") # Replace with a real key for actual test
            f.write("DEEPSEEK_API_KEY=your_deepseek_api_key_here\n") # Replace if using deepseek
        print("Created a dummy .env file. Please put your API keys there for testing.")

    # Create a dummy config.yml if it doesn't exist
    if not os.path.exists("../../config.yml"): # Relative to app/llm_clients.py
        # This path needs to be correct based on where load_config looks.
        # load_config uses os.path.dirname(os.path.dirname(os.path.abspath(__file__))) as project_root
        # So config.yml should be in langchain-openai-proxy/
        config_example_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yml")
        if not os.path.exists(config_example_path):
            with open(config_example_path, "w") as f:
                f.write("""
server:
  host: "0.0.0.0"
  port: 8000

llms:
  primary:
    provider: "openai" # or "deepseek"
    model: "gpt-3.5-turbo" # A model that can follow instructions for JSON thoughts
    api_key: "${OPENAI_API_KEY}" # or DEEPSEEK_API_KEY
    base_url: "https://api.openai.com/v1" # or deepseek base url
    temperature: 0.5

  target:
    provider: "openai"
    model: "gpt-3.5-turbo"
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    temperature: 0.7

prompts:
  system: "You are an advanced AI assistant."
  user_part1: "Based on the user's query, first outline your thought process in a JSON object under a 'thoughts' key, with each thought having 'step_name' and 'details'. Then, provide the final answer."
  user_part2: "Ensure the JSON for thoughts is complete and valid before starting the answer."
""")
            print(f"Created a dummy {config_example_path}. Please review and add API keys to .env.")
        else:
            print(f"Using existing {config_example_path}")


    # asyncio.run(test_clients()) # Comment out for non-execution during generation