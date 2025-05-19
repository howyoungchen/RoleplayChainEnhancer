import json
import os
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator, Optional, Union
import copy

from .models import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    ChoiceDelta,
    ChatCompletionStreamChoice,
    UsageInfo,
    PrimaryLLMStreamResponse, # For parsing what primary LLM client might yield
    ErrorResponse,
    ErrorDetail
)
from .llm_clients import get_llm_client # OpenAICompatibleClient, PrimaryLLMClient are not directly used here
from .llm_clients import PRIMARY_LLM_CONFIG, TARGET_LLM_CONFIG # Import configs for param merging
from .prompts import prompt_manager
from .utils import logger, generate_request_id, get_current_timestamp_ms, create_openai_error_response

# --- Custom Exceptions for flow control ---
class _BaseCoreError(Exception):
    """Base exception for core processing errors."""
    def __init__(self, message="Core processing error", details=None):
        super().__init__(message)
        self.details = details if details is not None else {}

class _PrimaryLLMInternalError(_BaseCoreError):
    """Custom exception for errors originating from primary LLM processing, carrying payload."""
    def __init__(self, error_payload: Dict[str, Any]):
        super().__init__(message=error_payload.get("error", {}).get("message", "Primary LLM internal error"), details=error_payload)
        self.error_payload = error_payload

class _TargetLLMInternalError(_BaseCoreError):
    """Custom exception for errors originating from target LLM processing, carrying payload."""
    def __init__(self, error_payload: Dict[str, Any]):
        super().__init__(message=error_payload.get("error", {}).get("message", "Target LLM internal error"), details=error_payload)
        self.error_payload = error_payload

class NonStreamFinalResponse(Exception):
    """Used to signal a complete, non-streamed response (success or error) that should be returned by api.py."""
    def __init__(self, response_data: Union[ChatCompletionResponse, ErrorResponse]):
        self.response_data = response_data
        if isinstance(response_data, ErrorResponse):
            super().__init__(response_data.error.message)
        else: # ChatCompletionResponse
            super().__init__("Successful non-stream response")


# Helper to convert Pydantic ChatMessage to dict for LLM client
def pydantic_messages_to_dict(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    return [msg.model_dump(exclude_none=True) for msg in messages]

# Helper to convert dict messages (from LLM or internal) to Pydantic ChatMessage
def dict_messages_to_pydantic(messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    return [ChatMessage(**msg) for msg in messages]

async def handle_chat_completion(
    request: ChatCompletionRequest, 
    request_id: str
) -> AsyncGenerator[str, None]: # Now always an async generator
    """
    Handles the main logic for /v1/chat/completions.
    Steps 1-9.
    This function will now always yield stream chunks for success or error.
    For non-stream requests, api.py will aggregate these if needed, or this function
    will raise a specific exception containing the full response.
    """
    start_time_ms = get_current_timestamp_ms()

    # --- Logging Setup ---
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S%f") # Removed underscore for a more compact timestamp
    # New directory format: logs/YYYYMMDDHHMMSSffffff_request_id/
    request_log_dir = os.path.join(log_dir, f"{timestamp_str}_{request_id}")
    if not os.path.exists(request_log_dir):
        os.makedirs(request_log_dir)

    # Log received request
    try:
        with open(os.path.join(request_log_dir, "received_request.json"), "w", encoding="utf-8") as f:
            json.dump(request.model_dump(exclude_none=True), f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to log received request", request_id=request_id, error=str(e))
    # --- End Logging Setup ---

    logger.info(
        "Processing chat completion request",
        request_id=request_id,
        model_requested=request.model,
        stream_requested=request.stream,
        num_messages=len(request.messages),
        log_path=request_log_dir
    )

    try:
        # --- Step 1: Receive Request (Done by FastAPI) ---
        # --- Step 2: Parse Messages ---
        original_messages_pydantic: List[ChatMessage] = copy.deepcopy(request.messages)

        # --- Step 3: Standardize Messages ---
        standardized_original_content_parts: List[str] = []
        for i, msg in enumerate(original_messages_pydantic):
            if isinstance(msg.content, list):
                text_content = []
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_content.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_content.append(part)
                    else:
                        logger.warning("Unsupported content part type in message",part_type=type(part))
                msg.content = "\n".join(text_content)
                standardized_original_content_parts.append(msg.content)
            elif isinstance(msg.content, str):
                standardized_original_content_parts.append(msg.content)
            else:
                msg.content = ""
                standardized_original_content_parts.append("")
        
        # --- Step 3b: Format original messages as a single string ---
        # Format: "role:\ncontent\n\nrole:\ncontent..."
        origin_messages_parts = []
        for msg in original_messages_pydantic:
            role = msg.role if msg.role else "unknown" # Ensure role is a string
            content = msg.content if isinstance(msg.content, str) else str(msg.content) # Ensure content is a string
            origin_messages_parts.append(f"{role}:\n{content}")
        origin_messages_str = "\n\n".join(origin_messages_parts)

        # --- Step 4: Reconstruct Prompt for Primary LLM ---
        # Pass the formatted origin_messages_str instead of full_original_user_content
        primary_llm_messages_dict: List[Dict[str, Any]] = \
            prompt_manager.construct_primary_llm_messages(origin_messages_str)
        
        logger.debug("Messages for Primary LLM", request_id=request_id, messages=primary_llm_messages_dict, origin_content_used=origin_messages_str)

        # --- Step 5 & 6: Call Primary LLM and Get Thoughts + Answer ---
        primary_llm_full_response_content = ""
        primary_llm_full_reasoning_content = "" # Added for reasoning_content
        try:
            primary_llm = await get_llm_client("primary")
            async for chunk_dict in primary_llm.agenerate_stream(primary_llm_messages_dict, stream_config=request.model_dump(exclude={'messages', 'model', 'stream'})):
                if chunk_dict.get("error"):
                    logger.error("Primary LLM client yielded an error during its stream", request_id=request_id, error_details=chunk_dict)
                    # This error dict is from create_openai_error_response via llm_client
                    raise _PrimaryLLMInternalError(chunk_dict)

                if chunk_dict.get("choices"):
                    delta = chunk_dict["choices"][0].get("delta", {})
                    content_part = delta.get("content")
                    reasoning_part = delta.get("reasoning_content") # Added for reasoning_content
                    if content_part:
                        primary_llm_full_response_content += content_part
                        # 立刻把 r1 的这块内容推给前端
                        # 立刻把 r1 的这块内容推给前端
                        # (确保 generate_request_id, get_current_timestamp_ms, PRIMARY_LLM_CONFIG 已在作用域内或正确导入)
                        r1_stream_id = f"chatcmpl-r1-{request_id[:12]}" # 创建一个唯一的 r1 块 ID
                        r1_model_name = PRIMARY_LLM_CONFIG.get("model", "primary_llm_phase")
                        
                        r1_chunk_for_client = {
                            "id": r1_stream_id,
                            "object": "chat.completion.chunk",
                            "created": get_current_timestamp_ms() // 1000,
                            "model": r1_model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": content_part}, # 这是关键，将 r1 内容放入 delta
                                "finish_reason": None # r1 阶段不是最终结束
                            }]
                            # 如果需要，可以添加一个自定义字段来区分，但SillyTavern可能不会直接使用它
                            # "custom_metadata": {"phase": "primary_llm_interaction"}
                        }
                        yield f"data: {json.dumps(r1_chunk_for_client)}\n\n"
                    if reasoning_part: # Added for reasoning_content
                        primary_llm_full_reasoning_content += reasoning_part # Added for reasoning_content
                    if chunk_dict["choices"][0].get("finish_reason"):
                        break
            
            logger.info("Primary LLM call completed successfully.", request_id=request_id, response_length=len(primary_llm_full_response_content), reasoning_length=len(primary_llm_full_reasoning_content))
            final_primary_llm_answer = primary_llm_full_response_content.strip()

            # Log Primary LLM interaction
            try:
                # Reconstruct effective parameters for logging, mimicking llm_clients.py logic
                # Parameters from original request that are passed to LLM client's stream_config/request_config
                request_params_for_llm = request.model_dump(exclude={'messages', 'model', 'stream'}, exclude_none=True)
                
                effective_primary_params = {
                    "temperature": request_params_for_llm.get("temperature", PRIMARY_LLM_CONFIG.get("temperature", 0.7)), # Default from OpenAI client if not in config
                    "top_p": request_params_for_llm.get("top_p", PRIMARY_LLM_CONFIG.get("top_p", 1.0)),
                    "max_tokens": request_params_for_llm.get("max_tokens", PRIMARY_LLM_CONFIG.get("max_tokens")),
                    # Add other relevant parameters if they are configurable and passed through
                }
                # Filter out None values if max_tokens wasn't set anywhere
                effective_primary_params = {k: v for k, v in effective_primary_params.items() if v is not None}

                primary_interaction_log = {
                    "primary_llm_model_configured": PRIMARY_LLM_CONFIG.get("model", "unknown_primary_model"),
                    "primary_llm_effective_params": effective_primary_params,
                    "request_to_primary_llm": primary_llm_messages_dict,
                    "response_from_primary_llm": final_primary_llm_answer,
                    "reasoning_from_primary_llm": primary_llm_full_reasoning_content.strip() # Added for reasoning_content
                }
                with open(os.path.join(request_log_dir, "primary_llm_interaction.json"), "w", encoding="utf-8") as f:
                    json.dump(primary_interaction_log, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error("Failed to log Primary LLM interaction", request_id=request_id, error=str(e))

        except _PrimaryLLMInternalError: # Re-raise to be caught by the main try-except
            raise
        except Exception as e_primary: # Catch other errors from primary LLM call
            logger.error("Unhandled error during Primary LLM call", request_id=request_id, error=str(e_primary), exc_info=True)
            err_payload = create_openai_error_response(
                message=f"Error interacting with primary LLM: {str(e_primary)}",
                err_type="internal_server_error", code="primary_llm_failure"
            )
            raise _PrimaryLLMInternalError(err_payload) # Wrap and raise

        # --- Step 7: Update origin_messages ---
        if original_messages_pydantic:
            last_msg_content = original_messages_pydantic[-1].content
            if not isinstance(last_msg_content, str):
                original_messages_pydantic[-1].content = str(last_msg_content)
            original_messages_pydantic[-1].content += f"\n{final_primary_llm_answer}"
        else:
            logger.warning("Original messages list empty, cannot append primary answer.", request_id=request_id)

        target_llm_messages_dict: List[Dict[str, Any]] = pydantic_messages_to_dict(original_messages_pydantic)

        # --- Step 8 & 9: Call Target LLM and Stream/Return Response ---
        target_llm = await get_llm_client("target")
        target_llm_params = request.model_dump(exclude={'messages', 'model', 'stream'}, exclude_none=True)

        if request.stream:
            stream_id = f"chatcmpl-{generate_request_id()}"
            target_model_name = TARGET_LLM_CONFIG.get("model", request.model)
            full_target_llm_response_content = ""
            target_llm_error_content = None

            async for chunk_dict in target_llm.agenerate_stream(target_llm_messages_dict, stream_config=target_llm_params):
                if chunk_dict.get("error"):
                    logger.error("Target LLM client yielded an error during its stream", request_id=request_id, error_details=chunk_dict)
                    error_message_content = chunk_dict.get("error", {}).get("message", "Error processing target LLM.")
                    target_llm_error_content = f"[ERROR_TARGET_LLM] {error_message_content}"
                    error_stream_chunk = ChatCompletionStreamResponse(
                        id=stream_id, created=get_current_timestamp_ms() // 1000, model=target_model_name,
                        choices=[ChatCompletionStreamChoice(index=0, delta=ChoiceDelta(content=target_llm_error_content), finish_reason="error")]
                    )
                    yield f"data: {error_stream_chunk.model_dump_json(exclude_none=True)}\n\n"
                    yield "data: [DONE]\n\n"
                    # Log Target LLM interaction (error case)
                    try:
                        # target_llm_params was defined before the stream/non-stream split in the original code
                        # request_params_for_llm is equivalent for this scope
                        request_params_for_llm = request.model_dump(exclude={'messages', 'model', 'stream'}, exclude_none=True)
                        effective_target_params = {
                            "temperature": request_params_for_llm.get("temperature", TARGET_LLM_CONFIG.get("temperature", 0.7)),
                            "top_p": request_params_for_llm.get("top_p", TARGET_LLM_CONFIG.get("top_p", 1.0)),
                            "max_tokens": request_params_for_llm.get("max_tokens", TARGET_LLM_CONFIG.get("max_tokens")),
                        }
                        effective_target_params = {k: v for k, v in effective_target_params.items() if v is not None}

                        target_interaction_log = {
                            "target_llm_model_configured": TARGET_LLM_CONFIG.get("model", request.model), # request.model is fallback if not in config
                            "target_llm_effective_params": effective_target_params,
                            "request_to_target_llm": target_llm_messages_dict,
                            "response_from_target_llm": target_llm_error_content,
                            "error_details": chunk_dict.get("error")
                        }
                        with open(os.path.join(request_log_dir, "target_llm_interaction.json"), "w", encoding="utf-8") as f:
                            json.dump(target_interaction_log, f, indent=4, ensure_ascii=False)
                    except Exception as e:
                        logger.error("Failed to log Target LLM error interaction", request_id=request_id, error=str(e))
                    return # Exit generator

                if chunk_dict.get("choices"):
                    choice = chunk_dict["choices"][0]
                    delta = choice.get("delta", {})
                    content_part = delta.get("content")
                    finish_reason = choice.get("finish_reason")

                    # 把 OpenAI 兼容接口给你的整块 chunk_dict 丢给前端
                    yield f"data: {json.dumps(chunk_dict)}\n\n"

                    if content_part is not None and content_part != "": # 仍然需要累积 content_part 用于日志记录
                        full_target_llm_response_content += content_part
                    
                    if finish_reason:
                        logger.info(f"Stream finished with reason: {finish_reason}", request_id=request_id)
                        break
                
                elif chunk_dict.get("error"): # Should be caught by the first if, but as a fallback
                    logger.error("Target LLM client yielded an error during its stream (re-check)", request_id=request_id, error_details=chunk_dict)
                    error_message_content = chunk_dict.get("error", {}).get("message", "Error processing target LLM.")
                    target_llm_error_content = f"[ERROR_TARGET_LLM] {error_message_content}"
                    yield f"data: {json.dumps(target_llm_error_content)}\n\n"
                    # Log Target LLM interaction (error case)
                    try:
                        request_params_for_llm = request.model_dump(exclude={'messages', 'model', 'stream'}, exclude_none=True)
                        effective_target_params = { # Duplicating logic for clarity in this block
                            "temperature": request_params_for_llm.get("temperature", TARGET_LLM_CONFIG.get("temperature", 0.7)),
                            "top_p": request_params_for_llm.get("top_p", TARGET_LLM_CONFIG.get("top_p", 1.0)),
                            "max_tokens": request_params_for_llm.get("max_tokens", TARGET_LLM_CONFIG.get("max_tokens")),
                        }
                        effective_target_params = {k: v for k, v in effective_target_params.items() if v is not None}
                        target_interaction_log = {
                            "target_llm_model_configured": TARGET_LLM_CONFIG.get("model", request.model),
                            "target_llm_effective_params": effective_target_params,
                            "request_to_target_llm": target_llm_messages_dict,
                            "response_from_target_llm": target_llm_error_content,
                            "error_details": chunk_dict.get("error")
                        }
                        with open(os.path.join(request_log_dir, "target_llm_interaction.json"), "w", encoding="utf-8") as f:
                            json.dump(target_interaction_log, f, indent=4, ensure_ascii=False)
                    except Exception as e:
                        logger.error("Failed to log Target LLM error interaction (re-check)", request_id=request_id, error=str(e))
                    break
            
            yield "data: [DONE]\n\n"
            logger.info("Streamed chat completion processed and [DONE] sent", request_id=request_id)
            # Log Target LLM interaction (success case for stream)
            try:
                request_params_for_llm = request.model_dump(exclude={'messages', 'model', 'stream'}, exclude_none=True)
                effective_target_params = { # Duplicating logic for clarity
                    "temperature": request_params_for_llm.get("temperature", TARGET_LLM_CONFIG.get("temperature", 0.7)),
                    "top_p": request_params_for_llm.get("top_p", TARGET_LLM_CONFIG.get("top_p", 1.0)),
                    "max_tokens": request_params_for_llm.get("max_tokens", TARGET_LLM_CONFIG.get("max_tokens")),
                }
                effective_target_params = {k: v for k, v in effective_target_params.items() if v is not None}
                target_interaction_log = {
                    "target_llm_model_configured": TARGET_LLM_CONFIG.get("model", request.model),
                    "target_llm_effective_params": effective_target_params,
                    "request_to_target_llm": target_llm_messages_dict,
                    "response_from_target_llm": full_target_llm_response_content
                }
                with open(os.path.join(request_log_dir, "target_llm_interaction.json"), "w", encoding="utf-8") as f:
                    json.dump(target_interaction_log, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error("Failed to log Target LLM stream interaction", request_id=request_id, error=str(e))

        else: # Non-streaming
            response_dict = await target_llm.agenerate(target_llm_messages_dict, stream_config=target_llm_params)
            
            # Log Target LLM interaction (non-stream)
            try:
                # target_llm_params (now request_params_for_llm) was defined before the stream/non-stream split
                request_params_for_llm = request.model_dump(exclude={'messages', 'model', 'stream'}, exclude_none=True)
                effective_target_params = {
                    "temperature": request_params_for_llm.get("temperature", TARGET_LLM_CONFIG.get("temperature", 0.7)),
                    "top_p": request_params_for_llm.get("top_p", TARGET_LLM_CONFIG.get("top_p", 1.0)),
                    "max_tokens": request_params_for_llm.get("max_tokens", TARGET_LLM_CONFIG.get("max_tokens")),
                }
                effective_target_params = {k: v for k, v in effective_target_params.items() if v is not None}

                target_interaction_log = {
                    "target_llm_model_configured": TARGET_LLM_CONFIG.get("model", request.model),
                    "target_llm_effective_params": effective_target_params,
                    "request_to_target_llm": target_llm_messages_dict,
                    "response_from_target_llm": response_dict # Log the entire response dict
                }
                with open(os.path.join(request_log_dir, "target_llm_interaction.json"), "w", encoding="utf-8") as f:
                    json.dump(target_interaction_log, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error("Failed to log Target LLM non-stream interaction", request_id=request_id, error=str(e))

            if response_dict.get("error"):
                logger.error("Target LLM non-stream returned an error", request_id=request_id, error_details=response_dict)
                raise _TargetLLMInternalError(response_dict)
            
            if "model" not in response_dict or not response_dict["model"]:
                response_dict["model"] = TARGET_LLM_CONFIG.get("model", request.model)
            
            if "usage" not in response_dict:
                 response_dict["usage"] = {"prompt_tokens":0, "total_tokens":0}

            final_response_obj = ChatCompletionResponse(**response_dict)
            logger.info("Non-streamed chat completion processed successfully", request_id=request_id)
            raise NonStreamFinalResponse(final_response_obj)

    except NonStreamFinalResponse as nsfR:
        # This is not an error, but a way to return a full response for non-streaming requests
        # api.py needs to catch this specifically.
        raise # Re-raise for api.py to handle
    except (_PrimaryLLMInternalError, _TargetLLMInternalError) as llm_err:
        logger.error(f"LLM Error caught in main handler: {type(llm_err).__name__}", request_id=request_id, details=llm_err.details)
        err_payload = llm_err.details # This is already a dict from create_openai_error_response
        if request.stream:
            error_model_name = request.model # Or determine from context
            error_message = err_payload.get("error",{}).get("message", "LLM processing error.")
            error_chunk = ChatCompletionStreamResponse(
                id=request_id, created=get_current_timestamp_ms() // 1000, model=error_model_name,
                choices=[ChatCompletionStreamChoice(index=0, delta=ChoiceDelta(content=f"[ERROR] {error_message}"), finish_reason="error")]
            )
            yield f"data: {error_chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"
        else:
            # For non-stream, wrap in NonStreamFinalResponse for api.py
            raise NonStreamFinalResponse(ErrorResponse(**err_payload))
    except Exception as e_generic:
        logger.error("Generic unhandled exception in handle_chat_completion", request_id=request_id, error=str(e_generic), exc_info=True)
        final_err_payload = create_openai_error_response(
            message=f"An unexpected error occurred in core processing: {str(e_generic)}",
            err_type="internal_server_error", code="core_unhandled_exception"
        )
        if request.stream:
            error_chunk = ChatCompletionStreamResponse(
                id=request_id, created=get_current_timestamp_ms() // 1000, model=request.model,
                choices=[ChatCompletionStreamChoice(index=0, delta=ChoiceDelta(content=f"[ERROR] {final_err_payload['error']['message']}"), finish_reason="error")]
            )
            yield f"data: {error_chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"
        else:
            raise NonStreamFinalResponse(ErrorResponse(**final_err_payload))