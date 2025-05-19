from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Union, AsyncGenerator
from contextlib import asynccontextmanager # Added for lifespan
import os # Added for setting environment variables

from .models import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from .core import handle_chat_completion, NonStreamFinalResponse # Import the custom exception
from .utils import logger, generate_request_id, create_openai_error_response, load_config, setup_logging
from .llm_clients import close_llm_clients # For graceful shutdown

# Initialize logging
setup_logging()

# Load server configuration
try:
    config = load_config()
    server_config = config.get("server", {})
    API_HOST = server_config.get("host", "0.0.0.0")
    API_PORT = server_config.get("port", 8000)
except Exception as e:
    logger.error("Failed to load server configuration for API", error=str(e), exc_info=True)
    API_HOST = "0.0.0.0"
    API_PORT = 8000


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # Startup logic
    logger.info("Application startup: Initializing resources...")

    # Setup LangSmith if enabled in config
    try:
        app_config = load_config() # Load config again or pass it if already loaded globally
        langsmith_config = app_config.get("langsmith", {})
        if langsmith_config.get("enabled", False):
            logger.info("LangSmith is enabled in config. Setting environment variables for LangSmith.")
            
            tracing_v2 = langsmith_config.get("LANGCHAIN_TRACING_V2")
            api_key = langsmith_config.get("LANGCHAIN_API_KEY")
            project = langsmith_config.get("LANGCHAIN_PROJECT", langsmith_config.get("project_name")) # Fallback to project_name
            endpoint = langsmith_config.get("LANGCHAIN_ENDPOINT") # Optional

            if tracing_v2 is not None: # Should be string "true" or "false" or boolean
                os.environ["LANGCHAIN_TRACING_V2"] = str(tracing_v2).lower()
                logger.info(f"Set LANGCHAIN_TRACING_V2 to {os.environ['LANGCHAIN_TRACING_V2']}")
            
            if api_key:
                os.environ["LANGCHAIN_API_KEY"] = api_key
                logger.info("Set LANGCHAIN_API_KEY from config.")
            elif not os.getenv("LANGCHAIN_API_KEY"):
                logger.warning("LANGCHAIN_API_KEY not found in config or environment for LangSmith.")

            if project:
                os.environ["LANGCHAIN_PROJECT"] = project
                logger.info(f"Set LANGCHAIN_PROJECT to '{project}' from config.")
            elif not os.getenv("LANGCHAIN_PROJECT"):
                logger.warning("LANGCHAIN_PROJECT not found in config or environment for LangSmith.")
            
            if endpoint: # Optional, e.g., for self-hosted LangSmith
                os.environ["LANGCHAIN_ENDPOINT"] = endpoint
                logger.info(f"Set LANGCHAIN_ENDPOINT to '{endpoint}' from config.")

            # Verify if the LangSmith client might pick these up now
            # (Actual client initialization happens within Langchain/Langsmith libraries when they are first used)
            logger.info("LangSmith environment variables have been set based on config.")
        else:
            logger.info("LangSmith is not enabled in config or langsmith config section missing.")
    except Exception as e_ls_setup:
        logger.error("Error during LangSmith setup in lifespan", error=str(e_ls_setup), exc_info=True)

    # LLM clients are loaded on demand by get_llm_client, so no specific init here.
    # Other resources can be initialized here.
    yield
    # Shutdown logic
    logger.info("Application shutdown: Closing LLM clients...")
    await close_llm_clients()
    logger.info("LLM clients closed.")

app = FastAPI(
    title="Langchain OpenAI Proxy",
    version="0.1.0",
    description="A proxy service to emulate OpenAI's Chat API with a Langchain backend.",
    lifespan=lifespan # Use the lifespan context manager
)

@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    request_id = generate_request_id() # Generate a request_id for this middleware log
    logger.info(
        "Request received at middleware",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        headers=dict(request.headers),
        client_host=request.client.host if request.client else "Unknown"
    )
    try:
        response = await call_next(request)
        logger.info(
            "Response sent from middleware",
            request_id=request_id, # Use the same request_id
            status_code=response.status_code
        )
        return response
    except Exception as e:
        logger.error(
            "Exception in middleware",
            request_id=request_id, # Use the same request_id
            error=str(e),
            exc_info=True
        )
        # Re-raise the exception to be handled by FastAPI's default error handling
        # or other specific exception handlers.
        # You might want to return a generic error response here if preferred.
        return JSONResponse(
            status_code=500,
            content={"detail": f"Middleware error processing request {request_id}: {str(e)}"}
        )

# @app.on_event("startup") # Deprecated
# async def startup_event():
#     logger.info("Application startup: Initializing resources...")

# @app.on_event("shutdown") # Deprecated
# async def shutdown_event():
#     logger.info("Application shutdown: Closing LLM clients...")
#     await close_llm_clients()
#     logger.info("LLM clients closed.")

@app.post(
    "/v1/chat/completions",
    response_model=None, 
    responses={ # Simplified schemas, to be restored/fixed later
        200: {
            "description": "Successful response. Can be a JSON object or a stream.",
            "content": {
                "application/json": {"schema": {"type": "object", "properties": {"message": {"type": "string"}}}},
                "text/event-stream": {"schema": {"type": "string"}}
            }
        },
        400: {"description": "Bad Request", "content": {"application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}}},
        401: {"description": "Unauthorized", "content": {"application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}}},
        403: {"description": "Forbidden", "content": {"application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}}},
        404: {"description": "Not Found", "content": {"application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}}},
        429: {"description": "Too Many Requests", "content": {"application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}}},
        500: {"description": "Internal Server Error", "content": {"application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}}},
    },
    summary="OpenAI Compatible Chat Completions Endpoint",
    tags=["Chat Completions"]
)
async def chat_completions(request_data: ChatCompletionRequest, http_request: Request):
    request_id = generate_request_id()
    logger.info(
        "Received request in chat_completions",
        endpoint=http_request.url.path,
        method=http_request.method,
        client_host=http_request.client.host if http_request.client else "Unknown",
        request_id=request_id,
        model=request_data.model,
        num_messages=len(request_data.messages),
        stream_requested=request_data.stream
    )
    try:
        logger.debug("Attempting to handle chat completion", request_id=request_id, stream=request_data.stream)
        if request_data.stream:
            logger.info("Processing as stream request", request_id=request_id)
            async def stream_generator_wrapper():
                logger.debug("Calling handle_chat_completion for stream", request_id=request_id)
                async for chunk in handle_chat_completion(request_data, request_id):
                    yield chunk
                logger.info("Stream finished for client", request_id=request_id)

            return StreamingResponse(stream_generator_wrapper(), media_type="text/event-stream")
        else:
            logger.info("Processing as non-stream request", request_id=request_id)
            final_response_data = None
            try:
                logger.debug("Calling handle_chat_completion for non-stream", request_id=request_id)
                # Consume the generator; expect NonStreamFinalResponse for non-stream success/error
                async for _ in handle_chat_completion(request_data, request_id):
                    # This part should ideally not be reached for non-stream if NonStreamFinalResponse is always raised.
                    logger.error("Unexpected yield from handle_chat_completion for non-stream request", request_id=request_id)
                    # This indicates a logic flaw in core.py if it yields for non-stream without raising NonStreamFinalResponse
                    generic_error = create_openai_error_response(message="Internal logic error: core module yielded unexpectedly for non-stream.", err_type="internal_server_error")
                    return JSONResponse(content=generic_error, status_code=500)
                
                # If the loop finishes without NonStreamFinalResponse being raised (e.g., generator exhausted without raising)
                logger.error("NonStreamFinalResponse not raised by core for non-stream request and generator exhausted", request_id=request_id)
                generic_error = create_openai_error_response(message="Core logic did not produce a final response or error for non-stream request.", err_type="internal_server_error")
                return JSONResponse(content=generic_error, status_code=500)

            except NonStreamFinalResponse as nsfR:
                logger.info("NonStreamFinalResponse caught", request_id=request_id, response_type=type(nsfR.response_data).__name__)
                final_response_data = nsfR.response_data
            
            if isinstance(final_response_data, ErrorResponse):
                status_code = 400 
                if final_response_data.error and final_response_data.error.type:
                    if "authentication" in final_response_data.error.type.lower(): status_code = 401
                    elif "permission" in final_response_data.error.type.lower(): status_code = 403
                    elif "not_found" in final_response_data.error.type.lower(): status_code = 404
                    elif "quota" in final_response_data.error.type.lower(): status_code = 429
                    elif "api_error" in final_response_data.error.type.lower(): status_code = 502 
                    elif "internal_server_error" == final_response_data.error.type: status_code = 500
                
                logger.warning("Returning ErrorResponse for non-stream request", request_id=request_id, status_code=status_code, error_type=final_response_data.error.type)
                return JSONResponse(content=final_response_data.model_dump(exclude_none=True), status_code=status_code)
            
            elif isinstance(final_response_data, ChatCompletionResponse):
                logger.info("Returning ChatCompletionResponse for non-stream request", request_id=request_id, response_id=final_response_data.id)
                return final_response_data
            
            else:
                logger.error("NonStreamFinalResponse carried unexpected data type or was None", request_id=request_id, data_type=type(final_response_data).__name__)
                generic_error = create_openai_error_response(message="Internal server error: Unexpected final response data.", err_type="internal_server_error")
                return JSONResponse(content=generic_error, status_code=500)

    except HTTPException as he:
        logger.warning("HTTPException caught in API layer", request_id=request_id, status_code=he.status_code, detail=he.detail)
        raise he
    except Exception as e: 
        logger.error("Unhandled exception in chat_completions endpoint", request_id=request_id, error=str(e), exc_info=True)
        err_resp = create_openai_error_response(
            message=f"An unexpected internal server error occurred. Request ID: {request_id}",
            err_type="internal_server_error",
            code="unhandled_api_exception"
        )
        return JSONResponse(content=err_resp, status_code=500)

@app.get("/health", summary="Health Check", tags=["Management"])
async def health_check():
    return {"status": "ok", "message": "Service is healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server with Uvicorn directly from __main__ on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)