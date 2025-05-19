import yaml
import os
import logging
import structlog
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union
import time
import uuid

# Load environment variables from .env file
load_dotenv()

CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "config.yml") # Default to config.yml in the project root

def load_config() -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    # Adjust path to be relative to this utils.py file if needed,
    # or ensure CONFIG_FILE_PATH is absolute or relative to CWD.
    # For simplicity, assuming config.yml is in the project root where the app is run.
    
    # Construct the path relative to the project root (langchain-openai-proxy)
    # Assuming utils.py is in langchain-openai-proxy/app/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, CONFIG_FILE_PATH)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    def _substitute_env_vars(data: Any) -> Any:
        if isinstance(data, dict):
            return {k: _substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [_substitute_env_vars(i) for i in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var_name = data[2:-1]
            env_var_value = os.getenv(env_var_name)
            if env_var_value is None:
                raise ValueError(f"Environment variable {env_var_name} not set, but required by config.")
            return env_var_value
        return data

    return _substitute_env_vars(config)

def setup_logging():
    """Configures structlog for structured logging."""
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.dev.ConsoleRenderer(), # Or structlog.processors.JSONRenderer() for JSON logs
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Initialize logger for use in other modules
setup_logging()
logger = structlog.get_logger()

def generate_request_id() -> str:
    """Generates a unique request ID."""
    return str(uuid.uuid4())

def get_current_timestamp_ms() -> int:
    """Returns the current timestamp in milliseconds."""
    return int(time.time() * 1000)

# Example of how to use the logger:
# from .utils import logger
# logger.info("event_name", key1="value1", key2="value2")

# Placeholder for OpenAI error mapping
# This should be expanded based on specific LLM client errors
def create_openai_error_response(
    message: str,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[Union[str, int]] = None, # Can be a string like "model_not_found" or an HTTP status
    status_code: int = 400 # Default HTTP status code for client errors
) -> Dict[str, Any]:
    """Creates an OpenAI-compatible error response dictionary."""
    from .models import ErrorDetail, ErrorResponse # Local import to avoid circular dependency
    
    error_detail = ErrorDetail(message=message, type=err_type, param=param, code=code)
    return ErrorResponse(error=error_detail).model_dump(exclude_none=True)

if __name__ == "__main__":
    # Test loading config
    try:
        cfg = load_config()
        logger.info("Config loaded successfully", config=cfg)
        
        # Example: Accessing a nested value that might use env var
        if cfg.get("llms", {}).get("primary", {}).get("api_key"):
            logger.info("Primary LLM API Key is set (substituted from env var).")
        else:
            logger.warning("Primary LLM API Key not found or not substituted.")

    except Exception as e:
        logger.error("Failed to load or parse config", error=str(e))