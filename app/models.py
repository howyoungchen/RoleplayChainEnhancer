from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]] # 支持 content 为字符串或 OpenAI v1.ChatCompletionContentPartType 数组
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None # For assistant messages
    tool_call_id: Optional[str] = None # For tool messages

class ChatCompletionRequest(BaseModel):
    model: str # 虽然我们的代理会处理模型路由，但请求中可能仍会带上
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1) # 我们可能只支持 n=1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Langchain/OpenAI specific fields that might be passed
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, str]] = None # e.g. {"type": "json_object"}

class ChoiceDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None # logprobs for stream not typically provided by OpenAI for delta

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
    system_fingerprint: Optional[str] = None # Added in newer OpenAI versions
    # usage is not part of stream response per chunk, but might be sent at the end by some implementations
    # usage: Optional[UsageInfo] = None 

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    logprobs: Optional[Dict[str, Any]] = None

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: Optional[int] = None # Optional because it might not be available (e.g. stream end)
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str # The model that generated the response
    choices: List[Choice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = None

# Error response models
class ErrorDetail(BaseModel):
    code: Optional[Union[str, int]] = None
    message: str
    param: Optional[str] = None
    type: str

class ErrorResponse(BaseModel):
    error: ErrorDetail

# Custom thought model for primary LLM (Step 6)
class Thought(BaseModel):
    step_name: str
    details: Dict[str, Any]

class PrimaryLLMStreamResponse(BaseModel):
    thoughts: Optional[List[Thought]] = None
    answer_delta: Optional[str] = None
    is_final_answer: bool = False # Indicates if this chunk contains the last part of the answer
    finish_reason: Optional[str] = None # e.g. "stop", "length" for the answer part