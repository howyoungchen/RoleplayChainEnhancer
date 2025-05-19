# RoleplayChainEnhancer

RoleplayChainEnhancer 是一个代理服务，旨在解决大型语言模型（如 V3 模型）在角色扮演时因上下文召回能力较弱而导致体验不佳的问题。它通过利用一个辅助 LLM 生成详细的“思维链”（Chain of Thought），来引导目标 LLM (例如 V3) 更准确、更沉浸地进行角色扮演。本项目主要设计用于 SillyTavern 等平台，以提升角色扮演的连贯性和深度。

## 项目背景与目标

在复杂的角色扮演场景中，一些大型语言模型（尤其是较早版本或特定优化版本，此处代称 V3）可能难以有效回忆和利用完整的对话历史与角色设定，导致角色行为偏离、对话不连贯。RoleplayChainEnhancer 通过引入一个“导演”LLM（在配置中称为 `primary` LLM）来分析当前对话上下文和用户输入，生成一段详细的思维链。这段思维链随后会作为增强提示的一部分，输入给负责实际角色扮演的“演员”LLM（在配置中称为 `target` LLM，例如 V3），从而帮助其更好地理解场景、人物状态和预期行为。

## 主要功能

*   **增强角色扮演上下文**: 通过辅助 LLM 生成的思维链，为目标 LLM 提供更丰富的上下文信息，弥补其自身记忆和推理能力的不足。
*   **提升角色扮演一致性**: 引导目标 LLM 更贴合角色设定和故事情节进行回应。
*   **OpenAI API 兼容**: 提供与 OpenAI `/v1/chat/completions` 端点兼容的接口，方便与 SillyTavern 等现有工具集成。
*   **多 LLM 支持**: 允许灵活配置用于生成思维链的 `primary` LLM 和执行角色扮演的 `target` LLM。支持 OpenAI、DeepSeek 等多种 LLM 提供商。
*   **流式响应**: 支持流式和非流式 API 响应，以适应不同客户端的需求。
*   **LangSmith 集成**: 可选集成 LangSmith 进行请求追踪和调试，方便开发者分析和优化提示流程。
*   **配置灵活**: 通过 `config.yml` 文件进行详细配置，包括 LLM 参数、提示词模板等。

## 配置

项目的主要配置在 `config.yml` 文件中。您可以配置以下内容：

*   **服务器设置**: 主机 (`host`) 和端口 (`port`)。
*   **LLM 提供商**:
    *   `primary`: 用于生成思维链的“导演”LLM。
    *   `target`: 最终执行角色扮演的“演员”LLM (例如 V3 模型)。
    *   每个 LLM 可以配置 `provider`, `model`, `api_key`, `base_url` 以及其他可选参数（如 `temperature`, `top_p`）。
*   **提示词 (Prompts)**: 配置用于指导 `primary` LLM 生成思维链的系统提示和用户提示部分。此处我留空了，你需要填一个破限提示词。
*   **LangSmith**: 启用/禁用 LangSmith 集成，并配置相关凭据和项目信息。

请根据您的需求修改 `config.yml` 文件。特别是 `api_key` 需要您自行填写。

## 如何运行

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **配置 `config.yml`**: 确保 `config.yml` 文件中的 API 密钥、LLM 模型选择（特别是 `target` LLM 指向您希望增强的 V3 模型）和其他设置已正确配置。
3.  **启动服务**:
    ```bash
    python -m app.api
    ```
    或者，如果您在项目根目录并且 `app` 是一个可发现的模块:
    ```bash
    uvicorn app.api:app --host <your_host> --port <your_port> --reload
    ```
    默认情况下，服务会根据 `config.yml` 中的 `server.host` 和 `server.port` 启动。
4.  **在 SillyTavern 中配置**:
    *   将 SillyTavern 的 API 端点指向本服务运行的地址和端口（例如 `http://localhost:8788/v1/chat/completions`，具体取决于您的 `config.yml` 设置）。
    *   选择一个兼容 OpenAI API 的模型接口。

## API 端点

*   **POST /v1/chat/completions**:
    *   描述: 模拟 OpenAI 的聊天补全 API，内置思维链增强逻辑。
    *   请求体: 遵循 OpenAI `ChatCompletionRequest` 结构。
    *   响应: 遵循 OpenAI `ChatCompletionResponse` 结构，支持流式 (`text/event-stream`) 和 JSON 响应。
*   **GET /health**:
    *   描述: 健康检查端点。
    *   响应: `{"status": "ok", "message": "Service is healthy"}`

## 注意事项

*   本项目旨在通过外部思维链注入的方式，提升特定 LLM 在角色扮演应用中的表现。
*   请确保您的 API 密钥安全，不要直接提交到版本控制中。考虑使用环境变量或其他安全方式管理密钥。
*   思维链的质量和 `primary` LLM 的选择对最终效果有显著影响，请仔细调优相关提示词和模型配置。