# AI Technical Analyst

AI Technical Analyst 是一个面向技术研究与项目资料分析的证据驱动 Agent。它可以根据用户问题选择本地知识库、Web Search 和 Web Reader，并在最终回答中展示来源、证据覆盖和验证 warning。

## 核心能力

- 使用 Router 将问题路由到直接回答、Local RAG 或 Web Research。
- 从本地 PDF 知识库检索项目资料。
- 通过 Web Search 发现候选资料，并用 Web Reader 获取网页正文。
- 支持本地资料、互联网资料和混合研究任务。
- 对成功 Evidence 做来源索引、URL 可追溯性和术语 Coverage 检查。
- 在工具失败、证据不足和 LLM timeout 时保留真实状态，不把失败伪装成成功。
- Evaluation Worker 实时记录 Router、Tool、Final 事件，超时也保留最后事件。

## 系统架构

```text
用户问题
   |
   v
Router (qwen2.5:3b)
   |------ DIRECT / FINAL ------> Final LLM (qwen3:8b)
   |------ RAG ------------------> ChromaDB + nomic-embed-text
   |------ SEARCH ---------------> DuckDuckGo Search
   |------ WEB_READ --------------> requests + BeautifulSoup
                                      |
                                      v
                         Observations -> Verification -> Final LLM
```

- `app/agent.py`：Agent 编排、Action 路由、迭代预算和最终回答。
- `app/rag.py`：PDF 文本提取、向量库构建和本地检索。
- `app/tools.py`：Search 与 Web Reader。
- `app/verification.py`：Evidence、Coverage、Citation Traceability 相关验证。
- `ui/app.py`：Streamlit Demo。
- `evaluation/evaluation_runner.py`：隔离 Worker、实时事件传递和结果记录。

## 三种工作模式

### 1. 直接技术分析

适合不要求外部资料的问题。Agent 可以直接进入 Final，由 `qwen3:8b` 生成技术解释。

### 2. 本地资料分析

问题包含“本地资料”“项目文档”等意图时，Agent 使用 Local RAG 从 `chroma_db/` 检索项目 PDF 内容，再基于 Evidence 回答。

### 3. Web / 混合研究

问题包含搜索、比较、论文或互联网资料时，Agent 使用 Web Search；有候选 URL 时可继续 Web Reader。若同时要求本地资料和网络研究，典型链路为：

```text
RAG -> SEARCH -> WEB_READ -> FINAL
```

## 技术栈

- Python 3.11
- Ollama
- `qwen2.5:3b`：Router
- `qwen3:8b`：Final 与 Baseline
- `nomic-embed-text:latest`：Embedding
- ChromaDB
- PyPDF2
- DuckDuckGo Search (`ddgs`)
- Requests + BeautifulSoup
- Streamlit

## 启动方式

### 1. 创建环境并安装依赖

Windows PowerShell：

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 准备 Ollama

先启动 Ollama，并准备项目使用的模型：

```powershell
ollama serve
ollama pull qwen2.5:3b
ollama pull qwen3:8b
ollama pull nomic-embed-text:latest
```

### 3. 启动命令行 Agent

请在项目根目录执行：

```powershell
venv\Scripts\python.exe app\agent.py
```

输入 `exit` 或 `quit` 退出。

## Streamlit Demo

在项目根目录执行：

```powershell
venv\Scripts\python.exe -m streamlit run ui\app.py
```

浏览器打开 Streamlit 输出的本地地址，在侧边栏选择示例问题，或输入自己的技术问题。页面会展示 Agent Workflow、最终报告、Evidence Verification 和来源内容。

## Evaluation

使用当前 6-case Evaluation：

```powershell
venv\Scripts\python.exe evaluation\run_evaluation.py
```

当前正式记录：

| 指标 | Baseline | Agent |
|---|---:|---:|
| 成功率 | 5/6 (83.3%) | 5/6 (83.3%) |
| timeout | 0 | 1 |
| exception | 1 | 0 |
| tool_failure | 0 | 0 |
| 平均延迟 | 88.567s | 121.236s |
| Evidence Coverage | N/A | 0.200 |
| Citation Traceability | N/A | 0.800 |
| Evidence Sufficiency | N/A | 0.200 |

Agent 唯一 case 级失败为 T4：Final `qwen3:8b` 达到 150 秒 timeout。Baseline 的 T1 exception 是一次 Ollama/CUDA 运行环境异常。Evaluation Worker 已修复 result queue 消费时序，并会保留实时 events 和 `agent_last_event`。

## 已知限制

- Web Search 依赖网络和第三方搜索服务，可能发生超时、空结果或连接错误。
- Web Reader 可能受到目标网站 403、TLS、robots 或反爬策略影响。
- Router 可能生成重复或近似的 Search query，当前去重仍不是完全可靠的语义去重。
- `qwen3:8b` Final 响应时间受本地 Ollama、GPU/CPU 和上下文长度影响；T4 曾发生 150 秒 Final timeout。
- Evidence Coverage 是术语覆盖率，不等价于事实正确率或来源可信度。
- `chroma_db/` 是本地持久化知识库；更换 `data/test.pdf` 后需要重新构建知识库。
- 当前项目面向本地演示和工程评估，不是生产级多用户服务。

## 项目结构

```text
app/                       Agent、LLM、RAG、工具和验证模块
ui/app.py                 Streamlit Demo
data/test.pdf             示例本地资料
chroma_db/                持久化 Chroma 知识库
evaluation/test_cases.json 6-case Evaluation 测试集
evaluation/results/       当前正式 Evaluation 结果
requirements.txt          Python 依赖
```
