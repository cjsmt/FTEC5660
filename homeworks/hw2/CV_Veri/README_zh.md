# CV Verification Agent

基于 LangGraph + MCP 的简历核验系统：从本地 PDF 简历提取信息，通过 MCP 工具查询 LinkedIn / Facebook 档案，并由 LLM 对比生成结构化验证报告。

---

## 项目目录结构

```
CV_Veri/
├── main.py                    # 入口：解析命令行、运行 pipeline、保存报告并打印
├── graph.py                   # 主图：extract_resume → fetch_social_profiles → compare_and_report
├── resume_extract_subgraph.py # 子图：PDF 原文提取 → LLM 结构化为 ResumeData
├── react_agent_subgraph.py    # 子图：ReAct 智能体（LLM + 工具调用，用于 LinkedIn/Facebook）
├── mcp_client.py              # MCP 客户端：从 MCP_BASE_URL 加载 LinkedIn/Facebook 工具
├── models.py                  # 数据模型：CVState, ResumeData, LinkedInProfile, FacebookProfile 等
├── utils.py                   # 工具函数：extract_json_from_text 等
├── requirements.txt           # Python 依赖
├── .env                       # 环境变量（API Key、MCP 地址等，需自行配置）
├── prompts/                   # 各节点使用的系统提示
│   ├── resume_extract_system.txt   # 简历结构化输出格式
│   ├── linkedin_agent_system.txt   # LinkedIn 智能体行为与输出格式
│   ├── facebook_agent_system.txt   # Facebook 智能体行为与输出格式
│   └── compare_and_report_system.txt # 对比报告生成与打分说明
├── output/                    # 运行后自动创建，存放按时间戳命名的报告 JSON
│   └── 20250211_143022.json   # 示例
└── datasets/                  # 可选：用于存放待核验的简历 PDF（路径可自定）
    └── CV_1.pdf
```

---

## 大图与子图架构（Workflow）

整体由 **主图** 串联三个节点，其中两个节点内部使用 **子图** 完成复杂逻辑。

### 主图（graph.py）

```
START → extract_resume → fetch_social_profiles → compare_and_report → END
```

| 节点 | 说明 |
|------|------|
| **extract_resume** | 调用「简历提取子图」，输入 `resume_path`，输出写入主图 state 的 `resume_data`。 |
| **fetch_social_profiles** | 并行运行两个 ReAct 子图：LinkedIn 智能体、Facebook 智能体；各自通过 MCP 工具搜索并拉取档案，结果写入 `linkedin_profile`、`facebook_profile`。 |
| **compare_and_report** | 将简历 + LinkedIn + Facebook 拼成上下文，调用 LLM 生成对比报告（技能/经历/教育打分 + 总结），写入 `report`，并计算 `average_score`。 |

### 子图 1：简历提取（resume_extract_subgraph.py）

在 **extract_resume** 节点内被调用：

```
START → extract_pdf_text → llm_to_resume → END
```

- **extract_pdf_text**：用 PyPDF2 从 PDF 抽取纯文本，写入 `ocr_text`。
- **llm_to_resume**：根据 system prompt 将 `ocr_text` 结构化为 `ResumeData`（name, city, country, skills, experience, education 等）。

### 子图 2：ReAct 智能体（react_agent_subgraph.py）

在 **fetch_social_profiles** 节点内被调用两次（LinkedIn / Facebook 各一次）：

- **llm_node**：根据当前 messages 生成 AIMessage（可带 tool_calls）。
- **tool_node**：执行 AIMessage 中的 tool_calls（MCP 工具），将 ToolMessage 追加回 messages。
- 外层循环多次调用该子图，直到得到「无 tool_calls 的 AIMessage」或达到轮数上限，再从最终 messages 中解析出 profile JSON。

### 数据流概览

```
resume_path (PDF)
    → [简历提取子图] → resume_data
    → [LinkedIn ReAct] + [Facebook ReAct]（并行）→ linkedin_profile, facebook_profile
    → [compare_and_report] resume_data + linkedin_profile + facebook_profile → report
```

---

## 从零跑通：安装、配置、放置简历、运行

### 1. 安装依赖

在项目根目录下执行：

```bash
pip install -r requirements.txt
```

主要依赖：`langgraph`、`langchain-core`、`langchain-openai`、`PyPDF2`、`python-dotenv`、`langchain_mcp_adapters` 等。

### 2. 配置环境变量

在项目根目录创建或编辑 `.env`，至少配置：

- **OPENAI_API_KEY**：调用 LLM / 兼容 API 的密钥。
- **OPENAI_BASE_URL**：API 地址（若使用第三方转发，如 `https://aihubmix.com/v1`）。
- **LLM_MODEL_NAME**：模型名（如 `deepseek-v3.2`）。
- **LLM_PROVIDER**：如 `openai`。
- **MCP_BASE_URL**：MCP 服务地址（提供 LinkedIn/Facebook 工具的 HTTP 入口，如 `https://xxx.ngrok.app/mcp`）。

其他可选项见 `.env` 内注释。

### 3. 放置简历 PDF

- 将待核验的简历 PDF 放在任意可访问路径，例如项目下的 `datasets/`：
  - `datasets/CV_1.pdf`、`datasets/CV_2.pdf` 等。
- 或使用绝对路径，如 `/path/to/resume.pdf`。

### 4. 运行项目

单份简历（必选参数为 PDF 路径）：

```bash
python main.py datasets/CV_1.pdf
```

或使用绝对路径：

```bash
python main.py /path/to/your/resume.pdf
```

可选参数：

- `--json`：在终端以 JSON 形式打印完整报告（便于对接其他系统）。

```bash
python main.py datasets/CV_1.pdf --json
```

运行结束后会：

1. 在终端打印一份可读的验证报告（姓名、城市、国家、总结、综合匹配分数、技能/工作/教育对比）。
2. 自动在 **output/** 下生成以时间戳命名的 JSON 文件（如 `output/20250211_143022.json`），内容为当次运行的完整报告，便于留存或二次分析。

---

## 生成的结果报告长什么样

### 保存的 JSON 文件（output/时间戳.json）

每次运行都会在 `output/` 下写入一个 JSON 文件，结构大致如下（字段可能为空或省略）：

```json
{
  "resume": {
    "name": "张三",
    "city": "Beijing",
    "country": "China",
    "headline": "...",
    "skills": [...],
    "experience": [...],
    "education": [...]
  },
  "linkedin_profile": { ... },
  "facebook_profile": { ... },
  "skills_comparison": {
    "score": 0.85,
    "common_skills": [...],
    "only_in_resume": [...],
    "only_in_social": [...],
    "summary": "..."
  },
  "experience_comparison": {
    "score": 0.9,
    "summary": "...",
    "details": [...]
  },
  "education_comparison": {
    "score": 0.8,
    "summary": "...",
    "details": [...]
  },
  "summary": "综合对比后的文字总结与可信度说明。",
  "average_score": 0.85
}
```

- **resume**：从 PDF 抽取并结构化后的简历信息。
- **linkedin_profile / facebook_profile**：MCP 工具查到的档案（未找到则为 `null` 或缺失）。
- **skills_comparison / experience_comparison / education_comparison**：三项对比及各自 **score**（0～1）。
- **average_score**：综合匹配分数，为上述三项 score 的算术平均值（保留 4 位小数）。
- **summary**：LLM 生成的总体结论与可信度分析。

### 终端可读输出

在不加 `--json` 时，终端会打印：

- 标题「简历验证报告」、姓名 / 城市 / 国家。
- 总结（summary）。
- 综合匹配分数（average_score）。
- 技能对比：匹配分数、共同/仅简历/仅社媒技能、技能对比总结。
- 工作经历对比：匹配分数与说明。
- 教育经历对比：匹配分数与说明。

整体流程可以概括为：**安装依赖 → 配置 .env → 放好 PDF → 执行 `python main.py <简历路径>` → 查看终端输出并到 `output/` 下查看对应时间戳的 JSON 报告。**

