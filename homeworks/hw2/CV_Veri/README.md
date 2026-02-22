# CV Verification Agent

LangGraph + MCP based CV verification system: extract information from a local PDF resume, query LinkedIn / Facebook profiles via MCP tools, and let an LLM compare them to produce a structured verification report.

---

## Project Structure

```
CV_Veri/
├── main.py                    # CLI entrypoint: parse args, run pipeline, save & print report
├── graph.py                   # Main graph: extract_resume → fetch_social_profiles → compare_and_report
├── resume_extract_subgraph.py # Subgraph: PDF text extraction → LLM → structured ResumeData
├── react_agent_subgraph.py    # Subgraph: ReAct-style agent (LLM + tools) for LinkedIn/Facebook
├── mcp_client.py              # MCP client: load LinkedIn/Facebook tools from MCP_BASE_URL
├── models.py                  # Data models: CVState, ResumeData, LinkedInProfile, FacebookProfile, etc.
├── utils.py                   # Utilities: extract_json_from_text, etc.
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API keys, MCP URL, etc. – must be configured)
├── prompts/                   # System prompts used by different nodes
│   ├── resume_extract_system.txt       # ResumeData output format & instructions
│   ├── linkedin_agent_system.txt       # LinkedIn agent behavior & output format
│   ├── facebook_agent_system.txt       # Facebook agent behavior & output format
│   └── compare_and_report_system.txt   # Final comparison report schema & scoring rules
├── output/                    # Auto-created; stores timestamped JSON reports
│   └── 20250211_143022.json   # Example
└── datasets/                  # Optional: place sample resumes here (paths are configurable)
    └── CV_1.pdf
```

---

## Evaluation Script (`test.py`)

This project ships with a small evaluation helper in `test.py`:

- It runs the full CV verification pipeline on **5 fixed test resumes**:
  - `datasets/CV_1.pdf`, `datasets/CV_2.pdf`, `datasets/CV_3.pdf`, `datasets/CV_4.pdf`, `datasets/CV_5.pdf`.
- For each resume it:
  - calls `main.run_pipeline(...)`  
  - collects the `average_score` from `state["report"]["average_score"]`.
- It then compares these scores against a **binary groundtruth label** list:
  - `groundtruth = [1, 1, 1, 0, 0]` (do not modify).
- Using a fixed threshold (default 0.5) in `evaluate(...)`:
  - `score > threshold` → predict `1` (\"match\" / \"credible\")  
  - `score <= threshold` → predict `0` (\"not match\" / \"not credible\")  
  - It computes accuracy (`final_score`) as correct / total.

You can run the evaluation with:

```bash
python test.py
```

You will see progress logs such as which CV is being processed and the `average_score` for each, followed by the final decisions, groundtruth, and overall accuracy.

---

## High-Level Workflow (Main Graph + Subgraphs)

The system uses a **main graph** composed of three nodes, where two nodes internally invoke **subgraphs** for more complex logic.

### Main Graph (`graph.py`)

```text
START → extract_resume → fetch_social_profiles → compare_and_report → END
```

| Node | Description |
|------|-------------|
| **extract_resume** | Calls the resume extraction subgraph. Input: `resume_path`; Output written into main-state `resume_data`. |
| **fetch_social_profiles** | Runs two ReAct subgraphs in parallel: a LinkedIn agent and a Facebook agent. Each uses MCP tools to search and fetch a candidate profile, stored as `linkedin_profile` / `facebook_profile`. |
| **compare_and_report** | Combines resume + LinkedIn + Facebook into a comparison context, calls the LLM to generate a structured report (skills/experience/education comparisons + narrative summary), and computes an overall `average_score`. |

### Subgraph 1: Resume Extraction (`resume_extract_subgraph.py`)

Invoked inside **extract_resume**:

```text
START → extract_pdf_text → llm_to_resume → END
```

- **extract_pdf_text**: Use PyPDF2 to extract plain text from the PDF and store it in `ocr_text` (\"raw resume text\").
- **llm_to_resume**: Use a system prompt to turn `ocr_text` into structured `ResumeData` (name, city, country, skills, experience, education, etc.).

### Subgraph 2: ReAct Agent (`react_agent_subgraph.py`)

Invoked twice inside **fetch_social_profiles** (once for LinkedIn, once for Facebook):

- **llm_node**: Given the current `messages`, generate the next `AIMessage` (optionally with `tool_calls`).  
- **tool_node**: Execute the tools requested in the last `AIMessage` (MCP tools) and append corresponding `ToolMessage`s back into `messages`.  
- The outer controller in `graph.py` repeatedly calls this ReAct subgraph until it gets an `AIMessage` **without** `tool_calls` or hits an iteration limit, then parses the final `messages` into a profile JSON.

### Overall Data Flow

```text
resume_path (PDF)
    → [Resume-extraction subgraph] → resume_data
    → [LinkedIn ReAct] + [Facebook ReAct] (in parallel) → linkedin_profile, facebook_profile
    → [compare_and_report] resume_data + linkedin_profile + facebook_profile → report
```

---

## Getting Started: Install, Configure, Add Resumes, Run

### 1. Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

Key libraries: `langgraph`, `langchain-core`, `langchain-openai`, `PyPDF2`, `python-dotenv`, `langchain_mcp_adapters`, etc.

### 2. Configure Environment Variables

Create or edit `.env` in the project root. At minimum, configure:

- **OPENAI_API_KEY** – API key for your LLM / OpenAI-compatible endpoint.  
- **OPENAI_BASE_URL** – Base URL for the API (e.g. `https://aihubmix.com/v1` for a proxy).  
- **LLM_MODEL_NAME** – Model name, e.g. `deepseek-v3.2`.  
- **LLM_PROVIDER** – Provider identifier, e.g. `openai`.  
- **MCP_BASE_URL** – MCP HTTP endpoint providing LinkedIn / Facebook tools, e.g. `https://xxx.ngrok.app/mcp`.  

See `.env` for more optional settings and comments.

### 3. Place Resume PDFs

- Put your resumes anywhere accessible; a common pattern is a `datasets/` folder in the project root:
  - `datasets/CV_1.pdf`, `datasets/CV_2.pdf`, etc.  
- Or use any absolute path, e.g. `/path/to/resume.pdf`.

### 4. Run the Pipeline

Single resume (required argument is the PDF path):

```bash
python main.py datasets/CV_1.pdf
```

Or with an absolute path:

```bash
python main.py /path/to/your/resume.pdf
```

Optional flag:

- `--json` – also print the full report as JSON in the terminal (for integration with other systems).

```bash
python main.py datasets/CV_1.pdf --json
```

After each run:

1. A human-readable verification report is printed to the terminal (name, city, country, summary, overall score, and detailed skills/experience/education comparisons).  
2. A full JSON report is written under **`output/`** with a timestamped filename (e.g. `output/20250211_143022.json`), which you can archive or post-process.

---

## What the Report Looks Like

### Saved JSON File (`output/<timestamp>.json`)

Each run writes a JSON file under `output/` with a structure similar to:

```json
{
  "resume": {
    "name": "John Doe",
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
    "score": 0.90,
    "summary": "...",
    "details": [...]
  },
  "education_comparison": {
    "score": 0.80,
    "summary": "...",
    "details": [...]
  },
  "summary": "Overall narrative summary and credibility assessment.",
  "average_score": 0.85
}
```

- **resume** – structured information extracted from the PDF.  
- **linkedin_profile / facebook_profile** – profiles retrieved via MCP tools (may be `null` or missing if not found).  
- **skills_comparison / experience_comparison / education_comparison** – three comparison sections, each with a **`score`** in \[0, 1\].  
- **average_score** – overall matching score, the arithmetic mean of the three section scores (rounded to 4 decimal places).  
- **summary** – natural-language conclusion about how well the resume matches the social profiles and how credible it appears.

### Terminal Output (Human-Readable)

When you do **not** pass `--json`, the CLI prints:

- Title `====== CV Verification Report ======` plus name / city / country.  
- The summary paragraph.  
- Overall matching score (`average_score`).  
- Skills comparison: score, common / only-in-resume / only-in-social skills, and a short summary if available.  
- Work experience comparison: score and narrative summary (plus per-item details if provided).  
- Education comparison: score and narrative summary (plus per-item details if provided).

In short: **install dependencies → configure `.env` → place your PDF → run `python main.py <resume_path>` → read the terminal output and inspect the timestamped JSON under `output/`.**

