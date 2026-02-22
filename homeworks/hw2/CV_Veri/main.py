from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

from graph import build_graph
from models import CVState

from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# The current workflow only supports a single resume: CVState.resume_path is a single path
# and the report represents a single verification result. Batch processing (e.g. a list
# like [CV_1.pdf, CV_2.pdf, CV_3.pdf]) is not yet implemented.

# Cached compiled graph so multiple run_pipeline calls (e.g. test.py) reuse it
# and avoid reconnecting to MCP on every resume.
_cached_app = None


def run_pipeline(resume_path: str) -> CVState:
    """
    Run the end-to-end LangGraph workflow and return the final state (containing a single VerificationReport).
    The compiled graph (including MCP tools) is cached; subsequent calls reuse it to avoid reconnecting.
    """
    global _cached_app
    if _cached_app is None:
        _cached_app = build_graph()
    initial_state: CVState = {"resume_path": str(resume_path)}
    final_state = _cached_app.invoke(initial_state)
    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="CV Verification Agent (LangGraph + MCP)")
    parser.add_argument(
        "resume_path",
        type=str,
        help="Local resume PDF file path, e.g. /path/to/resume.pdf",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full verification report as JSON (for integration use)",
    )

    args = parser.parse_args()
    resume_path = Path(args.resume_path)
    if not resume_path.exists():
        raise SystemExit(f"Resume file does not exist: {resume_path}")

    state = run_pipeline(str(resume_path))
    report = state.get("report")
    if report is None:
        raise SystemExit("Failed to generate verification report. Please check logs and input file.")

    # Persist report to local JSON under output/ with a timestamped filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {out_path}")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        resume = report.get("resume", {})
        print("====== CV Verification Report ======")
        print(f"Name: {resume.get('name', '-')}")
        print(f"City: {resume.get('city', '-')}")
        print(f"Country: {resume.get('country', '-')}")
        print("\n--- Summary ---")
        print(report.get("summary", "-"))
        avg = report.get("average_score")
        if avg is not None:
            print(f"\nOverall matching score (average_score): {avg:.4f}")

        skills_comparison = report.get("skills_comparison")
        if skills_comparison:
            print("\n--- Skills Comparison ---")
            print(f"Score: {skills_comparison.get('score', 0.0):.2f}")
            common_skills = skills_comparison.get("common_skills", [])
            only_in_resume = skills_comparison.get("only_in_resume", [])
            only_in_social = skills_comparison.get("only_in_social", [])
            print(f"Common skills: {', '.join(s.get('name', '') or '' for s in common_skills) or '-'}")
            print(f"Only in resume: {', '.join(s.get('name', '') or '' for s in only_in_resume) or '-'}")
            print(f"Only in social profiles: {', '.join(s.get('name', '') or '' for s in only_in_social) or '-'}")
            if skills_comparison.get("summary"):
                print(f"Skills comparison summary: {skills_comparison['summary']}")

        experience_comparison = report.get("experience_comparison")
        if experience_comparison:
            print("\n--- Work Experience Comparison ---")
            print(f"Score: {experience_comparison.get('score', 0.0):.2f}")
            print(experience_comparison.get("summary", "-"))
            for d in experience_comparison.get("details", []):
                print(f"  - {d}")

        education_comparison = report.get("education_comparison")
        if education_comparison:
            print("\n--- Education Comparison ---")
            print(f"Score: {education_comparison.get('score', 0.0):.2f}")
            print(education_comparison.get("summary", "-"))
            for d in education_comparison.get("details", []):
                print(f"  - {d}")


if __name__ == "__main__":
    main()
