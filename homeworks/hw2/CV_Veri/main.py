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

# 当前 workflow 仅支持单份简历：CVState.resume_path 为单个路径，report 为单份验证报告。
# 不支持批量（如传入 [CV_1.pdf, CV_2.pdf, CV_3.pdf] 返回多份报告）。若需批量校验需另行实现。


def run_pipeline(resume_path: str) -> CVState:
    """
    运行整套 LangGraph 工作流，返回最终状态（包含单份 VerificationReport）。
    """
    app = build_graph()
    initial_state: CVState = {"resume_path": str(resume_path)}
    final_state = app.invoke(initial_state)
    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="CV Verification Agent (LangGraph + MCP)")
    parser.add_argument(
        "resume_path",
        type=str,
        help="本地简历 PDF 文件路径，例如：/path/to/resume.pdf",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出完整验证报告（适合与其他系统对接）",
    )

    args = parser.parse_args()
    resume_path = Path(args.resume_path)
    if not resume_path.exists():
        raise SystemExit(f"简历文件不存在：{resume_path}")

    state = run_pipeline(str(resume_path))
    report = state.get("report")
    if report is None:
        raise SystemExit("未能生成验证报告，请检查日志或输入文件。")

    # 保存报告到本地 JSON，默认路径 output/ 下以时间戳命名的文件
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已保存至: {out_path}")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        resume = report.get("resume", {})
        print("====== 简历验证报告 ======")
        print(f"姓名: {resume.get('name', '-')}")
        print(f"城市: {resume.get('city', '-')}")
        print(f"国家: {resume.get('country', '-')}")
        print("\n--- 总结 ---")
        print(report.get("summary", "-"))
        avg = report.get("average_score")
        if avg is not None:
            print(f"\n综合匹配分数 (average_score): {avg:.4f}")

        skills_comparison = report.get("skills_comparison")
        if skills_comparison:
            print("\n--- 技能对比 ---")
            print(f"匹配分数: {skills_comparison.get('score', 0.0):.2f}")
            common_skills = skills_comparison.get("common_skills", [])
            only_in_resume = skills_comparison.get("only_in_resume", [])
            only_in_social = skills_comparison.get("only_in_social", [])
            print(f"共同技能: {', '.join(s.get('name', '') or '' for s in common_skills) or '-'}")
            print(f"仅在简历中的技能: {', '.join(s.get('name', '') or '' for s in only_in_resume) or '-'}")
            print(f"仅在社媒中的技能: {', '.join(s.get('name', '') or '' for s in only_in_social) or '-'}")
            if skills_comparison.get("summary"):
                print(f"技能对比总结: {skills_comparison['summary']}")

        experience_comparison = report.get("experience_comparison")
        if experience_comparison:
            print("\n--- 工作经历对比 ---")
            print(f"匹配分数: {experience_comparison.get('score', 0.0):.2f}")
            print(experience_comparison.get("summary", "-"))
            for d in experience_comparison.get("details", []):
                print(f"  - {d}")

        education_comparison = report.get("education_comparison")
        if education_comparison:
            print("\n--- 教育经历对比 ---")
            print(f"匹配分数: {education_comparison.get('score', 0.0):.2f}")
            print(education_comparison.get("summary", "-"))
            for d in education_comparison.get("details", []):
                print(f"  - {d}")


if __name__ == "__main__":
    main()
