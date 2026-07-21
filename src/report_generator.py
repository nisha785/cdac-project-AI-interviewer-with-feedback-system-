import json
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, needed for script/server use
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListFlowable, ListItem
)


def generate_radar_chart(session_summary: dict, output_path: str):
    """
    Radar chart comparing the 3 core dimensions: accuracy, speech confidence,
    and facial/body language (eye contact), using each dimension's mean score.
    """
    dimensions = ["Answer Accuracy", "Speech Confidence", "Eye Contact"]
    values = [
        session_summary.get("accuracy_weighted_accuracy_score", {}).get("mean", 0),
        session_summary.get("speech_confidence_score", {}).get("mean", 0),
        session_summary.get("facial_eye_contact_ratio", {}).get("mean", 0),
    ]
    values += values[:1]  # close the loop

    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#2563eb", linewidth=2)
    ax.fill(angles, values, color="#2563eb", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Session Overview", fontsize=11, pad=20)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_accuracy_bar_chart(per_question: list[dict], output_path: str):
    """
    Bar chart of accuracy score per question, colored by whether it was
    flagged as below threshold.
    """
    ids = [q["id"] for q in per_question]
    scores = [q.get("accuracy_weighted_accuracy_score", 0) for q in per_question]
    flagged = [q.get("accuracy_weighted_accuracy_score_flag", False) for q in per_question]
    colors_list = ["#dc2626" if f else "#16a34a" for f in flagged]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar([f"Q{i}" for i in ids], scores, color=colors_list)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy Score")
    ax.set_title("Per-Question Accuracy", fontsize=11)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Threshold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_pdf_report(
    session_report_path: str,
    feedback_report_path: str,
    output_pdf_path: str,
    charts_dir: str = "data/sample_resumes/charts"
):
    with open(session_report_path, "r") as f:
        session_report = json.load(f)
    with open(feedback_report_path, "r") as f:
        feedback = json.load(f)

    os.makedirs(charts_dir, exist_ok=True)
    radar_path = os.path.join(charts_dir, "radar_chart.png")
    bar_path = os.path.join(charts_dir, "accuracy_bar_chart.png")

    generate_radar_chart(session_report["session_summary"], radar_path)
    generate_accuracy_bar_chart(session_report["per_question"], bar_path)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=20)
    section_style = ParagraphStyle(
        "SectionStyle", parent=styles["Heading2"], fontSize=13,
        spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1a1a2e")
    )
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=10.5, leading=15)

    doc = SimpleDocTemplate(
        output_pdf_path, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch
    )
    story = []

    story.append(Paragraph("Mock Interview Performance Report", title_style))
    story.append(Spacer(1, 12))

    # Session overview table
    story.append(Paragraph("Session Overview", section_style))
    summary_rows = [["Dimension", "Mean", "Min", "Max", "Trend"]]
    for dim, stats in session_report["session_summary"].items():
        summary_rows.append([
            dim.replace("_", " ").title(),
            stats["mean"], stats["min"], stats["max"], stats["trend"]
        ])
    table = Table(summary_rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    # Radar chart
    story.append(Image(radar_path, width=3.2 * inch, height=3.2 * inch))
    story.append(Spacer(1, 10))

    # Bar chart
    story.append(Image(bar_path, width=5.5 * inch, height=2.75 * inch))
    story.append(Spacer(1, 14))

    # Narrative feedback sections
    story.append(Paragraph("Communication", section_style))
    story.append(Paragraph(
        f"{feedback['communication']['summary']} "
        f"<b>Score: {feedback['communication']['score_out_of_10']}/10</b>", body_style))

    story.append(Paragraph("Answer Quality", section_style))
    story.append(Paragraph(
        f"{feedback['answer_quality']['summary']} "
        f"<b>Score: {feedback['answer_quality']['score_out_of_10']}/10</b>", body_style))

    story.append(Paragraph("Body Language", section_style))
    story.append(Paragraph(
        f"{feedback['body_language']['summary']} "
        f"<b>Score: {feedback['body_language']['score_out_of_10']}/10</b>", body_style))

    story.append(Paragraph("Overall Trend", section_style))
    story.append(Paragraph(feedback["overall_trend"], body_style))

    story.append(Paragraph("Action Plan", section_style))
    action_items = [ListItem(Paragraph(item, body_style)) for item in feedback["action_plan"]]
    story.append(ListFlowable(action_items, bulletType="bullet", start="circle", leftIndent=18))

    doc.build(story)
    print(f"PDF report saved to {output_pdf_path}")


if __name__ == "__main__":
    build_pdf_report(
        session_report_path="data/sample_resumes/session_report.json",
        feedback_report_path="data/sample_resumes/feedback_report.json",
        output_pdf_path="data/sample_resumes/final_interview_report.pdf"
    )