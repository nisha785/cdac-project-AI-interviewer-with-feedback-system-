"""
report_generator.py

This module implements the Week 4 tasks for Member 5:
1. Loads normalized score data.
2. Filters metrics for a specific interview session.
3. Generates beautiful performance breakdown charts (radar chart and accuracy bar chart).
4. Assembles a professional PDF report containing candidate info, custom metrics, and dynamic narrative recommendations.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_radar_chart(categories, values, output_path):
    """
    Generates a polar radar chart showing performance across Accuracy, Speech, and Facial dimensions.
    """
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='navy', size=9, weight='bold')
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    val_plot = list(values)
    val_plot += val_plot[:1]
    ax.plot(angles, val_plot, color='#1A365D', linewidth=2, linestyle='solid')
    ax.fill(angles, val_plot, '#2B6CB0', alpha=0.3)
    
    plt.title("Performance Breakdown (0.0 - 1.0)", size=12, weight='bold', color='#1A365D', y=1.1)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def generate_bar_chart(question_ids, accuracy_scores, output_path):
    """
    Generates a bar chart showing score variations question-by-question.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    colors_list = ['#2B6CB0' if score >= 0.6 else '#E53E3E' for score in accuracy_scores]
    bars = ax.bar(question_ids, accuracy_scores, color=colors_list, width=0.5, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel("Question ID", fontsize=9, weight='bold', color='#1A365D')
    ax.set_ylabel("Accuracy Score", fontsize=9, weight='bold', color='#1A365D')
    ax.set_title("Question-by-Question Accuracy Score", fontsize=11, weight='bold', color='#1A365D')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(question_ids)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
        
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def generate_pdf_report(session_id, data_path, output_pdf_path, plots_dir):
    """
    Orchestrates the creation of the final ReportLab PDF.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found. Please run normalization_and_calibration.py first.")
        return False
        
    df = pd.read_csv(data_path)
    session_data = df[df["session_id"] == session_id]
    
    if session_data.empty:
        print(f"Error: No data found for Session ID {session_id}")
        return False
        
    # Calculate statistics
    avg_accuracy = session_data["accuracy_score"].mean()
    avg_speech = session_data["speech_score"].mean()
    avg_facial = session_data["facial_score"].mean()
    avg_overall = session_data["overall_score"].mean()
    
    # Define paths for temporary plots
    os.makedirs(plots_dir, exist_ok=True)
    radar_path = os.path.join(plots_dir, f"temp_radar_{session_id}.png")
    bar_path = os.path.join(plots_dir, f"temp_bar_{session_id}.png")
    
    # Generate charts
    generate_radar_chart(["Accuracy", "Speech", "Facial"], [avg_accuracy, avg_speech, avg_facial], radar_path)
    generate_bar_chart(session_data["question_id"].tolist(), session_data["accuracy_score"].tolist(), bar_path)
    
    # Ensure final output folder exists
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    # Define PDF Document Layout
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=15,
        alignment=1  # Centered
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2D3748')
    )
    feedback_style = ParagraphStyle(
        'Feedback',
        parent=body_style,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4A5568'),
        backColor=colors.HexColor('#F7FAFC'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=1,
        borderPadding=10,
        spaceBefore=8,
        spaceAfter=8
    )
    
    story = []
    
    # Header Info
    story.append(Paragraph("AI Mock Interview Performance Report", title_style))
    story.append(Spacer(1, 10))
    
    meta_data = [
        [Paragraph("<b>Session ID:</b>", body_style), Paragraph(str(session_id), body_style),
         Paragraph("<b>Interview Type:</b>", body_style), Paragraph("Technical & HR Combo", body_style)],
        [Paragraph("<b>Evaluation Date:</b>", body_style), Paragraph("2026-07-04", body_style),
         Paragraph("<b>Overall Candidate Score:</b>", body_style), Paragraph(f"<b>{avg_overall:.2%}</b>", body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[100, 150, 120, 150])
    t_meta.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))
    
    # Dimension summary grid
    story.append(Paragraph("Dimension Scoring Summary", section_heading))
    summary_data = [
        ["Metric Dimension", "Average Score (0.0 - 1.0)", "Performance Tier"],
        ["Technical Accuracy", f"{avg_accuracy:.2%}", "Excellent" if avg_accuracy >= 0.8 else "Proficient" if avg_accuracy >= 0.6 else "Needs Improvement"],
        ["Speech & Communication", f"{avg_speech:.2%}", "Excellent" if avg_speech >= 0.8 else "Proficient" if avg_speech >= 0.6 else "Needs Improvement"],
        ["Facial Expressions & Delivery", f"{avg_facial:.2%}", "Excellent" if avg_facial >= 0.8 else "Proficient" if avg_facial >= 0.6 else "Needs Improvement"],
        ["Overall Interview Index", f"{avg_overall:.2%}", "Outstanding" if avg_overall >= 0.8 else "Good Fit" if avg_overall >= 0.6 else "Further Practice Required"]
    ]
    t_summary = Table(summary_data, colWidths=[200, 160, 160])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F7FAFC')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')])
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 15))
    
    # Visual analysis graphs side-by-side
    story.append(Paragraph("Visual Analysis Diagrams", section_heading))
    img_radar = Image(radar_path, width=210, height=210)
    img_bar = Image(bar_path, width=250, height=160)
    
    t_charts = Table([[img_radar, img_bar]], colWidths=[250, 270])
    t_charts.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_charts)
    story.append(Spacer(1, 15))
    
    # Narrative Recommendations section
    story.append(Paragraph("Narrative Feedback & Actionable Recommendations", section_heading))
    
    strengths = []
    improvements = []
    
    # Dynamic rules matching
    if avg_accuracy >= 0.7:
        strengths.append("Demonstrated high technical command, yielding highly accurate answers to target criteria.")
    else:
        improvements.append("Enhance conceptual clarity and study technical definitions in core questions.")
        
    if avg_speech >= 0.7:
        strengths.append("Excellent communication quality with stable phrasing, smooth vocal pacing, and appropriate tempo.")
    else:
        improvements.append("Reduce speech conversational breaks, conversational pauses, and practice smooth speech transits.")
        
    if avg_facial >= 0.7:
        strengths.append("Maintained positive head alignments, professional posture, and persistent visual focus.")
    else:
        improvements.append("Ensure alignment with standard camera framing, maintain consistent eye-contact levels, and relax facial expressions.")
        
    feedback_text = "<b>Key Strengths:</b><br/>"
    for s in strengths:
        feedback_text += f"• {s}<br/>"
    if not strengths:
        feedback_text += "• Candidate met base guidelines across evaluation metrics.<br/>"
        
    feedback_text += "<br/><b>Ranked Action Items:</b><br/>"
    for idx, imp in enumerate(improvements, 1):
        feedback_text += f"{idx}. {imp}<br/>"
    if not improvements:
        feedback_text += "• Exceptional score sheet. No major improvement items found."
        
    story.append(Paragraph(feedback_text, feedback_style))
    
    # Build document
    doc.build(story)
    print(f"Report compiled successfully for Session {session_id} to: {output_pdf_path}")
    
    # Clean up temporary chart images
    if os.path.exists(radar_path):
        os.remove(radar_path)
    if os.path.exists(bar_path):
        os.remove(bar_path)
        
    return True

if __name__ == "__main__":
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.path.abspath(os.path.join(os.getcwd(), "analytics"))
        if not os.path.exists(os.path.join(script_dir, "sample_data")):
            script_dir = os.getcwd()
            
    normalized_data_path = os.path.join(script_dir, "sample_data", "normalized_scores.csv")
    output_report_pdf = os.path.join(script_dir, "plots", "session_1_report.pdf")
    temp_plots_dir = os.path.join(script_dir, "plots")
    
    generate_pdf_report(1, normalized_data_path, output_report_pdf, temp_plots_dir)
