"""
Script 24 — Convert paper_draft.md to Microsoft Word (.docx) and PDF (.pdf).

Usage:
    python scripts/24_convert_paper_to_docx_pdf.py
"""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

MD_FILE = Path("paper_draft.md")
DOCX_FILE = Path("paper_draft.docx")
PDF_FILE = Path("paper_draft.pdf")


def convert_to_docx():
    doc = Document()
    doc.add_heading("Dual-Stream Semantic-Frequency Fusion Network for Explainable Deepfake Detection and Cross-Domain Generalization", 0)
    doc.add_paragraph("Shashank Singh Singhania — Department of Computer Science & Engineering — July 2026")
    
    with open(MD_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("# Dual-Stream") or line_s.startswith("**Shashank"):
            continue

        if line_s.startswith("## "):
            doc.add_heading(line_s.replace("## ", ""), level=1)
        elif line_s.startswith("### "):
            doc.add_heading(line_s.replace("### ", ""), level=2)
        elif line_s.startswith("#### "):
            doc.add_heading(line_s.replace("#### ", ""), level=3)
        elif line_s.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(line_s[2:])
        elif not line_s.startswith("|") and not line_s.startswith("![") and not line_s.startswith("```"):
            doc.add_paragraph(line_s)

    doc.save(DOCX_FILE)
    print(f"Saved: {DOCX_FILE}")


def convert_to_pdf():
    doc = SimpleDocTemplate(str(PDF_FILE), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=16, leading=20, textColor=RGBColor(26, 43, 76))
    h1_style = ParagraphStyle("H1Style", parent=styles["Heading1"], fontSize=13, leading=16, textColor=RGBColor(39, 174, 96))
    body_style = ParagraphStyle("BodyStyle", parent=styles["BodyText"], fontSize=10, leading=14)

    story.append(Paragraph("Dual-Stream Semantic-Frequency Fusion Network for Explainable Deepfake Detection and Cross-Domain Generalization", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Shashank Singh Singhania — July 2026", body_style))
    story.append(Spacer(1, 15))

    with open(MD_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("# Dual-Stream") or line_s.startswith("**Shashank"):
            continue

        if line_s.startswith("## "):
            story.append(Spacer(1, 10))
            story.append(Paragraph(line_s.replace("## ", ""), h1_style))
            story.append(Spacer(1, 5))
        elif not line_s.startswith("|") and not line_s.startswith("![") and not line_s.startswith("```"):
            clean_text = line_s.replace("*", "").replace("#", "")
            if clean_text:
                story.append(Paragraph(clean_text, body_style))
                story.append(Spacer(1, 4))

    doc.build(story)
    print(f"Saved: {PDF_FILE}")


if __name__ == "__main__":
    convert_to_docx()
    convert_to_pdf()
