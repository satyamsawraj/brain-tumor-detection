"""
PDF Report Generator for Brain Tumor Detection
Generates a comprehensive PDF report with all visualizations
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import os


def generate_pdf_report(prediction, confidence, patient_name="Unknown", output_path="report.pdf"):
    """
    Generate a comprehensive PDF report with all visualizations
    
    Args:
        prediction: The tumor type prediction (e.g., "Meningioma")
        confidence: Confidence percentage (e.g., 51.64)
        patient_name: Optional patient identifier
        output_path: Where to save the PDF
    """
    
    # Create the document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    # Get standard styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Build the story (content)
    story = []
    
    # =====================
    # TITLE PAGE
    # =====================
    story.append(Spacer(1, 0.5 * inch))
    
    # Title
    title = Paragraph("🧠 Brain Tumor Detection Report", title_style)
    story.append(title)
    story.append(Spacer(1, 0.3 * inch))
    
    # Report info
    report_info = [
        ['Report Date:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
        ['Patient ID:', patient_name],
        ['Analysis Type:', 'MRI Brain Scan Classification'],
    ]
    
    info_table = Table(report_info, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5 * inch))
    
    # =====================
    # DIAGNOSIS SUMMARY
    # =====================
    story.append(Paragraph("Diagnosis Summary", heading_style))
    
    # Diagnosis box
    diagnosis_data = [
        ['Prediction:', prediction],
        ['Confidence:', f'{confidence}%'],
        ['Status:', 'Requires Medical Review' if confidence < 80 else 'High Confidence']
    ]
    
    diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor('#D62828') if prediction != "No Tumor" else colors.HexColor('#2E7D32')),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F0F0')),
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#2E86AB')),
        ('INNERGRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC')),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
    ]))
    story.append(diagnosis_table)
    story.append(Spacer(1, 0.3 * inch))
    
    # Add disclaimer
    disclaimer = Paragraph(
        "<i>Note: This is an AI-assisted analysis and should not replace professional medical diagnosis. "
        "Please consult with a qualified healthcare provider for medical advice.</i>",
        styles['Italic']
    )
    story.append(disclaimer)
    story.append(PageBreak())
    
    # =====================
    # VISUALIZATIONS PAGE
    # =====================
    story.append(Paragraph("Image Analysis & Visualizations", heading_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Path to images
    outputs_dir = 'static/outputs'
    
    # Add visualizations (2 per row)
    visualizations = [
        ('clahe.png', 'CLAHE Enhancement'),
        ('dtcwt.png', 'DTCWT Features'),
        ('loggabor.png', 'Log-Gabor Features'),
        ('entropy.png', 'Entropy Analysis'),
        ('roc.png', 'ROC Curve Analysis'),
        ('umap.png', 'Classification Results'),
    ]
    
    # Add images in pairs
    for i in range(0, len(visualizations), 2):
        row_data = []
        
        for j in range(2):
            if i + j < len(visualizations):
                img_file, img_title = visualizations[i + j]
                img_path = os.path.join(outputs_dir, img_file)
                
                if os.path.exists(img_path):
                    # Create cell with title and image
                    cell_content = []
                    cell_content.append(Paragraph(f"<b>{img_title}</b>", styles['Normal']))
                    
                    # Add image (resize to fit)
                    img = Image(img_path, width=2.5*inch, height=2*inch)
                    row_data.append([Paragraph(f"<b>{img_title}</b>", styles['Normal']), img])
                else:
                    row_data.append([Paragraph(f"<b>{img_title}</b>", styles['Normal']), 
                                   Paragraph("Image not available", styles['Italic'])])
            else:
                row_data.append(['', ''])
        
        # Create table for this row
        if len(row_data) == 2:
            viz_table = Table([[row_data[0][1], row_data[1][1]]], colWidths=[3*inch, 3*inch])
            viz_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            # Add titles
            title_table = Table([[row_data[0][0], row_data[1][0]]], colWidths=[3*inch, 3*inch])
            title_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            
            story.append(title_table)
            story.append(viz_table)
            story.append(Spacer(1, 0.3 * inch))
    
    story.append(PageBreak())
    
    # =====================
    # TECHNICAL DETAILS
    # =====================
    story.append(Paragraph("Technical Details", heading_style))
    story.append(Spacer(1, 0.2 * inch))
    
    tech_details = Paragraph(
        "<b>Analysis Method:</b> Deep Learning Classification using DTCWT Features<br/>"
        "<b>Preprocessing:</b> CLAHE (Contrast Limited Adaptive Histogram Equalization)<br/>"
        "<b>Feature Extraction:</b> Dual-Tree Complex Wavelet Transform (DTCWT)<br/>"
        "<b>Model Type:</b> Multi-class Classification (4 classes)<br/>"
        "<b>Classes:</b> No Tumor, Glioma, Meningioma, Pituitary<br/>"
        "<b>Image Size:</b> 224x224 pixels",
        styles['Normal']
    )
    story.append(tech_details)
    story.append(Spacer(1, 0.3 * inch))
    
    # =====================
    # FOOTER
    # =====================
    footer = Paragraph(
        f"<i>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
        "Brain Tumor Detection & Analysis System v1.0</i>",
        styles['Italic']
    )
    story.append(Spacer(1, 0.5 * inch))
    story.append(footer)
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test the PDF generator
    generate_pdf_report(
        prediction="Meningioma",
        confidence=51.64,
        patient_name="Test-Patient-001",
        output_path="static/outputs/test_report.pdf"
    )
    print("Test PDF created successfully!")