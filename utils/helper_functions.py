"""Report generation helper utilities for SentiFlow AI."""
import io
from datetime import datetime
import pandas as pd

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
except ImportError:
    pass


def generate_excel_report(reviews: list[dict]) -> bytes:
    """Generate Excel workbook byte contents for the given reviews."""
    if not reviews:
        df = pd.DataFrame(columns=["review", "corrected_review", "cleaned_review", "sentiment", "emotion", "confidence", "timestamp", "source", "domain"])
    else:
        df = pd.DataFrame(reviews)
        
    # Convert datetimes to strings to prevent serialization errors in openpyxl
    if "timestamp" in df:
        df["timestamp"] = df["timestamp"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if hasattr(x, "strftime") else str(x)
        )
        
    # Convert lists to strings (e.g. problems, recommendation)
    for col in ["problems", "recommendation"]:
        if col in df:
            df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
            
    # Clean output dict/object structures (e.g. probabilities)
    if "probabilities" in df:
        df["probabilities"] = df["probabilities"].apply(
            lambda x: "; ".join([f"{k}: {v}%" for k, v in x.items()]) if isinstance(x, dict) else str(x)
        )

    # Reorder columns to be user-friendly if they exist
    cols_order = [
        "timestamp", "source", "review", "corrected_review", "cleaned_review", 
        "sentiment", "emotion", "confidence", "domain", "problems", "recommendation"
    ]
    existing_cols = [c for c in cols_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols_order]
    df = df[existing_cols + other_cols]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sentiment Reviews')
    return output.getvalue()


def generate_pdf_report(reviews: list[dict]) -> bytes:
    """Generate a formatted PDF report with executive stats and review tables."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        rightMargin=36, 
        leftMargin=36, 
        topMargin=36, 
        bottomMargin=36
    )
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=15
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#334155')
    )

    bold_body_style = ParagraphStyle(
        'BoldBodyText',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    # Title & Metadata
    story.append(Paragraph("SentiFlow AI — Sentiment Intelligence Report", title_style))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Sample Size: {len(reviews)} Reviews", subtitle_style))
    story.append(Spacer(1, 5))
    
    # Calculate stats
    total = len(reviews)
    positive = sum(1 for r in reviews if r.get('sentiment') == 'Positive')
    negative = sum(1 for r in reviews if r.get('sentiment') == 'Negative')
    neutral = sum(1 for r in reviews if r.get('sentiment') == 'Neutral')
    
    csat = round((positive / total * 100), 1) if total > 0 else 0.0
    reputation = round(((positive - negative) / total * 50 + 50), 1) if total > 0 else 0.0
    
    # Stats table
    stats_data = [
        [
            Paragraph("Core Metric", bold_body_style),
            Paragraph("Count / Score", bold_body_style),
            Paragraph("Share %", bold_body_style)
        ],
        [
            Paragraph("Positive Sentiment", body_style),
            Paragraph(str(positive), body_style),
            Paragraph(f"{round(positive/total*100, 1) if total > 0 else 0.0}%", body_style)
        ],
        [
            Paragraph("Negative Sentiment", body_style),
            Paragraph(str(negative), body_style),
            Paragraph(f"{round(negative/total*100, 1) if total > 0 else 0.0}%", body_style)
        ],
        [
            Paragraph("Neutral Sentiment", body_style),
            Paragraph(str(neutral), body_style),
            Paragraph(f"{round(neutral/total*100, 1) if total > 0 else 0.0}%", body_style)
        ],
        [
            Paragraph("Customer Satisfaction (CSAT)", bold_body_style),
            Paragraph(f"{csat}%", bold_body_style),
            Paragraph("N/A", body_style)
        ],
        [
            Paragraph("Brand Reputation Index", bold_body_style),
            Paragraph(f"{reputation} / 100", bold_body_style),
            Paragraph("N/A", body_style)
        ]
    ]
    
    stats_table = Table(stats_data, colWidths=[220, 140, 140])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    
    story.append(Paragraph("Executive Summary Stats", heading_style))
    story.append(stats_table)
    story.append(Spacer(1, 10))
    
    # Insights Section
    story.append(Paragraph("Sentiment Diagnostics & Insights", heading_style))
    insights = []
    
    if total == 0:
        insights.append("No reviews available for insight compilation.")
    else:
        if reputation >= 70:
            insights.append("Customer sentiment is overwhelmingly positive. Brand equity is high; focus on leveraging positive testimonials in acquisition campaigns.")
        elif reputation >= 50:
            insights.append("Sentiment is stable but moderate. Track minor issues and negative reports to keep customer churn low.")
        else:
            insights.append("Critical: Sentiment is leaning negative. Review negative feedback immediately and inspect specific category operational issues.")
            
        if negative > 0:
            insights.append(f"Operations: Address the {negative} negative complaints to resolve customer friction and prevent churn.")
            
        # Check for recurring problems
        all_problems = []
        for r in reviews:
            all_problems.extend(r.get("problems", []))
        if all_problems:
            from collections import Counter
            top_prob = Counter(all_problems).most_common(2)
            prob_names = ", ".join([f"'{p[0]}' ({p[1]} times)" for p in top_prob])
            insights.append(f"Product Friction: The most frequent complaints are related to: {prob_names}.")
            
    insights_html = "".join([f"<li>{ins}</li>" for ins in insights])
    story.append(Paragraph(f"<ul>{insights_html}</ul>", body_style))
    story.append(Spacer(1, 12))
    
    # Review Details
    story.append(Paragraph("Recent Feedback Database Records", heading_style))
    
    review_headers = [
        Paragraph("Date", bold_body_style),
        Paragraph("Original User Review", bold_body_style),
        Paragraph("Sentiment", bold_body_style),
        Paragraph("Emotion", bold_body_style),
        Paragraph("Conf %", bold_body_style)
    ]
    
    review_rows = [review_headers]
    
    # Put top 25 reviews to avoid huge overflow, page limit is standard
    for rev in reviews[:25]:
        dt = rev.get("timestamp")
        dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        
        # Clean text
        review_text = rev.get("review", "")
        if len(review_text) > 90:
            review_text = review_text[:87] + "..."
            
        review_rows.append([
            Paragraph(dt_str, body_style),
            Paragraph(review_text, body_style),
            Paragraph(rev.get("sentiment", "N/A"), body_style),
            Paragraph(rev.get("emotion", "N/A"), body_style),
            Paragraph(f"{rev.get('confidence', 0.0)}%", body_style)
        ])
        
    review_table = Table(review_rows, colWidths=[65, 255, 65, 65, 50])
    review_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f8fafc')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(review_table)
    
    if len(reviews) > 25:
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"* Report limited to top 25 of {len(reviews)} total reviews.", subtitle_style))
        
    doc.build(story)
    return buffer.getvalue()
