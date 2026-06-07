from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd

from smartfabric.detector import Detection


def detections_to_frame(detections: list[Detection]) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame(
            columns=[
                "frame_id",
                "class_id",
                "label",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "width",
                "height",
                "area",
                "severity",
            ]
        )
    return pd.DataFrame([asdict(item) for item in detections])


def summarize(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "total_defects": 0,
            "types": {},
            "severity": {},
            "mean_confidence": 0.0,
            "highest_risk": "None",
        }

    severity_order = {"Low": 1, "Medium": 2, "High": 3}
    highest = max(df["severity"].tolist(), key=lambda value: severity_order.get(value, 0))
    return {
        "total_defects": int(len(df)),
        "types": df["label"].value_counts().to_dict(),
        "severity": df["severity"].value_counts().to_dict(),
        "mean_confidence": round(float(df["confidence"].mean()), 3),
        "highest_risk": highest,
    }


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def pdf_bytes(df: pd.DataFrame, title: str = "SmartFabric Inspector Report") -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    summary = summarize(df)

    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Spacer(1, 14),
        Paragraph(f"Total defects: {summary['total_defects']}", styles["Heading2"]),
        Paragraph(f"Highest risk: {summary['highest_risk']}", styles["Normal"]),
        Paragraph(f"Mean confidence: {summary['mean_confidence']}", styles["Normal"]),
        Spacer(1, 14),
    ]

    if df.empty:
        story.append(Paragraph("No defects detected for the selected threshold.", styles["Normal"]))
    else:
        display = df[["frame_id", "label", "confidence", "severity", "x1", "y1", "x2", "y2"]].copy()
        display["confidence"] = display["confidence"].map(lambda value: f"{value:.2f}")
        rows = [display.columns.tolist(), *display.round(2).astype(str).values.tolist()]
        table = Table(rows, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102A43")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BCCCDC")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
                ]
            )
        )
        story.append(table)

    doc.build(story)
    return buffer.getvalue()


def write_report_files(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"fabric_defect_report_{stamp}.csv"
    pdf_path = output_dir / f"fabric_defect_report_{stamp}.pdf"
    csv_path.write_bytes(csv_bytes(df))
    pdf_path.write_bytes(pdf_bytes(df))
    return csv_path, pdf_path
