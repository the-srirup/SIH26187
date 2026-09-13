"""
Printable event-log reports.

Why a PDF and not a CSV
-----------------------
A CSV is for a spreadsheet; this is for a file. A sealed event log is the
platform's evidentiary output, and the people who need it on paper — a duty
officer's handover, an incident file, a court bundle — need something that
carries its own context: which filters produced it, when it was produced, how
many events it covers, and whether the hash chain still verified at the moment
of printing. A bare table of rows proves nothing about the log it came from.

So every report states its own provenance, and the integrity line is measured
at generation time rather than assumed.

The dependency is optional. ``reportlab`` is pure Python and installs anywhere,
but a deployment that never prints should not fail to start over it, so the
import is deferred and :func:`pdf_available` lets callers degrade politely.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Optional

from core.config import settings
from core.timeutil import fmt_ist

log = logging.getLogger("ibvap.report")

#: Page furniture. A4 landscape, because the event table has eight columns and
#: portrait forces either a microscopic font or a wrapped, unreadable grid.
_MARGIN = 14

#: Severity -> (row background, text colour). Mirrors the dashboard so a printed
#: page and the screen agree about what is urgent.
_SEVERITY_COLOURS = {
    "CRITICAL": ("#7a1520", "#ffffff"),
    "HIGH": ("#8a4a00", "#ffffff"),
    "MEDIUM": ("#5a4a10", "#ffffff"),
    "LOW": ("#1f3a4a", "#ffffff"),
    "INFO": ("#2a2f38", "#d8dde6"),
}


def pdf_available() -> bool:
    """Is the PDF backend installed?"""
    try:
        import reportlab  # noqa: F401
        return True
    except Exception:
        return False


def _fmt(value: Any, dash: str = "—") -> str:
    text = "" if value is None else str(value)
    return text if text.strip() else dash


def _wrap(text: str, width: int, max_lines: int = 3) -> str:
    """
    Wrap a field so a table cell cannot run off the page.

    Long tokens are split mid-word rather than left intact. Wrapping only on
    whitespace is not enough: one unbroken 4,000-character token — a base64
    blob, a URL, a hash pasted into a description — stays a single line, and
    reportlab then aborts the entire document with a LayoutError because the
    row is taller than the page. A report that will not print because of one
    malformed row is far worse than a truncated cell.
    """
    text = " ".join(_fmt(text, "").split())          # collapse newlines and runs
    if len(text) <= width:
        return text

    words: list[str] = []
    for word in text.split():
        while len(word) > width:                     # break the unbreakable
            words.append(word[:width])
            word = word[width:]
        if word:
            words.append(word)

    out: list[str] = []
    line = ""
    for word in words:
        if line and len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
            if len(out) >= max_lines:
                break
        else:
            line = f"{line} {word}".strip()
    if line and len(out) < max_lines:
        out.append(line)

    clipped = out[:max_lines]
    if clipped and len(" ".join(clipped)) < len(text):
        clipped[-1] = clipped[-1][:max(1, width - 1)] + "…"
    return "\n".join(clipped)


def build_event_log_pdf(
    alerts: list[dict],
    *,
    total_matching: int,
    filters: Optional[dict] = None,
    integrity: Optional[dict] = None,
    title: str = "Event Log",
) -> bytes:
    """
    Render an event log to PDF bytes.

    ``alerts`` are serialized alert dicts as the API returns them.
    ``total_matching`` is how many rows matched the filter overall, which can
    exceed ``len(alerts)`` — the report says so explicitly rather than letting
    a reader assume a truncated page is the whole story.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
        TableStyle,
    )

    buffer = io.BytesIO()
    page = landscape(A4)
    doc = SimpleDocTemplate(
        buffer, pagesize=page,
        leftMargin=_MARGIN * mm, rightMargin=_MARGIN * mm,
        topMargin=_MARGIN * mm, bottomMargin=_MARGIN * mm,
        title=f"IBVAP {title}", author="IBVAP",
        subject="Border surveillance event log",
    )

    sheet = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=sheet["Title"], fontSize=16, leading=19,
                        alignment=TA_LEFT, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=sheet["Normal"], fontSize=8.5, leading=11,
                         textColor=colors.HexColor("#555f70"))
    note = ParagraphStyle("note", parent=sheet["Normal"], fontSize=8, leading=10.5)
    cell = ParagraphStyle("cell", parent=sheet["Normal"], fontSize=7.2, leading=8.8)

    story: list = []
    story.append(Paragraph("IBVAP — Intelligent Border Video Analytics Platform", h1))
    story.append(Paragraph(
        f"{title} &nbsp;·&nbsp; generated {fmt_ist()} &nbsp;·&nbsp; "
        f"all times Indian Standard Time", sub))
    story.append(Spacer(1, 5 * mm))

    # ---------------------------------------------------------------- scope
    filters = filters or {}
    applied = [f"{k.replace('_', ' ')}: {v}" for k, v in filters.items()
               if v not in (None, "", [])]
    scope_rows = [
        ["Events in this report", f"{len(alerts):,}"],
        ["Events matching the filter", f"{total_matching:,}"],
        ["Filters applied", ", ".join(applied) if applied else "none — full log"],
    ]
    if len(alerts) < total_matching:
        scope_rows.append([
            "Note",
            f"This report shows the most recent {len(alerts):,} of "
            f"{total_matching:,} matching events.",
        ])

    if integrity:
        verified = integrity.get("valid")
        if verified is None:
            chain = "not checked"
        elif verified:
            chain = (f"VERIFIED — {integrity.get('verified', 0):,} events form an "
                     f"unbroken SHA-256 chain")
        else:
            chain = (f"BROKEN — {len(integrity.get('breaks', []) or [])} break(s) "
                     f"detected; this log has been altered since it was sealed")
        scope_rows.append(["Hash-chain integrity at generation", chain])

    scope = Table([[Paragraph(f"<b>{r[0]}</b>", note), Paragraph(str(r[1]), note)]
                   for r in scope_rows], colWidths=[62 * mm, None])
    scope.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#dde2ea")),
    ]))
    story.append(scope)
    story.append(Spacer(1, 5 * mm))

    # -------------------------------------------------------------- summary
    by_severity: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for a in alerts:
        by_severity[(a.get("severity") or "INFO").upper()] = \
            by_severity.get((a.get("severity") or "INFO").upper(), 0) + 1
        by_type[a.get("alert_type") or "—"] = by_type.get(a.get("alert_type") or "—", 0) + 1

    if alerts:
        order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
        sev_cells = [["Severity", "Count"]] + [
            [s, f"{by_severity.get(s, 0):,}"] for s in order if by_severity.get(s)
        ]
        top_types = sorted(by_type.items(), key=lambda kv: kv[1], reverse=True)[:8]
        type_cells = [["Event type", "Count"]] + [
            [t.replace("_", " ").upper(), f"{c:,}"] for t, c in top_types
        ]

        def _mini(data):
            t = Table(data, colWidths=[52 * mm, 20 * mm])
            t.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f6")),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dde2ea")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]))
            return t

        summary = Table([[_mini(sev_cells), _mini(type_cells)]],
                        colWidths=[76 * mm, 76 * mm])
        summary.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(KeepTogether([Paragraph("<b>Summary</b>", note),
                                   Spacer(1, 2 * mm), summary]))
        story.append(Spacer(1, 6 * mm))

    # ---------------------------------------------------------------- table
    if not alerts:
        story.append(Paragraph(
            "<b>No events matched this filter.</b> An empty report is still a "
            "result: it records that the log was queried and held nothing for "
            "the period and filters stated above.", note))
    else:
        header = ["#", "Time (IST)", "Camera", "Event", "Sev", "Object",
                  "Plate", "Track", "Description", "Hash"]
        data = [header]
        for a in alerts:
            data.append([
                Paragraph(str(a.get("id", "")), cell),
                Paragraph(_fmt(a.get("timestamp_ist")), cell),
                Paragraph(_wrap(a.get("camera_name"), 18), cell),
                Paragraph(_fmt(a.get("alert_type")).replace("_", " ").upper(), cell),
                Paragraph(_fmt(a.get("severity"), "INFO"), cell),
                Paragraph(_fmt(a.get("object_class")).upper(), cell),
                Paragraph(_fmt(a.get("plate")), cell),
                Paragraph(_fmt(a.get("track_id")), cell),
                Paragraph(_wrap(a.get("description"), 70), cell),
                Paragraph(_fmt(a.get("hash"))[:12], cell),
            ])

        widths = [11, 33, 28, 29, 14, 16, 24, 11, 82, 21]
        total_w = sum(widths)
        avail = (page[0] - 2 * _MARGIN * mm) / mm
        widths = [w * avail / total_w * mm for w in widths]

        table = Table(data, colWidths=widths, repeatRows=1)
        style = [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 7.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2430")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dde2ea")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]
        # Colour the severity cell only. Shading whole rows turns a page of
        # routine traffic into a wall of colour and stops meaning anything.
        for i, a in enumerate(alerts, start=1):
            sev = (a.get("severity") or "INFO").upper()
            bg, fg = _SEVERITY_COLOURS.get(sev, _SEVERITY_COLOURS["INFO"])
            style.append(("BACKGROUND", (4, i), (4, i), colors.HexColor(bg)))
            style.append(("TEXTCOLOR", (4, i), (4, i), colors.HexColor(fg)))
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (3, i), colors.HexColor("#f6f8fb")))
                style.append(("BACKGROUND", (5, i), (-1, i), colors.HexColor("#f6f8fb")))
        table.setStyle(TableStyle(style))
        story.append(table)

    def _furniture(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#6b7688"))
        canvas.drawString(_MARGIN * mm, 8 * mm,
                          f"IBVAP {settings.VERSION} · tamper-evident event log · "
                          f"generated {fmt_ist()}")
        canvas.drawRightString(page[0] - _MARGIN * mm, 8 * mm,
                               f"Page {canvas.getPageNumber()}")
        canvas.setStrokeColor(colors.HexColor("#dde2ea"))
        canvas.line(_MARGIN * mm, 11 * mm, page[0] - _MARGIN * mm, 11 * mm)
        canvas.restoreState()

    doc.build(story, onFirstPage=_furniture, onLaterPages=_furniture)
    return buffer.getvalue()


__all__ = ["build_event_log_pdf", "pdf_available"]
