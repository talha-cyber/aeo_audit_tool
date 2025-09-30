"""Appendix section builder for listing grouped question/answer pairs."""

from __future__ import annotations

from typing import Any, Dict, List

from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer

from app.services.report_utils import (
    format_block_text,
    format_inline_text,
    group_records,
)

from ..accessibility import create_section_heading
from ..theme import Theme, create_paragraph_styles


def build(theme: Theme, data: Dict[str, Any]) -> List[Any]:
    """Build appendix section that groups responses by the configured dimension."""

    qa_entries = data.get("qa_entries") or []
    if not qa_entries:
        return []

    group_by = data.get("appendix_group_by", "provider")
    grouped_entries = group_records(qa_entries, group_by)

    styles = create_paragraph_styles(theme)
    story: List[Any] = []

    heading = create_section_heading("Appendix: Question & Answer Index", 1, theme)
    story.append(heading)
    story.append(Spacer(1, 0.2 * inch))

    for group_label, entries in grouped_entries.items():
        section_title = format_inline_text(group_by.replace("_", " ").title())
        group_heading = f"{section_title}: {format_inline_text(group_label)}"
        story.append(Paragraph(group_heading, styles["heading2"]))
        story.append(Spacer(1, 0.1 * inch))

        for idx, entry in enumerate(entries, start=1):
            question_text = entry.get("question") or "Question not available."
            answer_text = entry.get("answer") or "Answer not available."

            story.append(
                Paragraph(
                    f"{idx}. <b>Question:</b> {format_block_text(question_text)}",
                    styles["body"],
                )
            )
            story.append(
                Paragraph(
                    f"<b>Answer:</b> {format_block_text(answer_text)}",
                    styles["body"],
                )
            )

            metadata_parts = []
            if entry.get("category"):
                metadata_parts.append(
                    f"Category: {format_inline_text(entry['category'])}"
                )
            if entry.get("question_type"):
                metadata_parts.append(
                    f"Type: {format_inline_text(entry['question_type'])}"
                )
            if entry.get("platform"):
                metadata_parts.append(
                    f"Platform: {format_inline_text(entry['platform'])}"
                )

            if metadata_parts:
                meta_text = " | ".join(metadata_parts)
                story.append(
                    Paragraph(
                        f"<font size=9 color='#555555'>{meta_text}</font>",
                        styles["body_indent"],
                    )
                )

            story.append(Spacer(1, 0.12 * inch))

        story.append(Spacer(1, 0.2 * inch))

    return story
