"""Regression checks for the deterministic Loop 0 task fixtures."""

import hashlib
import json
import re
from collections import defaultdict
from datetime import date
from html.parser import HTMLParser

import pytest

from tools.create_implementation_fixtures import create_fixtures


class _InputParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inputs = {}

    def handle_starttag(self, tag, attrs):
        if tag != "input":
            return
        values = dict(attrs)
        self.inputs[values["id"]] = {
            "checked": "checked" in values,
            "type": values.get("type"),
        }


class _InvoiceTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.text_parts = []
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data):
        clean = data.strip()
        if clean:
            self.text_parts.append(clean)
        if self._cell is not None and clean:
            self._cell.append(clean)

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self._cell is not None:
            self._row.append(" ".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def test_implementation_fixture_oracles_are_independently_observable(tmp_path):
    fixture = create_fixtures(tmp_path, "fixture-test")
    expected = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    assert expected["fixture_version"] == 3
    rows = json.loads((fixture / "invoices.json").read_text(encoding="utf-8"))
    invoice_oracle = expected["oracles"]["invoices"]

    assert len(rows) == invoice_oracle["row_count"] == 56
    overdue_totals = defaultdict(int)
    overdue_ids = defaultdict(list)
    for row in rows:
        if row["status"] == "overdue":
            overdue_totals[row["supplier"]] += row["amount_cents"]
            overdue_ids[row["supplier"]].append(row["invoice_id"])
    supplier = max(overdue_totals, key=overdue_totals.get)
    assert supplier == invoice_oracle["supplier"] == "Juniper Labs"
    assert overdue_totals[supplier] == invoice_oracle["total_cents"] == 217500
    assert overdue_ids[supplier] == invoice_oracle["invoice_ids"] == ["INV-0055", "INV-0056"]
    page = (fixture / "invoice-report.html").read_text(encoding="utf-8")
    table = _InvoiceTableParser()
    table.feed(page)
    assert table.rows[0] == ["Invoice ID", "Supplier", "Status", "Amount"]
    assert len(table.rows[1:]) == 56
    assert len("\n".join(table.text_parts)) <= 6000
    rendered_totals = defaultdict(int)
    rendered_invoice_ids = defaultdict(list)
    for invoice_id, supplier, status, amount in table.rows[1:]:
        if status == "overdue":
            cents = int(amount.removeprefix("$").replace(",", "").replace(".", ""))
            rendered_totals[supplier] += cents
            rendered_invoice_ids[supplier].append(invoice_id)
    rendered_winner = max(rendered_totals, key=rendered_totals.__getitem__)
    assert rendered_winner == invoice_oracle["supplier"]
    assert rendered_invoice_ids[rendered_winner] == invoice_oracle["invoice_ids"]
    assert rendered_totals[rendered_winner] == invoice_oracle["total_cents"]

    work_order = expected["oracles"]["work_order"]
    assert expected["paths"]["work_order_file"] == str((fixture / "work-order.txt").resolve())
    assert (fixture / "work-order.txt").read_text(encoding="utf-8") == work_order["initial_text"]
    assert work_order["invoice_report_sha256"] == hashlib.sha256(page.encode("utf-8")).hexdigest()
    rendered_total = rendered_totals[rendered_winner]
    expected_work_order = (
        "WORK ORDER: overdue invoice review\n"
        f"SUPPLIER: {rendered_winner}\n"
        f"INVOICE IDS: {', '.join(rendered_invoice_ids[rendered_winner])}\n"
        f"OVERDUE TOTAL: ${rendered_total // 100:,}.{rendered_total % 100:02d}\n"
        "REVIEW: Complete\n"
        "OWNER: Adam\n"
    )
    assert work_order["final_text"] == expected_work_order
    work_order_prompt = expected["prompts"]["work_order_template"]
    assert "<LOCAL_REPORT_URL>" in work_order_prompt
    assert str((fixture / "work-order.txt").resolve()) in work_order_prompt
    assert "save the document, reopen it" in work_order_prompt

    source = (fixture / "source.txt").read_bytes()
    assert len(source) == expected["oracles"]["file_status"]["byte_length"]
    assert hashlib.sha256(source).hexdigest() == expected["oracles"]["terminal"]["source_sha256"]
    assert source.decode("utf-8").splitlines()[0] == expected["oracles"]["file_status"]["first_line"]
    assert (fixture / "document.txt").read_text(encoding="utf-8") == expected["oracles"]["document"]["initial_text"]

    parser = _InputParser()
    parser.feed((fixture / "settings.html").read_text(encoding="utf-8"))
    assert parser.inputs == {
        "compact": {"checked": False, "type": "checkbox"},
        "sync": {"checked": True, "type": "checkbox"},
        "notifications": {"checked": True, "type": "checkbox"},
    }
    settings = (fixture / "settings.html").read_text(encoding="utf-8")
    assert "STATE " in settings
    assert 'id="fixture-state"' in settings
    assert "function resetFixture()" in settings
    assert "window.addEventListener(\"pageshow\", resetFixture)" in settings
    assert expected["prompts"]["settings"] == "Set Compact mode to on, Sync to off, and Notifications to off."

    scratchpad = (fixture / "scratchpad.html").read_text(encoding="utf-8")
    scratchpad_oracle = expected["oracles"]["scratchpad"]
    assert 'for="note-text">Note text field' in scratchpad
    assert 'id="save-note" type="button">Save note' in scratchpad
    assert 'id="save-result"' in scratchpad
    assert 'saveResult.textContent = noteField.value.trim() ? "Saved: " + noteField.value : "Not saved"' in scratchpad
    assert "window.addEventListener(\"pageshow\", resetFixture)" in scratchpad
    assert scratchpad_oracle == {
        "initial_state": "Not saved",
        "saved_text": "Call the dentist Tuesday at 2 pm",
        "final_state": "Saved: Call the dentist Tuesday at 2 pm",
    }
    assert expected["prompts"]["scratchpad"] == (
        'In the local scratchpad, enter "Call the dentist Tuesday at 2 pm" in the Note text field, '
        "click Save note, and tell me the saved text."
    )

    memory = expected["oracles"]["memory"]
    assert len(memory["records"]) == 3
    assert all(re.search(r"\b20\d\d-\d\d-\d\d\b", record) for record in memory["records"])
    assert all(date.fromisoformat(value) for value in memory["expected_record_dates"])
    assert memory["timezone"] == expected["timezone"]
    assert set(memory["prompts"]) == {"day", "week", "year", "timezone"}


def test_generated_memory_fixture_retrieves_exact_day_week_year_and_zone(tmp_path):
    from src.memory.embedder import MemoryEmbedder
    from src.memory.manager import MemoryManager

    fixture = create_fixtures(tmp_path, "memory-fixture-test")
    spec = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))["oracles"]["memory"]
    manager = MemoryManager(
        storage_path=tmp_path / "isolated-memory.json",
        embedder=MemoryEmbedder(disabled=True),
    )
    for record in spec["records"]:
        manager.save(record)

    day, week, year, timezone = (
        manager.retrieve_context(spec["prompts"][name]) or ""
        for name in ("day", "week", "year", "timezone")
    )
    assert spec["records"][0] in day
    assert spec["expected_record_dates"][0] in day
    assert "09:30:00" in day and "10:15:00" in day
    assert spec["records"][0] in week and spec["records"][1] in week
    assert spec["records"][2] not in week
    assert spec["records"][2] in year
    assert spec["records"][0] not in year and spec["records"][1] not in year
    for context in (day, week, year, timezone):
        assert f"[event timezone: {spec['timezone']}]" in context
    assert spec["records"][0] in timezone


def test_implementation_fixture_ids_cannot_overwrite_existing_state(tmp_path):
    create_fixtures(tmp_path, "same-id")
    with pytest.raises(FileExistsError):
        create_fixtures(tmp_path, "same-id")
