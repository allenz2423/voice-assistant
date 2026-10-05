#!/usr/bin/env python3
"""Create disposable, deterministic fixtures for the implementation timeline."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import secrets
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.temporal import local_timezone_name


FIXTURE_VERSION = 3
SEED = 20261005
VENDORS = ("Amber Stationery", "Birch Support", "Cobalt Network", "Maple Office")


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _invoice_rows(seed: int) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for number in range(1, 55):
        amount_cents = 2500 + ((seed * 17 + number * 7919) % 6500)
        rows.append({
            "invoice_id": f"INV-{number:04d}",
            "supplier": VENDORS[(number - 1) % len(VENDORS)],
            "status": "overdue" if number % 3 else "current",
            "amount_cents": amount_cents,
        })
    # The unique maximum is deliberately near the end of the visible table.
    rows.extend((
        {"invoice_id": "INV-0055", "supplier": "Juniper Labs", "status": "overdue", "amount_cents": 120000},
        {"invoice_id": "INV-0056", "supplier": "Juniper Labs", "status": "overdue", "amount_cents": 97500},
    ))
    return rows


def _invoice_oracle(rows: list[dict[str, str | int]]) -> dict[str, object]:
    totals: dict[str, int] = {}
    invoice_ids: dict[str, list[str]] = {}
    for row in rows:
        if row["status"] != "overdue":
            continue
        supplier = str(row["supplier"])
        totals[supplier] = totals.get(supplier, 0) + int(row["amount_cents"])
        invoice_ids.setdefault(supplier, []).append(str(row["invoice_id"]))
    winner = max(totals, key=totals.__getitem__)
    total = totals[winner]
    return {
        "supplier": winner,
        "invoice_ids": invoice_ids[winner],
        "total_cents": total,
        "total_display": "$" + f"{total // 100:,}.{total % 100:02d}",
        "overdue_totals_cents": totals,
    }


def _invoice_html(rows: list[dict[str, str | int]]) -> str:
    body_rows = []
    for row in rows:
        amount = int(row["amount_cents"])
        amount_display = "$" + f"{amount // 100:,}.{amount % 100:02d}"
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['invoice_id']))}</td>"
            f"<td>{html.escape(str(row['supplier']))}</td>"
            f"<td>{html.escape(str(row['status']))}</td>"
            f"<td>{amount_display}</td>"
            "</tr>"
        )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Adam Fixture Invoice Report</title>
<style>body{font:16px sans-serif;margin:2rem}table{border-collapse:collapse}th,td{border:1px solid #777;padding:.3rem .6rem;text-align:left}</style>
</head><body><main><h1>Invoice Report</h1>
<p>Fixture report. Use the rows below to calculate overdue totals by supplier.</p>
<table><thead><tr><th>Invoice ID</th><th>Supplier</th><th>Status</th><th>Amount</th></tr></thead>
<tbody>""" + "\n".join(body_rows) + """</tbody></table>
</main></body></html>
"""


def _settings_html() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Adam Fixture Settings</title>
<style>body{font:18px sans-serif;margin:3rem;max-width:40rem}label{display:block;padding:1rem;border:1px solid #888;margin:.5rem 0}output{display:block;margin-top:2rem;font-weight:bold}</style>
</head><body><main><h1>Notification Preferences</h1>
<label><input id="compact" type="checkbox"> Compact mode</label>
<label><input id="sync" type="checkbox" checked> Sync</label>
<label><input id="notifications" type="checkbox" checked> Notifications</label>
<output id="fixture-state" aria-live="polite"></output>
<script>
const controls = ["compact", "sync", "notifications"];
function updateState() {
  const values = controls.map(function(id) {
    const name = id === "compact" ? "Compact" : id === "sync" ? "Sync" : "Notifications";
    return name + "=" + (document.getElementById(id).checked ? "on" : "off");
  });
  document.getElementById("fixture-state").textContent = "STATE " + values.join("; ");
}
controls.forEach(function(id) {
  document.getElementById(id).addEventListener("change", updateState);
});
function resetFixture() {
  controls.forEach(function(id) {
    document.getElementById(id).checked = id !== "compact";
  });
  updateState();
}
window.addEventListener("pageshow", resetFixture);
resetFixture();
</script></main></body></html>
"""


def _scratchpad_html() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Adam Fixture Scratchpad</title>
<style>body{font:18px sans-serif;margin:3rem;max-width:40rem}label,textarea,button,output{display:block;margin:.75rem 0}textarea{width:100%;height:6rem;font:inherit}button{font:inherit;padding:.5rem 1rem}output{margin-top:2rem;font-weight:bold}</style>
</head><body><main><h1>Local Scratchpad</h1>
<label for="note-text">Note text field</label>
<textarea id="note-text"></textarea>
<button id="save-note" type="button">Save note</button>
<output id="save-result" aria-live="polite">Not saved</output>
<script>
const noteField = document.getElementById("note-text");
const saveResult = document.getElementById("save-result");
document.getElementById("save-note").addEventListener("click", function() {
  saveResult.textContent = noteField.value.trim() ? "Saved: " + noteField.value : "Not saved";
});
function resetFixture() {
  noteField.value = "";
  saveResult.textContent = "Not saved";
}
window.addEventListener("pageshow", resetFixture);
resetFixture();
</script></main></body></html>
"""


def _memory_fixture(run_date: date, timezone_name: str) -> dict[str, object]:
    previous_week_monday = run_date - timedelta(days=run_date.weekday() + 7)
    first_date = previous_week_monday + timedelta(days=1)
    second_date = previous_week_monday + timedelta(days=4)
    prior_year_date = date(run_date.year - 1, 1, 15)
    records = [
        f"I worked on the Juniper migration on {first_date.isoformat()} from 9:30 to 10:15am.",
        f"I worked on the Juniper invoice export on {second_date.isoformat()} from 3:00 to 4:00pm.",
        f"I worked on the Juniper migration on {prior_year_date.isoformat()} from 11:00am to noon.",
    ]
    return {
        "timezone": timezone_name,
        "run_date": run_date.isoformat(),
        "records": records,
        "expected_record_dates": [first_date.isoformat(), second_date.isoformat(), prior_year_date.isoformat()],
        "prompts": {
            "day": f"What did I work on {first_date.isoformat()}, and what local time?",
            "week": "What Juniper work did I log last week?",
            "year": "What Juniper work did I log last year?",
            "timezone": f"What timezone did I use for the {first_date.isoformat()} work entry?",
        },
    }


def create_fixtures(output_root: Path, run_id: str | None = None) -> Path:
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    fixture_id = run_id or secrets.token_hex(5)
    if (
        not fixture_id.isascii()
        or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in fixture_id)
    ):
        raise ValueError("run ID may contain only ASCII letters, numbers, hyphens, and underscores")
    fixture = root / fixture_id
    fixture.mkdir(parents=True, exist_ok=False, mode=0o700)
    fixture.chmod(0o700)

    rows = _invoice_rows(SEED)
    oracle = _invoice_oracle(rows)
    invoice_path = fixture / "invoice-report.html"
    source_path = fixture / "source.txt"
    document_path = fixture / "document.txt"
    work_order_path = fixture / "work-order.txt"
    settings_path = fixture / "settings.html"
    scratchpad_path = fixture / "scratchpad.html"
    _write(invoice_path, _invoice_html(rows))

    source_text = "ITEM: olive\nNOTE: keep this line\nITEM: azure\nITEM: amber\nFOOTER: unchanged\n"
    document_text = "STATUS: DRAFT\nOWNER: Adam\n"
    work_order_initial = (
        "WORK ORDER: overdue invoice review\n"
        "SUPPLIER: TBD\n"
        "INVOICE IDS: TBD\n"
        "OVERDUE TOTAL: TBD\n"
        "REVIEW: Pending\n"
        "OWNER: Adam\n"
    )
    work_order_final = (
        "WORK ORDER: overdue invoice review\n"
        f"SUPPLIER: {oracle['supplier']}\n"
        f"INVOICE IDS: {', '.join(oracle['invoice_ids'])}\n"
        f"OVERDUE TOTAL: {oracle['total_display']}\n"
        "REVIEW: Complete\n"
        "OWNER: Adam\n"
    )
    _write(source_path, source_text)
    _write(document_path, document_text)
    _write(work_order_path, work_order_initial)
    _write(settings_path, _settings_html())
    _write(scratchpad_path, _scratchpad_html())

    timezone_name = local_timezone_name() or "UTC"
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        zone = ZoneInfo("UTC")
        timezone_name = "UTC"
    now = datetime.now(zone)
    memory = _memory_fixture(now.date(), timezone_name)
    prompt_file_status = f'Read "{source_path.resolve()}" and tell me its byte size and exact first line.'
    prompt_terminal = (
        f'In "{fixture}", count lines in source.txt beginning with ITEM: and write only the count plus a newline '
        "to answer.txt. Do not change source.txt."
    )
    prompt_document = (
        f'In "{document_path.resolve()}", change DRAFT to FINAL and save it. '
        "Leave every other character unchanged."
    )
    prompt_mixed = (
        "Find the supplier with the largest overdue invoice total in the local fixture report. "
        f'Write exactly one line to "{fixture / "task-output.txt"}", replacing <supplier> and <$amount> '
        "with the values you found. Then report current CPU core count and memory-use percentage."
    )
    prompt_work_order = (
        "In the visible browser report at <LOCAL_REPORT_URL>, find the supplier with the largest "
        "overdue invoice total and all its invoice IDs. In the open work order document at "
        f'"{work_order_path.resolve()}", replace the TBD supplier, invoice IDs, and total with those '
        "values; change REVIEW from Pending to Complete. Leave every other character unchanged, "
        "save the document, reopen it, and report the saved values."
    )
    invoice_text_chars = sum(
        len(str(row[key])) for row in rows for key in ("invoice_id", "supplier", "status", "amount_cents")
    )
    metadata: dict[str, object] = {
        "fixture_version": FIXTURE_VERSION,
        "run_id": fixture_id,
        "seed": SEED,
        "created_at_local": now.isoformat(),
        "timezone": timezone_name,
        "paths": {
            "root": str(fixture),
            "invoice_report": str(invoice_path.resolve()),
            "settings_page": str(settings_path.resolve()),
            "scratchpad_page": str(scratchpad_path.resolve()),
            "source_file": str(source_path.resolve()),
            "document_file": str(document_path.resolve()),
            "work_order_file": str(work_order_path.resolve()),
        },
        "prompts": {
            "factual": "Compare rigatoni and penne in two short sentences.",
            "file_status": prompt_file_status,
            "system_status": (
                "According to a fresh system status check, how many logical CPU cores does this system have "
                "and what percentage of memory is in use?"
            ),
            "memory": memory["prompts"],
            "settings": "Set Compact mode to on, Sync to off, and Notifications to off.",
            "scratchpad": (
                'In the local scratchpad, enter "Call the dentist Tuesday at 2 pm" in the Note text field, '
                "click Save note, and tell me the saved text."
            ),
            "browser_template": (
                "Among overdue invoices at <LOCAL_REPORT_URL>, which supplier has the largest total? "
                "Give the supplier, invoice IDs, and total."
            ),
            "terminal": prompt_terminal,
            "document": prompt_document,
            "mixed": prompt_mixed,
            "work_order_template": prompt_work_order,
        },
        "oracles": {
            "file_status": {"byte_length": len(source_text.encode("utf-8")), "first_line": "ITEM: olive"},
            "terminal": {
                "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                "answer_text": "3\n",
            },
            "document": {"initial_text": document_text, "final_text": "STATUS: FINAL\nOWNER: Adam\n"},
            "work_order": {
                "initial_text": work_order_initial,
                "final_text": work_order_final,
                "invoice_report_sha256": hashlib.sha256(invoice_path.read_bytes()).hexdigest(),
            },
            "settings": {
                "initial": {"Compact": "off", "Sync": "on", "Notifications": "on"},
                "final": {"Compact": "on", "Sync": "off", "Notifications": "off"},
            },
            "scratchpad": {
                "initial_state": "Not saved",
                "saved_text": "Call the dentist Tuesday at 2 pm",
                "final_state": "Saved: Call the dentist Tuesday at 2 pm",
            },
            "invoices": {"row_count": len(rows), "body_text_char_estimate": invoice_text_chars, **oracle},
            "memory": memory,
            "mixed_output_line": (
                f"Largest overdue supplier: {oracle['supplier']}; total: {oracle['total_display']}\n"
            ),
        },
    }
    (fixture / "invoices.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write(fixture / "expected.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Parent directory for the new disposable fixture folder")
    parser.add_argument("--run-id", help="Optional stable fixture ID; the target folder must not already exist")
    args = parser.parse_args()
    fixture = create_fixtures(Path(args.output_root), args.run_id)
    print(f"Fixture directory: {fixture}")
    print(f"Expected state: {fixture / 'expected.json'}")
    print(f"Local report: {fixture / 'invoice-report.html'}")
    print(f"Local settings page: {fixture / 'settings.html'}")
    print(f"Local scratchpad page: {fixture / 'scratchpad.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
