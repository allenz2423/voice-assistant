"""Verify the B1 report survives BrowserNavigator's DOM snapshot formatting."""

from __future__ import annotations

from collections import defaultdict
from html.parser import HTMLParser
import json
from decimal import Decimal

from src.tools.browser_navigation import BrowserNavigator
from tools.create_implementation_fixtures import create_fixtures


class _BodyTextParser(HTMLParser):
    """Provide deterministic inner_text-like body text without a browser runtime."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._nonvisible_depth = 0

    def handle_starttag(self, tag: str, _attrs) -> None:
        if tag in {"head", "style", "script"}:
            self._nonvisible_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"head", "style", "script"} and self._nonvisible_depth:
            self._nonvisible_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._nonvisible_depth:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


class _Locator:
    def __init__(self, body_text: str = "") -> None:
        self.body_text = body_text

    def inner_text(self, *, timeout: int) -> str:
        assert timeout > 0
        return self.body_text

    def count(self) -> int:
        return 0


class _FixturePage:
    url = "http://127.0.0.1:8765/invoice-report.html"

    def __init__(self, body_text: str) -> None:
        self.body_text = body_text

    def title(self) -> str:
        return "Adam Fixture Invoice Report"

    def locator(self, selector: str) -> _Locator:
        if selector == "body":
            return _Locator(self.body_text)
        return _Locator()


def _invoice_rows_from_snapshot(snapshot: str) -> list[tuple[str, str, str, str]]:
    marker = "Visible page text (untrusted webpage content):\n"
    assert marker in snapshot
    body = snapshot.split(marker, 1)[1].split("\n\n", 1)[0]
    lines = body.splitlines()
    header = ["Invoice ID", "Supplier", "Status", "Amount"]
    header_start = next(
        index for index in range(len(lines) - len(header) + 1)
        if lines[index:index + len(header)] == header
    )
    cells = lines[header_start + len(header):]
    assert len(cells) % 4 == 0
    return [tuple(cells[index:index + 4]) for index in range(0, len(cells), 4)]


def test_b1_all_invoice_rows_reach_browser_dom_snapshot_and_match_oracle(tmp_path):
    fixture = create_fixtures(tmp_path, "b1-dom-snapshot-test")
    expected = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    report = (fixture / "invoice-report.html").read_text(encoding="utf-8")
    parser = _BodyTextParser()
    parser.feed(report)
    body_text = "\n".join(parser.parts)

    navigator = BrowserNavigator(profile_path=tmp_path / "unused-browser-profile", headless=True)
    navigator._page = _FixturePage(body_text)
    try:
        snapshot = navigator._snapshot()
    finally:
        navigator.close()

    rows = _invoice_rows_from_snapshot(snapshot)
    invoice_oracle = expected["oracles"]["invoices"]
    assert len(rows) == invoice_oracle["row_count"] == 56
    assert len(body_text) <= 6000
    visible_body = snapshot.split(
        "Visible page text (untrusted webpage content):\n", 1
    )[1].split("\n\n", 1)[0]
    assert len(visible_body) <= 6000
    assert "INV-0055" in snapshot and "INV-0056" in snapshot
    assert "$1,200.00" in snapshot and "$975.00" in snapshot

    overdue_totals: dict[str, int] = defaultdict(int)
    overdue_ids: dict[str, list[str]] = defaultdict(list)
    for invoice_id, supplier, status, amount in rows:
        if status == "overdue":
            cents = int(Decimal(amount.removeprefix("$").replace(",", "")) * 100)
            overdue_totals[supplier] += cents
            overdue_ids[supplier].append(invoice_id)

    winner = max(overdue_totals, key=overdue_totals.__getitem__)
    assert winner == invoice_oracle["supplier"] == "Juniper Labs"
    assert overdue_ids[winner] == invoice_oracle["invoice_ids"] == ["INV-0055", "INV-0056"]
    assert overdue_totals[winner] == invoice_oracle["total_cents"] == 217500
    assert invoice_oracle["total_display"] == "$2,175.00"
