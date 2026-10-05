"""Provider-free X1 mixed-task dispatch using filesystem and synthetic status.

The report is read as a fixture file because the optional isolated-browser route
is disabled in the example configuration. This covers Brain routing, file
read/create dispatch, and status formatting; it does not test browser control,
model planning, or full X1 completion.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import re
from collections import defaultdict
from decimal import Decimal
from html.parser import HTMLParser
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _InvoiceRowsParser(HTMLParser):
    """Extract table cells from the rendered report, not invoices.json."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        clean = data.strip()
        if clean and self._cell is not None:
            self._cell.append(clean)

    def handle_endtag(self, tag) -> None:
        if tag in {"th", "td"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _rendered_report_summary(html: str) -> tuple[int, str, int]:
    parser = _InvoiceRowsParser()
    parser.feed(html)
    assert parser.rows[0] == ["Invoice ID", "Supplier", "Status", "Amount"]
    rows = parser.rows[1:]
    totals: dict[str, int] = defaultdict(int)
    for invoice_id, supplier, status, amount in rows:
        assert invoice_id.startswith("INV-")
        if status == "overdue":
            cents = int(Decimal(amount.removeprefix("$").replace(",", "")) * 100)
            totals[supplier] += cents
    supplier = max(totals, key=totals.__getitem__)
    return len(rows), supplier, totals[supplier]


def _source_rows_summary(rows: list[dict]) -> tuple[int, str, int]:
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["status"] == "overdue":
            totals[str(row["supplier"])] += int(row["amount_cents"])
    supplier = max(totals, key=totals.__getitem__)
    return len(rows), supplier, totals[supplier]


def _outer_tool_result(messages: list[dict], call_id: str) -> dict:
    message = next(
        item for item in messages
        if item.get("role") == "tool" and item.get("tool_call_id") == call_id
    )
    result = json.loads(message["content"])
    assert result["call_id"] == call_id
    assert result["status"] == "returned"
    assert result["goal_status"] == "not_assessed"
    return result


def _report_html_from_result(messages: list[dict], call_id: str, report_path: str) -> str:
    outer = _outer_tool_result(messages, call_id)
    inner = json.loads(outer["data"])
    assert inner["ok"] is True
    readback = inner["readback"]
    assert readback.startswith(f"File: {report_path}\n")
    assert "[truncated]" not in readback
    header, html = readback.split("\nText:\n", 1)
    assert f"Bytes: {len(html.encode('utf-8'))}" in header
    return html


class _ScriptedMixedClient:
    """Derive the create call from Brain's actual read_file result."""

    provider = "local"

    def __init__(self, report_path: str, output_path: str, repetition: int) -> None:
        self.report_path = report_path
        self.output_path = output_path
        self.call_ids = {
            "read": f"x1-read-{repetition}",
            "create": f"x1-create-{repetition}",
            "status": f"x1-status-{repetition}",
        }
        self.requests: list[dict] = []
        self.available_tool_names: set[str] = set()
        self.rendered_summary: tuple[int, str, int] | None = None
        self.output_line: str | None = None

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        self.requests.append({"messages": messages, "tools": tools})
        turn = len(self.requests)
        if turn == 1:
            self.available_tool_names = {tool.name for tool in tools or []}
            return {
                "content": "",
                "tool_calls": [{
                    "id": self.call_ids["read"],
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": self.report_path},
                    },
                }],
            }
        if turn == 2:
            rendered_html = _report_html_from_result(
                messages, self.call_ids["read"], self.report_path,
            )
            self.rendered_summary = _rendered_report_summary(rendered_html)
            row_count, supplier, total_cents = self.rendered_summary
            assert row_count == 56
            dollars, cents = divmod(total_cents, 100)
            self.output_line = (
                f"Largest overdue supplier: {supplier}; total: ${dollars:,}.{cents:02d}\n"
            )
            return {
                "content": "",
                "tool_calls": [{
                    "id": self.call_ids["create"],
                    "function": {
                        "name": "create_file",
                        "arguments": {
                            "path": self.output_path,
                            "content": self.output_line,
                        },
                    },
                }],
            }
        if turn == 3:
            created = _outer_tool_result(messages, self.call_ids["create"])
            assert created["data"].startswith(f"Created {self.output_path} (")
            return {
                "content": "",
                "tool_calls": [{
                    "id": self.call_ids["status"],
                    "function": {
                        "name": "get_system_status",
                        "arguments": {},
                    },
                }],
            }
        if turn == 4:
            status = _outer_tool_result(messages, self.call_ids["status"])["data"]
            cores = re.search(r"CPU has (\d+) logical cores", status)
            memory = re.search(r"Memory is (\d+) percent in use", status)
            assert cores is not None and memory is not None
            return {
                "content": (
                    f"I created the requested output line. The status check reports "
                    f"{cores.group(1)} logical CPU cores and {memory.group(1)} percent "
                    "memory use."
                ),
                "tool_calls": [],
            }
        raise AssertionError(f"Unexpected model turn {turn}.")

    def format_tool_response(self, tool_call_id, tool_name, result):
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }


class _SilentTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak_async(self, text: str) -> None:
        self.spoken.append(text)


class _NoCustomTools:
    """Keep host and repository custom-tool discovery out of this fixture."""

    tools: dict = {}

    def has_tool(self, _name: str) -> bool:
        return False

    def get_canonical_tools(self) -> list:
        return []


def _test_config():
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="local",
            local_model="test-model",
            cloud_model="test-model",
            ollama_host="http://localhost:11434",
            api_base="https://example.test/v1",
            api_key="",
            temperature=0,
            num_ctx=8192,
            max_tool_rounds=8,
        ),
        browser_navigation=SimpleNamespace(enabled=False),
        computer_control=SimpleNamespace(enabled=False),
        computer_vision=SimpleNamespace(enabled=False),
        desktop=SimpleNamespace(default_browser="fixture-browser"),
    )


@pytest.mark.asyncio
async def test_mixed_fixture_dispatches_report_file_creation_and_synthetic_status(
    tmp_path, monkeypatch,
):
    from src.llm.brain import AdamBrain
    from src.tools import system_telemetry
    from tools.create_implementation_fixtures import create_fixtures

    required_tools = {"read_file", "create_file", "get_system_status"}
    monkeypatch.setattr(
        "src.llm.brain.is_tool_enabled",
        lambda name: name in required_tools,
    )
    real_open = builtins.open
    memory = SimpleNamespace(retrieve_context=lambda _text: None)

    synthetic_cpu_count = 8
    synthetic_cpuinfo = "".join(
        f"processor\t: {index}\n\n" for index in range(synthetic_cpu_count)
    )
    mem_total_kib = 16_000_000
    mem_available_kib = 5_760_000
    synthetic_proc = {
        "/proc/stat": "",
        "/proc/loadavg": "0.10 0.20 0.30 1/100 1234\n",
        "/proc/cpuinfo": synthetic_cpuinfo,
        "/proc/meminfo": (
            f"MemTotal:       {mem_total_kib} kB\n"
            f"MemAvailable:    {mem_available_kib} kB\n"
        ),
    }
    expected_memory_percent = round(
        (mem_total_kib - mem_available_kib) / mem_total_kib * 100
    )
    assert expected_memory_percent == 64

    for repetition in range(1, 4):
        fixture = create_fixtures(
            tmp_path / "fixtures", f"mixed-brain-replay-{repetition}"
        )
        expected = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
        request = expected["prompts"]["mixed"]
        report_path = expected["paths"]["invoice_report"]
        output_path = str((fixture / "task-output.txt").resolve())
        report_file = fixture / "invoice-report.html"
        output_file = fixture / "task-output.txt"
        report_before = report_file.read_bytes()
        report_sha256 = hashlib.sha256(report_before).hexdigest()
        assert report_sha256 == expected["oracles"]["work_order"]["invoice_report_sha256"]
        assert not output_file.exists()

        invoice_rows = json.loads((fixture / "invoices.json").read_text(encoding="utf-8"))
        independent_summary = _source_rows_summary(invoice_rows)
        invoice_oracle = expected["oracles"]["invoices"]
        assert independent_summary == (
            invoice_oracle["row_count"],
            invoice_oracle["supplier"],
            invoice_oracle["total_cents"],
        ) == (56, "Juniper Labs", 217500)
        expected_output = expected["oracles"]["mixed_output_line"].encode("utf-8")
        assert expected_output == b"Largest overdue supplier: Juniper Labs; total: $2,175.00\n"

        client = _ScriptedMixedClient(report_path, output_path, repetition)
        tts = _SilentTTS()
        with patch(
            "src.llm.brain.CustomToolManager", return_value=_NoCustomTools(),
        ):
            brain = AdamBrain(
                _test_config(), None, None, None, tts, memory_mgr=memory,
            )
        brain.computer_controller = SimpleNamespace(
            available=False,
            drag_active=False,
            coordinate_mode="pixels",
            invalidate_snapshot=lambda: None,
        )
        brain.llm_client = client

        proc_reads: list[str] = []

        def open_synthetic_proc(path, mode="r", *args, **kwargs):
            path_text = os.fspath(path)
            if path_text.startswith("/proc/"):
                if path_text not in synthetic_proc or mode != "r":
                    raise AssertionError(f"Unexpected /proc access: {path_text} ({mode})")
                proc_reads.append(path_text)
                return io.StringIO(synthetic_proc[path_text])
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(system_telemetry, "open", open_synthetic_proc, raising=False)
        monkeypatch.setattr(
            system_telemetry,
            "os",
            SimpleNamespace(statvfs=lambda _path: SimpleNamespace(
                f_blocks=2, f_frsize=1024**3, f_bavail=1,
            )),
        )
        monkeypatch.setattr(
            system_telemetry, "shutil", SimpleNamespace(which=lambda _name: None)
        )

        with patch(
            "src.llm.brain.get_open_windows_prompt_context",
            return_value="Synthetic fixture context",
        ):
            await brain.process_user_utterance(request)

        assert len(client.requests) == 4
        assert client.available_tool_names == required_tools
        assert "browser_navigation" not in client.available_tool_names
        assert brain.browser_navigator is None
        assert sorted(proc_reads) == sorted(synthetic_proc)
        assert client.rendered_summary == independent_summary

        assistant_calls = [
            call for message in brain.messages
            if message.get("role") == "assistant"
            for call in message.get("tool_calls", [])
        ]
        assert [call["function"]["name"] for call in assistant_calls] == [
            "read_file", "create_file", "get_system_status",
        ]
        call_args = []
        for call in assistant_calls:
            arguments = call["function"]["arguments"]
            call_args.append(json.loads(arguments) if isinstance(arguments, str) else arguments)
        assert call_args[0] == {"path": report_path}
        assert call_args[1] == {"path": output_path, "content": expected_output.decode("utf-8")}
        assert call_args[2] == {}

        tool_messages = [
            message for message in brain.messages if message.get("role") == "tool"
        ]
        assert [message["name"] for message in tool_messages] == [
            "read_file", "create_file", "get_system_status",
        ]
        read_result = _outer_tool_result(brain.messages, client.call_ids["read"])
        read_inner = json.loads(read_result["data"])
        assert read_inner["ok"] is True
        assert read_inner["readback"].startswith(
            f"File: {report_path}\nBytes: {len(report_before)}\nText:\n"
        )

        create_result = _outer_tool_result(brain.messages, client.call_ids["create"])
        assert create_result["data"] == f"Created {output_path} ({len(expected_output)} bytes)."
        assert output_file.read_bytes() == expected_output
        assert hashlib.sha256(output_file.read_bytes()).hexdigest() == hashlib.sha256(
            expected_output
        ).hexdigest()

        status_result = _outer_tool_result(brain.messages, client.call_ids["status"])
        status_text = status_result["data"]
        assert status_text == (
            "CPU has 8 logical cores with load average 0.10, 0.20, 0.30. "
            "Memory is 64 percent in use (9.8 gigabytes used out of 15.3 gigabytes). "
            "Root storage has 1.0 gigabytes free out of 2.0 gigabytes. "
            "GPU telemetry is unavailable; no supported GPU driver returned data. "
            "CPU utilization percentage is unavailable; CPU load average is reported above."
        )
        assert _outer_tool_result(brain.messages, client.call_ids["read"])["status"] == "returned"
        assert create_result["status"] == status_result["status"] == "returned"
        assert tts.spoken[-1] == brain.messages[-1]["content"]
        assert "8 logical CPU cores and 64 percent memory use" in tts.spoken[-1]

        report_after = report_file.read_bytes()
        assert report_after == report_before
        assert hashlib.sha256(report_after).hexdigest() == report_sha256
