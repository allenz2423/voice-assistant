"""Safety tests for Chromium AT-SPI observation scope."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.tools import atspi_reader, observe_desktop


class _State:
    def __init__(self, focused=False):
        self.focused = focused

    def contains(self, state):
        return self.focused and state == 12


class _Node:
    def __init__(self, role, name="", *, children=(), active=None, focused=False, link_active=True):
        self.role = role
        self.name = name
        self.children = list(children)
        self.active = active
        self.focused = focused
        self.parent = None
        self.active_calls = 0
        for child in self.children:
            child.parent = self
        if link_active and active is not None and active.parent is None:
            active.parent = self

    def get_role(self):
        return self.role

    def get_role_name(self):
        return self.role

    def get_name(self):
        return self.name

    def get_parent(self):
        return self.parent

    def get_active_descendant(self):
        self.active_calls += 1
        return self.active

    def get_child_count(self):
        return len(self.children)

    def get_child_at_index(self, index):
        return self.children[index]

    def get_state_set(self):
        return _State(self.focused)


_ATSPI = SimpleNamespace(
    Role=SimpleNamespace(DOCUMENT_WEB="web document"),
    StateType=SimpleNamespace(FOCUSED=12),
)


def test_active_web_document_follows_only_the_active_chain():
    unrelated_page = _Node("web document", "Sibling tab")
    document = _Node("web document", "Current page")
    renderer = _Node("panel", children=[document], active=document)
    window = _Node("frame", "Current window", children=[renderer, unrelated_page], active=renderer)

    assert atspi_reader._active_web_document(window, _ATSPI) is document
    assert unrelated_page.active_calls == 0


def test_browser_window_requires_one_exact_title_match():
    app = _Node("application", children=[
        _Node("frame", "Current page — Chromium"),
        _Node("frame", "Current page — Chromium"),
    ])

    with pytest.raises(atspi_reader._ScopeUnavailable, match="unique exact-title"):
        atspi_reader._browser_window(app, "Current page — Chromium")


def test_active_web_document_fails_closed_without_a_contained_chain():
    window = _Node("frame", "Current window", children=[_Node("panel", "Unfocused")])
    outside = _Node("panel", "Outside window", active=_Node("web document", "Other page"))
    escaped_window = _Node("frame", "Other window", active=outside, link_active=False)

    with pytest.raises(atspi_reader._ScopeUnavailable, match="unique focused"):
        atspi_reader._active_web_document(window, _ATSPI)
    with pytest.raises(atspi_reader._ScopeUnavailable, match="escapes"):
        atspi_reader._active_web_document(escaped_window, _ATSPI)


def test_browser_document_child_traversal_error_fails_closed(monkeypatch):
    class BrokenDocument(_Node):
        def get_action_iface(self):
            return None

        def get_text_iface(self):
            return None

        def get_child_count(self):
            return 1

        def get_child_at_index(self, index):
            raise RuntimeError("AT-SPI child request failed")

    document = BrokenDocument("web document", "Current page")
    window = _Node("frame", "Current window", active=document)
    app = _Node("application", children=[window])
    app.get_process_id = lambda: 4242
    desktop = _Node("desktop", children=[app])

    atspi = SimpleNamespace(
        Role=SimpleNamespace(DOCUMENT_WEB="web document"),
        StateType=SimpleNamespace(FOCUSED=12),
        init=lambda: None,
        get_desktop=lambda index: desktop,
    )
    gi = ModuleType("gi")
    gi.require_version = lambda name, version: None
    gi_repository = ModuleType("gi.repository")
    gi_repository.Atspi = atspi
    gi.repository = gi_repository
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", gi_repository)

    result = atspi_reader.read_tree(4242, "Current window", browser_document_only=True)

    assert result["ok"] is False
    assert result["error"].startswith("Scope unavailable:")
    assert "document traversal failed" in result["error"]


def test_chromium_detection_does_not_match_unrelated_app_names():
    assert observe_desktop._is_chromium_browser("Microsoft Edge")
    assert observe_desktop._is_chromium_browser("google-chrome")
    assert not observe_desktop._is_chromium_browser("Edgecase Editor")
