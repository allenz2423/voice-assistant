import asyncio
import pytest
import os
from pathlib import Path
from src.tools.custom import (
    CustomToolManager,
    CustomToolDefinition,
    CustomCommandStep
)

@pytest.mark.asyncio
async def test_sequential_execution_foreground(tmp_path):
    # Create test tool definition
    test_yaml = tmp_path / "tools.yaml"
    test_yaml.write_text("""
- name: "test_seq"
  description: "Test sequential commands"
  timeout: -1
  parameters:
    tag:
      type: string
      default: "alpha"
  commands:
    - "echo step1_{tag}"
    - "echo step2_{tag}"
""")
    mgr = CustomToolManager(config_path="nonexistent.yaml", custom_dirs=[str(tmp_path)])
    assert mgr.has_tool("test_seq")

    tool = mgr.tools["test_seq"]
    steps = tool.get_steps()
    assert len(steps) == 2
    assert steps[0].timeout == -1

    # Execute
    res = await mgr.execute("test_seq", {"tag": "beta"})
    assert "step2_beta" in res

@pytest.mark.asyncio
async def test_yaml_command_arguments_are_shell_quoted(tmp_path):
    marker = tmp_path / "should-not-exist"
    test_yaml = tmp_path / "quoted.yaml"
    test_yaml.write_text(f"""
- name: "quoted_argument"
  description: "Prints a supplied value literally"
  command: "printf '%s' {{value}}"
  parameters:
    value:
      type: string
""")
    mgr = CustomToolManager(config_path="nonexistent.yaml", custom_dirs=[str(tmp_path)])
    payload = f"safe'; touch {marker}; echo '"
    result = await mgr.execute("quoted_argument", {"value": payload})

    assert result == payload
    assert not marker.exists()

@pytest.mark.asyncio
async def test_timeout_user_specified(tmp_path):
    # Test positive timeout triggering
    test_yaml = tmp_path / "timeout.yaml"
    test_yaml.write_text("""
- name: "test_timeout"
  description: "Test timeout trigger"
  timeout: 1
  commands:
    - "sleep 3"
""")
    mgr = CustomToolManager(config_path="nonexistent.yaml", custom_dirs=[str(tmp_path)])
    res = await mgr.execute("test_timeout", {})
    assert "timed out after 1" in res

@pytest.mark.asyncio
async def test_timeout_negative_one_waits_until_done(tmp_path):
    # Test -1 waits until completion
    test_yaml = tmp_path / "no_timeout.yaml"
    test_yaml.write_text("""
- name: "test_no_timeout"
  description: "Test -1 wait until done"
  timeout: -1
  commands:
    - "sleep 0.2 && echo finished_ok"
""")
    mgr = CustomToolManager(config_path="nonexistent.yaml", custom_dirs=[str(tmp_path)])
    res = await mgr.execute("test_no_timeout", {})
    assert "finished_ok" in res

@pytest.mark.asyncio
async def test_sequential_halts_on_failure(tmp_path):
    marker = tmp_path / "marker.txt"
    test_yaml = tmp_path / "fail.yaml"
    test_yaml.write_text(f"""
- name: "test_fail"
  description: "Test failure halts sequence"
  commands:
    - "false"
    - "touch {marker}"
""")
    mgr = CustomToolManager(config_path="nonexistent.yaml", custom_dirs=[str(tmp_path)])
    res = await mgr.execute("test_fail", {})
    assert "failed with code" in res
    assert not marker.exists()

@pytest.mark.asyncio
async def test_background_execution_announcement(tmp_path):
    announced = []

    class MockArbiter:
        async def enqueue_notification(self, priority, message):
            announced.append(message)

    class MockTTS:
        async def speak_async(self, text):
            announced.append(f"Spoken: {text}")

    test_yaml = tmp_path / "bg.yaml"
    test_yaml.write_text("""
- name: "test_bg"
  description: "Test background execution"
  background: true
  waiting_message: "Waiting for compile of {project}..."
  success_message: "{project} compilation finished."
  parameters:
    project:
      type: string
      default: "shin"
  commands:
    - "sleep 0.3"
""")
    mgr = CustomToolManager(config_path="nonexistent.yaml", custom_dirs=[str(tmp_path)])
    mock_arbiter = MockArbiter()
    mock_tts = MockTTS()

    res = await mgr.execute("test_bg", {"project": "desky"}, arbiter=mock_arbiter, tts=mock_tts)
    assert "Waiting for compile of desky" in res
    assert "Spoken: Waiting for compile of desky..." in announced

    # Wait for background task to finish and announce
    await asyncio.sleep(0.6)
    assert any("desky compilation finished." in m for m in announced)

def test_canonical_tool_generation(tmp_path):
    test_yaml = tmp_path / "schema.yaml"
    test_yaml.write_text("""
- name: "deploy_app"
  description: "Deploys a container"
  parameters:
    target:
      type: string
      description: "Target cluster"
      enum: ["prod", "staging"]
  command: "echo deploying"
""")
    mgr = CustomToolManager(config_path="nonexistent.yaml", custom_dirs=[str(tmp_path)])
    tools = mgr.get_canonical_tools()
    assert len(tools) == 1
    t = tools[0]
    assert t.name == "deploy_app"
    assert "prod" in t.parameters["properties"]["target"]["enum"]
    assert "target" in t.parameters["required"]

@pytest.mark.asyncio
async def test_python_tool_sync_and_async(tmp_path):
    py_tool = tmp_path / "my_tools.py"
    py_tool.write_text("""
from src.tools.custom import tool

@tool(description="Calculates compound growth")
def compound_interest(principal: float, rate: float = 0.05, years: int = 1) -> str:
    total = principal * ((1 + rate) ** years)
    return f"Total after {years} years: {total:.2f}"

@tool(description="Async simulated query", timeout=-1)
async def query_cluster(cluster_id: str) -> str:
    return f"Cluster {cluster_id} is healthy"
""")
    mgr = CustomToolManager(config_path="nonexistent.yaml", custom_dirs=[str(tmp_path)])
    assert mgr.has_tool("compound_interest")
    assert mgr.has_tool("query_cluster")

    # Verify schema
    tools = {t.name: t for t in mgr.get_canonical_tools()}
    ci_tool = tools["compound_interest"]
    assert ci_tool.parameters["properties"]["principal"]["type"] == "number"
    assert ci_tool.parameters["properties"]["years"]["type"] == "integer"
    assert "principal" in ci_tool.parameters["required"]
    assert "rate" not in ci_tool.parameters["required"]  # has default

    # Execute sync
    res_sync = await mgr.execute("compound_interest", {"principal": 1000.0, "rate": 0.10, "years": 2})
    assert "1210.00" in res_sync

    # Execute async
    res_async = await mgr.execute("query_cluster", {"cluster_id": "prod-east"})
    assert "Cluster prod-east is healthy" in res_async

@pytest.mark.asyncio
async def test_python_tool_background_execution(tmp_path):
    announced = []

    class MockArbiter:
        async def enqueue_notification(self, priority, message):
            announced.append(message)

    py_tool = tmp_path / "bg_tool.py"
    py_tool.write_text("""
import asyncio
from src.tools.custom import tool

@tool(
    description="Background train model",
    background=True,
    waiting_message="Starting training for {epochs} epochs...",
    success_message="Model trained successfully for {epochs} epochs."
)
async def train_model(epochs: int = 10) -> str:
    await asyncio.sleep(0.3)
    return "done"
""")
    mgr = CustomToolManager(config_path="nonexistent.yaml", custom_dirs=[str(tmp_path)])
    mock_arbiter = MockArbiter()

    res = await mgr.execute("train_model", {"epochs": 50}, arbiter=mock_arbiter)
    assert "Starting training for 50 epochs" in res

    await asyncio.sleep(0.6)
    assert any("Model trained successfully for 50 epochs." in m for m in announced)

@pytest.mark.asyncio
async def test_detached_execution_does_not_block_or_timeout(tmp_path):
    import time
    test_yaml = tmp_path / "detached.yaml"
    test_yaml.write_text("""
- name: "detached_tool"
  description: "Test detach flag"
  detach: true
  timeout: 1
  command: "sleep 5"
  success_message: "Launched detached."

- name: "ampersand_tool"
  description: "Test trailing ampersand"
  timeout: 1
  command: "sleep 5 &"
""")
    mgr = CustomToolManager(config_path="nonexistent.yaml", custom_dirs=[str(tmp_path)])

    start = time.monotonic()
    res1 = await mgr.execute("detached_tool", {})
    elapsed1 = time.monotonic() - start
    assert elapsed1 < 0.5
    assert "Launched detached." in res1
    assert "timed out" not in res1

    start = time.monotonic()
    res2 = await mgr.execute("ampersand_tool", {})
    elapsed2 = time.monotonic() - start
    assert elapsed2 < 0.5
    assert "Started" in res2
    assert "timed out" not in res2

