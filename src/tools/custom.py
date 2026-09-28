import os
import re
import shlex
import inspect
import asyncio
import importlib.util
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional, Union, get_origin, get_args
from pydantic import BaseModel, Field
import yaml

from src.llm.tools import CanonicalTool
from src.tools.desktop import ensure_gui_environment

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    confirm: bool = False,
    background: bool = False,
    timeout: int = -1,
    waiting_message: Optional[str] = None,
    success_message: Optional[str] = None
):
    """Decorator to mark a Python function as a user-defined tool for Shin."""
    def decorator(fn: Callable):
        fn.__shin_tool__ = {
            "name": name or fn.__name__,
            "description": description or (inspect.getdoc(fn) or f"Executes {fn.__name__}").split("\n")[0].strip(),
            "confirm": confirm,
            "background": background,
            "timeout": timeout,
            "waiting_message": waiting_message,
            "success_message": success_message,
            "handler": fn
        }
        return fn
    return decorator

class CustomCommandStep(BaseModel):
    """A single command step in a sequential execution chain."""
    run: str
    timeout: int = -1  # -1 means wait indefinitely until done; >0 is timeout in seconds
    detach: bool = False  # True means launch detached in background (do not block for GUI window exit)
    continue_on_error: bool = False
    waiting_message: Optional[str] = None

class CustomToolDefinition(BaseModel):
    """Declarative specification for a user-defined YAML command tool."""
    name: str
    description: str
    confirm: bool = False
    background: bool = False
    detach: bool = False  # True means spawn process detached without waiting for exit (for GUI apps/tools)
    timeout: int = -1  # -1 means wait indefinitely until done
    waiting_message: Optional[str] = None
    success_message: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    command: Optional[str] = None
    commands: list[Union[str, dict[str, Any]]] = Field(default_factory=list)

    def get_steps(self) -> list[CustomCommandStep]:
        """Normalizes single 'command' or list of 'commands' into a list of CustomCommandStep."""
        steps = []
        if self.command:
            steps.append(CustomCommandStep(run=self.command, timeout=self.timeout, detach=self.detach))

        for cmd in self.commands:
            if isinstance(cmd, str):
                steps.append(CustomCommandStep(run=cmd, timeout=self.timeout, detach=self.detach))
            elif isinstance(cmd, dict):
                steps.append(CustomCommandStep(
                    run=cmd.get("run") or cmd.get("command") or "",
                    timeout=cmd.get("timeout", self.timeout),
                    detach=cmd.get("detach", self.detach),
                    continue_on_error=cmd.get("continue_on_error", False),
                    waiting_message=cmd.get("waiting_message")
                ))
        return steps

    def to_canonical_tool(self) -> CanonicalTool:
        """Converts this custom tool definition to a CanonicalTool schema for LLM function calling."""
        props = {}
        required = []

        for p_name, p_spec in self.parameters.items():
            if isinstance(p_spec, dict):
                prop_entry = {
                    "type": p_spec.get("type", "string"),
                    "description": p_spec.get("description", f"Parameter {p_name}")
                }
                if "enum" in p_spec:
                    prop_entry["enum"] = p_spec["enum"]
                props[p_name] = prop_entry
                if not p_spec.get("default") and p_spec.get("required", True):
                    required.append(p_name)
            else:
                props[p_name] = {"type": "string", "description": f"Parameter {p_name}"}

        schema = {
            "type": "object",
            "properties": props
        }
        if required:
            schema["required"] = required

        return CanonicalTool(
            name=self.name,
            description=self.description,
            parameters=schema
        )

class CustomPythonToolDefinition:
    """Wraps a user-defined Python function into a CanonicalTool with automatic schema reflection."""
    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable,
        confirm: bool = False,
        background: bool = False,
        timeout: int = -1,
        waiting_message: Optional[str] = None,
        success_message: Optional[str] = None
    ):
        self.name = name
        self.description = description
        self.handler = handler
        self.confirm = confirm
        self.background = background
        self.timeout = timeout
        self.waiting_message = waiting_message
        self.success_message = success_message
        self.schema = self._extract_schema(handler)

    def _extract_schema(self, fn: Callable) -> dict[str, Any]:
        """Extracts JSON schema properties from Python type hints and docstrings."""
        sig = inspect.signature(fn)
        props = {}
        required = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        for param_name, param in sig.parameters.items():
            if param_name in ["self", "cls"]:
                continue

            param_type = "string"
            annotation = param.annotation
            origin = get_origin(annotation)
            args = get_args(annotation)

            enum_values = None
            if str(origin) == "typing.Literal" or origin is getattr(typing, "Literal", None) if "typing" in globals() else False:
                enum_values = list(args)
                param_type = "string"
            elif annotation in type_map:
                param_type = type_map[annotation]
            elif origin in type_map:
                param_type = type_map[origin]
            elif origin is Union and type(None) in args:
                # Optional[T]
                non_none = [a for a in args if a is not type(None)]
                if non_none and non_none[0] in type_map:
                    param_type = type_map[non_none[0]]

            p_def: dict[str, Any] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            if enum_values:
                p_def["enum"] = enum_values

            props[param_name] = p_def

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": props,
            "required": required
        }

    def to_canonical_tool(self) -> CanonicalTool:
        return CanonicalTool(
            name=self.name,
            description=self.description,
            parameters=self.schema
        )

class CustomToolManager:
    """Manages loading, schema registration, and execution of user-defined YAML tools and Python functions."""
    def __init__(self, config_path: str = "config.yaml", custom_dirs: Optional[list[str]] = None):
        self.config_path = config_path
        self.custom_dirs = custom_dirs or [
            "~/.config/shin/tools",
            "custom_tools"
        ]
        self.tools: dict[str, Union[CustomToolDefinition, CustomPythonToolDefinition]] = {}
        self.reload()

    def reload(self):
        """Scans config.yaml, custom tool directories, and Python modules for tool definitions."""
        self.tools.clear()

        # 1. Load from config.yaml
        try:
            cfg_p = Path(self.config_path).expanduser()
            if cfg_p.exists():
                with open(cfg_p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                raw_tools = data.get("custom_tools", [])
                for item in raw_tools:
                    if isinstance(item, dict) and "name" in item:
                        self.tools[item["name"]] = CustomToolDefinition(**item)
        except Exception as e:
            print(f"[CustomTools] Warning: Error parsing {self.config_path}: {e}")

        # 2. Load from tool directories (*.yaml, *.yml, and *.py)
        for dir_path in self.custom_dirs:
            p = Path(dir_path).expanduser()
            if not p.exists() or not p.is_dir():
                continue

            # Load YAML tools
            for f_path in list(p.glob("*.yaml")) + list(p.glob("*.yml")):
                try:
                    with open(f_path, "r", encoding="utf-8") as f:
                        doc = yaml.safe_load(f)
                    if isinstance(doc, list):
                        for item in doc:
                            if isinstance(item, dict) and "name" in item:
                                self.tools[item["name"]] = CustomToolDefinition(**item)
                    elif isinstance(doc, dict):
                        if "name" in doc:
                            self.tools[doc["name"]] = CustomToolDefinition(**doc)
                        elif "custom_tools" in doc:
                            for item in doc["custom_tools"]:
                                if isinstance(item, dict) and "name" in item:
                                    self.tools[item["name"]] = CustomToolDefinition(**item)
                except Exception as e:
                    print(f"[CustomTools] Warning: Error loading {f_path}: {e}")

            # Load Python tools (*.py)
            for py_path in p.glob("*.py"):
                try:
                    mod_name = f"shin_custom_tool_{py_path.stem}"
                    spec = importlib.util.spec_from_file_location(mod_name, str(py_path))
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        for attr_name in dir(mod):
                            obj = getattr(mod, attr_name)
                            if callable(obj) and hasattr(obj, "__shin_tool__"):
                                meta = obj.__shin_tool__
                                self.tools[meta["name"]] = CustomPythonToolDefinition(
                                    name=meta["name"],
                                    description=meta["description"],
                                    handler=obj,
                                    confirm=meta.get("confirm", False),
                                    background=meta.get("background", False),
                                    timeout=meta.get("timeout", -1),
                                    waiting_message=meta.get("waiting_message"),
                                    success_message=meta.get("success_message")
                                )
                            elif callable(obj) and attr_name.startswith("tool_"):
                                tool_name = attr_name[5:]
                                doc = (inspect.getdoc(obj) or f"Custom Python tool {tool_name}").split("\n")[0].strip()
                                self.tools[tool_name] = CustomPythonToolDefinition(
                                    name=tool_name,
                                    description=doc,
                                    handler=obj
                                )
                except Exception as e:
                    print(f"[CustomTools] Warning: Error loading Python tool {py_path}: {e}")

        if self.tools:
            print(f"[CustomTools] Registered {len(self.tools)} custom tool(s): {', '.join(self.tools.keys())}")

    def get_canonical_tools(self) -> list[CanonicalTool]:
        """Returns the list of CanonicalTool schemas for all registered custom tools."""
        return [tool.to_canonical_tool() for tool in self.tools.values()]

    def has_tool(self, name: str) -> bool:
        """Returns True if the tool name matches a registered custom tool."""
        return name in self.tools

    def interpolate_params(self, template: str, args: dict[str, Any], tool_def: Any) -> str:
        """Interpolates arguments and parameter defaults into command strings."""
        merged_args = {}
        params = getattr(tool_def, "parameters", {})
        if isinstance(params, dict):
            for p_name, p_spec in params.items():
                if isinstance(p_spec, dict) and "default" in p_spec:
                    merged_args[p_name] = p_spec["default"]
        merged_args.update(args or {})

        res = template
        for k, v in merged_args.items():
            res = res.replace(f"{{{k}}}", str(v))
        return res

    def interpolate_command_params(self, template: str, args: dict[str, Any], tool_def: Any) -> str:
        """Interpolate YAML command arguments as shell-quoted values."""
        merged_args = {}
        params = getattr(tool_def, "parameters", {})
        if isinstance(params, dict):
            for p_name, p_spec in params.items():
                if isinstance(p_spec, dict) and "default" in p_spec:
                    merged_args[p_name] = p_spec["default"]
        merged_args.update(args or {})

        res = template
        for key, value in merged_args.items():
            res = res.replace(f"{{{key}}}", shlex.quote(str(value)))
        return res

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        arbiter: Any = None,
        tts: Any = None,
        confirmation: Any = None
    ) -> str:
        """Executes a custom tool (YAML sequence or Python function)."""
        tool = self.tools.get(name)
        if not tool:
            return f"Custom tool '{name}' not found."

        # 1. Voice Confirmation check if confirm=True
        if tool.confirm and confirmation:
            prompt = f"Are you sure you want to run {tool.name.replace('_', ' ')}?"
            confirmed = await confirmation.request_confirmation(prompt)
            if not confirmed:
                return f"Action '{name}' was cancelled by user."

        # 2. Python Function Execution Path
        if isinstance(tool, CustomPythonToolDefinition):
            return await self._execute_python_tool(tool, args, arbiter, tts)

        # 3. YAML Sequential Execution Path
        return await self._execute_yaml_tool(tool, args, arbiter, tts)

    async def _execute_python_tool(
        self,
        tool: CustomPythonToolDefinition,
        args: dict[str, Any],
        arbiter: Any,
        tts: Any
    ) -> str:
        """Executes a custom Python tool (sync or async) with background and timeout support."""
        waiting_msg = None
        if tool.waiting_message:
            waiting_msg = self.interpolate_params(tool.waiting_message, args, tool)

        success_msg = None
        if tool.success_message:
            success_msg = self.interpolate_params(tool.success_message, args, tool)

        # Background mode for Python functions
        if tool.background:
            initial_msg = waiting_msg or f"Started {tool.name.replace('_', ' ')} in the background. Waiting for completion."
            if tts:
                await tts.speak_async(initial_msg)

            asyncio.create_task(
                self._run_python_background(
                    tool=tool,
                    args=args,
                    success_msg=success_msg,
                    arbiter=arbiter,
                    tts=tts
                )
            )
            return initial_msg

        # Foreground mode for Python functions
        return await self._run_python_foreground(tool, args, success_msg)

    async def _run_python_foreground(
        self,
        tool: CustomPythonToolDefinition,
        args: dict[str, Any],
        success_msg: Optional[str]
    ) -> str:
        """Runs a Python function foreground, respecting timeout=-1 (wait until done)."""
        print(f"[CustomTool:{tool.name}:Python] Running with args: {args}")
        try:
            # Match function signature arguments
            sig = inspect.signature(tool.handler)
            filtered_args = {}
            for param_name, param in sig.parameters.items():
                if param_name in args:
                    filtered_args[param_name] = args[param_name]
                elif param.default is not inspect.Parameter.empty:
                    filtered_args[param_name] = param.default

            if inspect.iscoroutinefunction(tool.handler):
                coro = tool.handler(**filtered_args)
                if tool.timeout == -1 or tool.timeout is None:
                    res = await coro
                else:
                    res = await asyncio.wait_for(coro, timeout=float(tool.timeout))
            else:
                func_call = lambda: tool.handler(**filtered_args)
                if tool.timeout == -1 or tool.timeout is None:
                    res = await asyncio.to_thread(func_call)
                else:
                    res = await asyncio.wait_for(asyncio.to_thread(func_call), timeout=float(tool.timeout))

            return success_msg or str(res) if res is not None else "Done."
        except asyncio.TimeoutError:
            err = f"Custom Python tool '{tool.name}' timed out after {tool.timeout}s."
            print(f"[CustomTool:{tool.name}] {err}")
            return err
        except Exception as e:
            err = f"Error in custom tool '{tool.name}': {e}"
            print(f"[CustomTool:{tool.name}] {err}")
            return err

    async def _run_python_background(
        self,
        tool: CustomPythonToolDefinition,
        args: dict[str, Any],
        success_msg: Optional[str],
        arbiter: Any,
        tts: Any
    ):
        """Runs Python tool in background and proactively announces completion via voice."""
        res_str = await self._run_python_foreground(tool, args, success_msg)
        completion_msg = success_msg or res_str or f"{tool.name.replace('_', ' ')} finished."

        print(f"[CustomTool:{tool.name}:Background] Finished. Announcing: \"{completion_msg}\"")
        if arbiter:
            await arbiter.enqueue_notification(priority=1, message=completion_msg)
        elif tts:
            await tts.speak_async(completion_msg)

    async def _execute_yaml_tool(
        self,
        tool: CustomToolDefinition,
        args: dict[str, Any],
        arbiter: Any,
        tts: Any
    ) -> str:
        """Executes a declarative YAML sequence tool."""
        ensure_gui_environment()
        steps = tool.get_steps()
        if not steps:
            return f"Custom tool '{tool.name}' has no commands to execute."

        waiting_msg = None
        if tool.waiting_message:
            waiting_msg = self.interpolate_params(tool.waiting_message, args, tool)

        success_msg = None
        if tool.success_message:
            success_msg = self.interpolate_params(tool.success_message, args, tool)

        # Background execution
        if tool.background:
            initial_msg = waiting_msg or f"Started {tool.name.replace('_', ' ')} in the background. Waiting for completion."
            if tts:
                await tts.speak_async(initial_msg)

            asyncio.create_task(
                self._run_sequence_background(
                    tool=tool,
                    steps=steps,
                    args=args,
                    success_msg=success_msg,
                    arbiter=arbiter,
                    tts=tts
                )
            )
            return initial_msg

        # Foreground straight sequential
        return await self._run_sequence_foreground(
            tool=tool,
            steps=steps,
            args=args,
            success_msg=success_msg
        )

    async def _run_sequence_foreground(
        self,
        tool: CustomToolDefinition,
        steps: list[CustomCommandStep],
        args: dict[str, Any],
        success_msg: Optional[str]
    ) -> str:
        """Executes steps sequentially one after another, respecting user-defined timeout (-1 = wait until done)."""
        last_output = ""
        for idx, step in enumerate(steps, 1):
            cmd_str = self.interpolate_command_params(step.run, args, tool)
            print(f"[CustomTool:{tool.name}] Step {idx}/{len(steps)}: {cmd_str}")

            clean_cmd = cmd_str.strip()
            is_detached = step.detach or getattr(tool, "detach", False) or clean_cmd.endswith("&")
            if clean_cmd.endswith("&"):
                clean_cmd = clean_cmd[:-1].strip()

            if is_detached:
                try:
                    proc = subprocess.Popen(
                        clean_cmd,
                        shell=True,
                        start_new_session=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        env=os.environ
                    )
                    await asyncio.sleep(0.05)
                    poll_code = proc.poll()
                    if poll_code is not None and poll_code != 0:
                        err_msg = f"Step {idx} ('{clean_cmd[:40]}') failed to launch with code {poll_code}."
                        print(f"[CustomTool:{tool.name}] {err_msg}")
                        if not step.continue_on_error:
                            return err_msg
                    last_output = f"Started '{clean_cmd[:40]}' in the background."
                    continue
                except Exception as e:
                    err_msg = f"Step {idx} failed to spawn: {e}"
                    print(f"[CustomTool:{tool.name}] {err_msg}")
                    if not step.continue_on_error:
                        return err_msg
                    continue

            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=os.environ
                )

                if step.timeout == -1 or step.timeout is None:
                    stdout_bytes, stderr_bytes = await proc.communicate()
                else:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=float(step.timeout)
                    )

                stdout_text = (stdout_bytes or b"").decode("utf-8", errors="ignore").strip()
                stderr_text = (stderr_bytes or b"").decode("utf-8", errors="ignore").strip()
                last_output = stdout_text or stderr_text

                if proc.returncode != 0:
                    err_msg = f"Step {idx} ('{cmd_str[:40]}') failed with code {proc.returncode}: {stderr_text or stdout_text}"
                    print(f"[CustomTool:{tool.name}] {err_msg}")
                    if not step.continue_on_error:
                        return err_msg

            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                err_msg = f"Step {idx} timed out after {step.timeout}s."
                print(f"[CustomTool:{tool.name}] {err_msg}")
                if not step.continue_on_error:
                    return err_msg
            except Exception as e:
                err_msg = f"Step {idx} error: {e}"
                print(f"[CustomTool:{tool.name}] {err_msg}")
                if not step.continue_on_error:
                    return err_msg

        if success_msg:
            return success_msg
        return last_output or "Done."

    async def _run_sequence_background(
        self,
        tool: CustomToolDefinition,
        steps: list[CustomCommandStep],
        args: dict[str, Any],
        success_msg: Optional[str],
        arbiter: Any,
        tts: Any
    ):
        """Runs the sequential steps in the background and proactively announces completion via arbiter/TTS."""
        last_output = ""
        failed_step = None

        for idx, step in enumerate(steps, 1):
            cmd_str = self.interpolate_command_params(step.run, args, tool)
            print(f"[CustomTool:{tool.name}:Background] Step {idx}/{len(steps)}: {cmd_str}")

            clean_cmd = cmd_str.strip()
            is_detached = step.detach or getattr(tool, "detach", False) or clean_cmd.endswith("&")
            if clean_cmd.endswith("&"):
                clean_cmd = clean_cmd[:-1].strip()

            if is_detached:
                try:
                    proc = subprocess.Popen(
                        clean_cmd,
                        shell=True,
                        start_new_session=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        env=os.environ
                    )
                    await asyncio.sleep(0.05)
                    poll_code = proc.poll()
                    if poll_code is not None and poll_code != 0:
                        print(f"[CustomTool:{tool.name}:Background] Step {idx} failed to launch with code {poll_code}")
                        if not step.continue_on_error:
                            failed_step = f"step {idx} failed to launch with code {poll_code}"
                            break
                    last_output = f"Started '{clean_cmd[:40]}' in the background."
                    continue
                except Exception as e:
                    if not step.continue_on_error:
                        failed_step = f"step {idx} failed to spawn: {e}"
                        break
                    continue

            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=os.environ
                )

                if step.timeout == -1 or step.timeout is None:
                    stdout_bytes, stderr_bytes = await proc.communicate()
                else:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=float(step.timeout)
                    )

                stdout_text = (stdout_bytes or b"").decode("utf-8", errors="ignore").strip()
                stderr_text = (stderr_bytes or b"").decode("utf-8", errors="ignore").strip()
                last_output = stdout_text or stderr_text

                if proc.returncode != 0:
                    print(f"[CustomTool:{tool.name}:Background] Step {idx} failed: {stderr_text or stdout_text}")
                    if not step.continue_on_error:
                        failed_step = f"step {idx} failed with exit code {proc.returncode}"
                        break
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                if not step.continue_on_error:
                    failed_step = f"step {idx} timed out after {step.timeout}s"
                    break
            except Exception as e:
                if not step.continue_on_error:
                    failed_step = f"step {idx} error: {e}"
                    break

        if failed_step:
            completion_msg = f"Custom task {tool.name.replace('_', ' ')} {failed_step}."
        else:
            completion_msg = success_msg or f"{tool.name.replace('_', ' ')} completed successfully."

        print(f"[CustomTool:{tool.name}:Background] Finished. Announcing: \"{completion_msg}\"")
        if arbiter:
            await arbiter.enqueue_notification(priority=1, message=completion_msg)
        elif tts:
            await tts.speak_async(completion_msg)
