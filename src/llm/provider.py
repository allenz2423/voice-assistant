import os
import io
import json
import re
import html
import asyncio
import base64
import math
import time
import uuid
from urllib.parse import urlparse
import ollama
from src.llm.tools import ADAM_TOOLS, CanonicalTool


def _telemetry_safe_text(value, max_length: int = 160) -> str | None:
    """Return a bounded provider identifier suitable for metadata telemetry."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > max_length or any(ord(char) < 32 for char in value):
        return None
    return value


def _nonnegative_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and math.isfinite(value) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _nonnegative_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _openrouter_usage_fields(data: dict) -> tuple[dict, str]:
    """Normalize only documented, numeric OpenRouter accounting fields."""
    raw_usage = data.get("usage")
    if not isinstance(raw_usage, dict):
        return {}, "missing"

    usage: dict[str, int | float] = {}
    for source, target in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = _nonnegative_int(raw_usage.get(source))
        if value is not None:
            usage[target] = value

    prompt_details = raw_usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        for source, target in (
            ("cached_tokens", "cached_input_tokens"),
            ("cache_write_tokens", "cache_write_tokens"),
        ):
            value = _nonnegative_int(prompt_details.get(source))
            if value is not None:
                usage[target] = value

    completion_details = raw_usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning_tokens = _nonnegative_int(completion_details.get("reasoning_tokens"))
        if reasoning_tokens is not None:
            usage["reasoning_tokens"] = reasoning_tokens

    cost = _nonnegative_number(raw_usage.get("cost"))
    if cost is not None:
        usage["provider_reported_cost"] = cost
        usage["currency"] = "credits"

    status = "available" if usage else "missing"
    return usage, status


def _emit_llm_event(event: str, *, status: str, provider: str, model: str | None,
                    trace_id: str,
                    span_id: str | None = None,
                    attributes: dict) -> None:
    """Emit metadata-only telemetry; instrumentation must never break a request."""
    try:
        from src.telemetry.events import emit_event

        emit_event(
            event,
            trace_id=trace_id,
            status=status,
            provider=provider,
            model=model,
            component="llm",
            span_id=span_id,
            attributes=attributes,
        )
    except Exception:
        # A missing/failed telemetry sink must not change user-visible behavior.
        return


def _optimize_image_for_llm(image_bytes: bytes, max_dim: int = 1600, quality: int = 85) -> tuple[bytes, str]:
    """Compress and downscale oversized screenshot images for fast and reliable LLM transmission."""
    from PIL import Image
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            needs_downscale = max(img.size) > max_dim
            needs_compress = len(image_bytes) > 500 * 1024 or img.format != "JPEG"
            if not needs_downscale and not needs_compress:
                mime_type = "image/jpeg" if img.format == "JPEG" else ("image/png" if img.format == "PNG" else "image/webp")
                return image_bytes, mime_type

            if needs_downscale:
                scale = max_dim / max(img.size)
                new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            rgb_img = img.convert("RGB")
            buf = io.BytesIO()
            rgb_img.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue(), "image/jpeg"
    except Exception:
        mime_type = "image/png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            mime_type = "image/jpeg"
        elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            mime_type = "image/webp"
        return image_bytes, mime_type


class UniversalLLMClient:
    """Unified LLM client supporting local Ollama models and cloud providers with a single switch."""
    def __init__(self, config):
        self.provider = config.llm.provider.lower()
        self.local_model = config.llm.local_model
        self.cloud_model = config.llm.cloud_model
        self.ollama_host = config.llm.ollama_host
        self.api_base = getattr(config.llm, "api_base", "")
        self.api_key = getattr(config.llm, "api_key", "")
        self.temperature = config.llm.temperature
        self.num_ctx = config.llm.num_ctx
        self.think = getattr(config.llm, "think", False)
        self.provider_only = getattr(config.llm, "provider_only", [])
        self.allow_provider_fallbacks = getattr(config.llm, "allow_provider_fallbacks", True)

    async def warmup(self):
        """Preloads and warms up the LLM model in VRAM during system startup."""
        if self.provider == "local":
            print(f"[LLM] Preloading and warming up '{self.local_model}' in Ollama (keep_alive=-1)...", flush=True)
            try:
                client = ollama.AsyncClient(host=self.ollama_host)
                await asyncio.wait_for(
                    client.generate(
                        model=self.local_model,
                        prompt="",
                        keep_alive=-1,
                        options={"num_ctx": self.num_ctx}
                    ),
                    timeout=45.0
                )
                print(f"[LLM] Model '{self.local_model}' is resident in VRAM and ready.", flush=True)
            except Exception as e:
                print(f"[LLM] Warning: Startup warmup failed ({e}). Model will load on first prompt.", flush=True)
        else:
            print(f"[LLM] Provider '{self.provider}' configured (no local GPU warmup required).", flush=True)

    async def chat(
        self,
        messages: list[dict],
        tools: list[CanonicalTool] = ADAM_TOOLS,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> dict:
        """Dispatches chat completion to the configured provider (local vs cloud)."""
        if max_tokens is not None:
            max_tokens = max(1, int(max_tokens))
        if self.provider == "local":
            return await self._chat_ollama(messages, tools, max_tokens, think)
        elif self.provider in ["groq", "cloud", "openai", "custom", "openai_compatible", "vllm", "llama_cpp"]:
            return await self._chat_openai_compatible(messages, tools, max_tokens, think)
        elif self.provider == "anthropic":
            return await self._chat_anthropic(messages, tools, max_tokens, think)
        elif self.provider == "gemini":
            return await self._chat_gemini(messages, tools, max_tokens, think)
        else:
            return await self._chat_ollama(messages, tools, max_tokens, think)

    async def _chat_ollama(
        self, messages: list[dict], tools: list[CanonicalTool] = ADAM_TOOLS,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> dict:
        """Executes tool calling via local Ollama daemon."""
        openai_tools = [t.to_openai() for t in tools] if tools else None
        try:
            client = ollama.AsyncClient(host=self.ollama_host)
            chat_kwargs = {
                "model": self.local_model,
                "messages": messages,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
                "think": self.think if think is None else think
            }
            if max_tokens is not None:
                chat_kwargs["options"]["num_predict"] = max_tokens
            if openai_tools:
                chat_kwargs["tools"] = openai_tools

            response = await client.chat(**chat_kwargs)
            msg = response.get("message", {})
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""

            # If tool calls are embedded in content or native tool_calls are empty
            if content:
                content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
                content = re.sub(r"<thought>[\s\S]*?</thought>", "", content, flags=re.IGNORECASE).strip()
                cleaned_content, extracted_tools = self._extract_embedded_tool_calls(content, tools)
                tool_calls = tool_calls or extracted_tools
                content = cleaned_content

            return {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls
            }
        except Exception as e:
            print(f"[LLM] Ollama call error: {e}")
            # Emergency offline rule-based fallback if Ollama is not running
            return self._emergency_rule_fallback(messages[-1].get("content", ""))

    async def _chat_openai_compatible(
        self, messages: list[dict], tools: list[CanonicalTool] = ADAM_TOOLS,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> dict:
        """Calls any OpenAI-compatible endpoint (Groq, OpenAI, vLLM, llama.cpp, LocalAI, LM Studio)."""
        import aiohttp

        url = self.api_base
        if not url:
            if self.provider == "groq":
                url = "https://api.groq.com/openai/v1/chat/completions"
            elif self.provider in ["openai", "cloud"]:
                url = "https://api.openai.com/v1/chat/completions"
            else:
                base = self.ollama_host.rstrip("/")
                url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"
        elif not url.endswith("/chat/completions"):
            url = f"{url.rstrip('/')}/chat/completions"

        key = self.api_key
        if not key:
            if self.provider == "groq":
                key = os.environ.get("GROQ_API_KEY", "")
            elif self.provider in ["openai", "cloud"]:
                key = os.environ.get("OPENAI_API_KEY", "")

        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        model = self.cloud_model if self.provider in ["groq", "cloud", "openai", "custom"] else self.local_model
        openai_tools = [t.to_openai() for t in tools] if tools else None

        formatted = []
        for m in messages:
            msg = dict(m)
            if msg.get("content") is None:
                msg["content"] = ""
            images = msg.pop("images", None)
            if images:
                content = msg["content"]
                parts = list(content) if isinstance(content, list) else []
                if isinstance(content, str) and content:
                    parts.append({"type": "text", "text": content})
                for image in images:
                    if isinstance(image, str) and image.startswith(("data:image/", "https://", "http://")):
                        image_url = image
                    else:
                        image_bytes = bytes(image)
                        image_started = time.perf_counter()
                        opt_bytes, mime_type = _optimize_image_for_llm(image_bytes)
                        encoded = base64.b64encode(opt_bytes).decode("ascii")
                        try:
                            from src.telemetry.events import get_trace_id
                            image_trace = get_trace_id() or "untraced"
                        except Exception:
                            image_trace = "untraced"
                        print(
                            f"[LLMTiming] trace={image_trace[:12]} stage=image_prepare "
                            f"duration_ms={(time.perf_counter() - image_started) * 1000:.1f} "
                            f"input_bytes={len(image_bytes)} output_bytes={len(opt_bytes)}",
                            flush=True,
                        )
                        image_url = f"data:{mime_type};base64,{encoded}"
                    parts.append({"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}})
                msg["content"] = parts
            formatted.append(msg)

        payload = {
            "model": model,
            "messages": formatted,
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if openai_tools:
            payload["tools"] = openai_tools

        # OpenRouter routing and reasoning are request-level options. Keep the
        # model pinned when configured instead of silently falling back to a
        # different inference provider.
        if "openrouter.ai" in url.lower():
            if self.provider_only:
                payload["provider"] = {
                    "only": self.provider_only,
                    "allow_fallbacks": self.allow_provider_fallbacks,
                }
            effective_think = self.think if think is None else think
            if effective_think:
                payload["reasoning"] = {"enabled": True}

        try:
            # Vision calls include screenshots and can take longer than a text turn.
            # Keep ordinary voice responses snappy while giving computer-use turns
            # enough time to reason over the full image and structured UI hints.
            has_image = any(bool(message.get("images")) for message in messages)
            timeout_seconds = 90 if has_image else 45
            request_span_id = uuid.uuid4().hex
            started_at = asyncio.get_running_loop().time()
            configured_provider = _telemetry_safe_text(self.provider, 64) or "unknown"
            configured_model = _telemetry_safe_text(model)
            endpoint_host = (urlparse(url).hostname or "").lower()
            is_openrouter_endpoint = (
                endpoint_host == "openrouter.ai" or endpoint_host.endswith(".openrouter.ai")
            )
            try:
                from src.telemetry.events import get_trace_id
                request_trace_id = get_trace_id() or uuid.uuid4().hex
            except Exception:
                request_trace_id = uuid.uuid4().hex
            route_attributes = {}
            if is_openrouter_endpoint:
                configured_routes = self.provider_only if isinstance(self.provider_only, (list, tuple)) else []
                safe_routes = [
                    safe for value in configured_routes
                    if (safe := _telemetry_safe_text(value, 80)) is not None
                ][:16]
                if safe_routes:
                    route_attributes["provider_only"] = ",".join(safe_routes)
                route_attributes["allow_fallbacks"] = bool(self.allow_provider_fallbacks)
            common_attributes = {
                "configured_provider": configured_provider,
                "configured_model": configured_model,
                "timeout_ms": timeout_seconds * 1000,
            }
            common_attributes.update(route_attributes)
            _emit_llm_event(
                "llm.request_started",
                status="started",
                provider=configured_provider,
                model=configured_model,
                trace_id=request_trace_id,
                span_id=request_span_id,
                attributes=common_attributes,
            )

            request_started_at = None
            response_headers_at = None
            response_body_done_at = None

            def complete_event(accounting_status: str, *, event_status: str = "error",
                               response: dict | None = None,
                               usage: dict | None = None, response_status: int | None = None):
                finished_at = asyncio.get_running_loop().time()
                usage = usage or {}
                attributes = {
                    **common_attributes,
                    "accounting_status": accounting_status,
                    "duration_ms": max(0, round((finished_at - started_at) * 1000)),
                }
                network_ms = None
                headers_ms = None
                body_ms = None
                if request_started_at is not None:
                    network_ms = max(0, round((finished_at - request_started_at) * 1000))
                    attributes["network_ms"] = network_ms
                if request_started_at is not None and response_headers_at is not None:
                    headers_ms = max(0, round((response_headers_at - request_started_at) * 1000))
                    attributes["response_headers_ms"] = headers_ms
                if response_headers_at is not None and response_body_done_at is not None:
                    body_ms = max(0, round((response_body_done_at - response_headers_at) * 1000))
                    attributes["response_body_ms"] = body_ms
                if response_status is not None:
                    attributes["http_status"] = response_status
                if isinstance(response, dict):
                    returned_model = _telemetry_safe_text(response.get("model"))
                    request_id = _telemetry_safe_text(response.get("id"))
                    if returned_model:
                        attributes["returned_model"] = returned_model
                    if request_id:
                        attributes["provider_request_id"] = request_id
                    # OpenRouter returns the selected model in `model`. The response
                    # schema does not guarantee a separately named provider route.
                if usage:
                    attributes["usage"] = usage
                choice_data = None
                if isinstance(response, dict):
                    choices = response.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        choice_data = choices[0]
                finish_reason = (
                    _telemetry_safe_text(choice_data.get("finish_reason"), 48)
                    if isinstance(choice_data, dict) else None
                )
                _emit_llm_event(
                    "llm.completed",
                    status=event_status,
                    provider=configured_provider,
                    model=(
                        _telemetry_safe_text(response.get("model"))
                        if isinstance(response, dict) else configured_model
                    ),
                    trace_id=request_trace_id,
                    span_id=request_span_id,
                    attributes=attributes,
                )
                print(
                    f"[LLMTiming] trace={request_trace_id[:12]} stage=openai_compatible_request provider={configured_provider} "
                    f"model={configured_model or 'unknown'} status={event_status} "
                    f"http_status={response_status if response_status is not None else 'unknown'} "
                    f"total_ms={attributes['duration_ms']} network_ms={network_ms if network_ms is not None else 'unknown'} "
                    f"headers_ms={headers_ms if headers_ms is not None else 'unknown'} "
                    f"body_ms={body_ms if body_ms is not None else 'unknown'} "
                    f"input_tokens={usage.get('input_tokens', 'unknown')} "
                    f"output_tokens={usage.get('output_tokens', 'unknown')} "
                    f"reasoning_tokens={usage.get('reasoning_tokens', 'unknown')} "
                    f"finish_reason={finish_reason or 'unknown'}",
                    flush=True,
                )

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as session:
                request_started_at = asyncio.get_running_loop().time()
                async with session.post(url, headers=headers, json=payload) as resp:
                    response_headers_at = asyncio.get_running_loop().time()
                    if resp.status != 200:
                        # Do not log provider response bodies: gateways can echo request data.
                        complete_event("http_error", response_status=resp.status)
                        print(f"[LLM] OpenAI-compatible call failed (HTTP {resp.status}).")
                        return self._emergency_rule_fallback(messages[-1].get("content", ""))
                    try:
                        data = await resp.json()
                        response_body_done_at = asyncio.get_running_loop().time()
                    except Exception:
                        complete_event("invalid_response", response_status=resp.status)
                        print("[LLM] OpenAI-compatible call returned invalid JSON.", flush=True)
                        return self._emergency_rule_fallback(messages[-1].get("content", ""))
                    if not isinstance(data, dict):
                        complete_event("invalid_response", response_status=resp.status)
                        print("[LLM] OpenAI-compatible call returned an invalid response shape.", flush=True)
                        return self._emergency_rule_fallback(messages[-1].get("content", ""))
                    if "error" in data:
                        complete_event("provider_error", response=data, response_status=resp.status)
                        print("[LLM] OpenAI-compatible call returned a provider error.", flush=True)
                        return self._emergency_rule_fallback(messages[-1].get("content", ""))
                    choices = data.get("choices")
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        complete_event("invalid_response", response=data, response_status=resp.status)
                        print("[LLM] OpenAI-compatible call returned no valid choice.", flush=True)
                        return self._emergency_rule_fallback(messages[-1].get("content", ""))
                    first_choice = choices[0]
                    choice = first_choice.get("message", {})
                    if not isinstance(choice, dict):
                        complete_event("invalid_response", response=data, response_status=resp.status)
                        print("[LLM] OpenAI-compatible call returned an invalid message shape.", flush=True)
                        return self._emergency_rule_fallback(messages[-1].get("content", ""))
                    content = choice.get("content") or ""
                    tool_calls = choice.get("tool_calls") or []
                    reasoning_details = choice.get("reasoning_details")

                    if not content and not tool_calls:
                        finish_reason = first_choice.get("finish_reason", "unknown")
                        fields = ",".join(sorted(choice.keys())) or "none"
                        print(
                            f"[LLM] Empty assistant completion (finish_reason={finish_reason}, "
                            f"message_fields={fields}, image_input={has_image}).",
                            flush=True,
                        )

                    if content:
                        content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
                        content = re.sub(r"<thought>[\s\S]*?</thought>", "", content, flags=re.IGNORECASE).strip()
                        cleaned_content, extracted = self._extract_embedded_tool_calls(content, tools)
                        tool_calls = tool_calls or extracted
                        content = cleaned_content

                    result = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls
                    }
                    if reasoning_details is not None:
                        # OpenRouter asks clients to return these opaque blocks
                        # unchanged on the next turn for supported reasoning models.
                        result["reasoning_details"] = reasoning_details
                    if is_openrouter_endpoint:
                        usage, accounting_status = _openrouter_usage_fields(data)
                    else:
                        usage, accounting_status = {}, "unsupported"
                    complete_event(
                        accounting_status,
                        event_status="ok",
                        response=data,
                        usage=usage,
                        response_status=resp.status,
                    )
                    return result
        except Exception as e:
            # Exclude exception text; client errors can contain URLs, headers, or payload data.
            if "started_at" in locals():
                complete_event("transport_error")
            print(f"[LLM] OpenAI-compatible request failed ({type(e).__name__}).")
            return self._emergency_rule_fallback(messages[-1].get("content", ""))

    async def _chat_anthropic(
        self, messages: list[dict], tools: list[CanonicalTool] = ADAM_TOOLS,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> dict:
        import aiohttp
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            print("[LLM] Error: ANTHROPIC_API_KEY not configured.")
            return self._emergency_rule_fallback(messages[-1].get("content", ""))

        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        system_prompt = "You are a voice-activated desktop assistant. Answer conversationally and clearly."
        ant_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_prompt = m.get("content", "")
            else:
                role = "assistant" if m.get("role") == "assistant" else "user"
                ant_messages.append({"role": role, "content": m.get("content", "")})

        payload = {
            "model": self.cloud_model or "claude-3-5-sonnet-20241022",
            "system": system_prompt,
            "messages": ant_messages,
            "max_tokens": max_tokens or 1024,
            "temperature": self.temperature
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
                async with session.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        print(f"[LLM] Anthropic API error ({resp.status}): {err}")
                        return self._emergency_rule_fallback(messages[-1].get("content", ""))
                    data = await resp.json()
                    content = "".join([b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"])
                    cleaned_content, extracted = self._extract_embedded_tool_calls(content, tools)
                    return {
                        "role": "assistant",
                        "content": cleaned_content,
                        "tool_calls": extracted
                    }
        except Exception as e:
            print(f"[LLM] Anthropic request failed: {e}")
            return self._emergency_rule_fallback(messages[-1].get("content", ""))

    async def _chat_gemini(
        self, messages: list[dict], tools: list[CanonicalTool] = ADAM_TOOLS,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> dict:
        import aiohttp
        key = self.api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            print("[LLM] Error: GEMINI_API_KEY not configured.")
            return self._emergency_rule_fallback(messages[-1].get("content", ""))

        model = self.cloud_model or "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        contents = []
        for m in messages:
            role = "model" if m.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        payload = {
            "contents": contents,
            "generationConfig": {"temperature": self.temperature}
        }
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        print(f"[LLM] Gemini API error ({resp.status}): {err}")
                        return self._emergency_rule_fallback(messages[-1].get("content", ""))
                    data = await resp.json()
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])
                    content = "".join([p.get("text", "") for p in parts if "text" in p])
                    cleaned_content, extracted = self._extract_embedded_tool_calls(content, tools)
                    return {
                        "role": "assistant",
                        "content": cleaned_content,
                        "tool_calls": extracted
                    }
        except Exception as e:
            print(f"[LLM] Gemini request failed: {e}")
            return self._emergency_rule_fallback(messages[-1].get("content", ""))

    def _extract_embedded_tool_calls(
        self, text: str, available_tools: list[CanonicalTool] | None = None
    ) -> tuple[str, list[dict]]:
        """Normalize textual tool-call formats emitted by models into existing calls.

        Structured provider tool calls remain preferred. This also normalizes
        DeepSeek DSML, Qwen, and Dots formats when a gateway leaks them as text.
        Only tools offered in this request are accepted.
        """
        if available_tools is None:
            allowed_names = {tool.name for tool in ADAM_TOOLS}
        else:
            allowed_names = {
                tool.name if hasattr(tool, "name") else str((tool.get("function") or {}).get("name", ""))
                for tool in available_tools
            }

        parsed_calls: list[dict] = []
        cleaned_text = text

        # Qwen commonly emits <tool_call><function=name>... and Dots emits
        # <dots_function_call><invoke name="...">...</invoke>.
        wrapper_re = re.compile(
            r"<(dots_function_call|tool_call)\b[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL
        )
        wrappers = list(wrapper_re.finditer(text))

        def parse_value(value: str):
            value = html.unescape(value.strip())
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        # DeepSeek V4/V4.1 uses special-token delimiters rendered as visible
        # strings by some OpenAI-compatible gateways. V4.1 adds spaces after
        # the DSML marker; older V4 used "tool_calls" without those spaces.
        dsml_call_re = re.compile(
            r"<｜DSML｜\s*(?:calls|tool_calls)\s*>(.*?)</｜DSML｜\s*(?:calls|tool_calls)\s*>",
            re.DOTALL,
        )
        dsml_blocks = list(dsml_call_re.finditer(text))
        invoke_re = re.compile(
            r'<｜DSML｜\s*invoke\s+name\s*=\s*(["\'])([A-Za-z0-9_.:-]+)\1\s*>'
            r'(.*?)</｜DSML｜\s*invoke\s*>',
            re.DOTALL,
        )
        parameter_re = re.compile(
            r'<｜DSML｜\s*parameter\s+name\s*=\s*(["\'])([A-Za-z0-9_]+)\1'
            r'\s+string\s*=\s*(["\'])(true|false)\3\s*>'
            r'(.*?)</｜DSML｜\s*parameter\s*>',
            re.DOTALL | re.IGNORECASE,
        )
        for block in dsml_blocks:
            for invoke in invoke_re.finditer(block.group(1)):
                name, fn_body = invoke.group(2), invoke.group(3)
                if name not in allowed_names:
                    continue
                arguments = {}
                for parameter in parameter_re.finditer(fn_body):
                    key = parameter.group(2)
                    is_string = parameter.group(4).lower() == "true"
                    raw_value = parameter.group(5)
                    arguments[key] = html.unescape(raw_value) if is_string else parse_value(raw_value)
                parsed_calls.append({
                    "function": {"name": name, "arguments": arguments},
                    "_origin": "dsml_fallback",
                })

        for wrapper in wrappers:
            body = wrapper.group(2)
            invokes = list(re.finditer(
                r"<invoke\s+name\s*=\s*(['\"])([A-Za-z0-9_]+)\1\s*>(.*?)</invoke>",
                body, re.IGNORECASE | re.DOTALL,
            ))
            functions = list(re.finditer(
                r"<function\s*=\s*([A-Za-z0-9_]+)\s*>(.*?)</function>",
                body, re.IGNORECASE | re.DOTALL,
            ))
            entries = [(m.group(2), m.group(3)) for m in invokes]
            entries.extend((m.group(1), m.group(2)) for m in functions)

            for name, fn_body in entries:
                if name not in allowed_names:
                    continue
                arguments = {}
                for param in re.finditer(
                    r"<parameter(?:\s+name\s*=\s*(['\"])([A-Za-z0-9_]+)\1|\s*=\s*([A-Za-z0-9_]+))\s*>(.*?)</parameter>",
                    fn_body, re.IGNORECASE | re.DOTALL,
                ):
                    parameter_name = param.group(2) or param.group(3)
                    arguments[parameter_name] = parse_value(param.group(4))
                parsed_calls.append({
                    "function": {"name": name, "arguments": arguments},
                    "_origin": "text_fallback",
                })

            if not entries:
                _, wrapped_json_calls = self._extract_json_embedded_tool_calls(body)
                for call in wrapped_json_calls:
                    fn = call.get("function", {})
                    if fn.get("name") in allowed_names:
                        call["_origin"] = "text_fallback"
                        parsed_calls.append(call)

        # Remove recognized wrapper blocks even when their function wasn't
        # offered, so unsupported pseudo-calls are never spoken back verbatim.
        # Remove all recognized wrapper blocks against the original text, in
        # reverse order so earlier offsets remain valid.
        spans = sorted(
            [(block.start(), block.end()) for block in dsml_blocks]
            + [(wrapper.start(), wrapper.end()) for wrapper in wrappers],
            reverse=True,
        )
        for start, end in spans:
            cleaned_text = cleaned_text[:start] + cleaned_text[end:]

        # Suppress truncated/unwrapped DSML markers too, rather than speaking
        # fragments of a provider's internal tool-call serialization.
        cleaned_text = re.sub(r"<｜DSML｜[^>]*>|</｜DSML｜[^>]*>", "", cleaned_text)

        # Keep the existing JSON/markdown/inline compatibility parser for other
        # providers, then filter its results against the current offered tools.
        cleaned_json, json_calls = self._extract_json_embedded_tool_calls(cleaned_text)
        for call in json_calls:
            fn = call.get("function", {})
            if fn.get("name") in allowed_names:
                call["_origin"] = "text_fallback"
                parsed_calls.append(call)

        return cleaned_json.strip(), parsed_calls

    def _extract_json_embedded_tool_calls(self, text: str) -> tuple[str, list[dict]]:
        """Recovers tool calls from raw JSON, markdown code blocks, or embedded objects with arbitrary nesting."""
        decoder = json.JSONDecoder()
        tool_calls = []
        cleaned_text = text

        # 1. Markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "name" in data:
                    tool_calls.append({"function": {"name": data["name"], "arguments": data.get("arguments", {})}})
                    cleaned_text = text[:match.start()].strip() + " " + text[match.end():].strip()
                    return cleaned_text.strip(), tool_calls
            except Exception:
                pass

        # 2. Raw JSON string
        trimmed = text.strip()
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
            try:
                data = json.loads(trimmed)
                if isinstance(data, dict) and "name" in data:
                    return "", [{"function": {"name": data["name"], "arguments": data.get("arguments", {})}}]
                elif isinstance(data, list):
                    calls = [{"function": {"name": item["name"], "arguments": item.get("arguments", {})}} for item in data if isinstance(item, dict) and "name" in item]
                    if calls:
                        return "", calls
            except Exception:
                pass

        # 3. Stream JSON scanner for embedded objects
        idx = 0
        while idx < len(text):
            m = re.search(r"\{\s*\"name\"\s*:", text[idx:])
            if not m:
                break
            start = idx + m.start()
            try:
                obj, end_offset = decoder.raw_decode(text[start:])
                if isinstance(obj, dict) and "name" in obj:
                    tool_calls.append({"function": {"name": obj["name"], "arguments": obj.get("arguments", {})}})
                    cleaned_text = text[:start].strip() + " " + text[start + end_offset:].strip()
                    idx = start + end_offset
                    continue
            except Exception:
                pass
        # 4. Handle inline model syntax: tool_name{...}
        if not tool_calls:
            m = re.search(r"\b([a-zA-Z0-9_]{3,30})\s*(\{[^{}]*\})", text, re.DOTALL)
            if m:
                fn_name = m.group(1)
                args_raw = m.group(2)
                try:
                    args_parsed = json.loads(args_raw)
                except json.JSONDecodeError:
                    # Preserve invalid input so the host can return a correction
                    # instead of executing a different call with empty args.
                    args_parsed = args_raw
                tool_calls.append({"function": {"name": fn_name, "arguments": args_parsed}})
                cleaned_text = (text[:m.start()].strip() + " " + text[m.end():].strip()).strip()

        return cleaned_text.strip(), tool_calls

    def _emergency_rule_fallback(self, user_text: str) -> dict:
        """Rule-based emergency fallback for common commands when no LLM is running."""
        # Brain requests include desktop and time context before the spoken command.
        # Never treat or repeat that injected context as user text.
        command_match = re.search(r"\[Local Time:[^\]]+\]\s*\n(.*)$", user_text, re.DOTALL)
        if command_match:
            user_text = command_match.group(1).strip()
        text = user_text.lower().strip()
        if "transcode" in text or "av1" in text:
            pattern = "*slime*" if "slime" in text else "*.mkv"
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "transcode_video",
                        "arguments": {"file_pattern": pattern, "target_codec": "av1"}
                    }
                }]
            }
        return {
            "role": "assistant",
            "content": "I couldn't reach the language model. Please try again shortly.",
            "tool_calls": [],
            "provider_error": True,
        }

    def format_tool_response(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        """Formats tool results for the current provider."""
        if self.provider == "anthropic":
            return {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": result}]
            }
        elif self.provider == "gemini":
            return {
                "role": "function",
                "parts": [{"functionResponse": {"name": tool_name, "response": {"result": result}}}]
            }
        else:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result
            }
