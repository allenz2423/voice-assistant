import os
import json
import re
import asyncio
import ollama
from src.llm.tools import SHIN_TOOLS, CanonicalTool

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

    async def chat(self, messages: list[dict], tools: list[CanonicalTool] = SHIN_TOOLS) -> dict:
        """Dispatches chat completion to the configured provider (local vs cloud)."""
        if self.provider == "local":
            return await self._chat_ollama(messages, tools)
        elif self.provider in ["groq", "cloud", "openai", "custom", "openai_compatible", "vllm", "llama_cpp"]:
            return await self._chat_openai_compatible(messages, tools)
        elif self.provider == "anthropic":
            return await self._chat_anthropic(messages, tools)
        elif self.provider == "gemini":
            return await self._chat_gemini(messages, tools)
        else:
            return await self._chat_ollama(messages, tools)

    async def _chat_ollama(self, messages: list[dict], tools: list[CanonicalTool] = SHIN_TOOLS) -> dict:
        """Executes tool calling via local Ollama daemon."""
        openai_tools = [t.to_openai() for t in tools] if tools else None
        try:
            client = ollama.AsyncClient(host=self.ollama_host)
            chat_kwargs = {
                "model": self.local_model,
                "messages": messages,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
                "think": self.think
            }
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
                cleaned_content, extracted_tools = self._extract_embedded_tool_calls(content)
                if extracted_tools:
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

    async def _chat_openai_compatible(self, messages: list[dict], tools: list[CanonicalTool] = SHIN_TOOLS) -> dict:
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
            formatted.append(msg)

        payload = {
            "model": model,
            "messages": formatted,
            "temperature": self.temperature,
        }
        if openai_tools:
            payload["tools"] = openai_tools

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        print(f"[LLM] OpenAI-compatible call error ({resp.status}): {err}")
                        return self._emergency_rule_fallback(messages[-1].get("content", ""))
                    data = await resp.json()
                    choice = data.get("choices", [{}])[0].get("message", {})
                    content = choice.get("content") or ""
                    tool_calls = choice.get("tool_calls") or []

                    if content:
                        content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
                        content = re.sub(r"<thought>[\s\S]*?</thought>", "", content, flags=re.IGNORECASE).strip()
                        cleaned_content, extracted = self._extract_embedded_tool_calls(content)
                        if extracted:
                            tool_calls = tool_calls or extracted
                            content = cleaned_content

                    return {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls
                    }
        except Exception as e:
            print(f"[LLM] OpenAI-compatible request failed: {e}")
            return self._emergency_rule_fallback(messages[-1].get("content", ""))

    async def _chat_anthropic(self, messages: list[dict], tools: list[CanonicalTool] = SHIN_TOOLS) -> dict:
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
            "max_tokens": 1024,
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
                    cleaned_content, extracted = self._extract_embedded_tool_calls(content)
                    return {
                        "role": "assistant",
                        "content": cleaned_content,
                        "tool_calls": extracted
                    }
        except Exception as e:
            print(f"[LLM] Anthropic request failed: {e}")
            return self._emergency_rule_fallback(messages[-1].get("content", ""))

    async def _chat_gemini(self, messages: list[dict], tools: list[CanonicalTool] = SHIN_TOOLS) -> dict:
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
                    cleaned_content, extracted = self._extract_embedded_tool_calls(content)
                    return {
                        "role": "assistant",
                        "content": cleaned_content,
                        "tool_calls": extracted
                    }
        except Exception as e:
            print(f"[LLM] Gemini request failed: {e}")
            return self._emergency_rule_fallback(messages[-1].get("content", ""))

    def _extract_embedded_tool_calls(self, text: str) -> tuple[str, list[dict]]:
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
                except Exception:
                    args_parsed = {}
                tool_calls.append({"function": {"name": fn_name, "arguments": args_parsed}})
                cleaned_text = (text[:m.start()].strip() + " " + text[m.end():].strip()).strip()

        return cleaned_text.strip(), tool_calls

    def _emergency_rule_fallback(self, user_text: str) -> dict:
        """Rule-based emergency fallback for common commands when no LLM is running."""
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
        elif any(k in text for k in ["pause", "stop music"]):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "media_control",
                        "arguments": {"action": "pause"}
                    }
                }]
            }
        elif any(k in text for k in ["play", "resume"]):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "media_control",
                        "arguments": {"action": "play"}
                    }
                }]
            }
        return {
            "role": "assistant",
            "content": f"Understood: '{user_text}'. (LLM daemon offline; fallback response).",
            "tool_calls": []
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
