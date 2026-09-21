"""Text-only native model calls, with raw receipts kept outside published results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "action": {"type": "string", "enum": ["hypothesis", "probe", "edit", "finish"]},
    "hypothesis": {"type": "string"}, "prediction": {"type": "string"},
    "source": {"type": "string"}, "probes_json": {"type": "string"}},
    "required": ["action", "hypothesis", "prediction", "source", "probes_json"]}


def complete(prompt, provider, out, timeout=150, stop=None):
    if provider not in ("opus", "astra", "sonnet", "haiku", "llama3.2:1b", "qwen2.5:7b"):
        raise ValueError("unknown model provider")
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    os.chmod(out, 0o700)
    (out / "prompt.txt").write_text(prompt)
    (out / "schema.json").write_text(json.dumps(SCHEMA))
    command = ["claude", "-p", "--model", "claude-opus-5", "--effort", "high",
               "--safe-mode", "--tools", "", "--strict-mcp-config", "--mcp-config",
               '{"mcpServers":{}}', "--no-session-persistence", "--output-format", "json",
               "--max-budget-usd", "0.80", "--json-schema", json.dumps(SCHEMA)]
    if provider in ("sonnet", "haiku"):
        command[3] = provider
        del command[4:6]  # Use provider default effort consistently across treatments.
    if provider in ("llama3.2:1b", "qwen2.5:7b"):
        payload = {"model": provider, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "format": SCHEMA, "keep_alive": "5m",
                   "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 6000}}
        (out / "prompt.txt").write_text(json.dumps(payload))
        command = ["curl", "--silent", "--show-error", "--fail", "--max-time", str(timeout),
                   "http://127.0.0.1:11434/api/chat", "-H", "Content-Type: application/json",
                   "--data-binary", "@-"]
    if provider == "astra":
        command = ["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                   "--sandbox", "read-only", "--enable", "skip_host_skill_discovery",
                   "--disable", "shell_tool", "--disable", "multi_agent", "--disable", "apps",
                   "--disable", "in_app_browser", "--disable", "image_generation", "--disable", "view_image",
                   "--disable", "skill_search", "-c", "project_doc_max_bytes=0", "-c", 'web_search="disabled"',
                   "-c", 'model_reasoning_effort="high"', "--model", "gpt-6-astra", "--json",
                   "--output-schema", str(out / "schema.json"), "-o", str(out / "answer.json"), "-"]
    request = {"provider": provider, "requested_model": {"opus": "claude-opus-5", "astra": "gpt-6-astra"}.get(provider, provider), "command": command, "owner_pid": os.getpid(),
               "started": time.monotonic(), "timeout": timeout,
               "stop": str(Path(stop).resolve()) if stop is not None else None}
    (out / "request.json").write_text(json.dumps(request))
    # This owner survives driver death and retains the native call's receipt.
    supervisor = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("model_supervisor.py")), str(out)],
        start_new_session=True,
    )
    try:
        code = supervisor.wait(timeout=timeout + 15)
    except BaseException:
        if supervisor.poll() is None:
            supervisor.terminate()
        raise
    receipt = out / "receipt.json"
    if code or not receipt.exists():
        raise RuntimeError("native supervisor failed; call outcome is uncertain")
    return json.loads(receipt.read_text())


def decode(receipt, out):
    if receipt["provider"] in ("llama3.2:1b", "qwen2.5:7b"):
        raw = json.loads((out / "stdout.log").read_text())
        receipt["usage"] = {k: raw.get(k) for k in ("model", "prompt_eval_count", "eval_count", "total_duration", "done_reason")}
        if receipt["exit_code"] or not raw.get("done") or raw.get("done_reason") == "length":
            receipt["error"] = "local_incomplete_response"; return
        receipt["response"] = json.loads(raw["message"]["content"])
        return
    if receipt["provider"] in ("opus", "sonnet", "haiku"):
        raw = json.loads((out / "stdout.log").read_text())
        receipt["usage"] = raw.get("modelUsage")
        receipt["estimated_cost_usd"] = raw.get("total_cost_usd")
        if raw.get("is_error") or raw.get("subtype") != "success" or receipt["exit_code"]:
            receipt["error"] = "provider_failure:" + str(raw.get("subtype")); return
        receipt["response"] = raw.get("structured_output")
        if receipt["response"] is None:
            receipt["response"] = json.loads(raw["result"])
        return
    events = [json.loads(line) for line in (out / "stdout.log").read_text().splitlines() if line]
    usages = [e["usage"] for e in events if e.get("type") == "turn.completed"]
    receipt["usage"] = usages[-1] if usages else None
    tools = [e for e in events if e.get("type") == "item.completed"
             and e.get("item", {}).get("type") not in ("agent_message", "reasoning", "error")]
    receipt["tool_events"] = len(tools)
    if tools or receipt["exit_code"] or not usages:
        receipt["error"] = "unexpected_tool_or_provider_failure"; return
    receipt["response"] = json.loads((out / "answer.json").read_text())
