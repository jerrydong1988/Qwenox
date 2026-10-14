"""Post-deploy API checks for a running text-only Qwen27 backend."""
import argparse
import json
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:18871")
    args = ap.parse_args()
    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.api+path, data, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()
    results = []
    def check(name, ok, detail):
        assert ok, (name, detail)
        results.append({"test": name, "pass": True, "detail": detail})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    status, raw = call("/health")
    h = json.loads(raw)
    check("text-only-capabilities", status == 200 and h["model"] == "qwen3.8-27b-hgn"
          and h["slots"] == 1 and h["vision"]["accepted"] is False
          and "image_url" not in h["supported"], h)
    status, raw = call("/v1/models")
    check("model-name", status == 200 and json.loads(raw)["data"][0]["id"] == "qwen3.8-27b-hgn", json.loads(raw))
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,invalid"}}
    for path, body in [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": [image]}]}),
        ("/v1/responses", {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,invalid"}]}]}),
    ]:
        status, raw = call(path, body)
        check("reject-images"+path, status == 400 and b"text only" in raw, json.loads(raw))
    body = {"messages": [{"role": "user", "content": "17乘以23等于多少？只回答数字。"}],
            "temperature": 0, "enable_thinking": False, "max_tokens": 32, "stream": True}
    status, raw = call("/v1/chat/completions", body)
    text = ""
    done = False
    for line in raw.decode().splitlines():
        if not line.startswith("data: "):
            continue
        if line[6:] == "[DONE]":
            done = True
        else:
            chunk = json.loads(line[6:])
            for choice in chunk.get("choices", []):
                text += choice.get("delta", {}).get("content", "") or ""
    check("streaming-chinese", status == 200 and text.strip() == "391" and done, text)
    status, raw = call("/v1/chat/completions", {
        "messages": [
            {"role": "user", "content": "调用 add 计算17加23，拿到结果后只回答数字。"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_q27", "type": "function",
                "function": {"name": "add", "arguments": '{"a":17,"b":23}'}}]},
            {"role": "tool", "tool_call_id": "call_q27", "content": "40"}],
        "temperature": 0, "enable_thinking": False, "max_tokens": 32})
    result = json.loads(raw)
    check("tool-result-roundtrip", status == 200 and result["choices"][0]["message"]["content"].strip() == "40", result)
    status, raw = call("/v1/responses", {"input": "2加2等于多少？只回答数字。", "temperature": 0,
                                         "enable_thinking": False, "max_output_tokens": 32})
    result = json.loads(raw)
    text = "".join(part.get("text", "") for item in result.get("output", [])
                   for part in item.get("content", []) if part.get("type") == "output_text")
    check("responses-text", status == 200 and text.strip() == "4", result)
    print(json.dumps({"pass": True, "checks": len(results)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
