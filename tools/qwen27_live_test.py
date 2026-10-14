"""Text-backend protocol and API regression tests against a running Qwen27.

Only sends inference requests to the supplied local endpoints. No process or
deployment changes. Output contains timings and assertions for reproducibility.
"""
import argparse
import json
import socket
import time
import urllib.request


class Client:
    def __init__(self, port, drafter=0):
        self.drafter = drafter
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.socket.settimeout(240)
        self.reader = self.socket.makefile("rb")

    def send(self, line):
        self.socket.sendall((line+"\n").encode())

    def read(self):
        line = self.reader.readline().decode().strip()
        if not line:
            raise RuntimeError("engine disconnected")
        return line

    def gen(self, req, ids, count, suffix="", eos=()):
        self.send(f"GEN {req} {count} {len(eos)} " + " ".join(map(str, eos)) +
                  f" {len(ids)} " + " ".join(map(str, ids)) + f" {self.drafter} " + suffix)
        tokens = []
        self.last_logprobs = []
        while True:
            line = self.read().split()
            if line[0] == "T":
                tokens.append(int(line[2]))
                if len(line) > 3:
                    self.last_logprobs.append(float(line[3]))
            if line[0] == "D":
                assert int(line[1]) == req
                return tokens, line

    def close(self):
        self.reader.close()
        self.socket.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-port", type=int, default=18870)
    ap.add_argument("--api", default="http://127.0.0.1:18871")
    ap.add_argument("--long", action="store_true", help="also verify 2056-token dense attention and continuation")
    ap.add_argument("--mtp", action="store_true", help="exercise MTP and expect its capability flag")
    ap.add_argument("--dflash", action="store_true", help="exercise DFlash2 and expect its capability flag")
    args = ap.parse_args()
    assert not (args.mtp and args.dflash), "select one drafter"
    results = []
    def check(name, ok, detail=None):
        assert ok, (name, detail)
        result = {"test": name, "pass": True, "detail": detail}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    draft = 2 if args.dflash else (1 if args.mtp else 0)
    c = Client(args.engine_port, draft)
    c.send("PING")
    check("ping", c.read() == "PONG")
    c.send("INFO")
    info = c.read().split()
    check("capabilities", info[:3] == ["I", str(draft), "0"] and info[10] == "1", info)
    t, d = c.gen(1, [1, 2, 3, 4], 1, eos=[5])
    check("eos-is-not-streamed", not t and d[2] == "done", d)
    t, d = c.gen(2, [248320], 1)
    check("invalid-token-rejected", not t and d[2] == "error", d)
    prompt = [1, 2, 3, 4]
    tokens, stats = c.gen(3, prompt, 8)
    check("greedy-generation", tokens == [5, 6, 24218, 10, 4838, 1665, 25, 33898], stats)
    continuation = prompt + tokens
    cached, cached_stats = c.gen(4, continuation, 8)
    c.gen(5, [100], 1)
    fresh, fresh_stats = c.gen(6, continuation, 8)
    check("cache-continuation-parity", cached == fresh and int(cached_stats[10]) > 0,
          {"tokens": fresh, "cached": cached_stats, "fresh": fresh_stats})
    suffix = "SAMPLE 0.7 20 0.9 0 12345 LOGPROBS PENALTY 0.2 0.1"
    first, _ = c.gen(7, prompt, 8, suffix)
    c.gen(8, [101], 1)
    second, _ = c.gen(9, prompt, 8, suffix)
    check("seeded-sampling-replay", first == second, second)
    # Cross a prefill-chunk boundary and then exercise causal cache continuation.
    long_prompt = ([1, 2, 3, 4, 5, 6, 7, 8] * 17)
    t, d = c.gen(10, long_prompt, 4)
    follow = long_prompt+t
    a, da = c.gen(11, follow, 4)
    c.gen(12, [102], 1)
    b, db = c.gen(13, follow, 4)
    check("multi-chunk-continuation", a == b, {"prefill": d, "cached": da, "fresh": db})
    if args.long:
        ids = [1, 2, 3, 4, 5, 6, 7, 8] * 257
        a, da = c.gen(101, ids, 16)
        continued, dc = c.gen(102, ids+a, 8)
        c.gen(103, [104], 1)
        fresh, df = c.gen(104, ids+a, 8)
        check("dense-attention-beyond-2051", continued == fresh,
              {"tokens": a, "continuation": continued, "prefill": da, "cached": dc, "fresh": df})
    c.send("GEN 14 1000 0 4 1 2 3 4 0")
    while not c.read().startswith("T 14 "):
        pass
    c.send("X 14")
    started = time.monotonic()
    while True:
        line = c.read()
        if line.startswith("D 14 "):
            break
    check("cancel", line.split()[2] == "cancel" and time.monotonic()-started < 10, line)
    c.send("PING")
    check("healthy-after-cancel", c.read() == "PONG")
    c.close()
    def chat(body):
        body.update(model="qwen3.8-27b-hgn", temperature=0, enable_thinking=False, max_tokens=128)
        request = urllib.request.Request(args.api+"/v1/chat/completions",
                    json.dumps(body).encode(), {"Content-Type": "application/json"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
        return result, time.monotonic()-started
    body, seconds = chat({"messages": [{"role": "user", "content": "17乘以23等于多少？只回答数字。"}]})
    text = body["choices"][0]["message"]["content"]
    check("chinese-arithmetic", text.strip() == "391", {"response": body, "seconds": seconds})
    body, seconds = chat({"messages": [{"role": "user", "content": "请调用 add 工具计算 17 加 23。"}],
        "tools": [{"type": "function", "function": {"name": "add", "description": "把两个整数相加",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]})
    msg = body["choices"][0]["message"]
    call = msg["tool_calls"][0]["function"]
    check("tool-call", call["name"] == "add" and json.loads(call["arguments"]) == {"a":17,"b":23}
          and "<|im_end|>" not in msg.get("content", ""), {"response":body,"seconds":seconds})
    print(json.dumps({"pass": True, "checks": len(results), "results": results}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
