"""Exercise production PLE ring pumping with delayed/partial/interrupted I/O."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX") or shutil.which("clang++"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if not args.cxx:
        parser.error("pass --cxx with a C++17 compiler")
    root = Path(__file__).resolve().parent.parent
    text = (args.source or root / "src/gpu/parts/10_ple_io.inc").read_text(encoding="utf-8")
    start = text.index("struct PleUring {")
    end = text.index("\n};", start)
    method = text[text.index("  void run(", start):end]
    with tempfile.TemporaryDirectory(prefix="qwenox-uring-") as temporary:
        work = Path(temporary)
        (work / "ple_uring_run.inc").write_text(method, encoding="utf-8")
        binary = work / ("test.exe" if os.name == "nt" else "test")
        subprocess.run([
            args.cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-D_CRT_SECURE_NO_WARNINGS",
            "-I", str(work), str(root / "tools/ple_uring_test.cpp"), "-o", str(binary),
        ], check=True, timeout=120)
        for mode in ("cold", "short-submit", "eintr", "short-read"):
            subprocess.run([str(binary), mode], check=True, timeout=10)


if __name__ == "__main__":
    main()
