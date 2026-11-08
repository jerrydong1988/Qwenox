#!/usr/bin/env python3
"""Run the production single-writer cache against file and allocation faults."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def section(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX"))
    args = parser.parse_args()
    compiler = args.cxx or shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        parser.error("pass --cxx with a C++17 compiler")
    root = Path(__file__).resolve().parent.parent
    source = (root / "src/gpu/parts/51_host_cfg.inc").read_text(encoding="utf-8")
    platform = (root / "src/gpu/parts/00_platform.inc").read_text(encoding="utf-8")
    (root / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="snap-writer-", dir=root / "build") as directory:
        work = Path(directory)
        cache = work / "cache"
        cache.mkdir()
        (work / "writer.inc").write_text(
            section(source, "static inline uint64_t mix64", "// c[i]") +
            section(source, "struct ChHdr", "static_assert") +
            section(source, "// ---- host buffers", "// ---- save"),
            encoding="utf-8")
        (work / "restore.inc").write_text(
            section(source, "static bool restore(GpuModel&", "// ---- startup"),
            encoding="utf-8")
        (work / "posix_io.inc").write_text(
            section(platform, "static inline int os_open_rd", "#endif"), encoding="utf-8")
        binary = work / ("writer.exe" if os.name == "nt" else "writer")
        subprocess.run([compiler, "-std=c++17", "-O1", "-D_CRT_SECURE_NO_WARNINGS",
                        "-I", str(work), str(root / "tools/kvsnap_writer_test.cpp"), "-o",
                        str(binary)], check=True, timeout=120)
        subprocess.run([str(binary), str(cache)], check=True, timeout=30)


if __name__ == "__main__":
    main()
