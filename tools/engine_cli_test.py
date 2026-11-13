"""Exercise missing-value failures through the real Flash CLI before model loading."""
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()
    options = [
        "--tokens", "--tokens-file", "--gen", "--split", "--ppl-window",
        "--ppl-stride", "--kld-base", "--kld-save", "--kld-chunks", "--kld-ctx",
        "--host", "--port", "--maxctx", "--mtp-test", "--spec-gen", "--gamma",
        "--mrope-grid", "--mrope-test", "--vision-tower", "--vision-test", "--patches",
        "--rope-factor", "--rope-original-ctx", "--rope-beta-fast",
        "--rope-beta-slow", "--rope-attn-scale",
    ]
    for option in options:
        result = subprocess.run(
            [args.engine, "missing-cli-fixture.hgn", option],
            capture_output=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        expected = option.encode() + b" requires"
        assert result.returncode == 2 and expected in result.stderr, (
            option, result.returncode, result.stderr.decode(errors="replace"))
    print(f"PASS {len(options)} missing-value CLI cases")


if __name__ == "__main__":
    main()
