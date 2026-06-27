#!/usr/bin/env python3
"""Extract hands-on lab scripts from ai-infra-docs MDX files."""
from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[2] / "hexoblog" / "docs" / "ai-infra-docs"
OUT_ROOT = Path(__file__).resolve().parents[1]

LAYER_DIRS = {
    "L0-foundations": "L0-foundations",
    "L1-compute-orchestration": "L1-compute-orchestration",
    "L2-data-operators-compilers": "L2-data-operators-compilers",
    "L3-training": "L3-training",
    "L4-inference-serving": "L4-inference-serving",
    "L5-mlops-llmops": "L5-mlops-llmops",
    "L6-application-architecture": "L6-application-architecture",
}

# Map comment/filename hints to output filenames
NAME_HINTS = [
    (r"# ([\w.-]+\.py)", 1),
    (r"python(?:3)? ([\w./-]+\.py)", 1),
    (r"python ([\w./-]+\.py)", 1),
    (r"([\w.-]+\.py) ——", 1),
    (r"([\w.-]+\.cu)", 1),
    (r"([\w.-]+\.cpp)", 1),
    (r"([\w.-]+\.yaml)", 1),
    (r"([\w.-]+\.yml)", 1),
    (r"([\w.-]+\.json)", 1),
    (r"([\w.-]+\.c)\b", 1),
]

EXT_LANG = {
    "python": ".py",
    "bash": ".sh",
    "cuda": ".cu",
    "cpp": ".cpp",
    "c": ".c",
    "yaml": ".yaml",
    "yml": ".yml",
    "json": ".json",
    "dockerfile": "Dockerfile",
}


def infer_name(code: str, lang: str, block_idx: int) -> str:
    for pat, grp in NAME_HINTS:
        m = re.search(pat, code)
        if m:
            return Path(m.group(grp)).name
    ext = EXT_LANG.get(lang, f".{lang}")
    if ext == "Dockerfile":
        if "cpu" in code.lower():
            return "Dockerfile.cpu"
        if "gpu" in code.lower() or "cuda" in code.lower():
            return "Dockerfile.gpu"
        return "Dockerfile"
    return f"snippet_{block_idx}{ext}"


def extract_blocks(text: str) -> list[tuple[str, str]]:
    if "## 动手实践" not in text:
        return []
    section = text.split("## 动手实践", 1)[1]
    # stop at next top-level section
    section = re.split(r"\n## [^#]", section, maxsplit=1)[0]
    blocks: list[tuple[str, str]] = []
    for m in re.finditer(r"```(\w+)\n(.*?)```", section, re.DOTALL):
        lang, code = m.group(1), m.group(2).rstrip() + "\n"
        if lang in ("text", "mermaid", "dockerfile"):
            lang = "dockerfile" if lang == "dockerfile" else lang
        if lang == "text":
            continue
        blocks.append((lang, code))
    return blocks


def main() -> int:
    docs = DOCS_ROOT
    if not docs.is_dir():
        print(f"Docs not found: {docs}", file=sys.stderr)
        return 1

    manifest: list[str] = []
    for src_dir, dst_layer in LAYER_DIRS.items():
        src_path = docs / src_dir
        if not src_path.is_dir():
            continue
        for mdx in sorted(src_path.glob("*.mdx")):
            text = mdx.read_text(encoding="utf-8")
            blocks = extract_blocks(text)
            if not blocks:
                continue
            chapter = mdx.stem
            out_dir = OUT_ROOT / dst_layer / chapter
            out_dir.mkdir(parents=True, exist_ok=True)
            seen: dict[str, int] = {}
            for i, (lang, code) in enumerate(blocks):
                if lang == "bash" and not any(
                    x in code for x in ("#!/", "def ", "class ", "import ", "apiVersion:")
                ):
                    continue
                name = infer_name(code, lang, i)
                if name in seen:
                    seen[name] += 1
                    stem, suffix = Path(name).stem, Path(name).suffix
                    name = f"{stem}_{seen[name]}{suffix}"
                else:
                    seen[name] = 0
                out_file = out_dir / name
                out_file.write_text(code, encoding="utf-8")
                manifest.append(f"{out_dir.relative_to(OUT_ROOT)}/{name}")
                print(f"wrote {out_file}")

    (OUT_ROOT / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"\nExtracted {len(manifest)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
