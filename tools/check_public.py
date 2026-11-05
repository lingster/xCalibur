#!/usr/bin/env python3
"""Check the public file surface without printing potentially sensitive values.

Defaults to tracked files plus non-ignored additions. --tracked checks only the
index's path list. Content is read from the working tree. Git history and visual
content inside images require separate review.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = {
    ".gitignore", ".github", "LICENSE", "README.md", "pyproject.toml", "MANIFEST.in",
    "assets", "data", "docs", "examples", "kernels", "notebooks", "tests",
    "tools", "xcalibur",
}
PRIVATE_PARTS = {"garage", ".archive", ".agents", ".codex", ".aws", ".secrets", ".kaggle"}
PATTERNS = {
    "home-directory path": r"/(?:home|Users)/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\",
    "credential token": r"\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})\b",
    "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "email address": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "private runtime URL": r"https?://[^\s<>\"']*(?:ngrok|trycloudflare|modal\.run)[^\s<>\"']*",
    "URL credentials": r"https?://[^\s/:]+:[^\s/@]+@",
}


def sensitive_text(text, private_terms=()):
    """Return only (line, category); never include matching values in diagnostics."""
    findings = []
    for category, pattern in PATTERNS.items():
        findings.extend((text.count("\n", 0, m.start()) + 1, category)
                        for m in re.finditer(pattern, text))
    for term in private_terms:
        if term:
            findings.extend((text.count("\n", 0, m.start()) + 1, "private term")
                            for m in re.finditer(re.escape(term), text, re.IGNORECASE))
    return sorted(set(findings))


def forbidden_path(path):
    p = Path(path)
    return (p.is_absolute() or ".." in p.parts or bool(PRIVATE_PARTS.intersection(p.parts))
            or p.name in {"credentials.json", "token.json"}
            or (p.name.startswith(".env") and p.name != ".env.example")
            or p.suffix in {".pem", ".key"})


def notebook_text(nb):
    """Inspect source, textual outputs and metadata; exclude image/base64 payloads."""
    parts = [json.dumps(nb.get("metadata", {}), ensure_ascii=False)]
    for cell in nb.get("cells", []):
        parts.append("".join(cell.get("source", [])))
        parts.append(json.dumps(cell.get("metadata", {}), ensure_ascii=False))
        for output in cell.get("outputs", []):
            if "text" in output:
                parts.append("".join(output["text"]))
            if "traceback" in output:
                parts.extend(output["traceback"])
            parts.append(json.dumps(output.get("metadata", {}), ensure_ascii=False))
            for mime, value in output.get("data", {}).items():
                if mime.startswith("text/") or mime in {"application/json", "image/svg+xml"}:
                    parts.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


def png_text(data):
    """Read PNG text metadata rather than searching compressed pixel bytes."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("invalid PNG signature")
    pos, values = 8, []
    while pos + 12 <= len(data):
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        kind, payload = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + size]
        if len(payload) != size:
            raise ValueError("truncated PNG")
        if kind == b"tEXt":
            values.append(payload.decode("latin1"))
        elif kind == b"zTXt":
            key, value = payload.split(b"\0", 1)
            values.append(key.decode("latin1") + " " + zlib.decompress(value[1:]).decode("latin1"))
        elif kind == b"iTXt":
            key, rest = payload.split(b"\0", 1)
            compressed, rest = rest[0], rest[2:]
            _, rest = rest.split(b"\0", 1)  # language
            _, value = rest.split(b"\0", 1)  # translated keyword
            if compressed:
                value = zlib.decompress(value)
            values.append(key.decode("latin1") + " " + value.decode("utf8"))
        elif kind == b"eXIf":
            # Fail closed: publishing GPS/camera metadata needs explicit review.
            raise ValueError("EXIF metadata requires removal or review")
        pos += size + 12
    return "\n".join(values)


def anchors(text):
    result = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', text))
    seen = {}
    for title in re.findall(r"^#{1,6}\s+(.+?)(?:\s+#+)?$", text, re.MULTILINE):
        title = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", title)
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        index = seen.get(slug, 0)
        seen[slug] = index + 1
        result.add(slug + (f"-{index}" if index else ""))
    return result


def local_links(text):
    # Inline Markdown and HTML links/images used by this repository.
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    links = re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", text)
    links += re.findall(r'(?:href|src)=["\']([^"\']+)', text)
    return [url.strip("<>") for url in links]


def check_links(root, name, text, files):
    findings = []
    for url in local_links(text):
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            continue
        target = (root / name).parent / unquote(parts.path) if parts.path else root / name
        try:
            rel = target.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            findings.append("link escapes checkout")
            continue
        if rel not in files and not any(p.startswith(rel.rstrip("/") + "/") for p in files):
            findings.append(f"link target not in public files: {rel}")
        elif parts.fragment and target.suffix == ".md" and target.is_file():
            if unquote(parts.fragment) not in anchors(target.read_text()):
                findings.append(f"missing anchor: {rel}#{parts.fragment}")
    return findings


def inspect(root, files, private_terms=()):
    findings = []
    for name in sorted(files):
        path = root / name
        if forbidden_path(name) or Path(name).parts[0] not in PUBLIC_ROOTS:
            findings.append((name, "unapproved public path"))
            continue
        if path.is_symlink():
            findings.append((name, "symlink requires explicit publication review"))
            continue
        try:
            if path.suffix == ".png":
                text = png_text(path.read_bytes())
            elif path.suffix == ".ipynb":
                nb = json.loads(path.read_text())
                text = notebook_text(nb)
                if any(o.get("output_type") == "error" for c in nb.get("cells", []) for o in c.get("outputs", [])):
                    findings.append((name, "notebook contains an error output"))
                if any(c.get("cell_type") == "code" and c.get("source") and c.get("execution_count") is None for c in nb.get("cells", [])):
                    findings.append((name, "notebook has an unexecuted code cell"))
                for c in nb.get("cells", []):
                    if c.get("cell_type") == "markdown":
                        findings.extend((name, f) for f in check_links(root, name, "".join(c.get("source", [])), files))
            else:
                text = path.read_text(encoding="utf8")
            findings.extend((name, f"line {line}: {category}") for line, category in sensitive_text(text, private_terms))
            if path.suffix == ".md":
                findings.extend((name, f) for f in check_links(root, name, text, files))
        except (ValueError, OSError, UnicodeError, KeyError, struct.error, zlib.error):
            findings.append((name, "unreadable or unsupported publication content"))
    return findings


def public_files(root, tracked=False):
    args = ["git", "ls-files", "--cached", "-z"]
    if not tracked:
        args += ["--others", "--exclude-standard"]
    selected = {p for p in subprocess.check_output(args, cwd=root).decode().split("\0") if p}
    deleted = set(subprocess.check_output(["git", "ls-files", "--deleted", "-z"], cwd=root).decode().split("\0"))
    return selected - deleted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true")
    parser.add_argument("--private-term", action="append", default=[], help="Additional case-insensitive term to reject; matching values are never printed")
    args = parser.parse_args()
    files = public_files(ROOT, args.tracked)
    findings = inspect(ROOT, files, args.private_term)
    for path, reason in findings:
        print(f"{path}: {reason}")
    print(f"Public surface: {len(files)} files; {len(findings)} findings. Git history is outside this check.")
    return bool(findings)


if __name__ == "__main__":
    sys.exit(main())
