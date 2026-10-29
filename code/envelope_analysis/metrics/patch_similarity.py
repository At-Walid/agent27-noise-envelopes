from __future__ import annotations

import re
from difflib import SequenceMatcher

_DIFF_HEADER_RE = re.compile(r"^--- .+\n\+\+\+ .+", re.MULTILINE)
# A fenced block; the closing fence is optional so a reply cut off by max_tokens still counts.
_FENCED_BLOCK_RE = re.compile(r"```[A-Za-z]*[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)
_SECTION_START_RE = re.compile(r"^(?=--- )", re.MULTILINE)
_FILE_HEADER_RE = re.compile(
    r"^--- (?:a/)?(?P<old>\S+)[^\n]*\n\+\+\+ (?:b/)?(?P<new>\S+)", re.MULTILINE
)
_FIRST_DIFF_LINE_RE = re.compile(r"^(?:diff --git |--- )", re.MULTILINE)
# What a line inside a unified diff can start with (a bare blank line is not one of them).
_DIFF_LINE_RE = re.compile(r"^(?:[ +\-\\]|@@|diff |index )")


def has_unified_diff(text: str | None) -> bool:
    """True if ``text`` holds a ---/+++ file-header pair and at least one @@ hunk marker."""
    return bool(text) and bool(_DIFF_HEADER_RE.search(text)) and "@@" in text


def _trim_trailing_prose(section: str) -> str:
    """Drop what follows a file's last hunk: blank lines, closing tags, explanations."""
    lines = section.split("\n")
    while lines and not _DIFF_LINE_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines)


def extract_final_patch(reply: str | None) -> str:
    """The patch a model reply ends up proposing, without prose or superseded drafts.

    The LLM Baseline is told to output only a unified diff, but in practice it answers with
    prose and often several drafts (```diff fences, or <patch> tags) that revise the same file
    one after another. This keeps the diff text only and, per file, only the last version, so
    that C1 compares patches rather than explanations or first attempts. A reply without
    fenced diffs is read as one unfenced diff, from its first file header on; a reply without
    any diff falls back to the stripped reply.

    Heuristic limit: a reply that explores several *different* files keeps all of them, since
    an abandoned attempt on another file cannot be told apart from a genuine multi-file fix.
    """
    if not reply:
        return ""
    text = reply.replace("\r\n", "\n")

    blocks = [b for b in _FENCED_BLOCK_RE.findall(text) if has_unified_diff(b)]
    if blocks:
        fallback = "\n".join(b.strip("\n") for b in blocks)
    else:
        start = _FIRST_DIFF_LINE_RE.search(text)
        if start is None:
            return text.strip()
        fallback = text[start.start() :].strip()
        blocks = [fallback]

    final: dict[str, str] = {}  # file path -> its last version; dict order = first appearance
    for block in blocks:
        for section in _SECTION_START_RE.split(block):
            header = _FILE_HEADER_RE.match(section)
            if header:
                path = header["new"] if header["new"] != "/dev/null" else header["old"]
                final[path] = _trim_trailing_prose(section.strip("\n"))
    return "\n".join(final.values()) or fallback


def patch_files(patch: str | None) -> list[str]:
    """Files a unified diff modifies, in first-appearance order, without ``a/``/``b/`` prefixes.

    Each file is named by its target path, or by its source path when it is deleted.
    """
    files: list[str] = []
    for header in _FILE_HEADER_RE.finditer((patch or "").replace("\r\n", "\n")):
        path = header["new"] if header["new"] != "/dev/null" else header["old"]
        if path not in files:
            files.append(path)
    return files


def file_overlap_jaccard(files_a: list[str], files_b: list[str]) -> float:
    """|A ∩ B| / |A ∪ B| over two file lists (metric C2 style); 0.0 when both are empty."""
    a, b = set(files_a), set(files_b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def c1_similarity(patch_a: str | None, patch_b: str | None) -> float:
    """Metric C1 — character-level diff similarity of two patches, in [0, 1].

    ``difflib.SequenceMatcher`` ratio over the raw patch text (2*M / T, where M is
    the number of matching characters and T the combined length). 1.0 means
    identical text; 0.0 means nothing in common. ``None`` is treated as an empty
    patch, so two empty patches are identical (1.0) while an empty patch against a
    non-empty one scores 0.0. Line endings are normalised and outer whitespace
    stripped so CRLF/LF or trailing-newline differences do not count as change.

    ``autojunk`` MUST stay disabled: with the default (True) difflib treats every
    character that makes up more than 1% of a 200+ character string as "junk",
    which collapses the ratio of two near-identical patches to ~0.
    """
    a = (patch_a or "").replace("\r\n", "\n").strip()
    b = (patch_b or "").replace("\r\n", "\n").strip()
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


class PatchSimilarity:
    """Metrics C1–C4 — Patch Similarity.

    C1: Character-level diff similarity (optional in the earlier pilot).
    C2: File overlap between patches.
    C3: Line overlap between patches.
    C4: AST similarity between patches.

    When C1 is available the StabilityScore (F) uses weights w_A=0.7, w_C=0.3.
    """

    def compute(self, run_data: list[dict]) -> None:
        # Deferred to future work
        return None
