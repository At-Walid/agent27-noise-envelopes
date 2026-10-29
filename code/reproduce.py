"""Reproduce one table or figure of the paper (or all of them) and check it against results/.

    python code/reproduce.py TARGET [--out DIR]

TARGET is one of the names in TARGETS, or "all". The whole analysis runs every time (one to two
minutes; paper_analysis.py); only TARGET's files are written to DIR (default: reproduced/), and each
is compared byte for byte with its copy in results/. Exit status 1 if any file differs.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
import tempfile
from pathlib import Path

import paper_analysis

ROOT = Path(__file__).resolve().parent.parent
ROBUSTNESS = paper_analysis.ROBUSTNESS_DIR + "/"

# paper item -> the files paper_analysis.py writes for it (a trailing "/": the whole folder)
TARGETS = {
    "rq1-noise": ["rq1_noise.csv", "rq1_noise.tex"],
    "rq1-noisy-issues": ["rq1_noisy_issues.csv", "rq1_noisy_issues.tex"],
    "rq1-backends": ["rq1_backends.csv", "rq1_backends.tex"],
    "fig1": ["fig1_repeat_similarity.pdf", "rq1_per_issue.csv"],
    "rq2-envelope": ["rq2_envelope.csv", "rq2_envelope.tex"],
    "rq2-stability": ["rq2_conclusion_stability.csv", "rq2_conclusion_stability.tex",
                      "rq2_conclusion_stability_ci.tex"],
    "fig2": ["fig2_conclusion_stability.pdf"],
    "rq2-posthoc-typo": ["rq2_posthoc_typo.csv", "rq2_posthoc_typo.tex"],
    "rq3-filelist": ["rq3_filelist.csv", "rq3_filelist.tex"],
    "fig3": ["fig3_rq3_deviation.pdf"],
    "robustness": ["robustness_changes.csv", "robustness_changes.tex", ROBUSTNESS],
    "numbers": ["NUMBERS.md"],
}


def selected(name: str, target: str) -> bool:
    if target == "all":
        return True
    return any(name == f or (f.endswith("/") and name.startswith(f)) for f in TARGETS[target])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("target", choices=[*TARGETS, "all"])
    parser.add_argument("--out", type=Path, default=ROOT / "reproduced")
    args = parser.parse_args(argv)
    differ = 0
    with tempfile.TemporaryDirectory() as tmp:
        paper_analysis.main(["--out", tmp])
        files = sorted(p.relative_to(tmp).as_posix() for p in Path(tmp).rglob("*") if p.is_file())
        for name in (f for f in files if selected(f, args.target)):
            dest = args.out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(tmp) / name, dest)
            expected = ROOT / "results" / name
            same = expected.is_file() and filecmp.cmp(dest, expected, shallow=False)
            differ += not same
            print(f"{'identical' if same else 'DIFFERS  '}  {name}")
    print(f"{args.target}: {'all files identical to results/' if not differ else f'{differ} differ'}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
