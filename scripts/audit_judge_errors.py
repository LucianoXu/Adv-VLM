"""
Audit every completed grading for silently mis-scored judge calls.

`_load_compliance_judge` catches any API exception and returns the verdict
"error". The grading task then counts compliance as `verdict == "compliant"`,
so an "error" is silently counted as a REFUSAL. A rate-limited or otherwise
failed request therefore biases the reported compliance rate DOWNWARD, with
nothing in results.json to indicate it happened.

That matters here: the OpenAI account has a 10,000 requests/day cap on
gpt-4o-mini, and a day of diagnostics can exhaust it mid-run.

This walks every compliance_results.json and reports the error counts, so any
contaminated number can be found and regraded rather than trusted.

Usage:
    python scripts/audit_judge_errors.py
"""

import glob
import json
import os
import sys


def main() -> int:
    rows, contaminated = [], []
    for f in sorted(glob.glob("results/**/compliance_results.json", recursive=True)):
        d = os.path.dirname(f)
        # already quarantined; listing them again is noise
        if "_contaminated-gradings" in d:
            continue
        try:
            payload = json.loads(open(f).read())
        except Exception as e:
            rows.append((d, -1, -1, -1, f"unreadable: {e}"))
            continue
        recs = payload.get("records", [])
        adv_err = sum(1 for r in recs if r.get("adv_verdict") == "error")
        clean_err = sum(1 for r in recs if r.get("clean_verdict") == "error")
        note = ""
        if adv_err or clean_err:
            contaminated.append(d)
            note = "CONTAMINATED"
        rows.append((d, len(recs), adv_err, clean_err, note))

    if not rows:
        print("no compliance_results.json found under results/")
        return 0

    w = max(len(r[0]) for r in rows)
    print(f"{'grading dir':<{w}} {'n':>5} {'adv_err':>8} {'clean_err':>10}  note")
    print("-" * (w + 30))
    for d, n, ae, ce, note in rows:
        print(f"{d:<{w}} {n:5} {ae:8} {ce:10}  {note}")

    print()
    if contaminated:
        print(f"{len(contaminated)} contaminated grading(s) -- these under-report "
              f"compliance and must be regraded:")
        for d in contaminated:
            print(f"  {d}")
        return 1
    print("no judge errors: every graded verdict came back from the API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
