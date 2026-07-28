#!/usr/bin/env python
"""LLM-as-judge harness for video descriptions (v0.8 round, pre-registered).

The judge never sees selector names or the video file -- it sees the clip's 8
FROZEN frames (docs/borissal/evalset_manifest.json, byte-frozen JPEGs) as
ground truth plus two blinded captions, and returns per-axis A/B/tie verdicts
as strict JSON. Backend-agnostic by design:

  prepare       captions.json + manifest -> judge_jobs.jsonl (each job =
                frames + prompt + blinded caption pair; every comparison is
                emitted TWICE with A/B swapped -- disagreement aggregates to
                a tie)
  prepare-qc    the 5-check judge validation suite (dense vs random@low,
                foreign-caption swap, sentence shuffle, verbosity probe)
  gemini        judge pending jobs via the Gemini API free tier (env
                GEMINI_API_KEY, paced; NOT a paid path)
  aggregate     jobs + verdicts -> win rates, sign tests, per-axis table,
                order-swap agreement, length-bias monitor (r>0.4 = gamed)

The Claude backend is intentionally NOT in this file: the Claude Code session
dispatches subagents that Read the frames listed in each job and append
verdict lines to the verdicts file (same schema as the gemini backend), so
judging costs no API money. Verdict line schema:
  {"id": <job id>, "axes": {"objects|actions|scene|temporal|hallucination":
   "A|B|tie"}, "overall": "A|B|tie", "judge": "<model label>"}

Rubric (fixed; the anti-gaming language is load-bearing -- see prereg):
the frames ARE the ground truth; length and unverifiable detail must not be
rewarded; unverifiable claims count AGAINST a caption (hallucination axis).
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "docs" / "borissal" / "evalset_manifest.json"

AXES = ["objects", "actions", "scene", "temporal", "hallucination"]

PROMPT_TEMPLATE = """You are judging two descriptions of the SAME short video.
The {n_frames} timestamped frames you are given ARE the ground truth -- judge only
against what is visible in them.

Rules (strict):
- Do NOT reward length, style, or vivid detail you cannot verify in the frames.
- Any claim not verifiable in the frames is a hallucination and counts AGAINST
  the caption that makes it.
- A shorter caption that is accurate beats a longer one with unverifiable claims.
- "tie" is a valid verdict for any axis where both are equally good or bad.

Compare Caption A and Caption B on these axes:
- objects: are the objects/people and their attributes correctly identified?
- actions: are movements and actions described correctly?
- scene: is the setting/situation/context right?
- temporal: is the order of events right (check frame timestamps)?
- hallucination: which caption makes FEWER unverifiable claims? (the one with
  fewer wins this axis)

Caption A:
{caption_a}

Caption B:
{caption_b}

Answer with STRICT JSON only, no other text:
{{"objects": "A|B|tie", "actions": "A|B|tie", "scene": "A|B|tie",
 "temporal": "A|B|tie", "hallucination": "A|B|tie", "overall": "A|B|tie",
 "reason": "<one short sentence>"}}"""


def _job_id(clip, spec_a, spec_b, ratio, order, tag):
    raw = f"{clip}|{spec_a}|{spec_b}|{ratio}|{order}|{tag}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _frames_for(manifest, clip):
    for r in manifest["clips"]:
        if r["name"] == clip:
            return [f["file"] for f in r["frames"]]
    raise SystemExit(f"clip {clip} not in manifest")


def _emit_pair(jobs, manifest, clip, spec_x, cap_x, spec_y, cap_y, ratio, tag):
    """Two blinded jobs per comparison: (x=A,y=B) and swapped. The canonical
    (unswapped) orientation records spec_x as 'first'."""
    frames = _frames_for(manifest, clip)
    for order, (sa, ca, sb, cb) in enumerate(
            [(spec_x, cap_x, spec_y, cap_y), (spec_y, cap_y, spec_x, cap_x)]):
        jobs.append({
            "id": _job_id(clip, spec_x, spec_y, ratio, order, tag),
            "tag": tag, "clip": clip, "ratio": ratio, "order": order,
            "first": spec_x, "second": spec_y,
            "a_is": sa, "b_is": sb, "frames": frames,
            "prompt": PROMPT_TEMPLATE.format(
                n_frames=len(frames), caption_a=ca, caption_b=cb),
        })


def _load_captions(path):
    """captions.json (eval_mllm_attach --generate) -> {clip: {key: caption}}
    where key is 'dense' or '<config>@<ratio>'."""
    rows = json.loads(Path(path).read_text())
    return {r["clip"]: {k: v for k, v in r.items() if k != "clip"} for r in rows}


def cmd_prepare(args):
    manifest = json.loads(MANIFEST.read_text())
    caps = _load_captions(args.captions)
    specs = [s for s in args.specs.split(";") if s]
    ratios = [float(r) for r in args.ratios.split(",")]
    jobs = []
    for clip, row in sorted(caps.items()):
        for ratio in ratios:
            rnd_key = f"random@{ratio}"
            for spec in specs:
                key = f"{spec}@{ratio}"
                if key not in row:
                    continue
                if rnd_key in row and spec != "random":
                    _emit_pair(jobs, manifest, clip, spec, row[key],
                               "random", row[rnd_key], ratio, "vs-random")
                if "dense" in row:
                    _emit_pair(jobs, manifest, clip, spec, row[key],
                               "dense", row["dense"], ratio, "vs-dense")
    Path(args.jobs).write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    print(f"wrote {len(jobs)} jobs ({len(jobs)//2} comparisons) -> {args.jobs}")


def cmd_prepare_qc(args):
    """The 5-check validation suite from captions.json of a QC run that
    contains 'dense' and 'random@0.1' captions."""
    manifest = json.loads(MANIFEST.read_text())
    caps = _load_captions(args.captions)
    clips = sorted(caps)[: args.n_clips]
    rng = random.Random(20260728)
    jobs = []
    pad = (" The lighting and composition give the footage a cinematic, atmospheric "
           "quality, and countless subtle details reward a careful, attentive viewer.")
    for i, clip in enumerate(clips):
        row = caps[clip]
        if "dense" not in row or "random@0.1" not in row:
            raise SystemExit(f"{clip}: QC captions need 'dense' and 'random@0.1'")
        dense, rnd = row["dense"], row["random@0.1"]
        # 1) dense must beat a starved random caption
        _emit_pair(jobs, manifest, clip, "dense", dense, "random@0.1", rnd, 0.1, "qc-dense-vs-random")
        # 2) a caption from a DIFFERENT clip must lose
        foreign = caps[clips[(i + 1) % len(clips)]]["dense"]
        _emit_pair(jobs, manifest, clip, "dense", dense, "foreign", foreign, 1.0, "qc-foreign")
        # 3) sentence-shuffled dense must lose on the temporal axis
        sents = [s.strip() for s in dense.split(". ") if s.strip()]
        if len(sents) >= 3:
            shuf = sents[:]
            rng.shuffle(shuf)
            if shuf != sents:
                _emit_pair(jobs, manifest, clip, "dense", dense, "shuffled",
                           ". ".join(shuf), 1.0, "qc-shuffle")
        # 5) verbose padding must not rescue a weak caption
        _emit_pair(jobs, manifest, clip, "dense", dense, "padded-random",
                   rnd + pad * 2, 0.1, "qc-verbose")
    Path(args.jobs).write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    print(f"wrote {len(jobs)} QC jobs on {len(clips)} clips -> {args.jobs}")


# ---------------------------------------------------------------- gemini ----

def _gemini_call(job, model, api_key):
    import base64
    import urllib.request
    parts = []
    for i, rel in enumerate(job["frames"]):
        t = rel.rsplit("_t", 1)[-1].rstrip(".jpg")
        parts.append({"text": f"Frame {i + 1} (t={t}):"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(
            (REPO_ROOT / rel).read_bytes()).decode()}})
    parts.append({"text": job["prompt"]})
    body = json.dumps({
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 512,
                             "responseMimeType": "application/json"},
    }).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=body, headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read())
    return out["candidates"][0]["content"]["parts"][0]["text"]


def cmd_gemini(args):
    import os
    import time
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("set GEMINI_API_KEY (AI Studio free-tier key; no billing)")
    done = set()
    vpath = Path(args.verdicts)
    if vpath.exists():
        done = {json.loads(l)["id"] for l in vpath.read_text().splitlines() if l}
    jobs = [json.loads(l) for l in Path(args.jobs).read_text().splitlines() if l]
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{len(todo)} pending of {len(jobs)} (pacing {args.rpm} rpm)")
    with vpath.open("a") as f:
        for i, job in enumerate(todo):
            t0 = time.time()
            try:
                text = _gemini_call(job, args.model, api_key)
                v = json.loads(text)
                rec = {"id": job["id"],
                       "axes": {a: v.get(a, "tie") for a in AXES},
                       "overall": v.get("overall", "tie"),
                       "reason": v.get("reason", ""), "judge": args.model}
            except Exception as e:  # noqa: BLE001 -- rate limits etc.: log & continue
                print(f"  [{job['id']}] {e}")
                if "429" in str(e):
                    time.sleep(60)
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            time.sleep(max(0.0, 60.0 / args.rpm - (time.time() - t0)))
            if (i + 1) % 20 == 0:
                print(f"  judged {i + 1}/{len(todo)}", flush=True)


def cmd_gemini_cli(args):
    """Judge via the gemini CLI (OAuth login, no API key/billing). Frames go
    in as @file references; -p runs non-interactive. Resumable like `gemini`."""
    import re
    import subprocess
    import time
    done = set()
    vpath = Path(args.verdicts)
    if vpath.exists():
        done = {json.loads(l)["id"] for l in vpath.read_text().splitlines() if l}
    jobs = [json.loads(l) for l in Path(args.jobs).read_text().splitlines() if l]
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{len(todo)} pending of {len(jobs)}")
    with vpath.open("a") as f:
        for i, job in enumerate(todo):
            refs = " ".join(f"@{REPO_ROOT / rel}" for rel in job["frames"])
            prompt = (f"These are 8 timestamped frames of one video: {refs}\n\n"
                      + job["prompt"])
            try:
                out = subprocess.run(
                    ["gemini", "-m", args.model, "-p", prompt],
                    capture_output=True, text=True, timeout=180, cwd=str(REPO_ROOT))
                m = re.search(r"\{.*\}", out.stdout, re.DOTALL)
                if not m:
                    raise ValueError(f"no JSON in output: {out.stdout[-200:]!r} "
                                     f"stderr={out.stderr[-200:]!r}")
                v = json.loads(m.group(0))
                rec = {"id": job["id"], "axes": {a: v.get(a, "tie") for a in AXES},
                       "overall": v.get("overall", "tie"),
                       "reason": v.get("reason", ""), "judge": f"gemini-cli:{args.model}"}
            except Exception as e:  # noqa: BLE001 -- quota etc.: log & continue
                print(f"  [{job['id']}] {e}")
                time.sleep(10)
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"  judged {i + 1}/{len(todo)}", flush=True)
            time.sleep(args.pause)


def cmd_verify_frames(args):
    """Integrity check after a cross-server copy: every manifest frame must
    exist under --root with a matching sha256-16. Exit 1 on any mismatch."""
    import hashlib as _h
    manifest = json.loads((Path(args.manifest)).read_text())
    root = Path(args.root)
    bad, n = [], 0
    for clip in manifest["clips"]:
        for fr in clip["frames"]:
            n += 1
            fp = root / fr["file"]
            if not fp.exists():
                bad.append(f"MISSING {fr['file']}")
            elif _h.sha256(fp.read_bytes()).hexdigest()[:16] != fr["sha256"]:
                bad.append(f"HASH MISMATCH {fr['file']}")
    for b in bad[:20]:
        print(b)
    print(f"{n - len(bad)}/{n} frames verified" + (f", {len(bad)} BAD" if bad else " -- all good"))
    if bad:
        raise SystemExit(1)


# ------------------------------------------------------------- aggregate ----

def _sign_test(wins, losses):
    """two-sided binomial sign test p-value (ties excluded upstream)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * p)


def cmd_aggregate(args):
    jobs = {j["id"]: j for l in [Path(args.jobs).read_text().splitlines()]
            for j in map(json.loads, filter(None, l))}
    verdicts = {}
    for line in filter(None, Path(args.verdicts).read_text().splitlines()):
        v = json.loads(line)
        verdicts.setdefault(v["id"], []).append(v)

    def first_wins(job, v):
        """verdict('A'|'B'|'tie') -> did job['first'] win this call?"""
        w = v["overall"]
        if w not in ("A", "B"):
            return None
        return (job["a_is"] == job["first"]) == (w == "A")

    # regroup swapped pairs by comparison key
    comps = {}
    for jid, job in jobs.items():
        key = (job["tag"], job["clip"], job["first"], job["second"], job["ratio"])
        comps.setdefault(key, {})[job["order"]] = jid
    n_missing, swap_agree, swap_total = 0, 0, 0
    results = {}
    len_bias = []
    for key, orders in sorted(comps.items()):
        tag = key[0]
        vs = []
        for order, jid in sorted(orders.items()):
            if jid not in verdicts:
                continue
            job = jobs[jid]
            calls = [first_wins(job, v) for v in verdicts[jid]]
            calls = [c for c in calls if c is not None]
            if calls:  # majority within repeats of one order
                vs.append(sum(calls) > len(calls) / 2)
        if len(vs) < len(orders):
            n_missing += 1
            continue
        if len(vs) == 2:
            swap_total += 1
            swap_agree += vs[0] == vs[1]
        outcome = "tie" if len(set(vs)) > 1 else ("win" if vs[0] else "loss")
        r = results.setdefault((tag, key[4]), {"win": 0, "loss": 0, "tie": 0})
        r[outcome] += 1
        jid0 = orders[min(orders)]
        pa, pb = jobs[jid0]["prompt"].split("Caption A:")[1].split("Caption B:")
        if outcome != "tie":
            len_bias.append((len(pa) - len(pb), 1 if outcome == "win" else 0))

    print(f"comparisons: {len(comps)} ({n_missing} incomplete)")
    if swap_total:
        print(f"order-swap agreement: {swap_agree}/{swap_total} = {swap_agree/swap_total:.0%}")
    report = {}
    for (tag, ratio), r in sorted(results.items()):
        n_dec = r["win"] + r["loss"]
        wr = r["win"] / n_dec if n_dec else float("nan")
        p = _sign_test(r["win"], r["loss"])
        report[f"{tag}@{ratio}"] = {**r, "win_rate_ex_ties": round(wr, 3),
                                    "sign_p": round(p, 4)}
        print(f"{tag}@{ratio}: W{r['win']}/T{r['tie']}/L{r['loss']}  "
              f"win-rate(ex-tie)={wr:.0%}  p={p:.3f}")
    if len(len_bias) >= 8:
        import statistics
        d = [x for x, _ in len_bias]
        w = [y for _, y in len_bias]
        if statistics.pstdev(d) > 0 and statistics.pstdev(w) > 0:
            r_pb = (sum(x * y for x, y in len_bias) / len(len_bias)
                    - statistics.mean(d) * statistics.mean(w)) / (
                statistics.pstdev(d) * statistics.pstdev(w))
            flag = "  ** GAMED (>0.4) **" if abs(r_pb) > 0.4 else ""
            print(f"length-bias r={r_pb:+.2f}{flag}")
            report["length_bias_r"] = round(r_pb, 3)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--captions", required=True)
    q.add_argument("--specs", required=True, help="';'-separated selector specs to judge")
    q.add_argument("--ratios", default="0.25,0.5")
    q.add_argument("--jobs", required=True)
    q = sub.add_parser("prepare-qc")
    q.add_argument("--captions", required=True)
    q.add_argument("--n-clips", type=int, default=20)
    q.add_argument("--jobs", required=True)
    q = sub.add_parser("gemini")
    q.add_argument("--jobs", required=True)
    q.add_argument("--verdicts", required=True)
    q.add_argument("--model", default="gemini-2.5-flash")
    q.add_argument("--rpm", type=float, default=8)
    q = sub.add_parser("gemini-cli")
    q.add_argument("--jobs", required=True)
    q.add_argument("--verdicts", required=True)
    q.add_argument("--model", default="gemini-2.5-flash")
    q.add_argument("--pause", type=float, default=2.0)
    q = sub.add_parser("verify-frames")
    q.add_argument("--manifest", default=str(MANIFEST))
    q.add_argument("--root", default=str(REPO_ROOT))
    q = sub.add_parser("aggregate")
    q.add_argument("--jobs", required=True)
    q.add_argument("--verdicts", required=True)
    q.add_argument("--out", default=None)
    args = p.parse_args()
    {"prepare": cmd_prepare, "prepare-qc": cmd_prepare_qc, "gemini": cmd_gemini,
     "gemini-cli": cmd_gemini_cli, "verify-frames": cmd_verify_frames,
     "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
