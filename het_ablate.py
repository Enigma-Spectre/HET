#!/usr/bin/env python3
# het_ablate.py (A0..A3; A3 uses --qrope_conjugate)
# Trains each variant, then runs inference (temp=0.65, top_p=0.95) on the best checkpoint.

import argparse, subprocess, sys, time, re, json, os, shutil, csv
from pathlib import Path
from datetime import datetime, timezone
import torch

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "het_train.py"
INFER = HERE / "het_inference.py"
RUNS_DIR = HERE / "ablate_runs"

EVAL_RE = re.compile(r"\[eval\]\s*step\s*(\d+)\s*\|\s*val_loss\s*([0-9.]+)")

def device_default():
    return "cuda" if torch.cuda.is_available() else "cpu"

def run_proc_capture(cmd, log_file: Path):
    with open(log_file, "w", encoding="utf-8") as logf:
        logf.write("# CMD: " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        buf = []
        for line in proc.stdout:
            buf.append(line)
            logf.write(line)
        ret = proc.wait()
    return ret, "".join(buf)

def run_train(test_name: str, args_list: list):
    outdir = RUNS_DIR / test_name
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train.log"

    t0 = time.time()
    cmd = [sys.executable, str(TRAIN)] + args_list + ["--out_dir", str(outdir)]
    last_eval = None

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("# CMD: " + " ".join(cmd) + "\n\n"); logf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            logf.write(line)
            if "[eval]" in line:
                m = EVAL_RE.search(line)
                if m:
                    step = int(m.group(1)); val = float(m.group(2))
                    last_eval = (step, val)
            if "step " in line or "[eval]" in line:
                print(f"[{test_name}] {line.strip()}")
        ret = proc.wait()
    dt = time.time() - t0

    best_ckpt = None; best_val = None
    if outdir.exists():
        cks = sorted(outdir.glob("model_step*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cks:
            best_ckpt = cks[0]
            try:
                obj = torch.load(best_ckpt, map_location="cpu")
                best_val = float(obj.get("best_val", "nan"))
            except Exception:
                pass
    if best_val is None and last_eval is not None:
        best_val = last_eval[1]

    return {
        "seconds": round(dt, 2),
        "best_val": best_val,
        "best_ckpt": str(best_ckpt) if best_ckpt else None,
        "final_eval_val": (None if last_eval is None else last_eval[1]),
        "train_log": str(log_path),
        "cmd": cmd,
    }

def run_inference(test_name: str, ckpt_path: Path, device: str, max_seq_len: int):
    """
    Uses het_inference.py one-shot mode:
      --ckpt --device --dtype --prompt --max_new --temperature --top_p --max_seq_len
    """
    outdir = RUNS_DIR / test_name
    inf_dir = outdir / "inference"
    inf_dir.mkdir(parents=True, exist_ok=True)

    if not INFER.exists():
        return {"error": f"het_inference.py not found at {INFER}", "generations": []}

    prompts = [
        "In one sentence, summarize the central themes of The Lord of the Rings.",
        "Write a vivid two-sentence description of Rivendell as if seen at dawn.",
    ]
    max_news = [64, 128, 192]
    temperature = 0.65
    top_p = 0.95
    dtype = "fp16" if device.startswith("cuda") else "fp32"

    generations = []
    for pi, prompt in enumerate(prompts):
        for mn in max_news:
            log_file = inf_dir / f"p{pi+1}_mn{mn}.log"
            cmd = [
                sys.executable, str(INFER),
                "--ckpt", str(ckpt_path),
                "--device", device,
                "--dtype", dtype,
                "--prompt", prompt,
                "--max_new", str(mn),
                "--temperature", str(temperature),
                "--top_p", str(top_p),
                "--max_seq_len", str(max_seq_len),
            ]
            ret, out_text = run_proc_capture(cmd, log_file)
            generations.append({
                "prompt_idx": pi+1,
                "prompt": prompt,
                "max_new": mn,
                "temperature": temperature,
                "top_p": top_p,
                "return_code": ret,
                "output": out_text.strip(),
                "log": str(log_file),
            })

    with open(inf_dir / "inference_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": generations}, f, indent=2)

    return {"generations": generations, "dir": str(inf_dir)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, required=True, help="Path to LOTR.txt")
    ap.add_argument("--device", type=str, default=device_default())
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=7e-5)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--train_split", type=float, default=0.995)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--amp", action="store_true", help="Enable AMP")
    ap.add_argument("--clean", action="store_true", help="Delete ./ablate_runs before starting")
    args = ap.parse_args()

    if not TRAIN.exists():
        sys.exit(f"Trainer not found at {TRAIN}. Put het_train.py next to this script.")
    if not INFER.exists():
        print(f"[warn] Inference script not found at {INFER}. Training will proceed; inference will be skipped.", file=sys.stderr)

    if args.clean and RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    shared = [
        "--corpus", str(Path(args.corpus).resolve()),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--d_model", "256",
        "--n_layers", "6",
        "--n_heads", "8",
        "--seq_len", str(args.seq_len),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--min_lr", str(args.min_lr),
        "--warmup", str(args.warmup),
        "--eval_every", str(args.eval_every),
        "--train_split", str(args.train_split),
        "--beta1", "0.9",
        "--beta2", "0.95",
        "--grad_clip", "1.0",
        "--dropout", "0.0",
        "--ffn_mult", "4",
        "--dtype", "fp16" if args.device.startswith("cuda") else "fp32",
        "--seed", str(args.seed),
        "--sliding_keep_pct", "0.14",
    ]
    if args.amp and args.device.startswith("cuda"):
        shared.append("--amp")

    # A0: Baseline — standard attention + standard RoPE
    A0 = list(shared)
    # A1: Quaternion Attention (no Q-RoPE)
    A1 = list(shared) + ["--quat_attention"]
    # A2: Q-RoPE (requires quat_attention; left-mul, cycle axis)
    A2 = list(shared) + ["--quat_attention", "--quat_rope", "--qrope_axis", "cycle"]
    # A3: Q-RoPE + conjugation (u*q*u^{-1})
    A3 = list(shared) + ["--quat_attention", "--quat_rope", "--qrope_axis", "cycle", "--qrope_conjugate"]

    plan = [
        ("A0_baseline", A0),
        ("A1_qattn", A1),
        ("A2_qrope", A2),
        ("A3_qrope_conj", A3),
    ]

    print("# Starting ablations...")
    results = []
    for name, argv in plan:
        print(f"\n=== {name} ===")
        train_res = run_train(name, argv)
        best_ckpt = train_res["best_ckpt"]
        inf_res = None
        if best_ckpt and INFER.exists():
            print(f"[{name}] Running inference on best checkpoint: {best_ckpt}")
            inf_res = run_inference(name, Path(best_ckpt), args.device, args.seq_len)
        else:
            print(f"[{name}] Skipping inference (no checkpoint or no het_inference.py).")

        results.append({
            "test": name,
            "seconds": train_res["seconds"],
            "best_val": train_res["best_val"],
            "final_eval_val": train_res["final_eval_val"],
            "best_ckpt": best_ckpt,
            "train_log": train_res["train_log"],
            "cmd": " ".join(train_res["cmd"]),
            "inference": inf_res,
        })

    # Persist results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = RUNS_DIR / "ablation_results.csv"
    json_path = RUNS_DIR / "ablation_results.json"
    fields = ["test", "seconds", "best_val", "final_eval_val", "best_ckpt", "train_log", "cmd"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            w.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "results": results}, f, indent=2)

    # Rank by best_val
    ranked = sorted(results, key=lambda r: (float("inf") if r["best_val"] is None else r["best_val"]))
    lines = []
    lines.append("Ablation Summary (lower val_loss is better)\n")
    lines.append(f"{'Rank':<4} {'Test':<22} {'best_val':>10} {'time_s':>10}")
    for i, r in enumerate(ranked, 1):
        bv = "NA" if r["best_val"] is None else f"{r['best_val']:.4f}"
        lines.append(f"{i:<4} {r['test']:<22} {bv:>10} {r['seconds']:>10.2f}")

    # Deltas vs A0
    base = next((r for r in results if r["test"] == "A0_baseline"), None)
    if base and base["best_val"] is not None:
        base_bv = base["best_val"]
        lines.append("\nDeltas vs A0_baseline (negative is better):")
        for r in results:
            if r["best_val"] is not None:
                delta = r["best_val"] - base_bv
                lines.append(f"  {r['test']:<22} Δbest_val={delta:+.4f}")

    # Inference digest
    lines.append("\nInference (temp=0.65, top_p=0.95) — first 120 chars per sample:")
    for r in results:
        lines.append(f"- {r['test']}:")
        inf = r.get("inference")
        if not inf or not inf.get("generations"):
            lines.append("    (no inference results)")
            continue
        for g in inf["generations"]:
            out = g.get("output","").replace("\r"," ").replace("\n"," ").strip()
            preview = (out[:120] + ("…" if len(out) > 120 else ""))
            lines.append(f"    p{g['prompt_idx']} mn={g['max_new']}: {preview}")

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(RUNS_DIR / "ablation_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"\nSaved:\n  {csv_path}\n  {json_path}\n  {RUNS_DIR/'ablation_summary.txt'}")

if __name__ == "__main__":
    main()
