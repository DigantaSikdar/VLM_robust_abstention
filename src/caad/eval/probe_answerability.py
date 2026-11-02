"""
caad/eval/probe_answerability.py
================================
E15: is "this question can't be answered from the image" still linearly readable
from a model whose behaviour no longer abstains?

extract   One forward pass per question (no generation). Saves the residual-stream
          hidden state at the LAST PROMPT TOKEN (the position right before the answer
          starts) at every layer (embeddings + 36 decoder layers), float16.
          Sets:
            vizwiz_train   VizWiz train pool (1,101 unanswerable / 800 answerable)  probe TRAIN
            vizwiz_val     VizWiz val (1,385 / 2,934), disjoint split + images        probe TEST
            noimg_*        the same prompts with the image withheld (VizWiz val, and
                           samples of GQA / VQAv2 / TextVQA)                          transfer
            gqa_clean      clean GQA questions with their image (answerable reference)
          --steer DIR:LAMBDA adds lambda * v_18 (steer_vectors_<DIR>.pt) to layer 18 at the
          last prompt token, so later layers show what the steering vector does to the
          representation the probe reads (our steering otherwise acts on generated tokens
          only, where this position is identical to the unsteered model).

analyze   Per layer: standardize on the train set, fit an L2 logistic regression
          (label 1 = unanswerable; strength by 5-fold CV on train), report test AUROC.
          Controls: the same probe on shuffled train labels (selectivity, Hewitt & Liang
          2019). Transfer: fraction of no-image / clean inputs the probe flags as
          unanswerable. Behaviour link: probe flag rate on the val questions labeled
          unanswerable that the model nonetheless answered. Geometry: cosine between the
          probe's read-out direction and the contrastive steering vectors.

    python -m caad.eval.probe_answerability extract --tag rkl --lora <adapter> --out outputs/probe
    python -m caad.eval.probe_answerability analyze --out outputs/probe --tags base rkl grpo
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .steer_probe import decoder_layers
from .tasks.registry import get_task
from .tasks.vqa_c import extract_answer
from ..data.vision import batch_images

EVAL_ROOT = "/gscratch/dsikdar/caad_data/vqa_c"
CAL_ROOT = "/gscratch/dsikdar/caad_data/steer_calib"
PF = "think_answer"


def sets(n_other):
    rng = random.Random(0)
    S = {"vizwiz_train": [(e, True) for e in get_task("vizwiz").load({"root": CAL_ROOT, "prompt_format": PF})],
         "vizwiz_val": [(e, True) for e in get_task("vizwiz").load({"root": EVAL_ROOT, "prompt_format": PF})]}
    S["noimg_vizwiz"] = [(e, False) for e, _ in S["vizwiz_val"]]
    for ds in ("gqa", "vqav2", "textvqa"):
        qs = get_task(ds).load({"root": EVAL_ROOT, "prompt_format": PF, "use_corrupted": False})
        pick = rng.sample(qs, min(n_other, len(qs)))
        S[f"noimg_{ds}"] = [(e, False) for e in pick]
        if ds == "gqa":
            S["gqa_clean"] = [(e, True) for e in pick]
    return S


@torch.inference_mode()
def cmd_extract(a):
    from PIL import Image
    from transformers import AutoProcessor
    from .steer_probe import BASE_MODEL, load_policy
    model = load_policy(a.lora)                    # None = base; LoRA adapter dir; or full checkpoint dir
    proc = AutoProcessor.from_pretrained(BASE_MODEL)
    proc.tokenizer.padding_side = "left"
    if a.steer:                                    # add lambda * v_18 at the last prompt token
        d, lam = a.steer.split(":")
        tag = {"contrast": "_contrast", "noimg": "_noimg", "all": ""}[d]
        v = torch.load(Path(a.vectors) / f"steer_vectors{tag}.pt")["v"][18].cuda()
        lam = float(lam)

        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.clone()
            hs[:, -1, :] += lam * v.to(hs.dtype)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        decoder_layers(model)[18].register_forward_hook(hook)
    # capture only the last-position hidden state per layer (keeping every layer's full
    # sequence via output_hidden_states does not fit on a 24 GB GPU with VizWiz photos)
    grab = {}
    layers = decoder_layers(model)
    layers[0].register_forward_pre_hook(
        lambda mod, args, kwargs: grab.__setitem__(0, (args[0] if args else kwargs["hidden_states"])[:, -1, :].float()),
        with_kwargs=True)
    for i, layer in enumerate(layers):
        layer.register_forward_hook(
            lambda mod, inp, out, i=i: grab.__setitem__(i + 1, (out[0] if isinstance(out, tuple) else out)[:, -1, :].float()))
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    for name, items in sets(a.n_other).items():
        if a.only and name not in a.only:
            continue
        f = out / f"{name}.npz"
        if f.exists():
            continue
        feats, meta, t0 = [], [], time.time()
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]
            texts, imgs = [], []
            for e, with_img in chunk:
                content = [{"type": "text", "text": e["prompt"]}]
                if with_img:
                    content.insert(0, {"type": "image"})
                    with Image.open(e["image"]) as im:
                        imgs.append([im.convert("RGB")])
                else:
                    imgs.append([])
                msgs = [{"role": "user", "content": content}]
                if e["system"]:
                    msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": e["system"]}]})
                texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            inp = proc(text=texts, images=batch_images(proc, imgs), padding=True, return_tensors="pt").to("cuda")
            grab.clear()
            model(**inp, logits_to_keep=1)                                   # hooks fill grab[0..L]
            feats.append(torch.stack([grab[k] for k in range(len(layers) + 1)], 1).cpu().numpy())   # float32: layer 36 overflows float16
            meta += [{"id": e["id"], "dataset": e["dataset"], "unanswerable": e.get("category") == "unanswerable"}
                     for e, _ in chunk]
            if (i // a.batch) % 20 == 0:
                print(f"[{a.tag}/{name}] {i + len(chunk)}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        np.savez(f, X=np.concatenate(feats), meta=json.dumps(meta))


# --------------------------------------------------------------------------- #
# linear probe (torch, GPU): standardized features, L2 logistic regression
# --------------------------------------------------------------------------- #
def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_lr(X, y, l2, iters=200):
    X = torch.as_tensor(X, dtype=torch.float32, device="cuda")
    y = torch.as_tensor(y, dtype=torch.float32, device="cuda")
    pw = (len(y) - y.sum()) / y.sum().clamp(min=1)                      # class balance
    w = torch.zeros(X.shape[1], device="cuda", requires_grad=True)
    b = torch.zeros((), device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y, pos_weight=pw) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def score(X, w, b):
    return (torch.as_tensor(X, dtype=torch.float32, device="cuda") @ w + b).cpu().numpy()


def choose_l2(X, y, grid=(1e-4, 1e-3, 1e-2, 1e-1), k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y)); folds = np.array_split(idx, k)
    best = None
    for l2 in grid:
        aucs = []
        for f in folds:
            tr = np.setdiff1d(idx, f)
            w, b = fit_lr(X[tr], y[tr], l2)
            aucs.append(auroc(score(X[f], w, b), y[f]))
        m = float(np.mean(aucs))
        if best is None or m > best[1]:
            best = (l2, m)
    return best


def load(d, name):
    z = np.load(Path(d) / f"{name}.npz")
    return z["X"].astype(np.float32), json.loads(str(z["meta"]))


def cmd_analyze(a):
    res, probes = {}, {}
    for tag in [t for t in a.tags if t not in a.parent]:
        d = Path(a.out) / tag
        Xtr, mtr = load(d, "vizwiz_train"); Xte, mte = load(d, "vizwiz_val")
        ytr = np.array([m["unanswerable"] for m in mtr]); yte = np.array([m["unanswerable"] for m in mte])
        L = Xtr.shape[1]
        shuf = np.random.default_rng(1).permutation(ytr)
        rows = []
        for l in range(L):
            mu, sd = Xtr[:, l].mean(0), Xtr[:, l].std(0) + 1e-6
            A, B = (Xtr[:, l] - mu) / sd, (Xte[:, l] - mu) / sd
            l2, cv = choose_l2(A, ytr) if l in a.cv_layers or a.cv_all else (a.l2, float("nan"))
            w, b = fit_lr(A, ytr, l2)
            probes.setdefault(tag, {})[l] = (mu, sd, w, b)
            ws, bs = fit_lr(A, shuf, l2)
            rows.append({"layer": l, "l2": l2, "cv_auroc": cv, "test_auroc": auroc(score(B, w, b), yte),
                         "shuffled_test_auroc": auroc(score(B, ws, bs), yte)})
        best = max(rows, key=lambda r: r["test_auroc"] if np.isnan(r["cv_auroc"]) else r["cv_auroc"])
        l = a.probe_layer if a.probe_layer is not None else best["layer"]
        mu, sd = Xtr[:, l].mean(0), Xtr[:, l].std(0) + 1e-6
        w, b = fit_lr((Xtr[:, l] - mu) / sd, ytr, rows[l]["l2"])
        flag = lambda X: float((score((X[:, l] - mu) / sd, w, b) > 0).mean())
        transfer = {}
        for name in ("noimg_vizwiz", "noimg_gqa", "noimg_vqav2", "noimg_textvqa", "gqa_clean"):
            p = d / f"{name}.npz"
            if p.exists():
                X, m = load(d, name)
                transfer[name] = flag(X)
        sv = score((Xte[:, l] - mu) / sd, w, b)
        transfer["vizwiz_val_unanswerable"] = float((sv[yte] > 0).mean())
        transfer["vizwiz_val_answerable"] = float((sv[~yte] > 0).mean())
        # behaviour link: val questions labeled unanswerable that the model answered anyway
        beh = {}
        if a.results.get(tag):
            import glob
            said = {r["id"]: "unanswerable" in extract_answer(r["completion"]).lower()
                    for f in glob.glob(a.results[tag]) for r in map(json.loads, open(f))
                    if r.get("group") in ("vizwiz_unans", "vizwiz_ans")}
            idx = [i for i, mm in enumerate(mte) if mm["unanswerable"] and mm["id"] in said and not said[mm["id"]]]
            beh = {"n_answered_although_unanswerable": len(idx),
                   "probe_flags_them_unanswerable": float((sv[idx] > 0).mean()) if idx else None}
        # geometry: read-out direction (original space) vs steering vectors at the same depth
        geo = {}
        w_raw = (w.cpu().numpy() / sd)
        for vt in ("contrast", "noimg", "all"):
            p = Path(a.vectors.get(tag, "")) / f"steer_vectors{ {'contrast': '_contrast', 'noimg': '_noimg', 'all': ''}[vt]}.pt"
            if a.vectors.get(tag) and p.exists() and l >= 1:
                v = torch.load(p)["v"][l - 1].numpy()        # v[l-1] = output of decoder layer l-1 = hidden_states[l]
                geo[vt] = float(w_raw @ v / (np.linalg.norm(w_raw) * np.linalg.norm(v)))
        res[tag] = {"layers": rows, "probe_layer": l, "transfer_flag_rate": transfer, "behaviour": beh, "cosine_probe_vs_steer": geo}
        print(f"\n== {tag}: probe layer {l} (hidden_states index; 0 = embeddings)")
        print("   layer  test AUROC  shuffled  (selected by CV)" )
        for r in rows[::a.print_every] + ([rows[l]] if l % a.print_every else []):
            cv = "" if np.isnan(r["cv_auroc"]) else f"cv {r['cv_auroc']:.3f}"
            print(f"   {r['layer']:5d}  {r['test_auroc']:.3f}      {r['shuffled_test_auroc']:.3f}   {cv}")
        print("   flag rate (probe says unanswerable):", {k: round(v, 3) for k, v in transfer.items()})
        print("   behaviour:", beh, "| cosine(probe, steering):", {k: round(v, 3) for k, v in geo.items()})
    # steered variants: the PARENT model's probes read the steered features. Steering is
    # added to decoder layer 18's output = hidden_states index 19, so only layers >= 19 see it.
    for tag, par in a.parent.items():
        d, dp = Path(a.out) / tag, Path(a.out) / par
        Xs, ms = load(d, "vizwiz_val"); Xp, _ = load(dp, "vizwiz_val")
        y = np.array([m["unanswerable"] for m in ms])
        out = {}
        for l in sorted(probes[par]):
            if l < 19:
                continue
            mu, sd, w, b = probes[par][l]
            rd = lambda X: score((X[:, l] - mu) / sd, w, b)
            row = {"auroc_steered": auroc(rd(Xs), y), "auroc_parent": auroc(rd(Xp), y),
                   "mean_shift_toward_unanswerable": float((rd(Xs) - rd(Xp)).mean())}
            for name in ("noimg_vizwiz", "noimg_gqa", "noimg_vqav2", "noimg_textvqa", "gqa_clean"):
                if (d / f"{name}.npz").exists():
                    row[f"flag_{name}_steered"] = float((rd(load(d, name)[0]) > 0).mean())
                    row[f"flag_{name}_parent"] = float((rd(load(dp, name)[0]) > 0).mean())
            row["flag_val_unans_steered"] = float((rd(Xs)[y] > 0).mean()); row["flag_val_ans_steered"] = float((rd(Xs)[~y] > 0).mean())
            row["flag_val_unans_parent"] = float((rd(Xp)[y] > 0).mean()); row["flag_val_ans_parent"] = float((rd(Xp)[~y] > 0).mean())
            out[l] = row
        res[tag] = {"parent": par, "by_layer": out}
        print(f"\n== {tag} (read with {par}'s probes)")
        for l, r in out.items():
            if l % a.print_every == 0 or l == max(out):
                print(f"   layer {l}: AUROC {r['auroc_parent']:.3f} -> {r['auroc_steered']:.3f} | shift {r['mean_shift_toward_unanswerable']:+.2f} | "
                      + " ".join(f"{k[5:]} {r[k]:.2f}" for k in r if k.startswith("flag_") and k.endswith("steered")))
    Path(a.out, "analysis.json").write_text(json.dumps(res, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--tag", required=True)
    e.add_argument("--lora", default=None)
    e.add_argument("--out", default="outputs/probe")
    e.add_argument("--steer", default=None, help="<contrast|noimg|all>:<lambda>, added to layer 18 at the last prompt token")
    e.add_argument("--vectors", default=None)
    e.add_argument("--n-other", type=int, default=1000)
    e.add_argument("--batch", type=int, default=4)
    e.add_argument("--only", nargs="*", default=None)
    z = sub.add_parser("analyze")
    z.add_argument("--out", default="outputs/probe")
    z.add_argument("--tags", nargs="+", required=True)
    z.add_argument("--l2", type=float, default=1e-2)
    z.add_argument("--cv-layers", type=int, nargs="*", default=[])
    z.add_argument("--cv-all", action="store_true")
    z.add_argument("--probe-layer", type=int, default=None)
    z.add_argument("--print-every", type=int, default=3)
    z.add_argument("--results", type=json.loads, default={}, help='{"rkl": "<image-view trained results jsonl>"}')
    z.add_argument("--vectors", type=json.loads, default={}, help='{"rkl": "outputs/steer_full/rkl"}')
    z.add_argument("--parent", type=json.loads, default={}, help='{"rkl_steer_contrast8": "rkl"}')
    a = ap.parse_args()
    {"extract": cmd_extract, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
