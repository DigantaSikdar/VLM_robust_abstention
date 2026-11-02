"""
caad/eval/probe_direction.py
============================
(1) Probe-direction steering vectors. For each trained model, fit the E15 linear
    probe (answerable vs unanswerable, VizWiz train) on the hidden state that the
    steering hook modifies: the output of decoder layer 18 (hidden_states index 19).
    Its weight vector, mapped back to the raw residual space and normalised, is the
    read-out direction w_hat (pointing toward "unanswerable"). Saved as
    steer_vectors_probe.pt with v[18] = w_hat * ||v_contrast[18]||, so lambda means the
    same steering magnitude as for the contrastive vector.
(2) E16 geometry, from the cached activations (VizWiz val, last prompt token):
    activation norm, separation of the two classes along w_hat (gap in units of the
    within-class spread), where the over-answered items sit, and how far each
    steering vector moves a state along w_hat relative to that gap.

    python -m caad.eval.probe_direction --probe outputs/probe --vectors outputs/steer_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .probe_answerability import choose_l2, fit_lr, load

IDX = 19          # hidden_states index of decoder layer 18's output (where steering is added)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="outputs/probe")
    ap.add_argument("--vectors", default="outputs/steer_full")
    ap.add_argument("--layers", type=int, nargs="+", default=[19, 24])
    ap.add_argument("--tags", nargs="+", default=["base", "rkl", "grpo"])
    a = ap.parse_args()
    # norm-matched lambdas (steer_full --norm-ref outputs/steer_full/rkl)
    lam = {"rkl": {"contrast": 8.0, "noimg": 1.0}, "grpo": {"contrast": 8.77, "noimg": 1.04},
           "frkl": {"contrast": 5.90, "noimg": 0.952}, "fgrpo": {"contrast": 9.25, "noimg": 1.90}}
    res = {}
    for tag in a.tags:
        d = Path(a.probe) / tag
        Xtr, mtr = load(d, "vizwiz_train"); Xte, mte = load(d, "vizwiz_val")
        ytr = np.array([m["unanswerable"] for m in mtr]); yte = np.array([m["unanswerable"] for m in mte])
        res[tag] = {}
        for l in a.layers:
            mu, sd = Xtr[:, l].mean(0), Xtr[:, l].std(0) + 1e-6
            l2, _ = choose_l2((Xtr[:, l] - mu) / sd, ytr)
            w, b = fit_lr((Xtr[:, l] - mu) / sd, ytr, l2)
            w_raw = w.cpu().numpy() / sd
            w_hat = w_raw / np.linalg.norm(w_raw)
            proj = Xte[:, l] @ w_hat                       # raw-space coordinate along the read-out direction
            gap = proj[yte].mean() - proj[~yte].mean()
            spread = np.sqrt(0.5 * (proj[yte].var() + proj[~yte].var()))
            row = {"l2": l2, "mean_norm": float(np.linalg.norm(Xte[:, l], axis=1).mean()),
                   "proj_mean_unans": float(proj[yte].mean()), "proj_mean_ans": float(proj[~yte].mean()),
                   "class_gap": float(gap), "within_class_sd": float(spread), "d_prime": float(gap / spread)}
            if tag in lam:
                for vt, lm in lam[tag].items():
                    p = Path(a.vectors) / tag / f"steer_vectors{'_contrast' if vt == 'contrast' else '_noimg'}.pt"
                    v = torch.load(p)["v"][l - 1].numpy()
                    row[f"{vt}_norm"] = float(np.linalg.norm(v))
                    row[f"{vt}_cos"] = float(v @ w_hat / np.linalg.norm(v))
                    row[f"{vt}_push_along_w_in_gaps"] = float(lm * (v @ w_hat) / gap)
                    row[f"{vt}_push_rel_to_norm"] = float(lm * np.linalg.norm(v) / row["mean_norm"])
            res[tag][l] = row
            if tag in lam and l == IDX:                     # save the probe-direction steering vector
                vc = torch.load(Path(a.vectors) / tag / "steer_vectors_contrast.pt")["v"]
                v = torch.zeros_like(vc)
                v[IDX - 1] = torch.tensor(w_hat, dtype=v.dtype) * vc[IDX - 1].norm()
                torch.save({"v": v, "stats": {"source": "E15 probe at hidden_states[19]", "l2": l2,
                                              "scaled_to_contrast_norm": float(vc[IDX - 1].norm())}},
                           Path(a.vectors) / tag / "steer_vectors_probe.pt")
    Path(a.probe, "geometry_e16.json").write_text(json.dumps(res, indent=2))
    for tag, rows in res.items():
        for l, r in rows.items():
            print(tag, f"L{l}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
