"""Image pipeline: data prep, eval scoring, and build_inputs routing.
Needs OpenCV + Pillow (skipped otherwise)."""
import json

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("PIL")

from caad.data.corruption import apply_corruption  # noqa: E402


def test_prepare_synthetic_manifest(tmp_path):
    from PIL import Image

    from caad.data.dataset import ManifestDataset, spec_from_manifest_row
    from caad.data.prepare_images import build_manifest, synthetic_items
    man = tmp_path / "m.jsonl"
    n = build_manifest(synthetic_items(3, 32), out_dir=tmp_path, manifest=man,
                       styles=["fog", "jpeg"], severities=[2], global_seed=0)
    assert n == 6
    ds = ManifestDataset(man)
    row = ds[0]
    clean = np.asarray(Image.open(row["clean_image"]))
    corr = np.asarray(Image.open(row["corrupted_image"]))
    assert clean.shape == corr.shape                    # alignment invariant on disk
    # the recorded spec re-renders the stored corrupted image bit-for-bit
    again = apply_corruption(clean, spec_from_manifest_row(row))
    assert np.array_equal(again, corr)


def test_image_mcq_task(tmp_path):
    from caad.eval.tasks.registry import get_task
    man = tmp_path / "m.jsonl"
    man.write_text(json.dumps({"image_id": "x", "question": "q", "answer": "b",
                               "clean_image": "c.png", "corrupted_image": "k.png"}) + "\n")
    task = get_task("image_mcq")
    ex = task.load({"manifest": str(man), "use_corrupted": True})[0]
    assert ex["image"] == "k.png" and ex["gold"] == "B"
    assert task.score(ex, "Looking closely, the answer is B.")
    assert not task.score(ex, "The answer is C.")


def test_build_inputs_routes_image_tensors(monkeypatch):
    torch = pytest.importorskip("torch")
    import caad.trainers.caad as caad_mod

    class Tok:
        eos_token = "<|im_end|>"

        def get_vocab(self):
            return {"<|im_end|>": 1}

        def __call__(self, text, return_tensors, add_special_tokens):
            assert text.endswith("<|im_end|>")      # end-of-turn is scored too
            return type("T", (), {"input_ids": torch.tensor([[7, 8, 9]])})()

    class Proc:
        tokenizer = Tok()

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages[1]["content"][0]["type"] == "image"
            return "prompt"

    class Out(dict):
        input_ids = torch.tensor([[1, 2, 3, 4]])

    def fake_process(processor, text, messages):
        o = Out(pixel_values=torch.zeros(4, 6), image_grid_thw=torch.tensor([[1, 2, 2]]))
        return o

    monkeypatch.setattr(caad_mod, "process_image", fake_process)
    batch, cmask, plen = caad_mod.build_inputs(
        Proc(), "sys", "q?", "img.png", "ans", "cpu", 1000)
    assert set(batch) == {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    assert plen == 4 and batch["input_ids"].shape == (1, 7)
    assert cmask.tolist() == [[0, 0, 0, 0, 1, 1, 1]]
