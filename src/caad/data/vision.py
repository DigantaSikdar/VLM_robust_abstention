"""
caad/data/vision.py
===================
Image input plumbing shared by the CAAD step (trainers/caad.py) and the
rollout engine: the chat messages and the processor call. Keeping both in one
place guarantees the student, teacher and rollout see the image identically.

The decoded image goes straight to the model's own processor (its default
resizing, or ``model.max_pixels`` if the recipe sets one), exactly as the
vLLM evaluation does, so training and evaluation see identical pixels.
Qwen-VL processors return ``pixel_values`` / ``image_grid_thw`` for images.
"""

from __future__ import annotations

PIXEL_KEY, GRID_KEY = "pixel_values", "image_grid_thw"


def image_item(path, max_pixels=None):
    """The chat-content item for an image."""
    item = {"type": "image", "image": path}
    if max_pixels:
        item["max_pixels"] = max_pixels
    return item


def build_messages(image_path, user_text, system=None, max_pixels=None):
    msgs = [{"role": "user", "content": [image_item(image_path, max_pixels),
                                         {"type": "text", "text": user_text}]}]
    if system:
        msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
    return msgs


def process_image(processor, text, messages):
    """Decode the image in ``messages`` and run the processor on it."""
    from PIL import Image
    path = next(c["image"] for m in messages if isinstance(m["content"], list)
                for c in m["content"] if c.get("type") == "image")
    with Image.open(path) as im:
        img = im.convert("RGB")
    return processor(text=[text], images=[img], return_tensors="pt")


def batch_images(processor, per_prompt):
    """
    The ``images=`` argument for a batched processor call, given one list of images per
    prompt (empty for a text-only prompt). Gemma 3's processor needs this per-prompt
    nesting; Qwen-VL and InternVL take the flat list. None when the batch has no image.
    """
    flat = [im for ims in per_prompt for im in ims]
    if not flat:
        return None
    return per_prompt if type(processor).__name__.startswith("Gemma3") else flat
