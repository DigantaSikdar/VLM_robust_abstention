"""
caad/eval/tasks/vqa_c.py
========================
TextVQA-C and GQA-C (Usama et al., 2025) over manifests from
caad.data.prepare_vqa_c. ``use_corrupted: true`` scores the question on its
assigned corrupted image; false on the clean one. Each example carries the
corruption metadata so results can be broken down by category / severity /
seen-vs-unseen.

The model is prompted to reason briefly and put its final short answer in
<answer></answer> tags (PROMPT). ``extract_answer`` pulls that answer out of a
free-form completion; scoring then follows the paper (Sec. 4.3):

  textvqa  correct if the normalized prediction matches ANY of the 10
           annotator answers (also exposes the standard VQA soft accuracy,
           min(#matches / 3, 1), as ``soft_score``)
  gqa      normalized exact match against the single answer

Normalization is the standard VQA answer processing (lowercase, punctuation,
number words -> digits, articles and common contractions).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .registry import register

PROMPT = ("{question}\n"
          "Look carefully at the image and reason step by step. Then give your final "
          "answer as a single word or short phrase inside <answer></answer> tags.")

SYSTEM_PROMPT = (
    "You are a careful visual question-answering assistant. For each question about "
    "an image, first reason step by step about what you see inside <think> </think> "
    "tags. Then give your final answer inside <answer> </answer> tags, as a single "
    "word or short phrase.\n"
    "Your response must use exactly this format:\n"
    "<think> your reasoning </think>\n"
    "<answer> your final answer </answer>")

# Same request without "single word or short phrase", which mirrors the stock
# short-answer VQA instruction and can make a model skip reasoning and tags.
PROMPT_BRIEF = ("{question}\n"
                "Look carefully at the image and reason step by step. Then write your final "
                "answer inside <answer></answer> tags. Keep the final answer brief.")

# InternVL3.5's own setups (model card / VLMEvalKit internvl_chat.py):
#   thinking mode   = this system prompt (verbatim R1_SYSTEM_PROMPT); answer after </think>, no tags
#   VQA benchmarks  = no system prompt, question + the short-answer cue below (non-thinking)
INTERNVL_R1_SYSTEM = (
    "You are an AI assistant that rigorously follows this response protocol:\n\n"
    "1. First, conduct a detailed analysis of the question. Consider different angles, potential "
    "solutions, and reason through the problem step-by-step. Enclose this entire thinking process "
    "within <think> and </think> tags.\n\n"
    "2. After the thinking section, provide a clear, concise, and direct answer to the user's "
    "question. Separate the answer from the think section with a newline.\n\n"
    "Ensure that the thinking process is thorough but remains focused on the query. The final "
    "answer should be standalone and not reference the thinking section.")
PROMPT_VQA_SHORT = "{question}\nAnswer the question using a single word or phrase."

# prompt_format -> (system prompt, user-turn template)
PROMPT_FORMATS = {
    "inline": (None, PROMPT),                        # instruction in the user turn
    "think_answer": (SYSTEM_PROMPT, "{question}"),   # instruction in the system prompt
    "inline_brief": (None, PROMPT_BRIEF),            # inline, without the short-answer cue
    "internvl_r1": (INTERNVL_R1_SYSTEM, "{question}"),   # InternVL3.5 thinking mode
    "internvl_r1_short": (INTERNVL_R1_SYSTEM, PROMPT_VQA_SHORT),   # thinking mode + short-answer cue
    "internvl_vqa": (None, PROMPT_VQA_SHORT),            # InternVL3.5 / VLMEvalKit VQA prompt
}


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #
_THINK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_STRICT = re.compile(r"^\s*<think>(.+?)</think>\s*<answer>(.+?)</answer>\s*$", re.S | re.I)
_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
_BOXED = re.compile(r"\\boxed\{\s*([^{}]*?)\s*\}")
_LINE = re.compile(r"(?:final answer|answer)\s*(?:is)?\s*[:：]\s*(.+)", re.I)


MAX_FALLBACK_WORDS = 8          # a bare last line longer than this is reasoning, not an answer


def split_reasoning(completion: str):
    """(reasoning, extracted_answer, method). ``method`` records which rule
    fired: tag | boxed | answer_line | last_line | none (output had no
    recognizable answer, e.g. a repetition loop cut off at max tokens) | empty.
    When the output has a <think> block, the reasoning is its content and the
    answer is looked for after it."""
    text = completion.strip()
    thought = None
    m = _THINK.search(text)
    if m:
        thought, rest = m.group(1).strip(), text[m.end():].strip()
    elif re.search(r"</think>", text, re.I):          # opening tag missing (e.g. in the template)
        thought, rest = [x.strip() for x in re.split(r"</think>", text, 1, flags=re.I)]
        thought = re.sub(r"^<think>", "", thought, flags=re.I).strip()
    else:
        rest = text
    if thought is not None and not rest:                 # everything was inside <think>
        rest = text
    tags = list(_TAG.finditer(rest))
    if tags:
        reasoning = thought if thought is not None else rest[:tags[-1].start()].strip()
        return reasoning, tags[-1].group(1).strip(), "tag"
    m = list(_BOXED.finditer(rest))
    if m:
        return (thought or rest[:m[-1].start()].strip()), m[-1].group(1).strip(), "boxed"
    lines = [l for l in rest.splitlines() if l.strip()]
    for i in range(len(lines) - 1, -1, -1):
        m = _LINE.search(lines[i])
        if m:
            return (thought or "\n".join(lines[:i]).strip()), m.group(1).strip(), "answer_line"
    if lines and len(lines[-1].split()) <= MAX_FALLBACK_WORDS:
        return (thought or "\n".join(lines[:-1]).strip()), lines[-1].strip(), "last_line"
    if lines:
        return (thought or rest), "", "none"
    return "", "", "empty"


def format_check(completion: str) -> dict:
    """Did the model emit the required tags? Independent of correctness.
      think   a complete <think>...</think> block is present
      answer  a complete, non-empty <answer>...</answer> block is present
      strict  the whole output is exactly <think>...</think> <answer>...</answer>
              (nothing before, between or after, one block each)"""
    t = completion.strip()
    n_think = len(_THINK.findall(t))
    answers = _TAG.findall(t)
    return {"think": n_think > 0,
            "answer": any(a.strip() for a in answers),
            "strict": bool(_STRICT.match(t)) and n_think == 1 and len(answers) == 1}


def extract_answer(completion: str) -> str:
    return split_reasoning(completion)[1]


# --------------------------------------------------------------------------- #
# VQA answer normalization (EvalAI / VQA v2 convention)
# --------------------------------------------------------------------------- #
_CONTRACTIONS = {"dont": "don't", "doesnt": "doesn't", "isnt": "isn't", "arent": "aren't",
                 "cant": "can't", "wont": "won't", "im": "i'm", "ive": "i've",
                 "whats": "what's", "thats": "that's", "theres": "there's"}
_NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
          ">", "<", "@", "`", ",", "?", "!"]                # official VQA eval list
_PERIOD = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_NUM = re.compile(r"(\d)(,)(\d)")


def normalize_answer(s: str) -> str:
    s = str(s).replace("\n", " ").replace("\t", " ").strip().lower()
    s = re.sub(r"[*_`]+", "", s)                         # markdown emphasis
    for p in _PUNCT:
        if (p + " " in s or " " + p in s) or re.search(_COMMA_NUM, s):
            s = s.replace(p, "")
        else:
            s = s.replace(p, " ")
    s = _PERIOD.sub("", s)
    words = []
    for w in s.split():
        w = _NUMBERS.get(w, w)
        if w not in _ARTICLES:
            words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words).strip()


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def _load(cfg, dataset):
    path = Path(cfg.get("manifest") or Path(cfg["root"]) / dataset / "manifest.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    corrupted = bool(cfg.get("use_corrupted"))
    system, user = PROMPT_FORMATS[cfg.get("prompt_format", "inline")]
    out = []
    for r in rows:
        ex = {"id": r["question_id"], "dataset": dataset, "question": r["question"],
              "system": system, "prompt": user.format(question=r["question"]),
              "image": r["corrupted_image"] if corrupted else r["clean_image"],
              "view": "corrupted" if corrupted else "clean",
              "corruption": r["corruption"], "image_id": r["image_id"]}
        if "category" in r:
            ex["category"] = r["category"]
        if dataset in ("textvqa", "vqav2"):
            ex["answers"] = r["answers"]
            ex["gold"] = max(set(r["answers"]), key=r["answers"].count)   # majority, for display
        else:
            ex["gold"] = r["answer"]
            ex["full_answer"] = r.get("full_answer")
        out.append(ex)
    if cfg.get("limit"):
        out = out[:int(cfg["limit"])]
    return out


@register("textvqa")
class TextVQA:
    @staticmethod
    def load(cfg):
        return _load(cfg, "textvqa")

    @staticmethod
    def score(example, completion) -> bool:
        pred = normalize_answer(extract_answer(completion))
        return bool(pred) and pred in {normalize_answer(a) for a in example["answers"]}

    @staticmethod
    def soft_score(example, completion) -> float:
        pred = normalize_answer(extract_answer(completion))
        hits = sum(normalize_answer(a) == pred for a in example["answers"])
        return min(hits / 3.0, 1.0) if pred else 0.0


@register("gqa")
class GQA:
    @staticmethod
    def load(cfg):
        return _load(cfg, "gqa")

    @staticmethod
    def score(example, completion) -> bool:
        pred = normalize_answer(extract_answer(completion))
        return bool(pred) and pred == normalize_answer(example["gold"])


# --------------------------------------------------------------------------- #
# VizWiz-VQA: real-world degraded photos, single view ("real")
# --------------------------------------------------------------------------- #
VIZWIZ_NOTE = " If the image does not contain enough information to answer, the answer is unanswerable."
VIZWIZ_PROMPT = PROMPT + VIZWIZ_NOTE


@register("vizwiz")
class VizWiz:
    @staticmethod
    def load(cfg):
        path = Path(cfg.get("manifest") or Path(cfg["root"]) / "vizwiz" / "manifest.jsonl")
        system, user = PROMPT_FORMATS[cfg.get("prompt_format", "inline")]
        out = []
        for l in path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            out.append({"id": r["question_id"], "dataset": "vizwiz", "question": r["question"],
                        "prompt": (user + VIZWIZ_NOTE).format(question=r["question"]), "system": system,
                        "image": r["image"], "view": "real", "corruption": None,
                        "category": r["category"], "image_id": r["image_id"],
                        "answers": r["answers"],
                        "gold": max(set(r["answers"]), key=r["answers"].count)})
        return out[:int(cfg["limit"])] if cfg.get("limit") else out

    score = staticmethod(lambda example, completion: TextVQA.score(example, completion))
    soft_score = staticmethod(lambda example, completion: TextVQA.soft_score(example, completion))


@register("vqav2")
class VQAv2:
    """VQAv2-C: a VQAv2 val subset with clean + one ImageNet-C corrupted view.
    Scored like TextVQA (any of the 10 annotator answers; VQA soft score too)."""
    @staticmethod
    def load(cfg):
        return _load(cfg, "vqav2")

    score = staticmethod(lambda example, completion: TextVQA.score(example, completion))
    soft_score = staticmethod(lambda example, completion: TextVQA.soft_score(example, completion))
