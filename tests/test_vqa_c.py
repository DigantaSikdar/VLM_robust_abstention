"""TextVQA-C / GQA-C: answer extraction, VQA normalization, scoring, and the
balanced assignment."""
from collections import Counter

from caad.eval.tasks.vqa_c import (GQA, TextVQA, extract_answer, format_check,
                                   normalize_answer, split_reasoning)


def test_extract_from_tags_and_fallbacks():
    assert split_reasoning("The sign says DAKOTA.\n<answer>Dakota</answer>") == \
        ("The sign says DAKOTA.", "Dakota", "tag")
    assert extract_answer("<think>hmm</think>It is red. <answer> red </answer>") == "red"
    assert extract_answer("so \\boxed{2010}") == "2010"
    assert split_reasoning("Reasoning here.\nFinal answer: yes")[1:] == ("yes", "answer_line")
    assert split_reasoning("blah\nno")[1:] == ("no", "last_line")
    assert split_reasoning("")[2] == "empty"
    # a runaway paragraph (e.g. repetition loop hitting max tokens) is not an answer
    loop = "The horse is moving. " * 50
    assert split_reasoning(loop)[1:] == ("", "none")
    # last tag wins when a model restates
    assert extract_answer("<answer>a</answer> wait <answer>b</answer>") == "b"


def test_normalize_matches_vqa_convention():
    assert normalize_answer("The Dakota.") == "dakota"
    assert normalize_answer("Two") == "2"
    assert normalize_answer("**Yes**") == "yes"
    assert normalize_answer("dont") == "don't"
    assert normalize_answer("1,000") == "1000"


def test_scoring():
    tv = {"answers": ["2010"] * 9 + ["unanswerable"]}
    assert TextVQA.score(tv, "The calendar shows 2010.\n<answer>2010</answer>")
    assert TextVQA.score(tv, "<answer>unanswerable</answer>")          # any-match (paper)
    assert TextVQA.soft_score(tv, "<answer>unanswerable</answer>") == 1 / 3
    assert not TextVQA.score(tv, "<answer>2013</answer>")
    assert GQA.score({"gold": "no"}, "It is sunny.\n<answer>No</answer>")
    assert not GQA.score({"gold": "no"}, "<answer>no, it is clear</answer>")


def test_balanced_assignment_is_balanced_and_fixed():
    from caad.data.prepare_vqa_c import CELLS, balanced_assignment
    ids = [str(i) for i in range(1000)]
    a = balanced_assignment(ids, 0, "gqa")
    counts = Counter(a.values())
    assert set(counts) == set(CELLS) and max(counts.values()) - min(counts.values()) <= 1
    assert a == balanced_assignment(list(reversed(ids)), 0, "gqa")      # order-independent
    assert a != balanced_assignment(ids, 1, "gqa")


def test_think_answer_format():
    out = "<think>The sky is clear and blue.</think>\n<answer>no</answer>"
    assert split_reasoning(out) == ("The sky is clear and blue.", "no", "tag")
    assert format_check(out) == {"think": True, "answer": True, "strict": True}
    # answer present but extra text after it -> extractable, not strict
    assert format_check(out + " Hope this helps!") == {"think": True, "answer": True, "strict": False}
    # no think block
    assert format_check("<answer>no</answer>") == {"think": False, "answer": True, "strict": False}
    # think only, no answer tags
    f = format_check("<think>hmm</think> It is no.")
    assert f == {"think": True, "answer": False, "strict": False}
    # empty answer tags do not count
    assert format_check("<think>x</think><answer> </answer>")["answer"] is False
    # missing opening <think> still splits reasoning from answer
    assert split_reasoning("looks red</think><answer>red</answer>") == ("looks red", "red", "tag")


def test_vllm_eval_summary_math():
    from caad.eval.vllm_eval import summarize
    def rec(ds, q, view, ok, ans=True, name="fog", cat="weather", sev=3, seen=True):
        r = {"dataset": ds, "question_id": q, "view": view, "correct": ok,
             "format": {"answer": ans, "think": False, "strict": False},
             "extraction": "tag" if ans else "none", "truncated": not ans,
             "corruption": {"name": name, "category": cat, "severity": sev, "seen_in_training": seen}}
        if ds == "textvqa":
            r["soft"] = 1.0 if ok else 0.0
        return r
    rs = [rec("gqa", "1", "clean", True), rec("gqa", "1", "corrupted", False),
          rec("gqa", "2", "clean", True), rec("gqa", "2", "corrupted", True, name="frost", seen=False),
          rec("gqa", "3", "clean", False, ans=False), rec("gqa", "3", "corrupted", True)]
    s = summarize(rs)["gqa"]
    assert abs(s["clean"]["accuracy"] - 2 / 3) < 1e-9 and abs(s["corrupted"]["accuracy"] - 2 / 3) < 1e-9
    assert abs(s["clean"]["extraction_rate"] - 2 / 3) < 1e-9
    assert s["paired"] == {"n": 3, "clean_right_corrupted_wrong": 1, "clean_wrong_corrupted_right": 1}
    assert s["by_seen"]["unseen"]["n"] == 1 and s["by_seen"]["seen"]["accuracy"] == 0.5
    assert summarize(rs)["overall"]["truncated"] == 1
