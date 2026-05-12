from lrd_reason.constitution import (
    COT_SYSTEM_PROMPT,
    FULL_CONSTITUTION,
    PROBLEM_GENERATOR_PROMPT,
    cot_messages,
    problem_gen_messages,
)


def test_cot_system_prompt_is_terse():
    n_words = len(COT_SYSTEM_PROMPT.split())
    assert n_words <= 100, f"COT_SYSTEM_PROMPT must be <=100 words (got {n_words})"


def test_cot_system_prompt_ends_with_answer_marker():
    # The parser at cot_generator.split_cot_answer relies on a trailing
    # "Final answer:" marker. The constitution must instruct the model to emit it.
    assert "Final answer:" in COT_SYSTEM_PROMPT


def test_full_constitution_has_all_seven_principles():
    for n in ["I.", "II.", "III.", "IV.", "V.", "VI.", "VII."]:
        assert n in FULL_CONSTITUTION


def test_cot_messages_uses_system_prompt():
    msgs = cot_messages("What is 2+2?")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == COT_SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"
    assert "2+2" in msgs[1]["content"]


def test_problem_generator_prompt_has_parser_contract():
    # The downstream extractor needs PROBLEM: and ANSWER: section markers, plus
    # a REJECT sentinel for cases where the generator can't meet the bar.
    for token in ["PROBLEM:", "ANSWER:", "REJECT"]:
        assert token in PROBLEM_GENERATOR_PROMPT


def test_problem_gen_messages_uses_system_prompt():
    msgs = problem_gen_messages(domain="math", difficulty="hard")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == PROBLEM_GENERATOR_PROMPT
    assert msgs[1]["role"] == "user"
    assert "math" in msgs[1]["content"]
    assert "hard" in msgs[1]["content"]
