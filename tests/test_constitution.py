from lrd_reason.constitution import COT_SYSTEM_PROMPT, FULL_CONSTITUTION, cot_messages


def test_cot_system_prompt_is_terse():
    n_words = len(COT_SYSTEM_PROMPT.split())
    assert n_words <= 40, f"COT_SYSTEM_PROMPT must be <=40 words (got {n_words})"


def test_full_constitution_has_all_seven_principles():
    for n in ["I.", "II.", "III.", "IV.", "V.", "VI.", "VII."]:
        assert n in FULL_CONSTITUTION


def test_cot_messages_uses_system_prompt():
    msgs = cot_messages("What is 2+2?")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == COT_SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"
    assert "2+2" in msgs[1]["content"]
