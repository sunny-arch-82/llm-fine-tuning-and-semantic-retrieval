import pytest

from modelforge.data import tickets
from modelforge.generation import answer_for, parse_answer, prompt_for
from modelforge.training import CompletionCollator, encode_example


class FakeTokenizer:
    eos_token = "[EOS]"
    pad_token_id = 0
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


def test_training_masks_instruction_not_completion():
    tokenizer = FakeTokenizer()
    row = tickets("data/sample", "train")[0]
    encoded = encode_example(tokenizer, row, 2000)
    prompt_length = len(prompt_for(tokenizer, row["text"]))
    assert encoded["labels"][:prompt_length] == [-100] * prompt_length
    assert encoded["labels"][prompt_length:] == encoded["input_ids"][prompt_length:]
    assert encoded["labels"][-5:] != [-100] * 5
    batch = CompletionCollator(0)([encoded, {key: value[:-10] for key, value in encoded.items()}])
    assert (batch["labels"][1, -10:] == -100).all()
    with pytest.raises(ValueError, match="raise max_length"):
        encode_example(tokenizer, row, 1)


def test_output_contract_distinguishes_missing_fields():
    row = tickets("data/sample", "train")[0]
    parsed = parse_answer(answer_for(row))
    assert parsed["category"] == row["category"]
    assert parsed["first_check"] == row["first_check"]
    assert parse_answer("The likely cause is a database.")["category"] is None
