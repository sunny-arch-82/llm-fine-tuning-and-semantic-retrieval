from modelforge.config import Config
from modelforge.data import passages, queries, tickets


def test_splits_and_passage_labels():
    root = "data/sample"
    training = tickets(root, "train")
    validation = tickets(root, "validation")
    testing = tickets(root, "test")
    all_ids = [row["id"] for rows in (training, validation, testing) for row in rows]
    assert len(all_ids) == len(set(all_ids)) == 64
    assert len(passages(root)) == 16
    known = {row["id"] for row in passages(root)}
    for split in ("validation", "test"):
        for row in queries(root, split):
            assert set(row["relevance"]) <= known
            assert set(row["relevance"].values()) == {1, 2}


def test_config_rejects_bad_budget():
    import pytest

    with pytest.raises(ValueError):
        Config(max_length=0)
    with pytest.raises(ValueError):
        Config(quantization="2bit")
