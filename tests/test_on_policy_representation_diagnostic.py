import torch

from scripts.diagnose_on_policy_representations_v1 import cosine, summarize


def test_diagnostic_summary_and_cosine_are_exact_on_fixture():
    assert cosine(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == 1.0
    assert cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == 0.0
    result = summarize([0.0, 1.0, 2.0, 3.0])
    assert result["count"] == 4
    assert result["median"] == 1.5
    assert result["mean"] == 1.5
