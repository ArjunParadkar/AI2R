import numpy as np
import pytest
torch = pytest.importorskip("torch")
from whiteboxtopo.activations import ActivationExtractor


def test_extractor_pools_and_stacks():
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 6), torch.nn.Linear(6, 2))
    ex = ActivationExtractor(net, ["0", "4", "5"])
    batches = [(torch.randn(5, 3, 8, 8), torch.zeros(5)) for _ in range(3)]
    acts = ex(batches)
    assert acts["0"].shape == (15, 4)   # conv map avg-pooled
    assert acts["4"].shape == (15, 6)
    assert acts["5"].shape == (15, 2)
    ex.remove()
    with pytest.raises(KeyError):
        ActivationExtractor(net, ["nope"])


def test_sequence_mask_pooling():
    class Seq(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.emb = torch.nn.Embedding(10, 4); self.lin = torch.nn.Linear(4, 4)
        def forward(self, input_ids, attention_mask=None):
            return self.lin(self.emb(input_ids))
    net = Seq(); ex = ActivationExtractor(net, ["lin"])
    ids = torch.tensor([[1, 2, 3], [4, 0, 0]]); mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    acts = ex([{"input_ids": ids, "attention_mask": mask}])
    ref = net.lin(net.emb(ids))[1, 0].detach().numpy()
    assert acts["lin"].shape == (2, 4)
    assert np.allclose(acts["lin"][1], ref, atol=1e-6)
