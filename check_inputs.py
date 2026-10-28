"""Input, mask, model and backward-pass checks; not performance evidence."""
from pathlib import Path
import json
import numpy as np
from missingness import make_mask, repair, gap_distance
from integrity import verify_input_hash

root = Path(__file__).resolve().parent
receipt = json.loads((root/'input_receipt.json').read_text())
for name, sha in receipt['sha256'].items():
    verify_input_hash(root/'data'/name, sha, text=name.endswith('.json'))
X = np.load(root/'data/band_streams.npy')
entries = json.loads((root/'data/manifest.json').read_text())['selected']
assert X.shape == (1260,18,500) and np.isfinite(X).all()
assert len(entries) == len({e['name'] for e in entries}) == 1260
mean = X[:10].mean((0,2))
for pattern in ['random', 'contiguous']:
    for seed in range(20261026,20261031):
        m = make_mask(entries[0]['name'], pattern, .5, seed)
        assert m.sum() == 250 and np.array_equal(m, make_mask(entries[0]['name'], pattern, .5, seed))
        changed = X[0].copy(); changed[:,~m] = 1e6
        for method in ['linear','nearest','source_mean']:
            a = repair(X[0],m,method,mean); b = repair(changed,m,method,mean)
            np.testing.assert_array_equal(a,b)  # Hidden values never influence repair.
        assert np.all(gap_distance(m)[m] == 0)
import torch
from mira_model import ActivityCNN
for aware in [False, True]:
    torch.manual_seed(1)
    model = ActivityCNN(aware)
    x = torch.randn(4,18,500); m = torch.ones(4,1,500); d = torch.zeros_like(m)
    y = model(x,m,d); assert y.shape == (4,7) and torch.isfinite(y).all()
    loss = torch.nn.functional.cross_entropy(y,torch.tensor([0,1,2,3])); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('Model check:', 'MIRA-CSI' if aware else 'CNN', sum(p.numel() for p in model.parameters()), 'parameters')
print('PASS: tensor hashes, 1260 identities, exact masks, hidden-value isolation and model gradients.')
