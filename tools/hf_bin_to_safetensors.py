"""Convert a cached .bin checkpoint to safetensors, in place.

transformers >= 4.56 refuses torch.load for any checkpoint that is not
safetensors unless torch >= 2.6 (CVE-2025-32434). This machine runs torch
2.5.1+cu124 against CUDA 12.4, and upgrading torch to satisfy a loader check
risks the whole GPU stack for no gain -- the weights are identical either way.
Converting the file is the smaller change.

Loads with weights_only=True, writes model.safetensors beside the .bin, and
leaves the .bin alone so nothing is destroyed.

    .venv-gpu/Scripts/python tools/hf_bin_to_safetensors.py <repo_id> [...]
"""
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file


def convert(repo):
    d = Path(snapshot_download(repo))
    st = d / "model.safetensors"
    if st.exists():
        print(f"  {repo}: already safetensors")
        return
    bins = sorted(d.glob("*.bin"))
    if not bins:
        print(f"  {repo}: no .bin found, nothing to do")
        return
    b = bins[0]
    sd = torch.load(b, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    # shared storage breaks safetensors; clone anything that aliases
    seen, out = {}, {}
    for k, v in sd.items():
        ptr = v.data_ptr()
        out[k] = v.clone() if ptr in seen else v
        seen[ptr] = k
    save_file(out, st, metadata={"format": "pt"})
    print(f"  {repo}: {b.name} -> model.safetensors "
          f"({st.stat().st_size / 1e6:.0f} MB, {len(out)} tensors)")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        convert(r)
