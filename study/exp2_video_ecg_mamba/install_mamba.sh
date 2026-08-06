#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate healthmirrorenv

EXPECTED_VERSION="2.2.6.post3"
WHEEL_URL="https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3%2Bcu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

pip install ninja==1.13.0 einops==0.8.2 transformers==4.44.2
if ! python -c "import mamba_ssm; assert mamba_ssm.__version__ == '${EXPECTED_VERSION}'" \
    >/dev/null 2>&1; then
    pip install --no-deps "${WHEEL_URL}"
fi

python - <<'PY'
import torch
import mamba_ssm
from mamba_ssm import Mamba

assert torch.__version__.startswith("2.4.")
assert torch.version.cuda.startswith("12.")
assert mamba_ssm.__version__ == "2.2.6.post3"
model = Mamba(d_model=32, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(1, 64, 32, device="cuda", requires_grad=True)
model(x).mean().backward()
assert torch.isfinite(x.grad).all()
print(f"Validated mamba-ssm {mamba_ssm.__version__} on {torch.cuda.get_device_name()}")
PY
