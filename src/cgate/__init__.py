"""Quality-aware expert routing with calibrated confidence gating for brain MRI.

Project-I scope: route between heterogeneous experts based on measured image
quality, calibrate the fused prediction, and gate low-confidence cases to human
review. Grounded report generation is deliberately out of scope -- see
docs/scope.md.
"""

import os as _os

# ---------------------------------------------------------------------------
# Windows OpenMP mitigation. MUST run before torch or scikit-learn import.
#
# scikit-learn/scipy load Intel MKL's OpenMP runtime and torch loads its own.
# With both spawning thread teams in one process, torch's backward pass can die
# with an access violation (exit 0xC0000005). This pipeline runs a scikit-learn
# radiomics expert alongside a torch CNN, so that is the normal configuration.
#
# The crash is intermittent -- the same command has both crashed and completed
# -- which is what a threading race looks like and why capping torch's own
# threads (see experts/torch_experts._require_torch) narrows the window without
# closing it. The environment variable has to be set before the native
# libraries load, which is why it lives here rather than in a function.
#
# This is a mitigation, not a root-cause fix. Set CGATE_NO_THREAD_LIMIT=1 to
# opt out, or set OMP_NUM_THREADS yourself and this leaves it alone.
# ---------------------------------------------------------------------------
if not _os.environ.get("CGATE_NO_THREAD_LIMIT"):
    _os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"

from .schema import FINDING_NAMES, FINDING_SCHEMA, N_FINDINGS

__all__ = ["FINDING_SCHEMA", "FINDING_NAMES", "N_FINDINGS", "__version__"]
