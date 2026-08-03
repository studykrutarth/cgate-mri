"""Quality-aware expert routing with calibrated confidence gating for brain MRI.

Project-I scope: route between heterogeneous experts based on measured image
quality, calibrate the fused prediction, and gate low-confidence cases to human
review. Grounded report generation is deliberately out of scope -- see
docs/scope.md.
"""

__version__ = "0.1.0"

from .schema import FINDING_NAMES, FINDING_SCHEMA, N_FINDINGS

__all__ = ["FINDING_SCHEMA", "FINDING_NAMES", "N_FINDINGS", "__version__"]
