"""forge-workbench — cross-stack tooling for the Forge / OpenCUDA / OpenPTXas / VortexSTARK toolchain.

The CLI cockpit lives in `workbench.cli`. For convenience, the most-used
public names are re-exported at package level so existing in-tree users
of the previous `openptxas/workbench.py` (catalog access, compile helpers)
keep working with `import workbench` after pip-installing this package.
"""
__version__ = "0.1.0"

# Re-export the public API. Anything in this list is considered stable;
# new code should import from `workbench.cli` directly.
from workbench.cli import (  # noqa: F401, E402
    KERNELS,
    SUITES,
    CUDAContext,
    compile_with_report,
    compile_openptxas,
    compile_ptxas,
    measure_kernel,
    metrics_from_cubin,
    cubin_metrics,
    STACK_ROOT,
    REPO_OPENPTXAS,
    REPO_FORGE,
    REPO_OPENCUDA,
)
