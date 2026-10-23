"""FUSED_FLUSH=1: run ReplaySSM as it was before the flush split.

The layer kernels come from `replayssm_fused.py`, the module as it was then
(only its import made absolute), which folds a full record buffer inside each
layer kernel; the builder's stacked flush becomes a no-op.
"""

import importlib.abc
import importlib.util
import os
import sys

TARGET = "atom.model_ops.fla_ops.replayssm"
FUSED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replayssm_fused.py")


class _FusedFlush(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            fused_spec = importlib.util.spec_from_file_location("replayssm_fused", FUSED)
            fused = importlib.util.module_from_spec(fused_spec)
            fused_spec.loader.exec_module(fused)

            def gdn(*a, fold=True, **k):
                return fused.replayssm_gated_delta_rule(*a, **k)

            def kda(*a, fold=True, **k):
                return fused.replayssm_sigmoid_gating_delta_rule(*a, **k)

            module.replayssm_gated_delta_rule = gdn
            module.replayssm_sigmoid_gating_delta_rule = kda
            module.replayssm_flush = lambda *a, **k: None
            print(f"FUSED_FLUSH pid {os.getpid()}: ReplaySSM folds inside the layer kernels",
                  file=sys.stderr, flush=True)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("FUSED_FLUSH") == "1":
    sys.meta_path.insert(0, _FusedFlush())
