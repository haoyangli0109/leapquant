import importlib.abc
import importlib.util
import os
import sys

TARGET = "atom.model_ops.fla_ops.replayssm"


class _NoReplay(importlib.abc.MetaPathFinder):
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
            module.replayssm_replay_records = lambda state, *a, **k: state
            print(f"ABLATION pid {os.getpid()}: records not replayed on resume",
                  file=sys.stderr, flush=True)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("ABLATE_REPLAY") == "1":
    sys.meta_path.insert(0, _NoReplay())
