from pathlib import Path
import importlib.util
import runpy
import sys


_SCRIPT_PATH = Path(__file__).resolve().parent / 'src' / 'main_sim.py'
_SCRIPT_DIR = str(_SCRIPT_PATH.parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _load_src_main_sim_module():
    spec = importlib.util.spec_from_file_location('_crazyflie_src_main_sim', _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'无法加载源模块: {_SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    runpy.run_path(str(_SCRIPT_PATH), run_name='__main__')
else:
    _module = _load_src_main_sim_module()
    for _name in dir(_module):
        if not _name.startswith('_'):
            globals()[_name] = getattr(_module, _name)