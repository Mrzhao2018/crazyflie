from pathlib import Path
import runpy
import sys


if __name__ == '__main__':
    script_path = Path(__file__).resolve().parent / 'src' / 'tune_pyramid_safety.py'
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    runpy.run_path(str(script_path), run_name='__main__')