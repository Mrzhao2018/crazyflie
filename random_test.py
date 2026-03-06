import os
import runpy
import sys


SRC_PATH = os.path.join(os.path.dirname(__file__), 'src')
TARGET = os.path.join(SRC_PATH, 'random_test.py')

if __name__ == '__main__':
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)
    runpy.run_path(TARGET, run_name='__main__')