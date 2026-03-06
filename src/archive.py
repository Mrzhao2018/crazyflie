"""
archive.py - 仿真结果存档系统

每次仿真结束后，将所有图表和数值数据打包为带时间戳的 zip 存档。
存档保存在项目根目录下的 saves/archives/ 目录中。

用法：
    from archive import SimArchive

    arch = SimArchive()                    # 自动生成时间戳目录名
    arch.save_figure(fig, 'fig1_xxx')      # 保存图表
    arch.save_array('positions', array)    # 保存 numpy 数组
    arch.save_params({...})               # 保存仿真参数
    arch.save_text('log', text)           # 保存文本
    arch.finalize()                       # 打包为 zip 并清理临时目录
"""

import os
import json
import shutil
import zipfile
from datetime import datetime

import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ARCHIVE_DIR = os.path.join(ROOT_DIR, 'saves', 'archives')


class SimArchive:
    """仿真结果存档器。"""

    def __init__(self, tag=''):
        """
        创建一次存档会话。

        Parameters
        ----------
        tag : str
            可选标记，附加到目录名（如 'sparse', 'saturated'）
        """
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'sim_{ts}'
        if tag:
            name += f'_{tag}'

        self._name = name
        self._tmp_dir = os.path.join(ARCHIVE_DIR, name)
        os.makedirs(self._tmp_dir, exist_ok=True)

        self._data_dir = os.path.join(self._tmp_dir, 'data')
        os.makedirs(self._data_dir, exist_ok=True)

        self._fig_dir = os.path.join(self._tmp_dir, 'figures')
        os.makedirs(self._fig_dir, exist_ok=True)

        self._manifest = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'figures': [],
            'data': [],
            'params': None,
        }

    @property
    def name(self):
        return self._name

    def save_figure(self, fig, name, dpi=200):
        """保存 matplotlib Figure 为 PNG。"""
        path = os.path.join(self._fig_dir, f'{name}.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        self._manifest['figures'].append(f'{name}.png')

    def save_array(self, name, arr):
        """保存 numpy 数组为压缩 npz。"""
        path = os.path.join(self._data_dir, name)
        np.savez_compressed(path, data=arr)
        fname = f'{name}.npz'
        self._manifest['data'].append(fname)

    def save_arrays(self, **arrays):
        """批量保存多个 numpy 数组到同一个 npz 文件。"""
        path = os.path.join(self._data_dir, 'sim_data')
        np.savez_compressed(path, **arrays)
        self._manifest['data'].append('sim_data.npz')

    def save_params(self, params):
        """保存仿真参数为 JSON。"""
        # 转换不可序列化的类型
        clean = _make_serializable(params)
        path = os.path.join(self._tmp_dir, 'params.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        self._manifest['params'] = 'params.json'

    def save_text(self, name, text):
        """保存文本文件。"""
        path = os.path.join(self._tmp_dir, f'{name}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    def save_matrix_csv(self, name, matrix, header=''):
        """保存矩阵为 CSV。"""
        path = os.path.join(self._data_dir, f'{name}.csv')
        np.savetxt(path, matrix, delimiter=',', header=header, comments='')
        self._manifest['data'].append(f'{name}.csv')

    def finalize(self):
        """
        将临时目录打包为 zip 并删除临时目录。

        Returns
        -------
        zip_path : str
            生成的 zip 文件路径
        """
        # 写入 manifest
        manifest_path = os.path.join(self._tmp_dir, 'manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self._manifest, f, indent=2, ensure_ascii=False)

        # 打包
        zip_path = os.path.join(ARCHIVE_DIR, f'{self._name}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(self._tmp_dir):
                for file in files:
                    full = os.path.join(root, file)
                    arcname = os.path.relpath(full, self._tmp_dir)
                    zf.write(full, arcname)

        # 清理临时目录
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"\n  📦 存档已保存: {zip_path}")
        print(f"     大小: {size_mb:.2f} MB")
        print(f"     图表: {len(self._manifest['figures'])} 张")
        print(f"     数据: {len(self._manifest['data'])} 份")

        return zip_path


def _make_serializable(obj):
    """递归转换为 JSON 可序列化类型。"""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj
