"""
plot_cs2_sim.py -- 可视化 Crazyswarm2 AFC 仿真轨迹

读取 /tmp/afc_sim_log.csv（WSL 中生成），绘制：
  1. 3D 轨迹图（所有无人机，leader/follower 颜色区分）
  2. 编队误差随时间变化曲线
  3. 3D 动画（可选）

用法：
  python integration/scripts/plot_cs2_sim.py [csv_path]

默认从 WSL 路径复制日志到 Windows 临时文件。
"""

import sys
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.animation as animation

# ── 配置 ──────────────────────────────────────────────────
LEADER_INDICES_5  = [0, 1, 2, 3]
LEADER_INDICES_10 = [0, 1, 2, 5]

LEADER_COLOR   = '#2196F3'   # blue
FOLLOWER_COLOR = '#FF5722'   # red-orange
LEADER_MARKER  = 'D'
FOLLOWER_MARKER = 'o'


def load_csv(path):
    """Load simulation log CSV, return dict of arrays."""
    import csv
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader]

    data = np.array(rows, dtype=float)
    t = data[:, 0]
    err = data[:, 1]

    # Detect number of agents from position columns
    # Header format: t_s, formation_error_m, agent_X_err_m..., x_0, y_0, z_0, ...
    pos_start = None
    for i, h in enumerate(header):
        if h == 'x_0':
            pos_start = i
            break

    if pos_start is None:
        print('Warning: no position data in CSV, only plotting error.')
        return {'t': t, 'err': err, 'positions': None, 'n_agents': 0}

    n_pos_cols = len(header) - pos_start
    n_agents = n_pos_cols // 3
    positions = data[:, pos_start:].reshape(len(t), n_agents, 3)

    return {'t': t, 'err': err, 'positions': positions, 'n_agents': n_agents}


def plot_trajectories(data, save_path=None):
    """Plot 3D trajectories for all drones."""
    positions = data['positions']
    n = data['n_agents']
    t = data['t']

    if n == 5:
        leaders = LEADER_INDICES_5
    else:
        leaders = LEADER_INDICES_10

    fig = plt.figure(figsize=(12, 5))

    # ── 3D trajectories ──
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(n):
        x, y, z = positions[:, i, 0], positions[:, i, 1], positions[:, i, 2]
        is_leader = i in leaders
        color = LEADER_COLOR if is_leader else FOLLOWER_COLOR
        label_prefix = 'L' if is_leader else 'F'
        ax1.plot(x, y, z, color=color, alpha=0.6, linewidth=0.8)
        # Start marker
        ax1.scatter(x[0], y[0], z[0], color=color, marker='^', s=30)
        # End marker
        ax1.scatter(x[-1], y[-1], z[-1], color=color,
                    marker=LEADER_MARKER if is_leader else FOLLOWER_MARKER,
                    s=50, label=f'{label_prefix}{i}')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'AFC {n}-drone 3D Trajectories')
    ax1.legend(fontsize=7, ncol=2, loc='upper left')

    # ── Formation error ──
    ax2 = fig.add_subplot(122)
    ax2.plot(t, data['err'], 'k-', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Formation Error (m)')
    ax2.set_title('Formation Error over Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()


def animate_trajectories(data, interval_ms=50, save_path=None):
    """Create 3D animation of the formation."""
    positions = data['positions']
    n = data['n_agents']
    t = data['t']
    n_frames = len(t)

    if n == 5:
        leaders = LEADER_INDICES_5
    else:
        leaders = LEADER_INDICES_10

    # Subsample for smooth animation
    step = max(1, n_frames // 400)
    idx = list(range(0, n_frames, step))
    positions = positions[idx]
    t = t[idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set fixed axis limits
    all_pos = data['positions']
    margin = 0.3
    ax.set_xlim(all_pos[:, :, 0].min() - margin, all_pos[:, :, 0].max() + margin)
    ax.set_ylim(all_pos[:, :, 1].min() - margin, all_pos[:, :, 1].max() + margin)
    ax.set_zlim(0, all_pos[:, :, 2].max() + margin)

    # Create scatter plots for current positions
    colors = [LEADER_COLOR if i in leaders else FOLLOWER_COLOR for i in range(n)]
    markers_list = [LEADER_MARKER if i in leaders else FOLLOWER_MARKER for i in range(n)]

    scatters = []
    trails = []
    for i in range(n):
        s = ax.scatter([], [], [], c=colors[i], s=80,
                       marker=markers_list[i], edgecolors='black', linewidth=0.5)
        scatters.append(s)
        trail, = ax.plot([], [], [], color=colors[i], alpha=0.3, linewidth=0.6)
        trails.append(trail)

    title = ax.set_title('')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    def update(frame):
        for i in range(n):
            x, y, z = positions[frame, i]
            scatters[i]._offsets3d = ([x], [y], [z])
            # Trail: show last 40 frames
            start = max(0, frame - 40)
            trails[i].set_data(positions[start:frame+1, i, 0],
                               positions[start:frame+1, i, 1])
            trails[i].set_3d_properties(positions[start:frame+1, i, 2])
        title.set_text(f't = {t[frame]:.1f}s')
        return scatters + trails + [title]

    ani = animation.FuncAnimation(fig, update, frames=len(t),
                                  interval=interval_ms, blit=False)
    if save_path:
        ani.save(save_path, writer='pillow', fps=20)
        print(f'Animation saved to {save_path}')
    plt.show()
    return ani


def main():
    # Get CSV path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Copy from WSL
        wsl_path = '/tmp/afc_sim_log.csv'
        local_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'afc_sim_log.csv')
        local_path = os.path.abspath(local_path)
        try:
            result = subprocess.run(
                ['wsl', '-d', 'Ubuntu-24.04', '--', 'cat', wsl_path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'w') as f:
                    f.write(result.stdout)
                print(f'Copied log from WSL -> {local_path}')
                csv_path = local_path
            else:
                print(f'Failed to copy from WSL: {result.stderr}')
                return
        except Exception as e:
            print(f'Error: {e}')
            return

    data = load_csv(csv_path)
    print(f'Loaded: {len(data["t"])} frames, {data["n_agents"]} agents, '
          f'{data["t"][-1]:.1f}s duration')

    if data['positions'] is not None:
        plot_trajectories(data)

        if '--animate' in sys.argv:
            gif_path = csv_path.replace('.csv', '.gif')
            animate_trajectories(data, save_path=gif_path)
    else:
        plt.plot(data['t'], data['err'])
        plt.xlabel('Time (s)')
        plt.ylabel('Formation Error (m)')
        plt.title('AFC Formation Error')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
