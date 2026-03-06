"""
animate_sim.py - 仿射编队控制 3D 运动动画

生成 MP4 动画，展示 10 架无人机编队的：
  Phase 1: 编队建立（随机 → 标称）
  Phase 2: 编队缩放 1.5x
  Phase 3: 编队旋转 45°

用法：
    cd e:/crazyflie/src
    e:/crazyflie/.venv/Scripts/python.exe animate_sim.py
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from stress_matrix import compute_stress_matrix
from formation import (
    double_pentagon, affine_transform, scale_matrix,
    rotation_matrix_z, create_leader_trajectory,
)
from afc_controller import AFCController

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_simulation():
    """运行仿真，返回轨迹数据。"""
    nominal_pos, leader_indices, adj = double_pentagon(radius=1.0, height=1.0)
    follower_indices = sorted(set(range(10)) - set(leader_indices))

    Omega, _ = compute_stress_matrix(nominal_pos, adj, leader_indices, method='optimize')
    controller = AFCController(Omega, leader_indices, gain=5.0)

    nominal_leaders = nominal_pos[leader_indices]
    A_scale = scale_matrix(3, 1.5)
    A_rotate = rotation_matrix_z(np.pi / 4)

    pos_phase1 = nominal_leaders.copy()
    pos_phase2 = affine_transform(nominal_leaders, A=A_scale)
    pos_phase3 = affine_transform(nominal_leaders, A=A_rotate @ A_scale)

    t_settle, t_trans, t_hold = 8.0, 5.0, 8.0
    phases = [
        {'start_positions': nominal_leaders, 't_start': 0.0, 't_end': 0.1, 'positions': pos_phase1},
        {'t_start': t_settle, 't_end': t_settle + t_trans, 'positions': pos_phase2},
        {'t_start': t_settle + t_trans + t_hold, 't_end': t_settle + 2*t_trans + t_hold, 'positions': pos_phase3},
    ]
    leader_traj = create_leader_trajectory(phases)
    T_total = t_settle + 2*t_trans + 2*t_hold

    np.random.seed(42)
    init_pos = nominal_pos.copy()
    init_pos[follower_indices] += np.random.randn(len(follower_indices), 3) * 0.5

    n, d = 10, 3
    n_f = controller.n_f
    f_idx = controller.follower_indices
    l_idx = controller.leader_indices

    def dynamics(t, state):
        p_f = state.reshape(n_f, d)
        p_l = leader_traj(t)
        return (-controller.gain * (controller.Omega_ff @ p_f + controller.Omega_fl @ p_l)).flatten()

    dt = 0.02
    t_eval = np.arange(0, T_total, dt)
    if t_eval[-1] < T_total:
        t_eval = np.append(t_eval, T_total)

    sol = solve_ivp(dynamics, (0, T_total), init_pos[f_idx].flatten(),
                    t_eval=t_eval, method='RK45', max_step=dt, rtol=1e-8, atol=1e-10)

    n_steps = len(sol.t)
    pos_hist = np.zeros((n_steps, n, d))
    errors = np.zeros(n_steps)
    for idx, t in enumerate(sol.t):
        p_l = leader_traj(t)
        pos_hist[idx, l_idx] = p_l
        pos_hist[idx, f_idx] = sol.y[:, idx].reshape(n_f, d)
        p_f_star = controller.steady_state(p_l)
        errors[idx] = np.linalg.norm(pos_hist[idx, f_idx] - p_f_star)

    return sol.t, pos_hist, errors, adj, leader_indices, follower_indices, t_settle, t_trans, t_hold


def main():
    print("运行仿真...")
    times, pos_hist, errors, adj, leaders, followers, t_settle, t_trans, t_hold = run_simulation()
    n_steps, n, d = pos_hist.shape
    print(f"仿真完成: {n_steps} 帧, 时长 {times[-1]:.1f}s")

    # 每隔几帧取一帧，控制动画帧率 ~30fps，视频 ~20fps 播放
    skip = max(1, n_steps // 600)
    frame_indices = list(range(0, n_steps, skip))
    n_frames = len(frame_indices)
    print(f"动画帧数: {n_frames} (每 {skip} 步取 1 帧)")

    # 计算场景范围
    all_pos = pos_hist[frame_indices]
    margin = 0.3
    x_min, x_max = all_pos[:,:,0].min() - margin, all_pos[:,:,0].max() + margin
    y_min, y_max = all_pos[:,:,1].min() - margin, all_pos[:,:,1].max() + margin
    z_min, z_max = all_pos[:,:,2].min() - margin, all_pos[:,:,2].max() + margin

    # 创建图形
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[3, 1],
                          hspace=0.3, wspace=0.3)

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_err = fig.add_subplot(gs[1, :])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')

    # 轨迹尾迹长度
    trail_len = min(50, n_frames)

    # 预分配绘图对象
    leader_scatter = ax3d.scatter([], [], [], c='red', s=120, marker='^',  # type: ignore[arg-type]
                                  edgecolors='darkred', linewidths=0.8, zorder=5)
    follower_scatter = ax3d.scatter([], [], [], c='dodgerblue', s=70, marker='o',  # type: ignore[arg-type]
                                    edgecolors='navy', linewidths=0.5, zorder=5)
    edge_lines = []

    # 误差曲线
    err_line, = ax_err.semilogy([], [], 'b-', linewidth=1.5)
    err_dot, = ax_err.semilogy([], [], 'ro', markersize=6, zorder=5)
    ax_err.set_xlim(0, times[-1])
    ax_err.set_ylim(max(errors.min() * 0.5, 1e-4), errors.max() * 2)
    ax_err.set_xlabel('时间 (s)', fontsize=10)
    ax_err.set_ylabel('编队误差', fontsize=10)
    ax_err.set_title('编队误差收敛曲线', fontsize=11)
    ax_err.grid(True, alpha=0.3)

    # 阶段背景色
    T_total = times[-1]
    phase_specs = [
        (0, t_settle, '#FFCCCC', '编队建立'),
        (t_settle, t_settle+t_trans, '#CCFFCC', '缩放变换'),
        (t_settle+t_trans, t_settle+t_trans+t_hold, '#CCCCFF', '缩放保持'),
        (t_settle+t_trans+t_hold, t_settle+2*t_trans+t_hold, '#FFFFCC', '旋转变换'),
        (t_settle+2*t_trans+t_hold, T_total, '#FFCCFF', '旋转保持'),
    ]
    for t0, t1, color, label in phase_specs:
        ax_err.axvspan(t0, t1, alpha=0.15, color=color)
        ax_err.text((t0+t1)/2, errors.max()*1.2, label, ha='center', fontsize=7, alpha=0.6)

    # 信息面板文字
    info_text = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes,
                              fontsize=11, verticalalignment='top',
                              fontfamily='Microsoft YaHei',
                              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 轨迹线（每个智能体一条）
    trail_lines = []
    for i in range(n):
        color = 'red' if i in leaders else 'dodgerblue'
        alpha = 0.5 if i in leaders else 0.3
        line, = ax3d.plot([], [], [], color=color, alpha=alpha, linewidth=0.8)
        trail_lines.append(line)

    def get_phase_name(t):
        for t0, t1, _, name in phase_specs:
            if t0 <= t < t1:
                return name
        return phase_specs[-1][3]

    def init():
        ax3d.set_xlim(x_min, x_max)
        ax3d.set_ylim(y_min, y_max)
        ax3d.set_zlim(z_min, z_max)
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        return []

    def update(frame_idx):
        nonlocal edge_lines

        fi = frame_indices[frame_idx]
        t = times[fi]
        pos = pos_hist[fi]

        # 更新无人机位置
        l_pos = pos[leaders]
        f_pos = pos[followers]
        leader_scatter._offsets3d = (l_pos[:,0], l_pos[:,1], l_pos[:,2])  # type: ignore[attr-defined]
        follower_scatter._offsets3d = (f_pos[:,0], f_pos[:,1], f_pos[:,2])  # type: ignore[attr-defined]

        # 更新通信边
        for line in edge_lines:
            line.remove()
        edge_lines = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] > 0:
                    line, = ax3d.plot(
                        [pos[i,0], pos[j,0]],
                        [pos[i,1], pos[j,1]],
                        [pos[i,2], pos[j,2]],
                        'gray', alpha=0.08, linewidth=0.3
                    )
                    edge_lines.append(line)

        # 更新轨迹尾迹
        start_fi = max(0, frame_idx - trail_len)
        for i in range(n):
            indices = [frame_indices[k] for k in range(start_fi, frame_idx+1)]
            trail_lines[i].set_data(pos_hist[indices, i, 0], pos_hist[indices, i, 1])
            trail_lines[i].set_3d_properties(pos_hist[indices, i, 2])

        ax3d.set_title(f'仿射编队控制动画  t = {t:.2f}s', fontsize=12)

        # 更新误差曲线
        err_line.set_data(times[:fi+1], errors[:fi+1])
        err_dot.set_data([t], [errors[fi]])

        # 更新信息面板
        phase = get_phase_name(t)
        info_str = (
            f"Time:  {t:.2f} s\n"
            f"Phase: {phase}\n"
            f"───────────────\n"
            f"Error: {errors[fi]:.4f}\n"
            f"Agents: {n}\n"
            f"  Leader:   {leaders}\n"
            f"  Follower: {followers}\n"
            f"───────────────\n"
            f"Gain Kp: 5.0\n"
            f"Frame: {frame_idx+1}/{n_frames}"
        )
        info_text.set_text(info_str)

        return []

    print("生成动画中...")
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_frames, interval=50, blit=False)

    # 保存动画
    output_path = 'e:/crazyflie/src/afc_animation.gif'
    try:
        writer = PillowWriter(fps=20)
        anim.save(output_path, writer=writer, dpi=100)
        print(f"GIF 动画已保存: {output_path}")
    except Exception as e:
        print(f"GIF 保存失败: {e}")

    # 尝试保存 MP4（需要 ffmpeg）
    mp4_path = 'e:/crazyflie/src/afc_animation.mp4'
    try:
        writer_mp4 = FFMpegWriter(fps=20, bitrate=2000)
        anim.save(mp4_path, writer=writer_mp4, dpi=120)
        print(f"MP4 动画已保存: {mp4_path}")
    except Exception as e:
        print(f"MP4 保存需要 ffmpeg，跳过: {e}")

    plt.show()
    print("完成！")


if __name__ == '__main__':
    main()
