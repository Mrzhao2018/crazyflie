"""
disturbance_observer.py - 扩展状态观测器 (ESO) 与风扰动模型

理论基础：
  - Han, J. "From PID to Active Disturbance Rejection Control" (IEEE Trans.
    Industrial Electronics, 2009)
  - Wang, Z. et al. "Robust time-varying formation design for multi-agent
    systems with disturbances: Extended-state-observer method" (Int. J. Robust
    Nonlinear Control, 30(7):2796-2808, 2020)

核心思想：
  将外部扰动视为系统扩展状态，通过 ESO 实时估计并前馈补偿，
  无需扰动模型的先验知识，实现对有界扰动的渐近抑制。

系统模型（含扰动）：
  ṗ_f = u_f + w_f(t)

  其中 w_f(t) 为有界外部扰动（如风场干扰）。

ESO 设计（对每个跟随者 i）：
  ż₁ᵢ = uᵢ + z₂ᵢ + β₁(pᵢ - z₁ᵢ)
  ż₂ᵢ = β₂(pᵢ - z₁ᵢ)

  其中 z₁ᵢ 估计位置 pᵢ，z₂ᵢ 估计扰动 wᵢ。
  β₁ = 2ω₀, β₂ = ω₀² 为观测器增益（Han 参数化），ω₀ 为观测器带宽。

补偿控制律：
  uᵢ = u_nom_i - z₂ᵢ
      = -K_p Σⱼ ωᵢⱼ(pᵢ - pⱼ) - z₂ᵢ

  补偿后闭环等效为：
    ṗᵢ ≈ u_nom_i - z₂ᵢ + wᵢ ≈ u_nom_i  (当 z₂ᵢ → wᵢ)

风扰动模型：
  w(t) = w_const + w_ou(t)

  - w_const: 恒定风场（如稳定侧风）
  - w_ou(t): Ornstein-Uhlenbeck 过程（有色噪声，模拟阵风）
      dw_ou = -θ · w_ou · dt + σ · dW
      θ = 1/τ_w（均值回复速率），σ（波动强度）
"""

import numpy as np


class WindDisturbance:
    """
    风场扰动模型：恒定分量 + Ornstein-Uhlenbeck 阵风。

    Parameters
    ----------
    n_agents : int
        受扰智能体数量
    dim : int
        空间维度
    w_const : ndarray (dim,) or (n_agents, dim)
        恒定风速分量 (m/s)
    ou_theta : float
        OU 过程均值回复速率 θ = 1/τ_w
    ou_sigma : float
        OU 过程波动强度 σ (m/s)
    seed : int or None
        随机种子
    """

    def __init__(self, n_agents, dim=3, w_const=None,
                 ou_theta=0.5, ou_sigma=0.1, seed=None):
        self.n_agents = n_agents
        self.dim = dim
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.rng = np.random.default_rng(seed)

        # 恒定分量
        if w_const is None:
            self.w_const = np.zeros((n_agents, dim))
        elif w_const.ndim == 1:
            self.w_const = np.tile(w_const, (n_agents, 1))
        else:
            self.w_const = w_const.copy()

        # OU 状态初始化
        self.w_ou = np.zeros((n_agents, dim))

    def reset(self):
        """重置 OU 状态。"""
        self.w_ou = np.zeros((self.n_agents, self.dim))

    def step(self, dt):
        """
        推进一步并返回当前扰动值。

        Parameters
        ----------
        dt : float
            时间步长

        Returns
        -------
        w : ndarray (n_agents, dim)
            当前风扰动
        """
        # OU 过程 Euler-Maruyama 离散化
        noise = self.rng.standard_normal((self.n_agents, self.dim))
        self.w_ou += (-self.ou_theta * self.w_ou * dt
                      + self.ou_sigma * np.sqrt(dt) * noise)
        return self.w_const + self.w_ou

    def current(self):
        """返回当前扰动值（不推进）。"""
        return self.w_const + self.w_ou


class ExtendedStateObserver:
    """
    扩展状态观测器 (ESO) — 估计并补偿未知外部扰动。

    采用 Han 参数化：β₁ = 2ω₀, β₂ = ω₀²，
    其中 ω₀ 为观测器带宽，越大估计越快但对噪声越敏感。

    Parameters
    ----------
    n_agents : int
        受观测智能体数量
    dim : int
        空间维度
    omega_o : float
        观测器带宽 ω₀ (rad/s)
    """

    def __init__(self, n_agents, dim=3, omega_o=8.0):
        self.n_agents = n_agents
        self.dim = dim
        self.omega_o = omega_o

        # Han 参数化
        self.beta1 = 2.0 * omega_o
        self.beta2 = omega_o ** 2

        # 观测器状态
        self.z1 = np.zeros((n_agents, dim))  # 位置估计
        self.z2 = np.zeros((n_agents, dim))  # 扰动估计

    def reset(self, p_f_init):
        """
        用当前位置初始化观测器。

        Parameters
        ----------
        p_f_init : ndarray (n_agents, dim)
        """
        self.z1 = p_f_init.copy()
        self.z2 = np.zeros((self.n_agents, self.dim))

    def update(self, p_f, u_applied, dt):
        """
        ESO 一步更新（Euler 离散化）。

        ż₁ = u + z₂ + β₁(p - z₁)
        ż₂ = β₂(p - z₁)

        Parameters
        ----------
        p_f : ndarray (n_agents, dim)
            当前测量位置
        u_applied : ndarray (n_agents, dim)
            实际施加的控制输入（饱和后）
        dt : float
            时间步长

        Returns
        -------
        w_hat : ndarray (n_agents, dim)
            当前扰动估计 z₂
        """
        e = p_f - self.z1  # 观测误差

        dz1 = u_applied + self.z2 + self.beta1 * e
        dz2 = self.beta2 * e

        self.z1 += dt * dz1
        self.z2 += dt * dz2

        return self.z2.copy()

    def disturbance_estimate(self):
        """返回当前扰动估计值 z₂。"""
        return self.z2.copy()
