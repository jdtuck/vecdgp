"""Default priors and proposal settings.

Values reproduce ``deepgp:::check_settings``.  They are calibrated for
``x`` scaled to :math:`[0,1]^d` and ``y`` scaled to mean zero, variance one.

Priors are ``Gamma(alpha, rate=beta)`` on ``theta - eps`` (and on ``g - eps``
when the nugget is estimated).  Proposals use a uniform sliding window
``Unif(l * v / u, u * v / l)`` with ``l = 1``, ``u = 2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Settings:
    """Prior / proposal configuration for a fit."""

    l: float = 1.0
    u: float = 2.0

    # Gamma(alpha, rate=beta) priors
    theta_alpha: float = 1.5
    theta_beta: float = 3.9 / 1.5
    theta_y_alpha: float = 1.5
    theta_y_beta: float = 3.9 / 6
    theta_w_alpha: float = 1.5
    theta_w_beta: float = 3.9 / 4
    theta_z_alpha: float = 1.5
    theta_z_beta: float = 3.9 / 4
    g_alpha: float = 1.01
    g_beta: float = 10.0

    # fixed scales of the latent layers
    tau2_w: float = 1.0
    tau2_z: float = 1.0

    # model flags
    sep: bool = False
    pmx: bool = False

    extra: dict = field(default_factory=dict)


def default_settings(layers=1, noisy=False, overrides=None):
    """Build a :class:`Settings` with ``deepgp``'s layer-specific defaults.

    ``noisy=True`` (i.e. the nugget is estimated rather than fixed) selects
    the alternative prior set used by ``deepgp`` for noisy data.
    """
    s = Settings()
    if noisy:
        s.g_alpha, s.g_beta = 1.01, 10.0  # mode 0.001, 95% quantile 0.3
        if layers == 1:
            s.theta_alpha, s.theta_beta = 1.2, 4.0  # mode 0.05, q95 0.84
        elif layers == 2:
            s.theta_w_alpha, s.theta_w_beta = 1.2, 2.0  # mode 0.1, q95 1.7
            s.theta_y_alpha, s.theta_y_beta = 1.2, 1.0  # mode 0.2, q95 3.4
        elif layers == 3:
            s.theta_z_alpha, s.theta_z_beta = 1.2, 2.0
            s.theta_w_alpha, s.theta_w_beta = 1.2, 0.8
            s.theta_y_alpha, s.theta_y_beta = 1.0, 1.2
    else:  # deterministic defaults
        if layers == 1:
            s.theta_alpha, s.theta_beta = 1.5, 3.9 / 1.5
        elif layers == 2:
            s.theta_w_alpha, s.theta_w_beta = 1.5, 3.9 / 4
            s.theta_y_alpha, s.theta_y_beta = 1.5, 3.9 / 6
        elif layers == 3:
            s.theta_z_alpha, s.theta_z_beta = 1.5, 3.9 / 4
            s.theta_w_alpha, s.theta_w_beta = 1.5, 3.9 / 12
            s.theta_y_alpha, s.theta_y_beta = 1.5, 3.9 / 6

    for k, val in (overrides or {}).items():
        if hasattr(s, k):
            setattr(s, k, val)
        else:
            s.extra[k] = val
    return s
