import numpy as np
import matplotlib.pyplot as plt
import adcs_toolbox as adcs
from psyche_model import tot_moicom as J
from mekf_with_controller import (
    tau_max, h_max, J_diag,
    mu_psyche, r_orbit, T_orbit,
    gravity_gradient_torque, solar_radiation_torque,
)

# ── Tracking gains (override regulator gains from attitude_controller.py) ──────
# The regulator used wn=0.05 rad/s, which is too slow for tracking a target
# that moves at Psyche's spin rate (~4.2e-4 rad/s). We need wn >> omega_psyche.
# Constraint: Kp_i * phi_e <= tau_max even for moderate errors (~10° = 0.17 rad)
#   => wn <= sqrt(tau_max / (J_max * 0.17)) = sqrt(0.05 / (2923 * 0.17)) ≈ 0.010
# Use wn=0.008 rad/s with zeta=1.0 (critically damped) to avoid oscillation.
wn_track   = 0.008              # [rad/s]
zeta_track = 1.0                # critically damped — no overshoot
Kp_track   = np.diag(J_diag * wn_track**2)
Kd_track   = np.diag(J_diag * 2 * zeta_track * wn_track)
# Also keep original matrices available under their old names for any
# function that still needs them
Kp_mat = Kp_track
Kd_mat = Kd_track

"""
Surface Spot Pointing Controller
==================================
Goal: keep body -X axis pointing at a fixed surface spot on Psyche
      whenever the spot is visible, while body +Z (solar panels) tracks
      the Sun as closely as possible given the primary constraint.

Primary constraint:   body -X  →  surface spot (inertial, time-varying)
Secondary constraint: body +Z  →  projection of Sun onto plane ⊥ to -X

Visibility: spot is visible when it is on the near hemisphere of Psyche
            relative to the spacecraft:
              dot(spot_inertial - psyche_center, r_sc - psyche_center) > 0
            which simplifies (psyche_center = origin) to:
              dot(spot_inertial, r_sc) > 0

Out-of-view behavior: hold last valid attitude until spot becomes visible again.

Psyche parameters:
  Spin period: 4.196 hours  (Shepard et al. 2021)
  Pole (J2000): RA = 35°, Dec = -7°
  Treated as spherical for visibility calculation.
"""

# ── Psyche rotation model ─────────────────────────────────────────────────────
####### I'M NOT SUPER SURE ABOUT THIS #######
T_psyche_spin = 4.196 * 3600.0          # [s]  spin period
omega_psyche  = 2 * np.pi / T_psyche_spin  # [rad/s]  spin rate

# Pole orientation in J2000 (RA=35°, Dec=-7°) → inertial unit vector
RA_pole  = np.radians(35.0)
Dec_pole = np.radians(-7.0)
PSYCHE_POLE = np.array([
    np.cos(Dec_pole) * np.cos(RA_pole),
    np.cos(Dec_pole) * np.sin(RA_pole),
    np.sin(Dec_pole)
])
PSYCHE_POLE = PSYCHE_POLE / np.linalg.norm(PSYCHE_POLE)

# Sun direction: fixed in inertial frame (Psyche orbital motion negligible
# over spacecraft orbital timescales — period ~5 yr vs ~12 hr)
SUN_VEC = adcs.unit_vec(np.array([1.0, 0.0, 0.0]))

# Orbital rate
omega_orbit = 2 * np.pi / T_orbit


# ── Coordinate utilities ──────────────────────────────────────────────────────

def psyche_rotation_matrix(t):
    """
    Rotation matrix from Psyche body frame to inertial frame at time t.
    Rodrigues' rotation formula: rotate about PSYCHE_POLE by omega_psyche * t.
    """
    angle = omega_psyche * t
    k     = PSYCHE_POLE
    K     = adcs.hat(k)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def spot_inertial(t, spot_body):
    """
    Inertial position of surface spot at time t.
    spot_body: (3,) unit vector in Psyche body frame (fixed on surface).
    Returns unit vector in inertial frame.
    """
    R = psyche_rotation_matrix(t)
    return adcs.unit_vec(R @ spot_body)


def spacecraft_position_inertial(t):
    """
    Spacecraft position unit vector in inertial frame.
    Polar orbit: orbit plane contains +X (Sun direction).
    Orbit normal = +Y, spacecraft rotates in X-Z plane.
    """
    theta = omega_orbit * t
    r_hat = np.array([np.cos(theta), 0.0, np.sin(theta)])
    return r_hat   # unit vector FROM Psyche center TO spacecraft


def is_visible(t, spot_body):
    """
    Returns True if the surface spot is on the near hemisphere of Psyche
    relative to the spacecraft.
    Visibility condition: dot(spot_inertial, r_sc) > 0
    (both measured from Psyche center, so near-side check reduces to this).
    """
    s = spot_inertial(t, spot_body)
    r = spacecraft_position_inertial(t)
    return np.dot(s, r) > 0


# ── Target attitude construction ──────────────────────────────────────────────

def compute_q_target(t, spot_body):
    """
    Desired attitude quaternion when the spot is visible.

    Frame construction:
      neg_x_b = unit vector toward spot (primary: body -X points at spot)
      z_b     = projection of Sun onto plane ⊥ neg_x_b (secondary: +Z toward Sun)
      y_b     = z_b × (-neg_x_b)  =  z_b × x_b  (right-hand rule)

    The rotation matrix R = [x_b | y_b | z_b] maps body → inertial.
    """
    # Primary: -X body axis points at spot
    spot = spot_inertial(t, spot_body)
    neg_x_b = spot                          # body -X in inertial = spot direction
    x_b     = -neg_x_b                     # body +X in inertial

    # Secondary: project Sun onto plane perpendicular to x_b to get z_b
    sun_proj = SUN_VEC - np.dot(SUN_VEC, neg_x_b) * neg_x_b
    sun_proj_norm = np.linalg.norm(sun_proj)
    if sun_proj_norm < 1e-10:
        # Degenerate: Sun is along the instrument axis. Fall back to orbit normal.
        fallback = np.array([0., 1., 0.])
        sun_proj = fallback - np.dot(fallback, neg_x_b) * neg_x_b
        sun_proj_norm = np.linalg.norm(sun_proj)
    z_b = sun_proj / sun_proj_norm          # body +Z in inertial (solar panels)

    # Complete the frame
    y_b = np.cross(z_b, x_b)
    y_b = y_b / np.linalg.norm(y_b)

    # Re-orthogonalise x_b (numerical safety)
    x_b = np.cross(y_b, z_b)
    x_b = x_b / np.linalg.norm(x_b)

    # Rotation matrix: columns = body axes in inertial frame
    R = np.column_stack([x_b, y_b, z_b])
    return adcs.q(R)


def compute_omega_target(t, spot_body, dt=1.0):
    """
    Angular rate of the target frame, estimated by finite difference.
    omega_target = 2 * G(q_target).T @ q_dot_target
    """
    q_t    = compute_q_target(t, spot_body)
    q_t_dt = compute_q_target(t + dt, spot_body)
    if np.dot(q_t, q_t_dt) < 0:
        q_t_dt = -q_t_dt
    q_dot  = (q_t_dt - q_t) / dt
    return 2.0 * adcs.G(q_t).T @ q_dot


# ── Pointing error metrics ────────────────────────────────────────────────────

def instrument_pointing_error_deg(q, t, spot_body):
    """Angle between body -X axis and spot direction [deg]."""
    neg_x_inertial = -(adcs.Q(q) @ np.array([1., 0., 0.]))
    spot            = spot_inertial(t, spot_body)
    cos_a           = np.clip(np.dot(neg_x_inertial, spot), -1, 1)
    return np.degrees(np.arccos(cos_a))


def solar_panel_efficiency(q):
    """
    cos(angle between body +Z and Sun) — proxy for solar panel power.
    1.0 = perfect, 0.0 = panels edge-on to Sun.
    """
    z_inertial = adcs.Q(q) @ np.array([0., 0., 1.])
    return max(0.0, np.dot(z_inertial, SUN_VEC))


# ── MEKF step ──────────────────────────────────────

def mekf_step(x_est, P_est, gyro_m, q_sru, dt, sigma_gyro, sigma_bias, sigma_sru):
    V_kf = np.block([
        [sigma_gyro**2 * np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)),          sigma_bias**2 * np.eye(3)]
    ])
    W_kf = sigma_sru**2 * np.eye(3)
    q_p, beta_p = x_est[:4], x_est[4:7]
    q_pred = adcs.unit_vec(adcs.L(q_p) @ adcs.qexp(0.5*dt*(gyro_m - beta_p)))
    x_pred = np.concatenate([q_pred, beta_p])
    qk1       = x_pred[:4]
    dphidphi  = adcs.G(qk1).T @ adcs.R(adcs.qexp(0.5*dt*(gyro_m-beta_p))) @ adcs.G(q_p)
    dphidbeta = -0.5*dt * adcs.G(qk1).T @ adcs.G(q_p)
    A_kf      = np.block([[dphidphi, dphidbeta], [np.zeros((3,3)), np.eye(3)]])
    P_pred    = A_kf @ P_est @ A_kf.T + V_kf
    z     = adcs.qlog(adcs.qmult(adcs.qinv(x_pred[:4]), q_sru))
    C_att = adcs.H.T @ adcs.R(q_sru) @ adcs.Tmat @ adcs.G(x_pred[:4])
    C     = np.hstack([C_att, np.zeros((3, 3))])
    S     = C @ P_pred @ C.T + W_kf
    K     = P_pred @ C.T @ np.linalg.inv(S)
    delta   = K @ z
    phi_upd = -delta[:3]
    q_upd   = adcs.L(x_pred[:4]) @ np.concatenate([
                  [np.sqrt(max(1 - phi_upd@phi_upd, 0.0))], phi_upd])
    x_new = np.concatenate([adcs.unit_vec(q_upd), x_pred[4:7] - delta[3:6]])
    P_new = (np.eye(6) - K@C) @ P_pred @ (np.eye(6) - K@C).T + K@W_kf@K.T
    return x_new, P_new


# ── Main simulation ───────────────────────────────────────────────────────────

def run_surface_pointing(n_orbits=3, dt=2.0):
    """
    Simulate the surface-spot pointing controller for n_orbits.

    Returns a dict of history arrays.
    """
    tf    = n_orbits * T_orbit
    t_arr = np.arange(0.0, tf, dt)
    N     = len(t_arr)

    # ── Random surface spot ────────────────────────────────────────────────────
    # Unit vector in Psyche body frame — fixed on the surface
    spot_body = adcs.unit_vec(np.random.randn(3))
    print(f"  Surface spot (Psyche body frame): {np.round(spot_body, 4)}")
    lat = np.degrees(np.arcsin(spot_body[2]))
    lon = np.degrees(np.arctan2(spot_body[1], spot_body[0]))
    print(f"  Lat = {lat:.1f}°,  Lon = {lon:.1f}°")

    # ── Noise parameters ───────────────────────────────────────────────────────
    sigma_gyro = np.radians(0.0035) / 60 / np.sqrt(dt)
    sigma_bias = np.radians(0.0035) / 3600
    sigma_sru  = np.radians(0.01)
    M_gyro     = np.diag(1 + 0.01 * np.random.randn(3))
    b_gyro     = sigma_bias * np.random.randn(3)

    # ── Initial condition ──────────────────────────────────────────────────────
    # Start pointing at spot if visible, else point -X at nadir as default
    if is_visible(0.0, spot_body):
        q0 = compute_q_target(0.0, spot_body)
    else:
        # Point -X at nadir, +Z at Sun
        nadir   = -spacecraft_position_inertial(0.0)
        q0      = compute_q_target(0.0, spot_body)   # will be overridden below
        # Build a safe default: -X nadir, +Z toward Sun
        neg_x_b = nadir
        x_b     = -neg_x_b
        sun_proj = SUN_VEC - np.dot(SUN_VEC, neg_x_b)*neg_x_b
        if np.linalg.norm(sun_proj) < 1e-10:
            sun_proj = np.array([0.,1.,0.])
        z_b = adcs.unit_vec(sun_proj)
        y_b = adcs.unit_vec(np.cross(z_b, x_b))
        x_b = np.cross(y_b, z_b)
        q0  = adcs.q(np.column_stack([x_b, y_b, z_b]))

    state = np.concatenate([q0, np.zeros(3), np.zeros(3)])
    x_est = np.concatenate([q0, np.zeros(3)])
    P_est = 0.01 * np.eye(6)

    # ── Storage ────────────────────────────────────────────────────────────────
    inst_err_h   = np.zeros(N)    # instrument pointing error [deg]
    solar_eff_h  = np.zeros(N)    # solar panel efficiency [0-1]
    visible_h    = np.zeros(N)    # visibility flag (1=visible, 0=not)
    tau_h        = np.zeros((N, 3))
    h_w_h        = np.zeros((N, 3))

    # ── Pre-compute next visibility transition times ───────────────────────────
    # For each timestep, store the time of the next visibility window start.
    # This avoids an O(N^2) search inside the main loop.
    next_vis_time = np.full(N, np.nan)
    for i in range(N - 1, -1, -1):
        t = t_arr[i]
        if is_visible(t, spot_body):
            next_vis_time[i] = t          # already visible
        else:
            # Look ahead: next_vis_time[i+1] may already have the answer
            if i + 1 < N and not np.isnan(next_vis_time[i + 1]):
                next_vis_time[i] = next_vis_time[i + 1]
            else:
                next_vis_time[i] = np.nan  # no future visibility found

    # Hold last valid target when no future visibility is found
    q_target_held = q0.copy()

    for i, t in enumerate(t_arr):
        q_true = adcs.unit_vec(state[0:4])
        omega  = state[4:7]
        h_w    = state[7:10]

        # ── Sensors ───────────────────────────────────────────────────────────
        w_gyro = sigma_gyro * np.random.randn(3)
        gyro_m = M_gyro @ omega + b_gyro + w_gyro
        q_sru  = adcs.qmult(q_true, adcs.qexp(sigma_sru * np.random.randn(3)))

        # ── MEKF ──────────────────────────────────────────────────────────────
        x_est, P_est = mekf_step(x_est, P_est, gyro_m, q_sru,
                                  dt, sigma_gyro, sigma_bias, sigma_sru)
        q_est     = x_est[:4]
        omega_est = gyro_m - x_est[4:7]

        # ── Visibility and target ──────────────────────────────────────────────
        visible = is_visible(t, spot_body)
        if visible:
            q_target     = compute_q_target(t, spot_body)
            omega_target = compute_omega_target(t, spot_body, dt=dt)
            q_target_held = q_target.copy()
        else:
            # Pre-slew: point toward where the spot will be at the next
            # visibility window start, so the spacecraft arrives pre-pointed.
            t_nv = next_vis_time[i]
            if not np.isnan(t_nv):
                q_target = compute_q_target(t_nv, spot_body)
            else:
                q_target = q_target_held   # no future visibility: hold last
            omega_target = np.zeros(3)     # no feedforward during pre-slew

        # ── PD control with feedforward rate ──────────────────────────────────
        q_e = adcs.qmult(adcs.qinv(q_target), q_est)
        if q_e[0] < 0:
            q_e = -q_e
        phi_e       = adcs.qlog(q_e)
        omega_error = omega_est - omega_target
        tau_cmd     = -(Kp_mat @ phi_e) - (Kd_mat @ omega_error)
        tau_cmd     = np.clip(tau_cmd, -tau_max, tau_max)

        # ── Disturbances ──────────────────────────────────────────────────────
        r_hat = spacecraft_position_inertial(t)
        nadir = -r_hat
        tau_gg  = gravity_gradient_torque(q_true, nadir)
        tau_srp = solar_radiation_torque(q_true, SUN_VEC)
        tau_dist = tau_gg + tau_srp

        # ── Log ───────────────────────────────────────────────────────────────
        inst_err_h[i]  = instrument_pointing_error_deg(q_true, t, spot_body)
        solar_eff_h[i] = solar_panel_efficiency(q_true)
        visible_h[i]   = 1.0 if visible else 0.0
        tau_h[i]       = tau_cmd
        h_w_h[i]       = h_w

        # ── Integrate ─────────────────────────────────────────────────────────
        def f(t_, s_):
            q_     = s_[0:4]
            omega_ = s_[4:7]
            h_w_   = s_[7:10]
            h_w_dot_  = -tau_cmd
            L_tot     = J @ omega_ + h_w_
            om_dot    = np.linalg.solve(J, -np.cross(omega_, L_tot)
                                        + tau_cmd + tau_dist)
            q_dot_    = 0.5 * adcs.G(q_) @ omega_
            return np.concatenate([q_dot_, om_dot, h_w_dot_])

        k1 = f(t, state)
        k2 = f(t + dt/2, state + dt*k1/2)
        k3 = f(t + dt/2, state + dt*k2/2)
        k4 = f(t + dt,   state + dt*k3)
        state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        state[0:4]  = adcs.unit_vec(state[0:4])
        state[7:10] = np.clip(state[7:10], -h_max, h_max)

    return {
        't':          t_arr,
        'inst_err':   inst_err_h,
        'solar_eff':  solar_eff_h,
        'visible':    visible_h,
        'tau':        tau_h,
        'h_w':        h_w_h,
        'spot_body':  spot_body,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results):
    t_min     = results['t'] / 60
    T_orb_min = T_orbit / 60
    visible   = results['visible']

    # Shade visible periods
    def shade_visible(ax):
        in_view = False
        t0 = 0.0
        for i in range(len(visible)):
            if visible[i] and not in_view:
                t0     = t_min[i]
                in_view = True
            elif not visible[i] and in_view:
                ax.axvspan(t0, t_min[i], alpha=0.12, color='green')
                in_view = False
        if in_view:
            ax.axvspan(t0, t_min[-1], alpha=0.12, color='green')
        # Orbit period markers
        for k in range(1, int(t_min[-1] / T_orb_min) + 1):
            ax.axvline(k * T_orb_min, color='grey', linestyle='--',
                       lw=0.8, alpha=0.5)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle('Surface Spot Pointing Controller\n'
                 '(green = spot visible, grey lines = orbit periods)',
                 fontsize=12, fontweight='bold')

    # ── Panel 1: Instrument pointing error ───────────────────────────────────
    # Only meaningful when spot is visible — mask out-of-view periods
    err_masked = np.where(visible > 0.5, results['inst_err'], np.nan)
    rms_visible = np.sqrt(np.nanmean(err_masked**2))
    axes[0].plot(t_min, err_masked, lw=0.9, color='steelblue',
                 label=f'Instrument error (−X vs spot)  RMS = {rms_visible:.4f}°')
    shade_visible(axes[0])
    axes[0].set_ylabel('Pointing Error [deg]')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ── Panel 2: Solar panel efficiency ──────────────────────────────────────
    mean_eff = np.mean(results['solar_eff']) * 100
    axes[1].plot(t_min, results['solar_eff'] * 100, lw=0.9, color='darkorange',
                 label=f'Solar panel efficiency  mean = {mean_eff:.1f}%')
    shade_visible(axes[1])
    axes[1].set_ylabel('Solar Panel Efficiency [%]')
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # ── Panel 3: Commanded torque ─────────────────────────────────────────────
    for j, (lbl, col) in enumerate(zip(['τ₁','τ₂','τ₃'],
                                        ['steelblue','darkorange','green'])):
        axes[2].plot(t_min, results['tau'][:, j], lw=0.7,
                     alpha=0.8, label=lbl, color=col)
    axes[2].axhline( tau_max, color='r', linestyle=':', lw=1.0, label='±τ_max')
    axes[2].axhline(-tau_max, color='r', linestyle=':', lw=1.0)
    # Only add orbit period lines here, skip per-window shading to reduce clutter
    for k in range(1, int(t_min[-1] / T_orb_min) + 1):
        axes[2].axvline(k * T_orb_min, color='grey', linestyle='--',
                        lw=0.8, alpha=0.5)
    axes[2].set_ylabel('τ_cmd per axis [N·m]')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # ── Panel 4: Wheel momentum ───────────────────────────────────────────────
    for j, lbl in enumerate(['h_w1', 'h_w2', 'h_w3']):
        axes[3].plot(t_min, results['h_w'][:, j], lw=0.8, label=lbl)
    axes[3].axhline( h_max, color='k', linestyle=':', lw=1.0)
    axes[3].axhline(-h_max, color='k', linestyle=':', lw=1.0, label='±h_max')
    shade_visible(axes[3])
    axes[3].set_ylabel('Wheel Momentum [N·m·s]')
    axes[3].set_xlabel('Time [min]')
    axes[3].legend(fontsize=8, ncol=4)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('surface_pointing_results.png', dpi=150)
    plt.show()

    # ── Visibility summary ────────────────────────────────────────────────────
    frac_visible = np.mean(visible) * 100
    print(f"\n  Spot visibility:           {frac_visible:.1f}% of simulation")
    print(f"  RMS pointing error (when visible): {rms_visible:.4f}°")
    print(f"  Mean solar panel efficiency:       {mean_eff:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("SURFACE SPOT POINTING CONTROLLER")
    print(f"  Orbital period  = {T_orbit/60:.1f} min")
    print(f"  Psyche spin period = {T_psyche_spin/3600:.3f} hr")
    print(f"  Psyche pole (inertial) = {np.round(PSYCHE_POLE, 4)}")
    print("=" * 60)

    print(f"  wn_track = {wn_track} rad/s,  zeta_track = {zeta_track}  (critically damped)")
    print(f"  Kp diag  = {np.round(np.diag(Kp_mat), 4)}")
    print(f"  Kd diag  = {np.round(np.diag(Kd_mat), 4)}")
    print(f"  Psyche spin rate = {omega_psyche*1e4:.2f}e-4 rad/s  "
          f"(wn/omega_psyche = {wn_track/omega_psyche:.1f})")

    results = run_surface_pointing(n_orbits=3, dt=2.0)
    plot_results(results)