import numpy as np
import matplotlib.pyplot as plt
import adcs_toolbox as adcs
from psyche_model import tot_moicom as J

from mekf_with_controller import (
    tau_max, h_max, Kp_mat, Kd_mat, J_diag, pd_control
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TRAJECTORY GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_maneuver_time(theta_f_vec, safety_factor=0.8):

    theta_f = np.linalg.norm(theta_f_vec)
    if theta_f < 1e-10:
        return 1.0   # trivial maneuver

    n_hat = theta_f_vec / theta_f   # unit eigenaxis

    # Required T per axis
    T_per_axis = np.pi * np.sqrt(
        J_diag * np.abs(n_hat) * theta_f / (2 * tau_max * safety_factor)
    )
    T = np.max(T_per_axis)
    return T


def versine_trajectory(q0, q_f, dt=1.0):

    # ── Eigenaxis from relative quaternion ───────────────────────────────────
    q_rel = adcs.qmult(adcs.qinv(q0), q_f)
    if q_rel[0] < 0:
        q_rel = -q_rel                          # short-rotation convention

    phi_f   = adcs.qlog(q_rel)                 # (3,)  = (θ_f/2) · n̂
    if not np.all(np.isfinite(phi_f)) or np.linalg.norm(phi_f) < 1e-10:
        v = q_rel[1:4]
        v_norm = np.linalg.norm(v)
        n_hat_fallback = v / v_norm if v_norm > 1e-10 else np.array([1., 0., 0.])
        phi_f = (np.pi / 2) * n_hat_fallback   # θ_f/2 = π/2 at 180°
    theta_f = 2 * np.linalg.norm(phi_f)        # total rotation angle [rad]
    n_hat   = phi_f / np.linalg.norm(phi_f)    # unit eigenaxis

    print(f"  Eigenaxis n̂ = {np.round(n_hat, 4)}")
    print(f"  Total angle θ_f = {np.degrees(theta_f):.2f}°")

    # ── Auto-compute maneuver time ────────────────────────────────────────────
    T = compute_maneuver_time(theta_f * n_hat)
    alpha_profile = np.pi / T                   # α in versine formula

    print(f"  Maneuver time T = {T:.1f} s  ({T/3600:.2f} hr)")
    tau_peak_check = J_diag * np.abs(n_hat) * theta_f * np.pi**2 / (2 * T**2)
    print(f"  Peak |ρ̇| per axis = {np.round(tau_peak_check, 5)} N·m  (limit {tau_max} N·m)")

    # ── Build time array ──────────────────────────────────────────────────────
    t_arr = np.arange(0.0, T + dt, dt)
    N     = len(t_arr)

    # ── Scalar versine profiles ───────────────────────────────────────────────
    theta_arr = theta_f / 2 * (1 - np.cos(alpha_profile * t_arr))
    omega_s   = theta_f * alpha_profile / 2 * np.sin(alpha_profile * t_arr)
    alpha_s   = theta_f * alpha_profile**2 / 2 * np.cos(alpha_profile * t_arr)

    # ── Vector profiles ───────────────────────────────────────────────────────
    q_nom     = np.zeros((N, 4))
    omega_nom = np.zeros((N, 3))
    alpha_nom = np.zeros((N, 3))
    rho_nom   = np.zeros((N, 3))   # wheel momentum  ρ(t)
    tau_nom   = np.zeros((N, 3))   # feedforward = ρ_dot(t)

    q_nom[0] = q0.copy()
    rho_nom[0] = np.zeros(3)

    for k in range(N):
        omega_nom[k] = omega_s[k] * n_hat
        alpha_nom[k] = alpha_s[k] * n_hat

        # Inverse dynamics:
        w   = omega_nom[k]
        a   = alpha_nom[k]
        rho = rho_nom[k]
        rho_dot = -J @ a - np.cross(w, J @ w + rho)
        tau_nom[k] = rho_dot   # feedforward command is rho_dot

        # Integrate wheel momentum:  rho_{k+1} = rho_k + dt * rho_dot
        if k < N - 1:
            rho_nom[k+1] = rho_nom[k] + dt * rho_dot
            # Integrate quaternion kinematics
            q_dot        = 0.5 * adcs.G(q_nom[k]) @ omega_nom[k]
            q_nom[k+1]   = adcs.unit_vec(q_nom[k] + dt * q_dot)

    return T, t_arr, q_nom, omega_nom, tau_nom, rho_nom, alpha_nom, theta_arr


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CLOSED-LOOP SIMULATION  (trajectory tracking + MEKF)
# ─────────────────────────────────────────────────────────────────────────────

def run_closed_loop(q0, q_f, dt=1.0):

    # ── Generate nominal trajectory ───────────────────────────────────────────
    print("\n── Generating nominal trajectory ──────────────────────────")
    T, t_arr, q_nom, omega_nom, tau_nom, rho_nom, alpha_nom, theta_arr = \
        versine_trajectory(q0, q_f, dt=dt)
    N = len(t_arr)

    # ── MEKF noise parameters (match attitude_controller.py / Test 2) ─────────
    sigma_gyro = np.radians(0.0035) / 60 / np.sqrt(dt)
    sigma_bias = np.radians(0.0035) / 3600
    sigma_sru  = np.radians(0.01)

    V_kf = np.block([
        [sigma_gyro**2 * np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)),          sigma_bias**2 * np.eye(3)]
    ])
    W_kf = sigma_sru**2 * np.eye(3)

    M_gyro = np.diag(1 + 0.01 * np.random.randn(3))
    b_gyro = sigma_bias * np.random.randn(3)

    # ── Storage ───────────────────────────────────────────────────────────────
    q_true_h   = np.zeros((N, 4))
    q_est_h    = np.zeros((N, 4))
    omega_h    = np.zeros((N, 3))
    omega_est_h= np.zeros((N, 3))
    h_w_h      = np.zeros((N, 3))
    tau_ff_h   = np.zeros((N, 3))   # feedforward
    tau_fb_h   = np.zeros((N, 3))   # feedback
    tau_tot_h  = np.zeros((N, 3))   # total commanded
    track_err_h= np.zeros(N)        # tracking error [deg]

    # ── Initial conditions ────────────────────────────────────────────────────
    state  = np.concatenate([q0, np.zeros(3), np.zeros(3)])  # [q, omega, h_w]
    x_est  = np.concatenate([q0, np.zeros(3)])               # [q, beta]
    P_est  = 0.1 * np.eye(6)

    print("\n── Running closed-loop simulation ─────────────────────────")

    for i in range(N):
        q_true = adcs.unit_vec(state[0:4])
        omega  = state[4:7]
        h_w    = state[7:10]

        # ── Sensor measurements ───────────────────────────────────────────────
        w_gyro  = sigma_gyro * np.random.randn(3)
        gyro_m  = M_gyro @ omega + b_gyro + w_gyro

        noise_phi = sigma_sru * np.random.randn(3)
        q_sru     = adcs.qmult(q_true, adcs.qexp(noise_phi))

        # ── MEKF predict ──────────────────────────────────────────────────────
        q_p    = x_est[:4]
        beta_p = x_est[4:7]
        q_pred = adcs.unit_vec(adcs.L(q_p) @ adcs.qexp(0.5 * dt * (gyro_m - beta_p)))
        x_pred = np.concatenate([q_pred, beta_p])

        qk1        = x_pred[:4]
        dphidphi   = adcs.G(qk1).T @ adcs.R(adcs.qexp(0.5*dt*(gyro_m - beta_p))) @ adcs.G(q_p)
        dphidbeta  = -0.5 * dt * adcs.G(qk1).T @ adcs.G(q_p)
        A_kf       = np.block([[dphidphi, dphidbeta], [np.zeros((3,3)), np.eye(3)]])
        P_pred     = A_kf @ P_est @ A_kf.T + V_kf

        # ── MEKF update ───────────────────────────────────────────────────────
        z     = adcs.qlog(adcs.qmult(adcs.qinv(x_pred[:4]), q_sru))
        C_att = adcs.H.T @ adcs.R(q_sru) @ adcs.Tmat @ adcs.G(x_pred[:4])
        C     = np.hstack([C_att, np.zeros((3, 3))])
        S     = C @ P_pred @ C.T + W_kf
        K     = P_pred @ C.T @ np.linalg.inv(S)

        delta    = K @ z
        phi_upd  = -delta[:3]
        beta_upd = x_pred[4:7] - delta[3:6]
        q_upd    = adcs.L(x_pred[:4]) @ np.concatenate([
                       [np.sqrt(max(1 - phi_upd @ phi_upd, 0.0))], phi_upd])
        x_est  = np.concatenate([adcs.unit_vec(q_upd), beta_upd])
        P_est  = (np.eye(6) - K@C) @ P_pred @ (np.eye(6) - K@C).T + K @ W_kf @ K.T

        q_est_use  = x_est[:4]
        omega_est  = gyro_m - x_est[4:7]   # bias-corrected rate

        # ── Control: feedforward + PD feedback on tracking error ──────────────
        # Tracking error = deviation from nominal trajectory at this timestep
        tau_ff = tau_nom[i]                                    # inverse dynamics
        tau_fb, phi_track = pd_control(q_est_use, omega_est,
                                       q_nom[i], h_w)          # PD on track error
        tau_total = np.clip(tau_ff + tau_fb, -tau_max, tau_max)

        # ── Log ───────────────────────────────────────────────────────────────
        q_true_h[i]    = q_true
        q_est_h[i]     = q_est_use
        omega_h[i]     = omega
        omega_est_h[i] = omega_est
        h_w_h[i]       = h_w
        tau_ff_h[i]    = tau_ff
        tau_fb_h[i]    = tau_fb
        tau_tot_h[i]   = tau_total
        track_err_h[i] = np.degrees(2 * np.linalg.norm(phi_track))

        # ── Integrate one RK4 step ────────────────────────────────────────────
        def f(t_, s_):
            q_     = s_[0:4]
            omega_ = s_[4:7]
            h_w_   = s_[7:10]
            h_w_dot_  = -tau_total
            L_tot     = J @ omega_ + h_w_
            om_dot    = np.linalg.solve(J, -np.cross(omega_, L_tot) + tau_total)
            q_dot_    = 0.5 * adcs.G(q_) @ omega_
            return np.concatenate([q_dot_, om_dot, h_w_dot_])

        k1 = f(t_arr[i], state)
        k2 = f(t_arr[i] + dt/2, state + dt*k1/2)
        k3 = f(t_arr[i] + dt/2, state + dt*k2/2)
        k4 = f(t_arr[i] + dt,   state + dt*k3)
        state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        state[0:4]  = adcs.unit_vec(state[0:4])
        state[7:10] = np.clip(state[7:10], -h_max, h_max)

    return (T, t_arr, q_nom, omega_nom, tau_nom,
            q_true_h, q_est_h, omega_h, omega_est_h,
            h_w_h, tau_ff_h, tau_fb_h, tau_tot_h,
            track_err_h, theta_arr)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ANGLE HISTORY FROM QUATERNION ARRAY
# ─────────────────────────────────────────────────────────────────────────────

def qarray_to_angle_from_q0(q_arr, q0):
    """Convert an (N,4) quaternion history to angle from q0 [deg]."""
    angles = np.zeros(len(q_arr))
    for i, q in enumerate(q_arr):
        q_rel = adcs.qmult(adcs.qinv(q0), q)
        if q_rel[0] < 0:
            q_rel = -q_rel
        angles[i] = np.degrees(2 * np.linalg.norm(adcs.qlog(q_rel)))
    return angles


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(T, t_arr, q_nom, omega_nom, tau_nom,
                 q_true_h, q_est_h, omega_h, omega_est_h,
                 h_w_h, tau_ff_h, tau_fb_h, tau_tot_h,
                 track_err_h, theta_arr, q0, q_f):

    t_min = t_arr / 60   # plot in minutes for readability

    theta_f_deg = np.degrees(np.max(theta_arr))
    nom_angle   = qarray_to_angle_from_q0(q_nom,    q0)
    true_angle  = qarray_to_angle_from_q0(q_true_h, q0)
    est_angle   = qarray_to_angle_from_q0(q_est_h,  q0)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"Eigenaxis Slew — {theta_f_deg:.1f}°  (T = {T/60:.1f} min)",
                 fontsize=13, fontweight='bold')

    # ── Panel 1: Slew angle ───────────────────────────────────────────────────
    axes[0].plot(t_min, nom_angle,  'k--', lw=1.5, label='Nominal')
    axes[0].plot(t_min, true_angle, lw=1.0, label='True (closed-loop)')
    axes[0].plot(t_min, est_angle,  lw=0.8, alpha=0.7, linestyle=':', label='Estimated')
    axes[0].set_ylabel('Slew Angle [deg]')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ── Panel 2: Body rates ───────────────────────────────────────────────────
    labels = ['ω₁', 'ω₂', 'ω₃']
    for j in range(3):
        axes[1].plot(t_min, np.degrees(omega_nom[:, j]),
                     '--', lw=1.2, label=f'{labels[j]} nom')
    for j in range(3):
        axes[1].plot(t_min, np.degrees(omega_h[:, j]),
                     lw=0.8, alpha=0.8, label=f'{labels[j]} true')
    axes[1].set_ylabel('Body Rate [deg/s]')
    axes[1].legend(fontsize=7, ncol=3)
    axes[1].grid(True, alpha=0.3)

    # ── Panel 3: Torques ──────────────────────────────────────────────────────
    axes[2].plot(t_min, np.linalg.norm(tau_nom,   axis=1), 'k--', lw=1.2,
                 label=r'$|\dot{\rho}_{ff}|$ feedforward (inv. dynamics)')
    axes[2].plot(t_min, np.linalg.norm(tau_fb_h,  axis=1), lw=0.9,
                 label=r'$|\tau_{fb}|$ PD feedback')
    axes[2].plot(t_min, np.linalg.norm(tau_tot_h, axis=1), lw=0.9,
                 label=r'$|\tau_{total}|$')
    axes[2].axhline(tau_max, color='r', linestyle=':', lw=1.0, label=r'$\tau_{max}$')
    axes[2].set_ylabel(r'$|\dot{\rho}|$ / Torque Magnitude [N·m]')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # ── Panel 4: Tracking error ───────────────────────────────────────────────
    rms_track = np.sqrt(np.mean(track_err_h**2))
    axes[3].plot(t_min, track_err_h, lw=0.9, label='Tracking error')
    axes[3].axhline(rms_track, color='k', linestyle=':', lw=1.2,
                    label=f'RMS = {rms_track:.4f}°')
    axes[3].set_ylabel('Tracking Error [deg]')
    axes[3].set_xlabel('Time [min]')
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eigenaxis_slew_results.png', dpi=150)
    plt.show()

    return rms_track


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPARISON: eigenaxis slew vs regulator for large initial error
# ─────────────────────────────────────────────────────────────────────────────

def run_regulator_only(q0, q_f, T, dt=1.0):

    N = int(T / dt) + 1
    t_arr = np.arange(N) * dt

    sigma_gyro = np.radians(0.0035) / 60 / np.sqrt(dt)
    sigma_bias = np.radians(0.0035) / 3600
    sigma_sru  = np.radians(0.01)
    V_kf = np.block([[sigma_gyro**2*np.eye(3), np.zeros((3,3))],
                     [np.zeros((3,3)),          sigma_bias**2*np.eye(3)]])
    W_kf = sigma_sru**2 * np.eye(3)
    M_gyro = np.diag(1 + 0.01*np.random.randn(3))
    b_gyro = sigma_bias * np.random.randn(3)

    state = np.concatenate([q0, np.zeros(3), np.zeros(3)])
    x_est = np.concatenate([q0, np.zeros(3)])
    P_est = 0.1 * np.eye(6)

    angle_h = np.zeros(N)

    for i in range(N):
        q_true = adcs.unit_vec(state[0:4])
        omega  = state[4:7]
        h_w    = state[7:10]

        w_gyro = sigma_gyro * np.random.randn(3)
        gyro_m = M_gyro @ omega + b_gyro + w_gyro
        q_sru  = adcs.qmult(q_true, adcs.qexp(sigma_sru * np.random.randn(3)))

        # MEKF predict
        q_p, beta_p = x_est[:4], x_est[4:7]
        q_pred = adcs.unit_vec(adcs.L(q_p) @ adcs.qexp(0.5*dt*(gyro_m - beta_p)))
        x_pred = np.concatenate([q_pred, beta_p])
        qk1    = x_pred[:4]
        dpp    = adcs.G(qk1).T @ adcs.R(adcs.qexp(0.5*dt*(gyro_m-beta_p))) @ adcs.G(q_p)
        dpb    = -0.5*dt * adcs.G(qk1).T @ adcs.G(q_p)
        A_kf   = np.block([[dpp, dpb], [np.zeros((3,3)), np.eye(3)]])
        P_pred = A_kf @ P_est @ A_kf.T + V_kf

        # MEKF update
        z     = adcs.qlog(adcs.qmult(adcs.qinv(x_pred[:4]), q_sru))
        C_att = adcs.H.T @ adcs.R(q_sru) @ adcs.Tmat @ adcs.G(x_pred[:4])
        C     = np.hstack([C_att, np.zeros((3,3))])
        S     = C @ P_pred @ C.T + W_kf
        K     = P_pred @ C.T @ np.linalg.inv(S)
        delta   = K @ z
        phi_upd = -delta[:3]
        q_upd   = adcs.L(x_pred[:4]) @ np.concatenate([
                      [np.sqrt(max(1-phi_upd@phi_upd, 0.0))], phi_upd])
        x_est = np.concatenate([adcs.unit_vec(q_upd), x_pred[4:7] - delta[3:6]])
        P_est = (np.eye(6)-K@C) @ P_pred @ (np.eye(6)-K@C).T + K@W_kf@K.T

        omega_est = gyro_m - x_est[4:7]
        tau_cmd, _ = pd_control(x_est[:4], omega_est, q_f, h_w)

        # Log angle from q0
        q_rel = adcs.qmult(adcs.qinv(q0), q_true)
        if q_rel[0] < 0: q_rel = -q_rel
        angle_h[i] = np.degrees(2 * np.linalg.norm(adcs.qlog(q_rel)))

        def f(t_, s_):
            q_, omega_, h_w_ = s_[0:4], s_[4:7], s_[7:10]
            L_tot  = J @ omega_ + h_w_
            om_dot = np.linalg.solve(J, -np.cross(omega_, L_tot) + tau_cmd)
            return np.concatenate([0.5*adcs.G(q_)@omega_, om_dot, -tau_cmd])

        k1 = f(0, state); k2 = f(0, state+dt*k1/2)
        k3 = f(0, state+dt*k2/2); k4 = f(0, state+dt*k3)
        state = state + (dt/6)*(k1+2*k2+2*k3+k4)
        state[0:4]  = adcs.unit_vec(state[0:4])
        state[7:10] = np.clip(state[7:10], -h_max, h_max)

    return t_arr, angle_h


def plot_comparison(T, t_arr, nom_angle, true_angle_slew, t_reg, angle_reg, theta_f_deg):
    """Plot eigenaxis slew vs regulator angle histories side by side."""
    fig, ax = plt.subplots(figsize=(11, 5))
    t_min     = t_arr  / 60
    t_min_reg = t_reg  / 60

    ax.plot(t_min,     nom_angle,          'k--', lw=1.5, label='Nominal trajectory')
    ax.plot(t_min,     true_angle_slew,    lw=1.2,        label='Eigenaxis slew (closed-loop)')
    ax.plot(t_min_reg, angle_reg,          lw=1.2,        label='Regulator only (no trajectory)')
    ax.axhline(theta_f_deg, color='grey', linestyle=':', lw=1.0, label=f'Target {theta_f_deg:.1f}°')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Slew Angle from q₀ [deg]')
    ax.set_title('Eigenaxis Slew vs Regulator — Large Initial Error Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('eigenaxis_vs_regulator.png', dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    q0  = adcs.unit_vec(np.random.randn(4))
    q_f = adcs.unit_vec(np.random.randn(4))

    # Ensure q_f is not the same rotation as q0 (q and -q are identical in SO3).
    # Compute actual separation and re-draw if too small.
    def separation_deg(qa, qb):
        q_rel = adcs.qmult(adcs.qinv(qa), qb)
        if q_rel[0] < 0:
            q_rel = -q_rel
        phi = adcs.qlog(q_rel)
        if not np.all(np.isfinite(phi)):
            return 0.0
        return np.degrees(2 * np.linalg.norm(phi))

    while separation_deg(q0, q_f) < 170.0:
        q_f = adcs.unit_vec(np.random.randn(4))

    actual_deg = separation_deg(q0, q_f)
    print("=" * 60)
    print("EIGENAXIS SLEW — 180° MANEUVER")
    print("=" * 60)
    print(f"  q0  = {np.round(q0, 4)}")
    print(f"  q_f = {np.round(q_f, 4)}")
    print(f"  Actual separation = {actual_deg:.2f}°")

    dt = 2.0   # [s]

    results = run_closed_loop(q0, q_f, dt=dt)
    (T, t_arr, q_nom, omega_nom, tau_nom,
     q_true_h, q_est_h, omega_h, omega_est_h,
     h_w_h, tau_ff_h, tau_fb_h, tau_tot_h,
     track_err_h, theta_arr) = results

    # Compute angle histories for plotting
    nom_angle        = qarray_to_angle_from_q0(q_nom,    q0)
    true_angle_slew  = qarray_to_angle_from_q0(q_true_h, q0)

    rms = plot_results(*results, q0=q0, q_f=q_f)

    final_err = np.degrees(2 * np.linalg.norm(
        adcs.qlog(adcs.qmult(adcs.qinv(q_f), q_true_h[-1]))))
    print(f"\n  RMS tracking error : {rms:.4f}°")
    print(f"  Final attitude error (vs q_f): {final_err:.4f}°")

    # ── Comparison: regulator for same duration ───────────────────────────────
    print("\n── Running regulator comparison ───────────────────────────")
    t_reg, angle_reg = run_regulator_only(q0, q_f, T, dt=dt)

    theta_f_deg = np.degrees(2 * np.linalg.norm(
        adcs.qlog(adcs.qmult(adcs.qinv(q0), q_f))))

    plot_comparison(T, t_arr, nom_angle, true_angle_slew,
                    t_reg, angle_reg, theta_f_deg)

    print("\nDone. Figures saved to outputs/")