import numpy as np
import matplotlib.pyplot as plt
import adcs_toolbox as adcs
from psyche_model import tot_moicom as J
import attitude_estimation as att_est

# ── Reaction wheel parameters ──────────────────────────────────────────────────
tau_max = 0.05      # [N·m]  peak torque per wheel
h_max   = 300.0     # [N·m·s] momentum storage per wheel

# ── PD gain selection ──────────────────────────────────────────────────────────
# Treat each axis independently; use the diagonal of J as the effective inertia.
J_diag = np.diag(J)          # (3,)  principal moments of inertia  [kg·m²]
wn     = 0.05                # [rad/s]  desired natural frequency
zeta   = 0.9                 # [-]      damping ratio

Kp = J_diag * wn**2          # (3,)  proportional gains
Kd = J_diag * 2 * zeta * wn  # (3,)  derivative gains

# Convert to diagonal matrices for clean multiplication
Kp_mat = np.diag(Kp)
Kd_mat = np.diag(Kd)


# ── Controller ─────────────────────────────────────────────────────────────────
def pd_control(q_est, omega_est, q_target, h_w):

    # Attitude error quaternion:  q_e = q_target^{-1} ⊗ q_est
    q_e   = adcs.qmult(adcs.qinv(q_target), q_est)

    # Ensure the scalar part is non-negative (short-rotation convention).
    if q_e[0] < 0:
        q_e = -q_e

    # Map to 3-vector error via quaternion log (≈ half the rotation axis × angle)
    phi_e = adcs.qlog(q_e)   # (3,)

    # PD law:  tau = -Kp * phi_e  -  Kd * omega
    tau_cmd = -(Kp_mat @ phi_e) - (Kd_mat @ omega_est)

    # Saturate torque to wheel capability
    tau_cmd = np.clip(tau_cmd, -tau_max, tau_max)

    return tau_cmd, phi_e


# ── RK4 with wheel momentum saturation ────────────────────────────────────────
def rk4_controlled(q_target, state0, t0, tf, dt,
                   use_mekf=False, disturbance_fn=None):
    
    t_vals    = np.arange(t0, tf, dt)
    N         = len(t_vals)
    q_hist    = np.zeros((N, 4))
    omega_hist = np.zeros((N, 3))
    h_w_hist  = np.zeros((N, 3))
    err_hist  = np.zeros(N)
    tau_hist  = np.zeros((N, 3))

    state = state0.copy()

    for i, t in enumerate(t_vals):
        q     = state[0:4]
        omega = state[4:7]
        h_w   = state[7:10]

        tau_cmd, phi_e = pd_control(q, omega, q_target, h_w)

        q_hist[i]     = q
        omega_hist[i] = omega
        h_w_hist[i]   = h_w
        err_hist[i]   = np.degrees(2 * np.linalg.norm(phi_e))  # error in deg
        tau_hist[i]   = tau_cmd

        # Build dynamics with optional disturbance
        def f(t_, s_):
            q_     = s_[0:4]
            omega_ = s_[4:7]
            h_w_   = s_[7:10]
            tau_, _ = pd_control(q_, omega_, q_target, h_w_)
            tau_ext_ = disturbance_fn(t_) if disturbance_fn is not None else np.zeros(3)
            h_w_dot_   = -tau_
            L_total_   = J @ omega_ + h_w_
            omega_dot_ = np.linalg.solve(J, -np.cross(omega_, L_total_) + tau_ + tau_ext_)
            q_dot_     = 0.5 * adcs.G(q_) @ omega_
            return np.concatenate([q_dot_, omega_dot_, h_w_dot_])

        k1 = f(t, state)
        k2 = f(t + dt/2, state + dt*k1/2)
        k3 = f(t + dt/2, state + dt*k2/2)
        k4 = f(t + dt,   state + dt*k3)
        state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        # Normalize quaternion and saturate wheel momentum
        state[0:4] = adcs.unit_vec(state[0:4])
        state[7:10] = np.clip(state[7:10], -h_max, h_max)

    return t_vals, q_hist, omega_hist, h_w_hist, err_hist, tau_hist


# ════════════════════════════════════════════════════════════════════════════════
# TEST 1 — Random initial conditions up to ±90° error
# ════════════════════════════════════════════════════════════════════════════════
def test_random_initial_conditions(n_trials=10, max_angle_deg=90.0,
                                   tf=None, dt=0.5):

    if tf is None:
        J_max     = np.max(np.diag(J))
        alpha_max = tau_max / J_max              # max angular acceleration [rad/s²]
        # Time to slew 90° from rest at constant alpha_max (lower bound), ×5 margin
        t_slew = np.sqrt(2 * np.radians(max_angle_deg) / alpha_max)
        tf = 5 * t_slew
        print(f"  Auto tf = {tf:.0f} s  (slew estimate = {t_slew:.0f} s)")
    print("=" * 60)
    print("TEST 1: Random initial conditions (±90° error)")
    print("=" * 60)

    # Fixed target: identity quaternion (hold inertial reference frame)
    q_target = np.array([1.0, 0.0, 0.0, 0.0])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_err, ax_tau = axes

    final_errors = []

    for trial in range(n_trials):
        # Random rotation axis and angle up to max_angle_deg
        axis  = adcs.unit_vec(np.random.randn(3))
        angle = np.radians(np.random.uniform(0, max_angle_deg))
        q0    = adcs.quat(axis, angle)

        # Zero initial body rate — Test 1 isolates attitude error recovery.
        omega0 = np.zeros(3)

        # Zero initial wheel momentum
        h_w0  = np.zeros(3)
        state0 = np.concatenate([q0, omega0, h_w0])

        t, q_h, om_h, hw_h, err_h, tau_h = rk4_controlled(
            q_target, state0, 0, tf, dt)

        ax_err.plot(t, err_h, alpha=0.7, lw=1.2,
                    label=f'Trial {trial+1} (init {np.degrees(angle):.1f}°)')
        ax_tau.plot(t, np.linalg.norm(tau_h, axis=1), alpha=0.7, lw=1.0)

        final_errors.append(err_h[-1])
        converged = "✓" if err_h[-1] < 1.0 else "✗"
        print(f"  Trial {trial+1:2d}: init error = {np.degrees(angle):5.1f}°  →"
              f"  final error = {err_h[-1]:.4f}°  {converged}")

    ax_err.set_ylabel("Attitude Error [deg]")
    ax_err.set_title("Test 1 — Attitude Error from Random ICs (±90°)")
    ax_err.legend(fontsize=7, ncol=2)
    ax_err.grid(True, alpha=0.3)

    ax_tau.set_ylabel("|τ| [N·m]")
    ax_tau.set_xlabel("Time [s]")
    ax_tau.set_title("Commanded Torque Magnitude")
    ax_tau.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("test1_random_ic.png", dpi=150)
    plt.show()
    print(f"\n  Mean final error: {np.mean(final_errors):.4f}°")
    print(f"  Max  final error: {np.max(final_errors):.4f}°\n")


# ════════════════════════════════════════════════════════════════════════════════
# TEST 2 — Several orbits with disturbances and sensor noise (MEKF in the loop)
# ════════════════════════════════════════════════════════════════════════════════
 
# Orbital / disturbance parameters
G_grav = 6.67430e-11
M_psyche = 2.287e19
mu_psyche = G_grav * M_psyche
R_psyche = 111e3
altitude = 303e3
r_orbit  = R_psyche + altitude
T_orbit  = 2 * np.pi * np.sqrt(r_orbit**3 / mu_psyche)  # orbital period [s]
 
def gravity_gradient_torque(q, r_hat):

    coeff   = 3 * mu_psyche / r_orbit**3
    r_hat_b = adcs.Q(q).T @ r_hat
    return coeff * np.cross(r_hat_b, J @ r_hat_b)
 
 
# ── SRP physical constants ─────────────────────────────────────────────────────
P0          = 1367.0          # Solar flux at Earth [W/m^2]
c_light     = 2.998e8         # Speed of light [m/s]
AU          = 1.496e11        # 1 AU in metres
R_sun_AU    = 2.92            # Psyche semi-major axis [AU]
P_srp       = (P0 / c_light) * (1.0 / R_sun_AU)**2   # SRP at Psyche [N/m^2]
 
# Material coefficients for unpainted aluminium alloy 6061-T6
rho_a = 0.7    # absorption
rho_s = 0.2    # specular reflection
rho_d = 0.1    # diffuse reflection  (rho_a + rho_s + rho_d = 1)
 
# Simplified spacecraft geometry: model as a single flat plate facing the Sun.
A_plate = 4.0             # effective sunlit area [m^2]
r_cp    = np.array([0.1, 0.05, 0.0])   # centre-of-pressure offset from CoM [m]
 
 
def solar_radiation_torque(q, sun_vec_inertial):

    # Sun vector in body frame
    r_sun_b = adcs.Q(q).T @ sun_vec_inertial   # body-frame sun unit vector
    n_hat   = np.array([1.0, 0.0, 0.0])        # plate normal (body frame, +x face)
 
    cos_theta = np.dot(n_hat, r_sun_b)
    if cos_theta <= 0:
        # Plate is in shadow — no SRP on this face
        return np.zeros(3)
 
    F_srp = P_srp * A_plate * cos_theta * (
        (1 - rho_s) * r_sun_b
        + 2 * (rho_s * cos_theta + rho_d / 3.0) * n_hat
    )
    return np.cross(r_cp, F_srp)   # torque in body frame
 
 
def test_orbital_disturbances(n_orbits=3, dt=1.0):

    print("=" * 60)
    print(f"TEST 2: {n_orbits} orbits with disturbances + sensor noise")
    print("=" * 60)
 
    tf = n_orbits * T_orbit
    print(f"  Orbital period = {T_orbit:.1f} s  |  Total sim time = {tf:.0f} s")
 
    q_target = np.array([1.0, 0.0, 0.0, 0.0])
 
    # Start close to target with a small initial error
    axis  = adcs.unit_vec(np.random.randn(3))
    angle = np.radians(5.0)   # 5° initial error
    q0    = adcs.quat(axis, angle)
    omega0 = np.zeros(3)
    h_w0   = np.zeros(3)
 
    t_vals    = np.arange(0, tf, dt)
    N         = len(t_vals)
    q_true_h  = np.zeros((N, 4))
    q_est_h   = np.zeros((N, 4))
    err_true_h = np.zeros(N)    # error using true state
    err_est_h  = np.zeros(N)    # error using estimated state
    tau_hist  = np.zeros((N, 3))
    h_w_hist  = np.zeros((N, 3))
 
    # ── MEKF initialisation ──────────────────────────────────────────
    # State: x = [q (4), beta (3)]
    x_est = np.concatenate([q0, np.zeros(3)])   # initial estimate
    P_est = 0.1 * np.eye(6)                     # initial covariance
 
    # Noise covariances (match attitude_estimation.py)
    sigma_gyro   = np.radians(0.0035) / 60 / np.sqrt(dt)
    sigma_bias   = np.radians(0.0035) / 3600
    V_kf = np.block([
        [sigma_gyro**2 * np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)),          sigma_bias**2 * np.eye(3)]
    ])
 
    # Single SRU measurement noise
    sigma_sru = np.radians(0.01)   # 0.01° per axis
    W_kf = sigma_sru**2 * np.eye(3)
 
    # Fixed star-tracker reference vector (body frame, arbitrary)
    sru_ref_N = adcs.unit_vec(np.array([0.6, 0.5, 0.6]))   # inertial
 
    # Gyro scale/bias model
    M_gyro = np.diag(1 + 0.01 * np.random.randn(3))
    b_gyro = np.radians(0.0035) / 3600 * np.random.randn(3)
 
    # ── Nadir and Sun directions (inertial, fixed for circular orbit) ────
    r_hat_N      = np.array([1.0, 0.0, 0.0])          # nadir unit vector (inertial)
    sun_vec_iner = adcs.unit_vec(np.array([1.0, 0.2, 0.05]))  # Sun direction (inertial)
 
    state = np.concatenate([q0, omega0, h_w0])
 
    for i, t in enumerate(t_vals):
        q_true = adcs.unit_vec(state[0:4])
        omega  = state[4:7]
        h_w    = state[7:10]
 
        # ── Sensor measurements ──────────────────────────────────────
        # Gyro
        w_gyro  = sigma_gyro * np.random.randn(3)
        gyro_m  = M_gyro @ omega + b_gyro + w_gyro
 
        # SRU: noisy quaternion from noisy body-frame star vector
        noise_phi = sigma_sru * np.random.randn(3)
        delta_q   = adcs.qexp(noise_phi)
        q_sru     = adcs.qmult(q_true, delta_q)
 
        # ── MEKF predict ─────────────────────────────────────────────
        q_p    = x_est[:4]
        beta_p = x_est[4:7]
        q_pred = adcs.L(q_p) @ adcs.qexp(0.5 * dt * (gyro_m - beta_p))
        x_pred = np.concatenate([adcs.unit_vec(q_pred), beta_p])
 
        # Jacobian A
        qk1 = x_pred[:4]
        dphidphi  = adcs.G(qk1).T @ adcs.R(adcs.qexp(0.5*dt*(gyro_m - beta_p))) @ adcs.G(q_p)
        dphidbeta = -0.5 * dt * adcs.G(qk1).T @ adcs.G(q_p)
        A_kf = np.block([[dphidphi, dphidbeta], [np.zeros((3, 3)), np.eye(3)]])
        P_pred = A_kf @ P_est @ A_kf.T + V_kf
 
        # ── MEKF update (SRU) ────────────────────────────────────────
        z  = adcs.qlog(adcs.qmult(adcs.qinv(x_pred[:4]), q_sru))
        C_att = adcs.H.T @ adcs.R(q_sru) @ adcs.Tmat @ adcs.G(x_pred[:4])  # (3,3)
        C  = np.hstack([C_att, np.zeros((3, 3))])                            # (3,6)
        S  = C @ P_pred @ C.T + W_kf
        K  = P_pred @ C.T @ np.linalg.inv(S)
 
        delta    = K @ z
        phi_upd  = -delta[:3]
        beta_upd = x_pred[4:7] - delta[3:6]
        q_upd    = adcs.L(x_pred[:4]) @ np.concatenate([
            [np.sqrt(max(1 - phi_upd @ phi_upd, 0.0))], phi_upd])
        x_est = np.concatenate([adcs.unit_vec(q_upd), beta_upd])
        P_est = (np.eye(6) - K @ C) @ P_pred @ (np.eye(6) - K @ C).T + K @ W_kf @ K.T
 
        # ── Controller uses MEKF estimate ────────────────────────────
        q_est_use   = x_est[:4]
        # Bias-corrected angular rate estimate
        omega_est   = gyro_m - x_est[4:7]
 
        tau_cmd, phi_e = pd_control(q_est_use, omega_est, q_target, h_w)
 
        # ── Disturbance torques ──────────────────────────────────────
        # Gravity gradient: depends on current true attitude and nadir direction
        tau_gg   = gravity_gradient_torque(q_true, r_hat_N)
        # SRP: N-plate model at 2.92 AU, aluminium 6061-T6 coefficients
        tau_srp  = solar_radiation_torque(q_true, sun_vec_iner)
        # Note: Psyche has no significant magnetic field so magnetic torque is omitted
        tau_dist  = tau_gg + tau_srp
        tau_total = np.clip(tau_cmd + tau_dist, -tau_max, tau_max)
 
        # ── Log ──────────────────────────────────────────────────────
        q_true_h[i]   = q_true
        q_est_h[i]    = q_est_use
        # True error: computed from actual spacecraft quaternion
        phi_e_true    = adcs.qlog(adcs.qmult(adcs.qinv(q_target), q_true))
        err_true_h[i] = np.degrees(2 * np.linalg.norm(phi_e_true))
        # Estimated error: what the controller thinks the error is
        err_est_h[i]  = np.degrees(2 * np.linalg.norm(phi_e))
        tau_hist[i]   = tau_cmd
        h_w_hist[i]   = h_w
 
        # ── Integrate one step ───────────────────────────────────────
        def f(t_, s_):
            q_     = s_[0:4]
            omega_ = s_[4:7]
            h_w_   = s_[7:10]
            # h_w_dot = -tau_cmd  →  spacecraft feels +tau_cmd  ✓
            h_w_dot_  = -tau_cmd
            L_tot     = J @ omega_ + h_w_
            om_dot    = np.linalg.solve(J, -np.cross(omega_, L_tot) + tau_cmd + tau_dist)
            q_dot_    = 0.5 * adcs.G(q_) @ omega_
            return np.concatenate([q_dot_, om_dot, h_w_dot_])
 
        k1 = f(t, state)
        k2 = f(t + dt/2, state + dt*k1/2)
        k3 = f(t + dt/2, state + dt*k2/2)
        k4 = f(t + dt,   state + dt*k3)
        state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        state[0:4] = adcs.unit_vec(state[0:4])
        state[7:10] = np.clip(state[7:10], -h_max, h_max)
 
    # ── Metrics ──────────────────────────────────────────────────────
    # Discard the first 10% as transient
    i_ss = N // 10
    rms_err = np.sqrt(np.mean(err_true_h[i_ss:]**2))
    print(f"\n  RMS pointing error (steady-state): {rms_err:.4f}°")
    print(f"  Peak pointing error:               {err_true_h[i_ss:].max():.4f}°")
 
    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
 
    t_hrs = t_vals / 3600
 
    axes[0].plot(t_hrs, err_true_h, lw=0.8, label="True error", color="steelblue")
    axes[0].plot(t_hrs, err_est_h,  lw=0.8, label="Estimated error", color="tomato",
                 linestyle="--", alpha=0.8)
    axes[0].axhline(rms_err, color='k', linestyle=':', lw=1.2,
                    label=f"RMS = {rms_err:.4f}°")
    axes[0].set_ylabel("Attitude Error [deg]")
    axes[0].set_title(f"Test 2 — {n_orbits} Orbits with Disturbances + Noise")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
 
    axes[1].plot(t_hrs, np.linalg.norm(tau_hist, axis=1), lw=0.8, color="darkorange")
    axes[1].set_ylabel("|τ_cmd| [N·m]")
    axes[1].set_title("Commanded Torque Magnitude")
    axes[1].grid(True, alpha=0.3)
 
    axes[2].plot(t_hrs, h_w_hist[:, 0], lw=0.8, label="h_w1")
    axes[2].plot(t_hrs, h_w_hist[:, 1], lw=0.8, label="h_w2")
    axes[2].plot(t_hrs, h_w_hist[:, 2], lw=0.8, label="h_w3")
    axes[2].axhline( h_max, color='k', linestyle=':', lw=1.0, label="±h_max")
    axes[2].axhline(-h_max, color='k', linestyle=':', lw=1.0)
    axes[2].set_ylabel("Wheel Momentum [N·m·s]")
    axes[2].set_xlabel("Time [hours]")
    axes[2].set_title("Reaction Wheel Momentum")
    axes[2].legend(fontsize=8, ncol=4)
    axes[2].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("test2_orbital_disturbances.png", dpi=150)
    plt.show()
 
    return rms_err


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n── Controller parameters ──────────────────────────────")
    print(f"  wn = {wn} rad/s,  zeta = {zeta}")
    print(f"  Kp = {np.round(Kp, 6)}")
    print(f"  Kd = {np.round(Kd, 6)}")
    print(f"  tau_max = {tau_max} N·m,  h_max = {h_max} N·m·s\n")

    test_random_initial_conditions(n_trials=8, max_angle_deg=90.0, dt=0.5)
    rms = test_orbital_disturbances(n_orbits=3, dt=1.0)