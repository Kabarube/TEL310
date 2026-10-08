# =============================================================================
# Author:       Kai Ruben Hamlander Enerhaugen
# Email:        kai.r.h.enerhaugen@nmbu.no
# Created:      2026-10-08
# Description:  TEL310 Labwork 1 - Motion model functions
# =============================================================================

import numpy as np

def motion_model_velocity(x_t, x_prev, u_t, alpha, dt):
    """Algorithm for computing p(x_t | u_t, x_t-1) based on velocity information
    """
    x, y, theta = x_t
    xp, yp, thetap = x_prev
    v, omega = u_t
    a1, a2, a3, a4, a5, a6 = alpha  # Specific robot error parameters
    
    # Coefficient
    numerator = (
        (xp - x) * np.cos(thetap)
        + (yp - y) * np.sin(thetap)
    )

    denominator = (
        (yp - y) * np.cos(thetap)
        - (xp - x) * np.sin(thetap)
    )

    # Avoid singularity for straight-line motion
    if np.isclose(denominator, 0.0):
        return 0.0
    
    mu = 0.5 * numerator / denominator

    # Center of rotation
    x_star = 0.5 * (xp + x) + mu * (yp - y)
    y_star = 0.5 * (yp + y) + mu * (x - xp)

    # Radius
    r_star = np.sqrt(
        (x-x_star)**2 +
        (y-y_star)**2
    )

    dtheta = (
        np.arctan2(y - y_star, x - x_star)
        - np.arctan2(yp - y_star, xp - x_star)
    )
    
    # Velocities obtained from the hypothesis
    omega_hat = dtheta / dt
    v_hat = omega_hat * r_star

    gamma_hat = ((theta - thetap) / dt) - omega_hat
    
    # Probabilities
    p1 = prob_normal_distribution(
        v - v_hat,
        a1 * v**2 + a2 * omega**2
    )
    p2 = prob_normal_distribution(
        omega - omega_hat,
        a3 * v**2 + a4 * omega**2
    )
    p3 = prob_normal_distribution(
        gamma_hat,
        a5 * v**2 + a6 * omega**2
    )

    return p1 * p2 * p3

def sample_motion_model_velocity(u_t, x_prev, alpha, dt):
    x, y, theta = x_prev
    v, omega = u_t
    a1, a2, a3, a4, a5, a6 = alpha

    var_v = a1 * v**2 + a2 * omega**2
    var_omega = a3 * v**2 + a4 * omega**2
    var_gamma = a5 * v**2 + a6 * omega**2

    v_hat = v + sample_normal_distribution(var_v)
    omega_hat = omega + sample_normal_distribution(var_omega)
    gamma_hat = sample_normal_distribution(var_gamma)

    # Radius
    r_hat = v_hat / omega_hat
    theta_hat = theta + omega_hat * dt

    # Sample coordinates
    x_t = x - r_hat * (
        np.sin(theta) - np.sin(theta_hat)
    )
    y_t = y + r_hat * (
        np.cos(theta) - np.cos(theta_hat)
    )
    theta_t = theta_hat + gamma_hat * dt

    return (x_t, y_t, theta_t)

def prob_triangular_distribution(error, variance):
    if abs(error) > np.sqrt(6 * variance):
        return 0
    else:
        p = 1 / np.sqrt(6 * variance) - abs(error) / (6 * variance)
        return p

def prob_normal_distribution(error, variance):
    """Computes the probability of its argument a (error) under zero-centered distribution with variance b**2
    """
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * error**2 / variance)

def sample_normal_distribution(variance):
    b = np.sqrt(variance)
    return 0.5 * sum(
        np.random.uniform(-b, b)
        for _ in range(1, 13)
    )

def sample_triangular_distribution(variance):
    b = np.sqrt(variance)
    return 0.5 * np.sqrt(6) * (
        np.random.uniform(-b, b, size=2).sum()
    )

def motion_model_odometry(x_prev, x_t, x_bar_prev, x_bar_t, alpha):
    x_old, y_old, theta_old = x_prev                        # Previous pose
    x_new, y_new, theta_new = x_t                           # Current pose
    x_pred_old, y_pred_old, theta_pred_old = x_bar_prev         # Previous predicted pose
    x_pred_new, y_pred_new, theta_pred_new = x_bar_t    # New predicted pose 
    a1, a2, a3, a4, a5, a6 = alpha                          # Specific robot error parameters

    sigma_rot1 = np.atan2(y_pred_new - y_pred_old, x_pred_new - x_pred_old) - theta_pred_old

    sigma_trans = np.sqrt(
        (x_pred_new - x_pred_old)**2 +
        (y_pred_new - y_pred_old)**2
    )

    sigma_rot2 = theta_pred_new - theta_pred_old - sigma_rot1

    sigma_hat_rot1 = np.atan2(
        y_new - y_old,
        x_new - x_old
    ) - theta_old

    sigma_hat_trans = np.sqrt(
        (x_new - x_old)**2 +
        (y_new - y_old)**2
    )

    sigma_hat_rot2 = theta_new - theta_old - sigma_hat_rot1

    p1_error = sigma_rot1 - sigma_hat_rot1
    p2_error = sigma_trans - sigma_hat_trans
    p3_error = sigma_rot2 - sigma_hat_rot2

    p1_var = a1 * abs(sigma_rot1) + a2 * sigma_trans
    p2_var = a3 * abs(sigma_rot1) + a4 * (abs(sigma_rot1) + abs(sigma_rot2))
    p3_var = a5 * abs(sigma_rot2) + a6 * sigma_trans

    p1 = prob_normal_distribution(p1_error, p1_var)
    p2 = prob_normal_distribution(p2_error, p2_var)
    p3 = prob_normal_distribution(p3_error, p3_var)

    return p1 * p2 * p3

def sample_motion_model_odometry(u_t, x_prev, alpha):
    # Robot pose at t-1
    x, y, theta = x_prev

    # Odometry poses
    odom_prev, odom_curr = u_t
    x_bar_prev, y_bar_prev, theta_bar_prev = odom_prev
    x_bar_curr, y_bar_curr, theta_bar_curr = odom_curr

    # Noise parameters
    a1, a2, a3, a4, a5, a6 = alpha

    # Odometry components
    delta_rot1 = np.atan2(
        y_bar_curr - y_bar_prev,
        x_bar_curr - x_bar_prev
    ) - theta_bar_prev

    delta_trans = np.sqrt(
        (x_bar_curr - x_bar_prev)**2 +
        (y_bar_curr - y_bar_prev)**2
    )

    delta_rot2 = (
        theta_bar_curr
        - theta_bar_prev
        - delta_rot1
    )

    # Calculate variance
    sigma_hat_rot1_var = a1 * delta_rot1**2 + a2 * delta_trans**2
    sigma_hat_trans_var = a3 * delta_trans**2 + a4 * (delta_rot1**2 + delta_rot2**2)
    sigma_hat_rot2_var = a5 * delta_rot2**2 + a6 * delta_trans**2

    sigma_hat_rot1 = delta_rot1 - sample_normal_distribution(sigma_hat_rot1_var)
    sigma_hat_trans = delta_trans - sample_normal_distribution(sigma_hat_trans_var)
    sigma_hat_rot2 = delta_rot2 - sample_normal_distribution(sigma_hat_rot2_var)

    x_current = x + sigma_hat_trans * np.cos(theta + sigma_hat_rot1)
    y_current = y + sigma_hat_trans * np.sin(theta + sigma_hat_rot1)
    theta_current = theta + sigma_hat_rot1 + sigma_hat_rot2

    return (x_current, y_current, theta_current)