# Kalman Filter for Object Tracking in NED Frame

This document describes the Kalman Filter (KF) used to filter noisy object position measurements and estimate object velocities, given a UAV with perfectly known states.

---

## 1. Problem Statement

| Component | Status |
|-----------|--------|
| UAV position, velocity, heading | **Known (100% reliable)** |
| Object positions | Measured with unknown noise |
| Object velocities | **Unknown (to be estimated)** |
| Object motion | Stationary or dynamic |

**Objective**: Apply KF to reduce noisy positional geolocalisation and estimate precise object velocities.

---

## 2. Coordinate Frame: NED

| Axis | Direction |
|------|-----------|
| **N** | North |
| **E** | East |
| **D** | Down |

All object states are expressed in the NED frame relative to a fixed origin.

---

## 3. State Vector

For each tracked object:

$$
\mathbf{x} = \begin{bmatrix} p_N \\ p_E \\ p_D \\ v_N \\ v_E \\ v_D \end{bmatrix}
$$

| State | Description | Unit |
|-------|-------------|------|
| $p_N, p_E, p_D$ | Object position in NED | m |
| $v_N, v_E, v_D$ | Object velocity in NED | m/s |

**Dimension**: 6 states per object

---

## 4. Prediction Step

### 4.1 Motion Model

Constant velocity model (suitable for both stationary and dynamic objects):

$$
\mathbf{x}_{k+1} = \mathbf{F} \mathbf{x}_k + \mathbf{w}_k
$$

Where $\mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q})$ is process noise.

### 4.2 State Transition Matrix

$$
\mathbf{F} = \begin{bmatrix}
1 & 0 & 0 & \Delta t & 0 & 0 \\
0 & 1 & 0 & 0 & \Delta t & 0 \\
0 & 0 & 1 & 0 & 0 & \Delta t \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

### 4.3 Process Noise Covariance (Requires Tuning)

The process noise $\mathbf{Q}$ accounts for unmodeled accelerations. Since the true measurement covariance is unknown, $\mathbf{Q}$ must be **manually optimised**.

**Discrete white noise acceleration model**:

$$
\mathbf{Q} = \sigma_a^2 \begin{bmatrix}
\frac{\Delta t^4}{4} & 0 & 0 & \frac{\Delta t^3}{2} & 0 & 0 \\
0 & \frac{\Delta t^4}{4} & 0 & 0 & \frac{\Delta t^3}{2} & 0 \\
0 & 0 & \frac{\Delta t^4}{4} & 0 & 0 & \frac{\Delta t^3}{2} \\
\frac{\Delta t^3}{2} & 0 & 0 & \Delta t^2 & 0 & 0 \\
0 & \frac{\Delta t^3}{2} & 0 & 0 & \Delta t^2 & 0 \\
0 & 0 & \frac{\Delta t^3}{2} & 0 & 0 & \Delta t^2
\end{bmatrix}
$$

Where $\sigma_a$ is the assumed acceleration standard deviation (tuning parameter).

**Simplified diagonal form** (alternative):

$$
\mathbf{Q} = \text{diag}\left(q_p, q_p, q_p, q_v, q_v, q_v\right)
$$

Where:
- $q_p$ = position process noise variance
- $q_v$ = velocity process noise variance

### 4.4 Prediction Equations

**State prediction**:
$$
\hat{\mathbf{x}}_{k|k-1} = \mathbf{F} \hat{\mathbf{x}}_{k-1|k-1}
$$

**Covariance prediction**:
$$
\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}
$$

---

## 5. Update Step

### 5.1 Measurement Model (Cartesian)

The UAV measures object position in NED frame:

$$
\mathbf{z}_k = \mathbf{H} \mathbf{x}_k + \mathbf{v}_k
$$

Where:
- $\mathbf{z}_k = \begin{bmatrix} z_N \\ z_E \\ z_D \end{bmatrix}$ = measured position
- $\mathbf{v}_k \sim \mathcal{N}(0, \mathbf{R})$ = measurement noise

### 5.2 Observation Matrix

$$
\mathbf{H} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0
\end{bmatrix}
$$

This extracts position states from the full state vector.

### 5.3 Measurement Noise Covariance

$$
\mathbf{R} = \begin{bmatrix}
\sigma_N^2 & 0 & 0 \\
0 & \sigma_E^2 & 0 \\
0 & 0 & \sigma_D^2
\end{bmatrix}
$$

Since measurement noise is unknown, $\mathbf{R}$ can be:
- Estimated from sensor specifications
- Tuned alongside $\mathbf{Q}$ for optimal performance

### 5.4 Update Equations

**Innovation (measurement residual)**:
$$
\boldsymbol{\nu}_k = \mathbf{z}_k - \mathbf{H} \hat{\mathbf{x}}_{k|k-1}
$$

**Innovation covariance**:
$$
\mathbf{S}_k = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R}
$$

**Kalman gain**:
$$
\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T \mathbf{S}_k^{-1}
$$

**State update**:
$$
\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \boldsymbol{\nu}_k
$$

**Covariance update**:
$$
\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1}
$$

---

## 6. Algorithm Summary

```
Kalman Filter for Object Tracking
─────────────────────────────────────────────────────

Initialisation:
  x̂ = [z_N, z_E, z_D, 0, 0, 0]ᵀ   (first measurement, zero velocity)
  P = diag(σ²_init)               (initial uncertainty)

For each timestep k:

  1. PREDICT
     x̂ₖ|ₖ₋₁ = F · x̂ₖ₋₁|ₖ₋₁
     Pₖ|ₖ₋₁ = F · Pₖ₋₁|ₖ₋₁ · Fᵀ + Q

  2. UPDATE (when measurement available)
     ν = zₖ - H · x̂ₖ|ₖ₋₁           (innovation)
     S = H · Pₖ|ₖ₋₁ · Hᵀ + R       (innovation covariance)
     K = Pₖ|ₖ₋₁ · Hᵀ · S⁻¹         (Kalman gain)
     x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + K · ν        (state update)
     Pₖ|ₖ = (I - K · H) · Pₖ|ₖ₋₁   (covariance update)

  3. OUTPUT
     Filtered position: [x̂₁, x̂₂, x̂₃]
     Estimated velocity: [x̂₄, x̂₅, x̂₆]
```

---

## 7. Tuning Guidelines

Since the measurement covariance is unknown, tune $\mathbf{Q}$ and $\mathbf{R}$ to balance:

| Condition | Effect |
|-----------|--------|
| $\mathbf{Q}$ too small | Filter trusts predictions too much → slow response to changes |
| $\mathbf{Q}$ too large | Filter trusts predictions too little → noisy estimates |
| $\mathbf{R}$ too small | Filter trusts measurements too much → noisy output |
| $\mathbf{R}$ too large | Filter trusts measurements too little → sluggish tracking |

### Tuning Approaches

1. **Grid search**: Test combinations of $\sigma_a$ and $\mathbf{R}$ values
2. **Innovation-based**: Adjust until innovation sequence is white with covariance $\approx \mathbf{S}$
3. **RMSE minimisation**: If ground truth available, minimise position/velocity RMSE

### Typical Starting Values

| Parameter | Stationary Objects | Dynamic Objects |
|-----------|-------------------|-----------------|
| $\sigma_a$ | 0.01 - 0.1 m/s² | 0.5 - 2.0 m/s² |
| $\sigma_N, \sigma_E$ | 1 - 5 m | 1 - 5 m |
| $\sigma_D$ | 1 - 10 m | 1 - 10 m |

---

## 8. Key Outputs

| Output | Source | Description |
|--------|--------|-------------|
| Filtered position | $\hat{p}_N, \hat{p}_E, \hat{p}_D$ | Smoothed object position |
| Estimated velocity | $\hat{v}_N, \hat{v}_E, \hat{v}_D$ | Derived from position changes |
| State covariance | $\mathbf{P}$ | Uncertainty of estimates |

---

## 9. Implementation Notes

1. **Initialisation**: Use first measurement as initial position, set initial velocity to zero
2. **Multiple objects**: Maintain separate KF instance per tracked object
3. **Missing measurements**: Run prediction step only (no update)
4. **Object classification**: Low velocity magnitude ($|\mathbf{v}| < \epsilon$) indicates stationary object
