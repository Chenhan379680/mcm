"""
Problem 5 — heuristic optimization (differential evolution) for smoke/cloud deployment
Output: printed best-found strategy and two plots (detonation XY, coverage timeline).
No Excel / file I/O.

Assumptions (tunable in PARAMETERS):
- free_fall_time: time between release and detonation (s)
- cloud_effective_duration: seconds after detonation the cloud can provide coverage
- cloud_effective_radius: horizontal radius (m) from cloud center that covers the true target
- UAV motion: straight-line constant speed at chosen heading and constant altitude
- cloud horizontal center at detonation equals UAV horizontal position after free-fall horizontal travel
- max drops per UAV: configurable (default 3)
- objective: maximize union length (total seconds) that true target is covered by any cloud;
             penalty for same-UAV drop intervals < min_separation_s
"""

import math
import time
import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS (edit here)
# -------------------------
# initial positions (example from provided PDF/previous message)
missiles = {
    'M1': np.array([20000.0,   0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0,-600.0, 1900.0]),
}

uavs_init = {
    'FY1': np.array([17800.0,   0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0,-2000.0, 1300.0]),
}

true_target_center = np.array([0.0, 200.0, 0.0])  # given
missile_speed = 300.0  # m/s (for Tmax heuristic; tune to real value if known)

# cloud & drop parameters (change these to reflect problem statement)
free_fall_time = 3.0              # seconds from release to detonation (assumed)
cloud_effective_duration = 20.0   # seconds cloud provides coverage after detonation
cloud_effective_radius = 10.0     # meters horizontal radius to cover true target

# optimization settings
n_drops_per_uav = 3               # permit up to 3 drops per UAV (tunable)
min_separation_s = 1.0            # minimum separation between drops from same UAV
uav_speed_bounds = (70.0, 140.0)  # m/s bounds
maxiter = 200                     # DE iterations (increase for better results)
popsize = 20                      # DE population size (increase for better results)
seed = 42

# plotting toggles
PLOT = True

# -------------------------
# Derived constants
# -------------------------
uav_names = list(uavs_init.keys())
n_uav = len(uav_names)
n_vars = 2*n_uav + n_uav * n_drops_per_uav

# compute Tmax (upper bound for release times): use the farthest missile distance / speed + margin
distances_to_true = [np.linalg.norm(pos - true_target_center) for pos in missiles.values()]
Tmax = max(distances_to_true) / (missile_speed if missile_speed>0 else 300.0) + 120.0  # margin 120s

# -------------------------
# Helper functions
# -------------------------
def uav_position(uav_idx, heading_rad, speed, t):
    """Return UAV xyz at time t (constant altitude)."""
    name = uav_names[uav_idx]
    x0, y0, z0 = uavs_init[name]
    dx = math.cos(heading_rad)
    dy = math.sin(heading_rad)
    return np.array([x0 + dx*speed*t, y0 + dy*speed*t, z0])

def detonation_info(uav_idx, heading_rad, speed, release_time):
    """Return detonation time and xyz given UAV motion parameters and release time.
       Uses free_fall_time and assumes horizontal displacement during free-fall from UAV horizontal speed."""
    t_det = release_time + free_fall_time
    pos_release = uav_position(uav_idx, heading_rad, speed, release_time)
    # horizontal displacement during free-fall (approx: UAV horizontal speed * free_fall_time)
    horiz_disp = np.array([math.cos(heading_rad)*speed*free_fall_time,
                           math.sin(heading_rad)*speed*free_fall_time,
                           -0.5*9.81*free_fall_time**2])
    det_pos = pos_release + horiz_disp
    # clamp altitude to ground >= 0
    if det_pos[2] < 0:
        det_pos[2] = 0.0
    return t_det, det_pos

def union_length(intervals):
    if not intervals:
        return 0.0, []
    iv = sorted(intervals, key=lambda x: x[0])
    merged = []
    cur_s, cur_e = iv[0]
    for s, e in iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    total = sum(e - s for s, e in merged)
    return total, merged


def coverage_intervals_from_vector(x):
    """Given decision vector x, return list of coverage intervals (for true target), and penalty."""
    intervals = []
    penalty = 0.0
    # extract headings & speeds
    headings = []
    speeds = []
    for i in range(n_uav):
        headings.append(float(x[2*i]))
        speeds.append(float(x[2*i+1]))
    # release times
    base = 2*n_uav
    for i in range(n_uav):
        # collect times for this UAV
        times = []
        for k in range(n_drops_per_uav):
            times.append(float(x[base + i*n_drops_per_uav + k]))
        # penalize times outside [0, Tmax]
        for t in times:
            if t < 0.0:
                penalty += (0.0 - t) * 1000.0
            if t > Tmax:
                penalty += (t - Tmax) * 1000.0
        # penalize spacing < min_separation_s
        times_sorted = sorted(times)
        for a,b in zip(times_sorted, times_sorted[1:]):
            if b - a < min_separation_s:
                penalty += (min_separation_s - (b-a)) * 500.0
        # compute detonation info for each drop
        for t_rel in times:
            if t_rel < 0.0 or t_rel > Tmax:
                continue
            t_det, det_pos = detonation_info(i, headings[i], speeds[i], t_rel)
            horiz_dist = np.linalg.norm(det_pos[:2] - true_target_center[:2])
            if horiz_dist <= cloud_effective_radius:
                intervals.append((t_det, t_det + cloud_effective_duration))
    return intervals, penalty

def objective(x):
    """Objective to MINIMIZE: negative covered_time + penalty -> so optimizer maximizes covered_time."""
    # penalize bad speeds as guard
    for i in range(n_uav):
        s = x[2*i+1]
        if s < uav_speed_bounds[0] - 1e-6 or s > uav_speed_bounds[1] + 1e-6:
            return 1e6 + abs(s - 100.0)
    intervals, penalty = coverage_intervals_from_vector(x)
    covered, _ = union_length(intervals)
    return -covered + penalty

# -------------------------
# bounds and initial guess
# -------------------------
bounds = []
for i in range(n_uav):
    bounds.append((0.0, 2*math.pi))          # heading
    bounds.append(uav_speed_bounds)          # speed
for i in range(n_uav):
    for k in range(n_drops_per_uav):
        bounds.append((0.0, Tmax))           # release time bounds

# initial guess: point toward true target and mid speed, staggers releases
x0 = []
for i,name in enumerate(uav_names):
    vec = uavs_init[name]
    dx = true_target_center[0] - vec[0]
    dy = true_target_center[1] - vec[1]
    heading = math.atan2(dy, dx)
    if heading < 0: heading += 2*math.pi
    x0.append(heading)
    x0.append(sum(uav_speed_bounds)/2.0)
for i in range(n_uav):
    # stagger releases in a heuristic window
    for k in range(n_drops_per_uav):
        x0.append(10.0 + i*6.0 + k*3.5)

# -------------------------
# Run optimization (differential evolution)
# -------------------------
print("Starting optimization with DE: maxiter={}, popsize={}, n_vars={}".format(maxiter, popsize, n_vars))
start_time = time.time()
result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, tol=1e-3, polish=True, seed=seed)
elapsed = time.time() - start_time
print("Optimization finished in {:.2f}s; success={} msg='{}' fun={:.6f}".format(elapsed, result.success, result.message, result.fun))

best_x = result.x
# compute final coverage intervals and merged intervals
intervals, penalty = coverage_intervals_from_vector(best_x)
covered_time, merged_intervals = union_length(intervals)

# -------------------------
# Prepare readable output (pandas DataFrame printed)
# -------------------------
rows = []
base = 2*n_uav
for i,name in enumerate(uav_names):
    heading = float(best_x[2*i])
    speed = float(best_x[2*i+1])
    for k in range(n_drops_per_uav):
        t_rel = float(best_x[base + i*n_drops_per_uav + k])
        t_det, det_pos = detonation_info(i, heading, speed, t_rel)
        horiz_dist = float(np.linalg.norm(det_pos[:2] - true_target_center[:2]))
        cover = horiz_dist <= cloud_effective_radius
        rows.append({
            'UAV': name,
            'drop_id': k+1,
            'heading_deg': heading*180.0/math.pi,
            'speed_m_s': speed,
            'release_time_s': t_rel,
            'detonation_time_s': t_det,
            'detonation_x': det_pos[0],
            'detonation_y': det_pos[1],
            'detonation_z': det_pos[2],
            'horiz_dist_to_true_m': horiz_dist,
            'covers_true_target': bool(cover)
        })

df = pd.DataFrame(rows)
pd.set_option('display.float_format', '{:.3f}'.format)
print("\nBest strategy (per drop):")
print(df)

print("\nSummary:")
print(f"Total covered time (union) = {covered_time:.3f} s")
print(f"Penalty = {penalty:.3f}")
print(f"Objective (fun) = {result.fun:.6f}")
print(f"Optimization elapsed = {elapsed:.2f} s")
print(f"Merged coverage intervals: {merged_intervals}")

# -------------------------
# Optional: more realistic objective variant (commented)
# You can swap objective() with a weighted objective that prioritizes coverage around missile impact windows.
# Example (not active): compute missile arrival times to true target and weight coverage within ±T_window more.
# -------------------------

# -------------------------
# Visualization (if PLOT True)
# -------------------------
if PLOT:
    # scatter detonation XY positions, color-coded by covers_true_target
    fig, ax = plt.subplots(figsize=(8,6))
    covers = df['covers_true_target'].astype(bool)
    ax.scatter(df['detonation_x'], df['detonation_y'], s=40, marker='o', label='detonation points')
    for idx, row in df.iterrows():
        txt = f"{row['UAV']}-{int(row['drop_id'])}"
        ax.annotate(txt, (row['detonation_x'], row['detonation_y']), textcoords="offset points", xytext=(3,3), fontsize=8)
    # true target marker and effective radius
    ax.scatter([true_target_center[0]], [true_target_center[1]], c='red', marker='*', s=120, label='True target')
    circle = plt.Circle((true_target_center[0], true_target_center[1]), cloud_effective_radius, color='red', fill=False, linestyle='--', label=f'coverage radius {cloud_effective_radius} m')
    ax.add_patch(circle)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Detonation positions (XY plane)')
    ax.legend()
    ax.axis('equal')
    plt.grid(True)
    plt.show()

    # coverage timeline plot: show each cloud's interval and merged union
    fig2, ax2 = plt.subplots(figsize=(10,4))
    y = 0
    for idx,(s,e) in enumerate(intervals):
        ax2.hlines(y, s, e, linewidth=6, label='cloud intervals' if idx==0 else "")
        y += 1
    # plot merged union (bold)
    for s,e in merged_intervals:
        ax2.hlines(-1, s, e, linewidth=10, color='red', label='merged coverage' if s==merged_intervals[0][0] else "")
    ax2.set_xlabel('Time (s)')
    ax2.set_yticks([])
    ax2.set_title('Cloud coverage intervals and merged coverage (red)')
    ax2.legend()
    plt.show()

# -------------------------
# End of script
# -------------------------
