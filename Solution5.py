import time
import numpy as np
from scipy.optimize import differential_evolution
import functions
import multiprocessing

# 定义不同无人机的搜索范围
bounds_single = {
    "FY1": [
        (170, 190),    # angle
        (70, 140),   # speed
        (1, 8),     # t_d1
        (1, 5)      # t_f1
    ],
    "FY2": [
        (186.6597, 328.5280),   # angle
        (70, 140),   # speed
        (0.01, 10),     # t_d1
        (0.01, 16.733)      # t_f1
    ],
    "FY3": [
        (38.6161, 75.2986),   # angle
        (70, 140),   # speed
        (0.01, 100),     # t_d1
        (0.01, 11.832)      # t_f1
    ],
    "FY4": [
        (190.3138, 319.5953),   # angle
        (70, 140),   # speed
        (0.01, 56.25),     # t_d1
        (0.01, 18.973)      # t_f1
    ],
    "FY5": [
        (52.0360, 171.2473),   # angle
        (70, 140),   # speed
        (0.01, 43.75),     # t_d1
        (0.01, 16.124)      # t_f1
    ]
}

bounds_multi = {
    "FY1": [
        (170, 190),    # angle
        (70, 140),   # speed
        (0.01, 13.75),     # t_d1
        (1, 13.75),     # dt2
        (1, 13.75),     # dt3
        (0.01, 18.973),      # t_f1
        (0.01, 18.973),      # t_f2
        (0.01, 18.973)       # t_f3
    ],
    "FY2": [
        (186.6597, 328.5280),   # angle
        (70, 140),   # speed
        (0.01, 50),     # t_d1
        (1, 10),      # dt2
        (1, 10),      # dt3
        (0.01, 16.733),      # t_f1
        (0.01, 16.733),      # t_f2
        (0.01, 16.733)       # t_f3
    ],
    "FY3": [
        (38.6161, 75.2986),   # angle
        (70, 140),   # speed
        (0.01, 100),     # t_d1
        (1, 10),      # dt2
        (1, 10),      # dt3
        (0.01, 11.832),      # t_f1
        (0.01, 11.832),      # t_f2
        (0.01, 11.832)       # t_f3
    ],
    "FY4": [
        (190.3138, 319.5953),   # angle
        (70, 140),   # speed
        (0.01, 56.25),     # t_d1
        (1, 10),      # dt2
        (1, 10),      # dt3
        (0.01, 18.973),      # t_f1
        (0.01, 18.973),      # t_f2
        (0.01, 18.973)       # t_f3
    ],
    "FY5": [
        (52.0360, 171.2473),   # angle
        (70, 140),   # speed
        (0.01, 43.75),     # t_d1
        (1, 10),      # dt2
        (1, 10),      # dt3
        (0.01, 16.124),      # t_f1
        (0.01, 16.124),      # t_f2
        (0.01, 16.124)       # t_f3
    ]
}




# --- 目标函数1：用于第一阶段单烟幕弹的粗略搜索 ---
def objective_single_bomb(params, uav_pos, missile_pos, target_pos):
    try:
        if np.isnan(params).any():
            print(f"[DEBUG][WARN] 优化器传入了nan值: {params}")
            return 1e12

        full_params = list(params) + [0.2]  # Angle, Speed, Release, Fuse, Precision

        total_time_neg, intervals, _, times = functions.calculate_obscuration_time(
            full_params, uav_pos, missile_pos
        )

        if total_time_neg > 0:
            return -float(total_time_neg)

        min_dists_to_line = []
        if isinstance(times, np.ndarray) and len(times) > 0:
            for t in times[::max(1, len(times) // 20)]:
                cloud_center = functions.calculate_cloud_center(
                    uav_pos, full_params[0], full_params[1],
                    full_params[2], full_params[3], t
                )
                missile_p = functions.calculate_missile_positon(missile_pos, t)
                d, _ = functions.point_to_segment_distance(
                    np.array(cloud_center),
                    np.array(missile_p),
                    np.array(target_pos)
                )
                min_dists_to_line.append(d)

        guide_val = min(min_dists_to_line) if min_dists_to_line else 1e3
        return guide_val * 0.01

    except Exception as e:
        print("\n" + "=" * 20 + " [DEBUG][CRASH] " + "=" * 20)
        print(f"函数 objective_single_bomb 发生意外错误。")
        print(f"  - 错误类型: {type(e).__name__}")
        print(f"  - 错误信息: {e}")
        print(f"  - 导致错误的参数 (params): {np.round(params, 4)}")
        print("=" * 55 + "\n")
        return 1e12


# --- 目标函数2：用于第二阶段三烟幕弹的精细搜索 ---
def objective_multi_bomb(params, uav_pos, missile_pos):
    try:
        (angle, speed, t_d1, dt2, dt3, t_f1, t_f2, t_f3) = params
        t_d2 = t_d1 + dt2
        t_d3 = t_d2 + dt3
        precision = 0.2

        p1 = [angle, speed, t_d1, t_f1, precision]
        p2 = [angle, speed, t_d2, t_f2, precision]
        p3 = [angle, speed, t_d3, t_f3, precision]

        _, i1, _, _ = functions.calculate_obscuration_time(p1, uav_pos, missile_pos)
        _, i2, _, _ = functions.calculate_obscuration_time(p2, uav_pos, missile_pos)
        _, i3, _, _ = functions.calculate_obscuration_time(p3, uav_pos, missile_pos)

        all_intervals = []
        if i1 != (None, None): all_intervals.append(i1)
        if i2 != (None, None): all_intervals.append(i2)
        if i3 != (None, None): all_intervals.append(i3)

        merged = functions.merge_intervals(all_intervals)
        total_duration = sum(end - start for start, end in merged)
        return -float(total_duration)

    except Exception as e:
        print(f"[ERROR] objective_multi_bomb 出错: params={params}, err={e}")
        return 1e6


# --- 主执行程序 ---
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    TARGET_UAV = 'FY2'  # 设置为特定无人机名称以仅优化该无人机, 或 None 以优化所有无人机
    TARGET_MISSILE = None  # 设置为特定导弹名称以仅优化该导弹, 或 None 以优化所有导弹
    UAV_LIST = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
    MISSILES_TO_INTERCEPT = ['M1', 'M2', 'M3']

    final_results = []

    for UAV_TO_ANALYZE in UAV_LIST:
        if TARGET_UAV is not None and UAV_TO_ANALYZE != TARGET_UAV:
            continue
        bound_single = bounds_single[UAV_TO_ANALYZE]
        bound_multi = bounds_multi[UAV_TO_ANALYZE]
        uav_pos = functions.UAV_POSITIONS[UAV_TO_ANALYZE]
        print(f"\n开始为无人机 {UAV_TO_ANALYZE} 执行多阶段优化...")

        all_run_results = []

        for missile_name in MISSILES_TO_INTERCEPT:
            if TARGET_MISSILE is not None and missile_name != TARGET_MISSILE:
                continue
            missile_pos = functions.MISSILE_POSITIONS[missile_name]
            print("\n" + "=" * 50)
            print(f"当前优化目标: {UAV_TO_ANALYZE} vs {missile_name}")

            # --- 阶段 1: 单烟幕弹粗略搜索 ---
            result_single = differential_evolution(
                func=objective_single_bomb,
                args=(uav_pos, missile_pos, functions.true_target),
                bounds=bound_single,
                strategy='best1bin', maxiter=1000, popsize=50, tol=0.001,
                mutation=(0.5, 1), recombination=0.7,
                disp=False, seed=42,workers=-1
            )
            initial_best_params = result_single.x
            print(f"  阶段 1 完成, 单弹最大遮蔽时长: {-result_single.fun:.3f} s")
            print(f"    初始参数: {np.round(initial_best_params, 4)}")
            params = initial_best_params
            print(functions.calculate_cloud_center(uav_pos, params[0], params[1], params[2], params[3], params[2]+params[3]))

    #         # --- 阶段 2: 三烟幕弹精细搜索 ---
    #         angle_center, speed_center, t_d_center, t_f_center = initial_best_params
    #
    #         original_bounds = bounds_multi[UAV_TO_ANALYZE]
    #
    #         refined_bounds = []
    #         for i, (low, high) in enumerate(original_bounds):
    #             if i == 0:  # angle
    #                 refined_bounds.append((max(angle_center - 20, low), min(angle_center + 20, high)))
    #             # elif i == 1:  # speed
    #             #     refined_bounds.append((max(speed_center - 15, low), min(speed_center + 15, high)))
    #             # elif i == 2:  # t_d1
    #             #     refined_bounds.append((max(t_d_center - 10, low), min(t_d_center + 10, high)))
    #             # elif i == 5:  # t_f1
    #             #     refined_bounds.append((max(t_f_center - 10, low), min(t_f_center + 10, high)))
    #             else:
    #                 # 其他参数保持原始 bound
    #                 refined_bounds.append((low, high))
    #
    #         for i, (low, high) in enumerate(refined_bounds):
    #             orig_low, orig_high = original_bounds[i]
    #             refined_bounds[i] = (max(low, orig_low), min(high, orig_high))
    #
    #         result_multi = differential_evolution(
    #             func=objective_multi_bomb,
    #             args=(uav_pos, missile_pos),
    #             bounds=refined_bounds,
    #             strategy='best1bin', maxiter=600, popsize=30, tol=0.01,
    #             mutation=(0.5, 1), recombination=0.7,
    #             disp=False, workers=-1
    #         )
    #
    #         max_duration = -result_multi.fun
    #         best_params_multi = result_multi.x
    #
    #         print(max_duration)
    #         print(best_params_multi)
    #         all_run_results.append({
    #             'missile': missile_name,
    #             'duration': max_duration,
    #             'params': best_params_multi
    #         })
    #
    #     # --- 阶段 3: 选出该无人机的最佳导弹 ---
    #     if all_run_results:
    #         best_overall_result = max(all_run_results, key=lambda x: x['duration'])
    #         (angle, speed, t_d1, dt2, dt3, t_f1, t_f2, t_f3) = best_overall_result['params']
    #         t_d2 = t_d1 + dt2
    #         t_d3 = t_d2 + dt3
    #
    #         flight_params = {
    #             'angle': angle,
    #             'speed': speed,
    #             't_d1': t_d1, 't_d2': t_d2, 't_d3': t_d3,
    #             't_f1': t_f1, 't_f2': t_f2, 't_f3': t_f3
    #         }
    #
    #         final_results.append({
    #             'uav': UAV_TO_ANALYZE,
    #             'missile': best_overall_result['missile'],
    #             'duration': best_overall_result['duration'],
    #             'flight_params': flight_params
    #         })
    #
    # # --- 打印所有 UAV–Missile 的最佳对应关系 ---
    # print("\n" + "#" * 60)
    # print("最终无人机–导弹对应关系:")
    # for res in final_results:
    #     print(f"  UAV {res['uav']}  ->  Missile {res['missile']}  "
    #           f"(遮蔽 {res['duration']:.3f} s)")
    #     print("  飞行参数:", res['flight_params'])
    # print("#" * 60)
