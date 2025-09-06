import time
import numpy as np
from scipy.optimize import differential_evolution
import functions
import multiprocessing


# --- 目标函数1：用于第一阶段单烟幕弹的粗略搜索 ---
def objective_single_bomb(params, uav_pos, missile_pos, target_pos):
    """
    单烟幕弹优化目标函数（已增加错误处理和Debug提示）。
    """
    try:
        # --- 检查传入参数的有效性 ---
        if np.isnan(params).any():
            print(f"[DEBUG][WARN] 优化器传入了nan值: {params}")
            return 1e12  # 返回巨大惩罚

        full_params = list(params) + [0.1]  # Angle, Speed, Release, Fuse, Precision

        # --- 主要计算逻辑 ---
        # calculate_obscuration_time 返回: 负时长, 区间, 最大距离, 时间数组
        total_time_neg, intervals, _, times = functions.calculate_obscuration_time(
            full_params, uav_pos, missile_pos
        )

        # 返回的total_time_neg是负数，所以有效遮蔽时<0
        if total_time_neg < 0:
            # 有遮蔽 → 主要目标
            return float(total_time_neg)

        # 没遮蔽 → 引导项
        min_dists_to_line = []
        if isinstance(times, np.ndarray) and len(times) > 0:
            for t in times[::max(1, len(times) // 20)]:  # 抽样降低计算量
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
        # 返回一个正值，确保它比任何有效的负遮蔽时间都“差”
        return guide_val * 0.01

    except Exception as e:
        # --- 错误处理与Debug提示 ---
        # 如果 try 块中的任何代码行出错，程序将跳转到这里
        print("\n" + "=" * 20 + " [DEBUG][CRASH] " + "=" * 20)
        print(f"函数 objective_single_bomb 发生意外错误。")
        print(f"  - 错误类型: {type(e).__name__}")
        print(f"  - 错误信息: {e}")
        print(f"  - 导致错误的参数 (params): {np.round(params, 4)}")
        print("=" * 55 + "\n")

        # 返回一个巨大的惩罚值，让优化器放弃这个解
        return 1e12


# --- 目标函数2：用于第二阶段三烟幕弹的精细搜索 ---
def objective_multi_bomb(params, uav_pos, missile_pos):
    """
    三烟幕弹优化的目标函数。
    """
    try:
        (angle, speed, t_d1, dt2, dt3, t_f1, t_f2, t_f3) = params
        t_d2 = t_d1 + dt2
        t_d3 = t_d2 + dt3
        precision = 0.1

        p1 = [angle, speed, t_d1, t_f1, precision]
        p2 = [angle, speed, t_d2, t_f2, precision]
        p3 = [angle, speed, t_d3, t_f3, precision]

        _, i1, _, _, = functions.calculate_obscuration_time(p1, uav_pos, missile_pos)
        _, i2, _, _, = functions.calculate_obscuration_time(p2, uav_pos, missile_pos)
        _, i3, _, _, = functions.calculate_obscuration_time(p3, uav_pos, missile_pos)

        all_intervals = []
        # 健壮性检查：只把非空的区间列表加入
        if i1 != (None, None): all_intervals.append(i1)
        if i2 != (None, None): all_intervals.append(i2)
        if i3 != (None, None): all_intervals.append(i3)

        merged = functions.merge_intervals(all_intervals)
        total_duration = sum(end - start for start, end in merged)
        return -float(total_duration)

    except Exception as e:
        print(f"[ERROR] objective_multi_bomb 出错: params={params}, err={e}")
        return 1e6  # 返回一个大惩罚值


# --- 主执行程序 ---
if __name__ == '__main__':
    # 确保这行代码在 main 块的最前面，以兼容多进程
    multiprocessing.set_start_method("spawn", force=True)

    UAV_TO_ANALYZE = 'FY2'
    MISSILES_TO_INTERCEPT = ['M1', 'M2', 'M3']

    uav_pos = functions.UAV_POSITIONS[UAV_TO_ANALYZE]

    print(f"开始为无人机 {UAV_TO_ANALYZE} 执行多阶段优化...")

    all_run_results = []

    for missile_name in MISSILES_TO_INTERCEPT:
        missile_pos = functions.MISSILE_POSITIONS[missile_name]

        print("\n" + "=" * 50)
        print(f"当前优化目标: {UAV_TO_ANALYZE} vs {missile_name}")

        # --- 阶段 1: 单烟幕弹粗略搜索，找到一个好的初始解区域 ---
        print("  --- 阶段 1: 单烟幕弹初始搜索 ---")
        bounds_single = [(0, 360), (70, 140), (1, 40), (1, 18)]  # 角度, 速度, 投放时间, 引信时间

        result_single = differential_evolution(
            func=objective_single_bomb,
            args=(uav_pos, missile_pos, functions.true_target),
            bounds=bounds_single,
            strategy='best1bin', maxiter=500, popsize=20, tol=0.01,
            mutation=(0.5, 1), recombination=0.7,
            disp=False, seed=42, workers=-1
        )

        initial_best_params = result_single.x
        print(f"  阶段 1 完成. 找到最优单弹策略参数。")

    #     # --- 阶段 2: 三烟幕弹精细搜索，使用阶段1的结果来收窄搜索范围 ---
    #     print("  --- 阶段 2: 三烟幕弹精细搜索 ---")
    #
    #     angle_center, speed_center, t_d_center, t_f_center = initial_best_params
    #
    #     # 定义一个原始的、较宽的边界，用于后续范围检查
    #     original_bounds = [
    #         (0, 360), (70, 140), (1, 40), (1, 10), (1, 10), (1, 8), (1, 8), (1, 8)
    #     ]
    #
    #     # 基于阶段1的结果创建更窄的搜索范围
    #     refined_bounds = [
    #         (angle_center - 20, angle_center + 20),
    #         (speed_center - 15, speed_center + 15),
    #         (t_d_center - 5, t_d_center + 5),
    #         (1, 5),  # 投放间隔2
    #         (1, 5),  # 投放间隔3
    #         (t_f_center - 2, t_f_center + 2),
    #         (t_f_center - 2, t_f_center + 2),
    #         (t_f_center - 2, t_f_center + 2),
    #     ]
    #
    #     # 检查并修正新范围，确保它们不超出原始的合法边界
    #     for i, (low, high) in enumerate(refined_bounds):
    #         orig_low, orig_high = original_bounds[i]
    #         refined_bounds[i] = (max(low, orig_low), min(high, orig_high))
    #
    #     result_multi = differential_evolution(
    #         func=objective_multi_bomb,
    #         args=(uav_pos, missile_pos),
    #         bounds=refined_bounds,
    #         strategy='best1bin', maxiter=500, popsize=20, tol=0.01,
    #         mutation=(0.5, 1), recombination=0.7,
    #         disp=False, seed=42, workers=-1
    #     )
    #
    #     max_duration = -result_multi.fun
    #     best_params_multi = result_multi.x
    #
    #     print(f"  阶段 2 完成. 对 {missile_name} 的最大遮蔽时长: {max_duration:.4f} s")
    #
    #     all_run_results.append({
    #         'missile': missile_name,
    #         'duration': max_duration,
    #         'params': best_params_multi
    #     })
    #
    # # --- 阶段 3: 从所有导弹的优化结果中，选出效果最好的一个 ---
    # if not all_run_results:
    #     print("所有优化任务均未找到有效解。")
    # else:
    #     best_overall_result = max(all_run_results, key=lambda x: x['duration'])
    #
    #     print("\n" + "=" * 50)
    #     print(f"最终结果: 无人机 {UAV_TO_ANALYZE} 对导弹 {best_overall_result['missile']} 的拦截效果最佳")
    #     print("=" * 50)
    #
    #     print(f"最大有效遮蔽时长: {best_overall_result['duration']:.4f} s")
    #
    #     (angle, speed, t_d1, dt2, dt3, t_f1, t_f2, t_f3) = best_overall_result['params']
    #     t_d2 = t_d1 + dt2
    #     t_d3 = t_d2 + dt3
    #
    #     print("\n最优策略详情:")
    #     print(f"  - 飞行方向: {angle:.2f} 度")
    #     print(f"  - 飞行速度: {speed:.2f} m/s")
    #     print("  - 烟幕弹投放时间:")
    #     print(f"    - 第1枚: {t_d1:.2f} s")
    #     print(f"    - 第2枚: {t_d2:.2f} s")
    #     print(f"    - 第3枚: {t_d3:.2f} s")
    #     print("  - 烟幕弹引信时间:")
    #     print(f"    - 第1枚: {t_f1:.2f} s")
    #     print(f"    - 第2枚: {t_f2:.2f} s")
    #     print(f"    - 第3枚: {t_f3:.2f} s")