import time
import numpy as np
import functions
from scipy.optimize import differential_evolution

uav_initial_pos_1 = functions.UAV_POSITIONS['FY1']
uav_initial_pos_2 = functions.UAV_POSITIONS['FY2']
uav_initial_pos_3 = functions.UAV_POSITIONS['FY3']
missile_initial_pos = functions.MISSILE_POSITIONS['M1']


def safe_calculate(params, uav_pos, missile_pos, uav_name):
    """ 包装 calculate_obscuration_time，加入合法性检查 """
    try:
        result = functions.calculate_obscuration_time(params, uav_pos, missile_pos)
        mask_time, mask_interval, _, _ = result

        if not np.isfinite(mask_time):
            print(f"[ERROR] {uav_name} 遮蔽时间为非有限值, params={params}")
            return 0, [], None, None
        if mask_time <= 0:
            print(f"[WARN] {uav_name} 遮蔽时间 <= 0, params={params}")
            return 0, [], None, None

        return result

    except Exception as e:
        print(f"[CRASH] {uav_name} 计算失败, params={params}, error={e}")
        return 0, [], None, None


def objective_for_optimizer(params):
    # 解包参数
    (angle_degrees_1, uav_speed_1, t_release_delay_1, t_free_fall_1,
     angle_degrees_2, uav_speed_2, t_release_delay_2, t_free_fall_2,
     angle_degrees_3, uav_speed_3, t_release_delay_3, t_free_fall_3) = params

    # 设置参数
    params_1 = [angle_degrees_1, uav_speed_1, t_release_delay_1, t_free_fall_1, 0.2]
    params_2 = [angle_degrees_2, uav_speed_2, t_release_delay_2, t_free_fall_2, 0.2]
    params_3 = [angle_degrees_3, uav_speed_3, t_release_delay_3, t_free_fall_3, 0.2]

    # 分别计算遮蔽时长
    mask_time_1, mask_interval_1, _, _ = safe_calculate(params_1, uav_initial_pos_1, missile_initial_pos, "UAV1")
    mask_time_2, mask_interval_2, _, _ = safe_calculate(params_2, uav_initial_pos_2, missile_initial_pos, "UAV2")
    mask_time_3, mask_interval_3, _, _ = safe_calculate(params_3, uav_initial_pos_3, missile_initial_pos, "UAV3")

    if (mask_time_1 <= 0 or mask_time_2 <= 0 or mask_time_3 <= 0):
        return np.inf

    # 合并区间
    try:
        all_intervals = [mask_interval_1, mask_interval_2, mask_interval_3]
        merged_intervals = functions.merge_intervals(all_intervals)
        total_mask_time = sum(end - start for start, end in merged_intervals)
        return -total_mask_time
    except Exception as e:
        print(f"[CRASH] 区间合并失败, error={e}, params={params}")
        return np.inf


if __name__ == '__main__':
    # 定义 12 个决策变量的边界
    bounds = [
        (170, 190), (70, 140), (0, 8), (0, 5),
        (260, 330), (70, 140), (0, 8), (0, 5),
        (40, 100), (70, 140), (10, 40), (0, 5)
    ]

    print("开始运行差分进化算法求解 问题4...")
    start_time = time.time()

    result = differential_evolution(
        func=objective_for_optimizer,
        bounds=bounds,
        strategy='best1bin',
        maxiter=500,
        popsize=20,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False,
        workers=-1  # 使用所有CPU核心并行计算
    )

    end_time = time.time()
    print(f"\n优化完成，耗时: {end_time - start_time:.2f} 秒")
    best_params = result.x
    max_duration = -result.fun

    print("\n" + "=" * 30)
    print("Final Refined Optimal Strategy (Problem 4)")
    print("=" * 30)
    print(f"Maximum Total Effective Obscuration Time: {max_duration:.4f} s")
