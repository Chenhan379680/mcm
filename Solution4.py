import time
import numpy as np
import functions
import random
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
        return random.uniform(1e12, 1e13)

    # 合并区间
    try:
        all_intervals = [mask_interval_1, mask_interval_2, mask_interval_3]
        merged_intervals = functions.merge_intervals(all_intervals)
        total_mask_time = sum(end - start for start, end in merged_intervals)
        return -total_mask_time
    except Exception as e:
        print(f"[CRASH] 区间合并失败, error={e}, params={params}")
        return random.uniform(1e12, 1e13)


if __name__ == '__main__':
    # 定义 12 个决策变量的边界
    bounds = [
        (179, 180), (110, 120), (1, 2), (3, 4),
        (296, 297), (130, 137), (6, 7), (4, 5),
        (73, 75), (105, 110), (28, 30), (0, 1)
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

    # 解包最优数据
    (angle_degrees_1, uav_speed_1, t_release_delay_1, t_free_fall_1,
     angle_degrees_2, uav_speed_2, t_release_delay_2, t_free_fall_2,
     angle_degrees_3, uav_speed_3, t_release_delay_3, t_free_fall_3) = best_params



    uav_data = [
        {'name': 'FY1', 'pos': uav_initial_pos_1, 'params': [angle_degrees_1, uav_speed_1, t_release_delay_1, t_free_fall_1, 0.2]},
        {'name': 'FY2', 'pos': uav_initial_pos_2, 'params': [angle_degrees_2, uav_speed_2, t_release_delay_2, t_free_fall_2, 0.2]},
        {'name': 'FY3', 'pos': uav_initial_pos_3, 'params': [angle_degrees_3, uav_speed_3, t_release_delay_3, t_free_fall_3, 0.2]}
    ]

    print("\n--- 每个无人机的参数 ---")

    all_final_intervals = []
    for data in uav_data:
        # 通过最优参数计算分别遮蔽时间和遮蔽区间
        time_neg, intervals, _, _ = functions.calculate_obscuration_time(data['params'], data['pos'],
                                                                         missile_initial_pos)

        # 计算投弹点,引爆点
        angle_radians = np.radians(data['params'][0])
        uav_speed = data['params'][1]
        t_release = data['params'][2]
        t_free_fall = data['params'][3]

        v_uav = np.array([uav_speed * np.cos(angle_radians), uav_speed * np.sin(angle_radians), 0])
        p_release = data['pos'] + v_uav * t_release
        p_detonation = p_release + v_uav * t_free_fall + np.array([0, 0, -0.5 * functions.g * t_free_fall ** 2])

        # 打印无人机信息
        print(f"\n[UAV {data['name']}]")
        print(f"  - 飞行方向: {data['params'][0]:.4f} degrees")
        print(f"  - 飞行速度:     {uav_speed:.4f} m/s")
        print(
            f"  - 投弹点:    ({p_release[0]:.2f}, {p_release[1]:.2f}, {p_release[2]:.2f}) at {t_release:.4f} s")
        print(
            f"  - 引爆点: ({p_detonation[0]:.2f}, {p_detonation[1]:.2f}, {p_detonation[2]:.2f}) at {t_release + t_free_fall:.4f} s")
        print(f"  - 单独的遮蔽时长: {time_neg:.4f} s")

    print("\n" + "=" * 30)
    print("最终遮蔽时间 (Problem 4)")
    print("=" * 30)
    print(f"最大遮蔽时间: {max_duration:.4f} s")