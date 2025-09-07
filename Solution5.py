import time
import numpy as np
# from scipy.optimize import differential_evolution # 不再需要DE
from skopt import gp_minimize  # 导入贝叶斯优化器
from skopt.space import Real  # 导入定义搜索空间的工具
import functions  # 确保您的 functions.py 文件在同一个目录下


# --- 1. 修改目标函数 ---
# 移除了针对DE的 random.uniform() 惩罚。
# 当没有遮蔽时，直接返回0。贝叶斯优化会把这里当作一个平坦区域来学习。
def objective_for_optimizer(params, uav_pos, missile_pos):
    # 解包决策变量
    angle_degrees, uav_speed, t_release_delay_1, t_release_delay_2, t_release_delay_3, t_free_fall_1, t_free_fall_2, t_free_fall_3 = params

    # 时间处理
    t_release_1 = t_release_delay_1
    t_release_2 = t_release_1 + t_release_delay_2
    t_release_3 = t_release_2 + t_release_delay_3

    # 设置参数
    params_1 = [angle_degrees, uav_speed, t_release_1, t_free_fall_1, 0.2]
    params_2 = [angle_degrees, uav_speed, t_release_2, t_free_fall_2, 0.2]
    params_3 = [angle_degrees, uav_speed, t_release_3, t_free_fall_3, 0.2]

    # 得到遮蔽区间
    res_1 = functions.calculate_obscuration_time(params_1, uav_pos, missile_pos)
    res_2 = functions.calculate_obscuration_time(params_2, uav_pos, missile_pos)
    res_3 = functions.calculate_obscuration_time(params_3, uav_pos, missile_pos)

    # 处理遮蔽区间
    all_intervals = [res_1[1], res_2[1], res_3[1]]
    merged_intervals = functions.merge_intervals(all_intervals)
    total_mask_time = sum(end - start for start, end in merged_intervals if start is not None)

    # 返回目标函数值
    if total_mask_time > 0.0:
        # 我们的目标是最大化遮蔽时间, 而gp_minimize是最小化工具, 所以返回负值
        return -total_mask_time
    else:
        # 对于无效解（遮蔽时间为0），直接返回0即可
        return 0.0


# --- 2. 修改主函数以使用贝叶斯优化 ---
def find_optimal_strategy(uav_key, missile_key, bounds, maxiter=300, popsize=40, seed=None):
    print("\n" + "#" * 50)
    print(f"开始为 无人机 '{uav_key}' 对抗 导弹 '{missile_key}' 进行策略优化")
    print("#" * 50)

    try:
        uav_initial_pos = functions.UAV_POSITIONS[uav_key]
        missile_initial_pos = functions.MISSILE_POSITIONS[missile_key]  # 确保这里的key是正确的
    except KeyError as e:
        print(f"错误: 无法找到指定的key: {e}。请检查 functions.py 中的定义。")
        return

    # --- 3. 将 bounds 格式转换为 scikit-optimize 所需的格式 ---
    # 这样您就不需要修改 if __name__ == '__main__': 中的 bounds 定义了
    skopt_dimensions = [Real(low, high, name=f'p{i}') for i, (low, high) in enumerate(bounds)]

    print(f"开始运行贝叶斯优化 (总评估次数={maxiter}, 初始探索点数={popsize})...")
    start_time = time.time()

    # --- 4. 将 differential_evolution 替换为 gp_minimize ---
    # 我们使用一个lambda函数来包装目标函数，以便传入固定的 uav_pos 和 missile_pos
    result = gp_minimize(
        func=lambda params: objective_for_optimizer(params, uav_initial_pos, missile_initial_pos),
        dimensions=skopt_dimensions,
        x0=[plausible_solution],
        n_calls=maxiter,  # 总评估次数
        n_initial_points=popsize,  # 初始随机探索的点数
        acq_func='LCB',
        kappa=0.1,
        random_state=seed,  # 随机种子
        verbose=True  # 设置为True可以在运行时看到每一步的进展
    )

    end_time = time.time()
    print(f"\n优化完成，耗时: {end_time - start_time:.2f} 秒")

    # --- 5. 输出结果部分几乎不需要修改 ---
    # gp_minimize返回的结果对象与DE的非常相似 (result.x, result.fun)
    best_params = result.x

    # 检查result.fun是否是一个有效的数值
    if result.fun is not None and np.isfinite(result.fun):
        max_duration = -result.fun
        print(f"最大总有效遮蔽时长: {max_duration:.4f} 秒")
    else:
        print("优化未能找到有效的非零解。")
        max_duration = 0

    # 解析最优参数
    angle_degrees, uav_speed, t_release_delay_1, t_release_delay_2, t_release_delay_3, t_free_fall_1, t_free_fall_2, t_free_fall_3 = best_params

    # ... (后续所有的结果分析和打印代码与您之前的版本完全相同，无需改动) ...
    # ... (为了简洁，这里省略了和之前版本完全一样的结果输出代码) ...
    t_release_1 = t_release_delay_1
    t_release_2 = t_release_1 + t_release_delay_2
    t_release_3 = t_release_2 + t_release_delay_3
    print("\n--- 无人机飞行策略 ---")
    print(f"  - 飞行方向角: {angle_degrees:.4f} 度")
    print(f"  - 飞行速度:   {uav_speed:.4f} m/s")
    params_1 = [angle_degrees, uav_speed, t_release_1, t_free_fall_1, 0.2]
    params_2 = [angle_degrees, uav_speed, t_release_2, t_free_fall_2, 0.2]
    params_3 = [angle_degrees, uav_speed, t_release_3, t_free_fall_3, 0.2]
    res_1 = functions.calculate_obscuration_time(params_1, uav_initial_pos, missile_initial_pos)
    res_2 = functions.calculate_obscuration_time(params_2, uav_initial_pos, missile_initial_pos)
    res_3 = functions.calculate_obscuration_time(params_3, uav_initial_pos, missile_initial_pos)
    interval_1 = res_1[1]
    interval_2 = res_2[1]
    interval_3 = res_3[1]
    duration_1 = interval_1[1] - interval_1[0] if interval_1 and interval_1[0] is not None else 0
    duration_2 = interval_2[1] - interval_2[0] if interval_2 and interval_2[0] is not None else 0
    duration_3 = interval_3[1] - interval_3[0] if interval_3 and interval_3[0] is not None else 0
    print("\n--- 单独遮蔽效果分析 ---")
    if duration_1 > 0:
        print(f"  - 烟雾弹1单独遮蔽时长: {duration_1:.4f} s, 时间区间: [{interval_1[0]:.2f}, {interval_1[1]:.2f}]")
    if duration_2 > 0:
        print(f"  - 烟雾弹2单独遮蔽时长: {duration_2:.4f} s, 时间区间: [{interval_2[0]:.2f}, {interval_2[1]:.2f}]")
    if duration_3 > 0:
        print(f"  - 烟雾弹3单独遮蔽时长: {duration_3:.4f} s, 时间区间: [{interval_3[0]:.2f}, {interval_3[1]:.2f}]")
    flight_angle_rad = np.deg2rad(angle_degrees)
    v_uav_best = np.array([uav_speed * np.cos(flight_angle_rad), uav_speed * np.sin(flight_angle_rad), 0])
    p_d1 = uav_initial_pos + v_uav_best * t_release_1
    p_d2 = uav_initial_pos + v_uav_best * t_release_2
    p_d3 = uav_initial_pos + v_uav_best * t_release_3
    p_det1 = p_d1 + v_uav_best * t_free_fall_1 + np.array([0, 0, -0.5 * functions.g * t_free_fall_1 ** 2])
    p_det2 = p_d2 + v_uav_best * t_free_fall_2 + np.array([0, 0, -0.5 * functions.g * t_free_fall_2 ** 2])
    p_det3 = p_d3 + v_uav_best * t_free_fall_3 + np.array([0, 0, -0.5 * functions.g * t_free_fall_3 ** 2])
    bombs_info = [
        {'id': 1, 't_release': t_release_1, 't_free_fall': t_free_fall_1, 'p_drop': p_d1, 'p_det': p_det1},
        {'id': 2, 't_release': t_release_2, 't_free_fall': t_free_fall_2, 'p_drop': p_d2, 'p_det': p_det2},
        {'id': 3, 't_release': t_release_3, 't_free_fall': t_free_fall_3, 'p_drop': p_d3, 'p_det': p_det3},
    ]
    print("\n--- 烟幕弹投放策略详情 ---")
    for info in bombs_info:
        print(f"\n  [烟幕弹 {info['id']}]")
        print(f"    投放时间: {info['t_release']:.4f} s")
        print(f"    引信时间: {info['t_free_fall']:.4f} s")
        print(f"    绝对起爆时间: {info['t_release'] + info['t_free_fall']:.4f} s")
        print(f"    投放点: ({info['p_drop'][0]:.4f}, {info['p_drop'][1]:.4f}, {info['p_drop'][2]:.4f})")
        print(f"    起爆点: ({info['p_det'][0]:.4f}, {info['p_det'][1]:.4f}, {info['p_det'][2]:.4f})")

    print("#" * 50 + "\n")
    return best_params, max_duration


# --- 主程序入口: 在这里定义并运行你的不同场景 ---
if __name__ == '__main__':
    print("--- 开始进行手动验证 ---")

    # 推荐的、逻辑上可能产生非零解的参数
    plausible_solution = [87, 138, 18, 1.0, 1.0, 2.894, 7.0, 7.0]

    # 获取初始位置
    uav_pos = functions.UAV_POSITIONS['FY3']
    missile_pos = functions.MISSILE_POSITIONS['M3']

    # 直接调用目标函数进行测试
    objective_value = objective_for_optimizer(plausible_solution, uav_pos, missile_pos)

    if objective_value < 0:
        print(f"✅ 验证成功！这组参数产生了 {-objective_value:.4f} 秒的有效遮蔽时间。")
    else:
        print(f"❌ 验证失败。这组参数的遮蔽时间为0。可能需要微调参数。")

    print("--- 手动验证结束 ---\n")
    # bound1 = [  # FY1 to M1
    #     (170, 190),  # angle
    #     (70, 140),  # speed
    #     (1, 8),  # t_d1
    #     (1, 5),  # dt2
    #     (1, 5),  # dt3
    #     (1, 8),  # t_f1
    #     (1, 8),  # t_f2
    #     (1, 8)# t_f3
    # ]
    #
    # # 运行场景一
    # find_optimal_strategy(
    #     uav_key='FY1',
    #     missile_key='M1',
    #     bounds=bound1,
    # )
    #



    # bound2 = [  # FY2 to M2
    #     (186.6581, 335.4666),  # angle
    #     (70, 140),  # speed
    #     (1, 8),  # t_d1
    #     (1, 5),  # dt2
    #     (1, 5),  # dt3
    #     (1, 8),  # t_f1
    #     (1, 8),  # t_f2
    #     (1, 8)  # t_f3
    # ]
    #
    # find_optimal_strategy(
    #     uav_key='FY2',
    #     missile_key='M2',
    #     bounds=bound2,
    # )


    bound3 = [                  #FY3 to M3
        (85, 90),     # angle
        (135, 140),              # speed
        (16, 20),            # t_d1
        (1, 5),                # dt2
        (1, 5),                # dt3
        (2, 3.5),         # t_f1
        (0.01, 11.832),         # t_f2
        (0.01, 11.832)          # t_f3






    ]

    find_optimal_strategy(
        uav_key='FY3',
        missile_key='M3',
        bounds=bound3,
        maxiter=500,
        popsize=40,
        seed=None
    )


    # bound4 = [                  #FY4 to M2
    #     (237, 243),   # angle
    #     (130, 135),              # speed
    #     (1, 5),          # t_d1
    #     (1, 10),                # dt2
    #     (1, 10),                # dt3
    #     (10, 13),         # t_f1
    #     (1, 18.973),         # t_f2
    #     (1, 18.973)          # t_f3
    # ]
    #
    # find_optimal_strategy(
    #     uav_key='FY4',
    #     missile_key='M2',
    #     bounds=bound4,
    # )

    # bound5 = [  # FY5 to M3
    #     (125, 135),  # angle
    #     (130, 135),  # speed
    #     (10, 15),  # t_d1
    #     (1, 10),  # dt2
    #     (1, 10),  # dt3
    #     (3, 4),  # t_f1
    #     (0.01, 16.124),  # t_f2
    #     (0.01, 16.124)  # t_f3
    # ]
    #
    # find_optimal_strategy(
    #     uav_key='FY5',
    #     missile_key='M3',
    #     bounds=bound5,
    # )

    ##2-2
    # 飞行速度: 138m / s
    # 飞行方向: 280° (与x轴正方向夹角)
    # 投放时间: 4s
    # 起爆时间: 7.111111111111111s


    ##3-3
    # 飞行速度: 138 m / s
    # 飞行方向: 87° (与x轴正方向夹角)
    # 投放时间: 18.0 s
    # 起爆时间: 20.894736842105264 s

    ##4-2
    ## 飞行方向: 240° (与x轴正方向夹角)
    ## 飞行速度: 132 m/s
    ## 投放时间: 3 s
    # 起爆时间: 14.555555555555555 s



    ##5-3
    ##飞行方向: 131° (与x轴正方向夹角)
    ##飞行速度: 132 m/s
    ##投放时间: 12.5 s
    ##起爆时间: 16.342105263157894 s