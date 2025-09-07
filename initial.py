import numpy as np
from numba import jit, prange
import time
import matplotlib.pyplot as plt
import functions
from cylinder import theta

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 常量定义
g = 9.8  # 重力加速度
# ================== 优化主函数 ==================
def optimize_strategy(uav, missile):

    """多阶段参数优化"""
    # 阶段1: 粗粒度全局搜索
    best_time = 0.0
    best_params = (0, 0, 0, 0)
    
    v_list = np.arange(70, 140, 1)  # 速度步长5m/s
    theta_list = np.arange(0, 360, 5)  # 角度步长10°
    t_drop_list = np.arange(1, 30, 1)  # 投放时间步长2s
    
    print("开始粗粒度搜索...")
    start_time = time.time()
    
    for v in v_list:
        for theta_deg in theta_list:

            for t_drop in t_drop_list:
                # 起爆时间范围 (1-20s后)
                t_expl_list = np.linspace(t_drop+1, t_drop+20, 10)
                
                for t_expl in t_expl_list:
                    cover_time = functions.calculate_obscuration_time([theta_deg, v, t_drop, t_expl - t_drop, 0.01], functions.UAV_POSITIONS[uav], functions.MISSILE_POSITIONS[missile])[0]

                    if cover_time > best_time:
                        best_time = cover_time
                        best_params = (v, theta_deg, t_drop, t_expl)
                        print(f"新最优: {cover_time:.2f}s, 参数: v={v}, θ={theta_deg}°, t_drop={t_drop}s, t_expl={t_expl}s")
    
    print(f"粗搜索完成! 耗时: {time.time()-start_time:.1f}s")
    print(f"最优解: t={best_time:.2f}s, params={best_params}")
    
    # 阶段2: 细粒度局部搜索
    v_opt, theta_opt, t_drop_opt, t_expl_opt = best_params
    
    v_range = np.arange(max(70, v_opt-10), min(140, v_opt+10)+1, 1)
    theta_range = np.arange(max(0, theta_opt-10), min(360, theta_opt+10)+1, 1)
    t_drop_range = np.arange(max(1, t_drop_opt-5), min(50, t_drop_opt+5)+0.5, 0.5)
    
    print("开始细粒度搜索...")
    start_time = time.time()
    
    for v in v_range:
        for theta_deg in theta_range:

            for t_drop in t_drop_range:
                t_expl_range = np.linspace(t_drop+1, t_drop+10, 20)
                
                for t_expl in t_expl_range:
                    cover_time = functions.calculate_obscuration_time([theta_deg, v, t_drop, t_expl - t_drop, 0.01], functions.UAV_POSITIONS['FY5'], functions.MISSILE_POSITIONS['M3'])[0]
                    
                    if cover_time > best_time:
                        best_time = cover_time
                        best_params = (v, theta_deg, t_drop, t_expl)
                        print(f"优化后: {cover_time:.2f}s, 参数: v={v}, θ={theta_deg}°, t_drop={t_drop}s, t_expl={t_expl}s")
    
    print(f"细搜索完成! 耗时: {time.time()-start_time:.1f}s")
    
    # 计算最优参数对应位置
    v_opt, theta_opt, t_drop_opt, t_expl_opt = best_params
    print(f"------------{uav}, {missile}---------")
    print(f"v={v_opt}")
    print(f"theta_deg={theta_deg}")
    print(f"t_drop={t_drop}")
    print(f"t_expl={t_expl}")
