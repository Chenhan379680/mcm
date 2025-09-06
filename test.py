import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

import functions

uav_initial_pos_3 = functions.UAV_POSITIONS['FY3']
missile_initial_pos = functions.MISSILE_POSITIONS['M1']


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

t = functions.calculate_obscuration_time(params=[np.float64(74.69719556205483),
                                                 np.float64(106.61334718035776),
                                                 np.float64(29.318946358743553),
                                                 np.float64(0.42651023296990787),
                                                 0.2],
                                         uav_initial_pos=uav_initial_pos_3,
                                         missile_initial_pos=missile_initial_pos)[0]


angle_radians = np.radians(np.float64(74.69719556205483))
uav_speed = np.float64(106.61334718035776)
t_release = np.float64(29.318946358743553)
t_free_fall = np.float64(0.42651023296990787)

v_uav = np.array([uav_speed * np.cos(angle_radians), uav_speed * np.sin(angle_radians), 0])
p_release = uav_initial_pos_3 + v_uav * t_release
p_detonation = p_release + v_uav * t_free_fall + np.array([0, 0, -0.5 * functions.g * t_free_fall ** 2])

print(t)

print(p_release)
print(p_detonation)