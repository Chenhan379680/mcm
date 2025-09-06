import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# --- Initial Data and Parameters ---
MISSILE_POSITIONS = {
    'M1': np.array([20000.0, 0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0, -600.0, 1900.0])
}
fake_target = np.array([0.0, 0.0, 0.0])
missile_speed = 300.0  # m/s


# --- Calculation Function ---
def calculate_missile_position(initial_pos, t):
    """Calculates the position of a missile at time t."""
    direction_vector = fake_target - initial_pos
    if np.linalg.norm(direction_vector) == 0:
        return initial_pos  # Avoid division by zero if already at target
    unit_vector = direction_vector / np.linalg.norm(direction_vector)
    position = initial_pos + unit_vector * missile_speed * t
    return position


# --- Plotting Setup ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# To handle Chinese characters in labels
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Plot the static fake target
ax.scatter(fake_target[0], fake_target[1], fake_target[2], c='black', marker='P', s=150, label='假目标 (原点)')
ax.text(fake_target[0], fake_target[1], fake_target[2] - 200, ' 假目标', color='black')

# Initialize plot elements for the missiles
missile_points = {}
missile_lines = {}
missile_texts = {}

for name, pos in MISSILE_POSITIONS.items():
    # Initialize the scatter plot point for each missile
    missile_points[name], = ax.plot([pos[0]], [pos[1]], [pos[2]], 'rx', markersize=10,
                                    label=f'导弹 ({name})' if name == 'M1' else "")
    # Initialize the line (path) for each missile
    missile_lines[name], = ax.plot([pos[0], fake_target[0]], [pos[1], fake_target[1]], [pos[2], fake_target[2]], 'r--',
                                   linewidth=1)
    # Initialize the text label for each missile
    missile_texts[name] = ax.text(pos[0], pos[1], pos[2] + 100, f' {name}', color='red')


# --- Animation Function ---
def update(frame):
    """This function is called for each frame of the animation."""
    # 'frame' is the current time in seconds (for this animation)
    t = frame

    # Update the position of each missile
    for name, initial_pos in MISSILE_POSITIONS.items():
        # Calculate new position
        new_pos = calculate_missile_position(initial_pos, t)

        # Check if the missile has reached the target
        if np.linalg.norm(new_pos - initial_pos) >= np.linalg.norm(fake_target - initial_pos):
            new_pos = fake_target

        # --- CORRECTED LINE HERE ---
        # Pass each coordinate as a single-element list (a sequence)
        missile_points[name].set_data_3d([new_pos[0]], [new_pos[1]], [new_pos[2]])

        # Update the line's data (path from current position to target)
        missile_lines[name].set_data_3d([new_pos[0], fake_target[0]], [new_pos[1], fake_target[1]],
                                        [new_pos[2], fake_target[2]])

        # Update the text label's position
        missile_texts[name].set_position((new_pos[0], new_pos[1], new_pos[2] + 100))

    # Update the title to show the current time
    ax.set_title(f'导弹飞行态势 (时间: {t:.1f} s)')

    # Return the updated plot elements
    return list(missile_points.values()) + list(missile_lines.values()) + list(missile_texts.values())


# --- Customize and Run the Animation ---
ax.set_xlabel('X 坐标 (m)')
ax.set_ylabel('Y 坐标 (m)')
ax.set_zlabel('Z 坐标 (m)')

# Set fixed axis limits to prevent the view from shifting
ax.set_xlim([0, 22000])
ax.set_ylim([-3000, 3000])
ax.set_zlim([0, 2500])

# Create a clean legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left')

# Create the animation
# frames: number of frames (here, simulating 70 seconds with a step of 0.5s)
# interval: delay between frames in milliseconds
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 70, 0.5),
                              blit=False, interval=50)

# To save the animation as a gif, you might need to install 'imagemagick' or 'ffmpeg'
# ani.save('missile_animation.gif', writer='imagemagick', fps=20)

plt.tight_layout()
plt.show()