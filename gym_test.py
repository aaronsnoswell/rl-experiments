
import gym
from pyglet.window import key

env = gym.make('MountainCarContinuous-v0')
env.reset()

# Render once to construct visualisation objects
env.render()

# Attach keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.viewer.window.push_handlers(key_handler)

# Collect a few expert trajectories
expert_trajectories = []
num_trajectories = 5

for i in range(num_trajectories):
    expert_trajectories.append([])

    next_action = [0]
    per_step_action_inc = 0.05
    action_decay = 0.99

    t = 0
    cumulative_reward = 0
    while True:
        env.render()

        # Take a random step
        #env.step(env.action_space.sample())

        # Get next action from user
        next_action[0] *= action_decay
        if key_handler[key.LEFT] and not key_handler[key.RIGHT]:
            next_action[0] = min(next_action[0] - per_step_action_inc, env.unwrapped.min_action)
        elif not key_handler[key.LEFT] and key_handler[key.RIGHT]:
            next_action[0] = max(next_action[0] + per_step_action_inc, env.unwrapped.max_action)
        observation, reward, done, info = env.step(next_action)
        cumulative_reward += reward

        print("@t={}, r={:.2f}, c={:.2f}".format(t, reward, cumulative_reward))

        # Record taken action
        expert_trajectories[i].append((observation, next_action[0]))

        if done:
            break

        t += 1

    env.reset()

print("Done")


# Plot sampled expert policy

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory
for traj in expert_trajectories:
    ax.plot(
        [pt[0][0] for pt in traj],
        [pt[0][1] for pt in traj],
        [0 for pt in traj],
        '*-'
    )

ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_zlabel('Action')

plt.axis([
        env.unwrapped.min_position,
        env.unwrapped.max_position,
        -env.unwrapped.max_speed,
        env.unwrapped.max_speed
    ]
)
plt.show()
