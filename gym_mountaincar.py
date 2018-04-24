
import gym
import copy
import numpy as np
import matplotlib.pyplot as plt

from pyglet.window import key


def run_episode(policy, *, continuous=False):
    """
    Runs one episode of mountain car using the given policy

    @param policy - A function f(observation, *, kh) -> action mapping
        observations to actions. The kh parameter is a
        pyglet.window.key.KeyStateHandler instance that can be used to detect
        key presses in the rendering window
    """
    env = gym.make('MountainCar-v0')

    if continuous:
        env = gym.make('MountainCarContinuous-v0')

    # Shorten episode length
    env._max_episode_steps = 200
    
    env.reset()

    # Render once to construct visualisation objects
    env.render()

    # Attach keyboard handler
    key_handler = key.KeyStateHandler()
    env.unwrapped.viewer.window.push_handlers(key_handler)

    t = 0
    cumulative_reward = 0
    observation = [0, 0]
    trajectory = []
    while True:
        env.render()

        # Get next action from policy
        action = policy(observation, env, key_handler)
        step_action = None

        if not continuous:
            # Round action to nearest int, and offset to get an index
            action = min(max(round(action), -1), 1)
            step_action = action + 1

        if continuous:
            # Continuous version uses actions on [-1, 1]
            action = min(max(action, -1), 1)
            step_action = [float(action)]

        observation, reward, done, info = env.step(step_action)
        cumulative_reward += reward

        trajectory.append((observation, copy.copy(action)))

        if done:
            break

        t += 1

    env.close()

    return t, cumulative_reward, trajectory


def simple_policy(observation, env, key_handler, *, q=-0.003):
    """
    A simple policy that solves the mountain car problem
    """

    # Sign function
    sgn = lambda x: -1 if x < 0 else 1

    position, velocity = observation
    
    return sgn(velocity + q)


def manual_policy(observation, env, key_handler):
    """
    Manual control policy
    """

    # Constants
    step_inc = 0.1
    decay = 0.9

    # Set static variable for next action
    if "action" not in manual_policy.__dict__:
        manual_policy.action = 0

    # Get next action from user
    manual_policy.action *= 0.9
    if key_handler[key.LEFT] and not key_handler[key.RIGHT]:
        manual_policy.action = max(
            manual_policy.action - step_inc,
            -1
        )
    elif not key_handler[key.LEFT] and key_handler[key.RIGHT]:
        manual_policy.action = min(
            manual_policy.action + step_inc,
            1
        )

    return manual_policy.action


def plot(traj):
    """
    Plots a sampled trajectory
    """

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    positions = [pt[0][0] for pt in traj]
    velocities = [pt[0][1] for pt in traj]
    actions = [pt[1] for pt in traj]

    ax.plot(
        positions,
        velocities,
        actions,
        '*-'
    )

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Action')

    plt.axis([
            -1.2,
            0.6,
            -0.07,
            0.07
        ]
    )
    plt.show()


if __name__ == "__main__":

    t, cr, traj = run_episode(simple_policy, continuous=False)
    #t, cr, traj = run_episode(manual_policy, continuous=False)
    plot(traj)
