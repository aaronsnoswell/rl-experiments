
import gym
import copy
import numpy as np
import matplotlib.pyplot as plt

from pyglet.window import key


def run_episode(policy):
    """
    Runs one episode of mountain car using the given policy

    @param policy - A function f(observation, *, kh) -> action mapping
        observations to actions. The kh parameter is a
        pyglet.window.key.KeyStateHandler instance that can be used to detect
        key presses in the rendering window
    """
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
        observation, reward, done, info = env.step(action)
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
    
    return [sgn(velocity + q)]


def manual_control_policy(observation, env, key_handler):
    """
    Manual control policy
    """

    # Constants
    action_decay = 0.9
    per_step_action_inc = 0.1

    # Set static variable for next action
    if "next_action" not in manual_control_policy.__dict__:
        manual_control_policy.next_action = [0]

    # Get next action from user
    manual_control_policy.next_action[0] *= action_decay
    if key_handler[key.LEFT] and not key_handler[key.RIGHT]:
        manual_control_policy.next_action[0] = max(
            manual_control_policy.next_action[0] - per_step_action_inc,
            env.unwrapped.min_action
        )
    elif not key_handler[key.LEFT] and key_handler[key.RIGHT]:
        manual_control_policy.next_action[0] = min(
            manual_control_policy.next_action[0] + per_step_action_inc,
            env.unwrapped.max_action
        )

    return manual_control_policy.next_action


def plot(traj):
    """
    Plots a sampled trajectory
    """

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    positions = [pt[0][0] for pt in traj]
    velocities = [pt[0][1] for pt in traj]
    actions = [pt[1][0] for pt in traj]

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

    t, cr, traj = run_episode(simple_policy)
    #plot(traj)
    #plt.title("Simple policy trace")
