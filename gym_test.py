
import gym
from pyglet.window import key

env = gym.make('MountainCarContinuous-v0')
env.reset()

# Render once to construct visualisation objects
env.render()

# Attach keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.viewer.window.push_handlers(key_handler)

next_action = [0]
per_step_action_inc = 0.05
max_action = 1
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
        next_action[0] = min(next_action[0] - per_step_action_inc, -max_action)
    elif not key_handler[key.LEFT] and key_handler[key.RIGHT]:
        next_action[0] = max(next_action[0] + per_step_action_inc, +max_action)
    observation, reward, done, info = env.step(next_action)
    cumulative_reward += reward

    print("@t={}, r={:.2f}, c={:.2f}".format(t, reward, cumulative_reward))

    if done:
        break

    t += 1

print("Done")
