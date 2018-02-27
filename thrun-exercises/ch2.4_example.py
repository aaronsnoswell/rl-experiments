"""
Demonstrates an application of the Bayes filter
From Thrun, 2005 Probabilistic Robotics, ch2.4, p28
A robot tries to sense a door and then open or close it
"""


# The sensor model
# A mapping from true states to sensed states
# The sensor is fairly good if the door is truly closed, but if the door is
# open it's not much better than chance
sensor_model = {
    "open": {
        "open": 0.6,
        "closed": 0.4
    },
    "closed": {
        "open": 0.2,
        "closed": 0.8
    }
}


# The world dynamics model
# A mapping from current true state to action to new true states
# The robot has an 80% chance of opening a closed door. For other state-action
# pairs everything is deterministic
dynamics_model = {
    "open": {
        "push": {
            "open": 1,
            "closed": 0
        },
        "nothing": {
            "open": 1,
            "closed": 0
        }
    },
    "closed": {
        "push": {
            "open": 0.8,
            "closed": 0.2
        },
        "nothing": {
            "open": 0,
            "closed": 1
        }
    }
}


# Initialize state set
state_set = list(sensor_model.keys())


def bayes_filter(belief, action, observation):
    """
    Applies the Bayes filter
    Updates a belief based on an applied aciton, and a recieved observation
    (assumed to have happened in that order)
    """

    # Initialise mappings
    prediction = {}
    new_belief = {}

    # For all states
    for current_state in state_set:

        # Step 1 - compute prediction by integrating over states
        prediction[current_state] = 0
        for new_state in state_set:
            new_state_probability = dynamics_model[current_state][action][new_state]
            print(current_state, action, new_state, new_state_probability, belief[current_state])
            prediction[current_state] += new_state_probability * belief[current_state]

        # Step 2 - compute measurement update / integrate observation
        observation_probability = sensor_model[current_state][observation]
        new_belief[current_state] = observation_probability * prediction[current_state]

    # Finally, apply eta normalizing constant
    belief_sum = sum(list(new_belief.values()))
    for key in new_belief:
        new_belief[key] /= belief_sum

    return prediction, new_belief


# Initialize the boundary condition - initial belief is uniform random
belief = {
    "open": 0.5,
    "closed": 0.5
}

# Initial conditions
t = 0
print(t, belief)

# Initially the robot takes no action but senses an open door
t += 1
prediction, belief = bayes_filter(belief, "nothing", "open")
print(t, prediction, belief)

# Then is pushes and senses open
t += 1
prediction, belief = bayes_filter(belief, "push", "open")
print(t, prediction, belief)
