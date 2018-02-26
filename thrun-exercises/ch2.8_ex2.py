
from enum import Enum
from random import uniform

"""A class to simulate markovian weather"""
class MarkovWeather:

    """A state enum for the weather in a simple world"""
    class State(Enum):
        Sunny = 1
        Cloudy = 2
        Rainy = 3

    """
    Transition table for the weather
    First element is 'from' weather, second is 'to'
    """
    Transitions = {
        State.Sunny : {
            State.Sunny : 0.8,
            State.Cloudy : 0.2,
            State.Rainy : 0.0
        },
        State.Cloudy : {
            State.Sunny : 0.4,
            State.Cloudy : 0.4,
            State.Rainy : 0.2
        },
        State.Rainy : {
            State.Sunny : 0.2,
            State.Cloudy : 0.6,
            State.Rainy : 0.2
        }
    }

    """c-tor"""
    def __init__(self, S0 = State.Sunny):
        self.S = S0

    """Shortcut for getting a transition probability"""
    def p(self, To):
        return MarkovWeather.Transitions[self.S][To]

    """Step the Markov chain once"""
    def step(self):

        roll = uniform(0, 1)

        a = self.p(MarkovWeather.State.Sunny)
        b = self.p(MarkovWeather.State.Cloudy)

        self.S = MarkovWeather.State.Sunny if (roll <= a) else \
            (MarkovWeather.State.Cloudy if (roll <= a + b) else MarkovWeather.State.Rainy)

        return self.S


if __name__ == "__main__":
    
    # Simulate some weather
    N = int(1E5)

    w = MarkovWeather(MarkovWeather.State.Sunny)

    Counts = {
        MarkovWeather.State.Sunny : 0,
        MarkovWeather.State.Cloudy : 0,
        MarkovWeather.State.Rainy : 0
    }

    total = 0
    for i in range(N):
        Counts[w.step()] += 1
        total += 1
    
    # Compute the stationary distribution probabilities
    stationary_distribution = [(v * 100.0 / total) for v in Counts.values()]
    print(Counts)
    print(stationary_distribution)
