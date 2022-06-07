# -*- coding: utf-8 -*-
import numpy as np
import random

def _custom_pmf(population, weights, size):
    # TODO: improve this code:
    if len(size) == 2:
        ans = np.zeros(shape=size)
        for i in range(size[0]):
            ans[i,:] = random.choices(
                population=population,
                weights=weights,
                k=size[1]
            )
    else:
        ans = np.array(random.choices(
            population=population,
            weights=weights,
            k=size[0]
        ))
    return ans

class StochasticDemandModel():
    def __init__(self, settings):
        self.settings = settings
        distr_dict = {
            'normal': lambda x: np.random.normal(
                self.settings['demand_distribution']['mu'],
                self.settings['demand_distribution']['sigma'],
                size=x
            ),
            'discrete_uniform': lambda x: np.random.randint(
                low=self.settings['demand_distribution']['low'],
                high=self.settings['demand_distribution']['high'],
                size=x
            ),
            'binomial': lambda x: np.random.binomial(
                self.settings['demand_distribution']['n'],
                self.settings['demand_distribution']['p'],
                size=x
            ),
            'probability_mass_function': lambda x: _custom_pmf(
                population=self.settings['demand_distribution']['vals'],
                weights=self.settings['demand_distribution']['probs'],
                size=x
            )
        }
        self.name_distribution = self.settings['demand_distribution']['name']
        self.generate = distr_dict[self.settings['demand_distribution']['name']]
        self.n_items = self.settings['n_items']

    def fit(self, data):
        pass

    def generate_scenario(self, history=None, n_time_steps=1):
        if n_time_steps == 1:
            return self.generate( (self.n_items, ) )
        else:
            return self.generate( (self.n_items, n_time_steps) )

        