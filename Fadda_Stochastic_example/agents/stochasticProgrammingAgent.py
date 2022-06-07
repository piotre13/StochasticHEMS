# -*- coding: utf-8 -*-
import os
import time
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from models.scenarioTree import ScenarioTree
from models.multistageOptimization import MultistageOptimization


class StochasticProgrammingAgent():

    def __init__(self, env, branching_factors, stoch_model):
        super(StochasticProgrammingAgent, self).__init__()
        self.env = env
        self.stoch_model = stoch_model
        self.branching_factors = branching_factors
        self.time_horizon = len(self.branching_factors)
        scenario_tree = ScenarioTree(
            name="scenario_tree",
            depth=self.time_horizon,
            branching_factors=self.branching_factors,
            dim_observations=self.env.n_items,
            initial_value=stoch_model.generate_scenario(),
            stoch_model=self.stoch_model
        )
        self.prb = MultistageOptimization(env, scenario_tree)

    def get_action(self, obs, debug=False):
        # 1. create the scenario tree
        new_scenario_tree = ScenarioTree(
            name="scenario_tree",
            depth=self.time_horizon,
            branching_factors=self.branching_factors,
            dim_observations=self.env.n_items,
            initial_value=obs['demand'],
            stoch_model=self.stoch_model
        )
        self.prb.update_data(obs, new_scenario_tree)
        # 2. solve the problem        
        _, sol, _ = self.prb.solve(debug_model=debug)
        return sol

    def save(self):
        pass

    def plot_policy(self):
        self._check_requisite()	
        possible_states = self.env.n_items + 1

        policy = np.zeros(
            shape=(
                possible_states,
                self.env.max_inventory_level[0] + 1,
                self.env.max_inventory_level[1] + 1
            )
        )
        for setup in range(possible_states):
            for inv0 in range(self.env.max_inventory_level[0] + 1):
                for inv1 in range(self.env.max_inventory_level[1] + 1):
                    obs = {
                        'demand': np.array([3, 2]),
                        'inventory_level': [inv0, inv1],
                        'machine_setup': [setup]
                    }
                    policy[setup, inv0, inv1] = self.get_action(
                        obs
                    )

        fig, axs = plt.subplots(1, possible_states)
        fig.suptitle('Found Policy')	
        for i, ax in enumerate(axs):	
            ax.set_title(f'Stato {i}')
            im = ax.pcolormesh(	
                policy[i,:,:], edgecolors='k', linewidth=2	
            )	
            im.set_clim(0, possible_states - 1)	
            if i == 0:	
                ax.set_ylabel('I1')	
            ax.set_xlabel('I2')	
        # COLOR BAR:	
        fig.subplots_adjust(right=0.85)	
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])	
        fig.colorbar(im, cax=cbar_ax)	
        plt.show()
	
    def _check_requisite(self):	
        if self.env.n_machines > 1:	
            raise Exception('ValueIteration is defined for one machine environment')
