# -*- coding: utf-8 -*-
import json
import random
import logging
import numpy as np
from envs import *
from agents import *
from models import *
from test_functions import *


np.random.seed(123)
random.seed(0)


if __name__ == '__main__':
    log_name = "./logs/main.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )

    fp = open("./cfg/setting_2items_VI.json", 'r')
    settings = json.load(fp)
    fp.close()
    settings["time_horizon"] = 100

    stoch_model = StochasticDemandModel(
        settings
    )

    env = SimplePlant(settings, stoch_model)

    vi_agent = ValueIteration(env)
    vi_agent.learn(iterations=10)
    # vi_agent.plot_policy()

    stoch_agent = StochasticProgrammingAgent(
        env,
        branching_factors=[4, 4, 2],
        stoch_model=stoch_model
    )

    obs = {
        'demand': np.array([3, 2]),
        'inventory_level': [10, 10],
        'machine_setup': [1]
    }
    # act = stoch_agent.get_action(
    #     obs
    # )
    # print(act)

    # stoch_agent.plot_policy()

    test_agents_and_plot(
        env,
        [
            ("RP", stoch_agent, 'b'),
            ("VI", vi_agent, 'g')
        ]   
    )
