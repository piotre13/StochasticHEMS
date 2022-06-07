# -*- coding: utf-8 -*-
import gym
import numpy as np


class SimplePlant(gym.Env):

    def __init__(self, settings, stoch_model):
        
        super(SimplePlant, self).__init__()
        # Basic cardinalities:
        self.T = settings['time_horizon']
        self.n_items = settings['n_items']
        self.n_machines = settings['n_machines']

        # Caracteristics:
        self.machine_production_matrix = settings['machine_production']
        self.max_inventory_level = settings['max_inventory_level']

        # Costs:
        self.holding_costs = settings['holding_costs']
        self.lost_sales_costs = settings['lost_sales_costs']
        self.setup_costs = settings['setup_costs']
        self.setup_loss = settings['setup_loss']

        # Initial State:
        self.machine_initial_setup = settings['initial_setup'] # This is the vector indicating the index of the item type for which the initial setup of the machine is prepared
        self.initial_inventory = settings['initial_inventory']

        # Demand generation:
        self.stoch_model = stoch_model
        # Generate demand scenario:
        self.generate_scenario_realization()

        self.reset()

        # Action Space        
        self.action_space = gym.spaces.MultiDiscrete([self.n_items+1 for _ in range(self.n_machines)])

        # Observation Space
        self.observation_space = gym.spaces.Dict({
            'demand': gym.spaces.Box(low = 0, high = 3, shape = (self.n_items,)),
            'inventory_level': gym.spaces.Box(low = 0, high = np.max(self.max_inventory_level), shape = (self.n_items,)),
            'machine_setup': gym.spaces.MultiDiscrete([self.n_items+1 for _ in range(self.n_machines)])
        })


    def generate_scenario_realization(self):
        self.scenario_demand = self.stoch_model.generate_scenario(n_time_steps=self.T + 1)

    def random_initial_state(self):
        self.reset()
        for i in range(self.n_items):
            self.inventory_level[i] = np.random.randint(0, self.max_inventory_level[i])
        for m in range(self.n_machines):
            non_zero_prod = [i for i, e in enumerate(self.machine_production_matrix[m]) if e != 0]
            self.machine_setup[m] = np.random.choice(non_zero_prod)

    def _next_observation(self):
        """
        Returns the next demand
        """
        self.demand = self.scenario_demand[:, self.current_step]
        return {
            'demand': self.demand,
            'inventory_level': self.inventory_level,
            'machine_setup': self.machine_setup
        }
    
    def _take_action(self, action, machine_setup, inventory_level, demand):
        """
        This method needs to return the cost on each lot decision devided in three main costs:
        
        Inputs
        ----------
            -action: action taken by the agent
    
        Returns
        -------
            - state updated component: the new inventory, machine setup, and effective setup
                - next inventory level: the inventory level changes with the demand, lost-setup production, production
                - next machine setup: gives the next machine set (usefull when we have setup time)
                - next effective setup: the setup that will be used for the production (usefull when we have setup time)
            - total_cost: the sum of all costs
            - next setup time counter: used to control the setup time
          
        """
        self.production = 0
        self.total_cost = np.array([0,0,0,0,0])
        setup_costs = np.zeros(self.n_machines)
        setup_loss = np.zeros(self.n_machines, dtype=int)
        lost_sales = np.zeros(self.n_items)
        holding_costs = np.zeros(self.n_items)

        # TODO: check time consequence
        # if we are just changing the setup, we use the setup cost matrix with the corresponding position given by the actual setup and the new setup
        for m in range(0, self.n_machines):   
            if action[m] != 0: # if the machine is not iddle
                # 1. IF NEEDED CHANGE SETUP
                if machine_setup[m] != action[m] and action[m] != 0:
                    setup_costs[m] = self.setup_costs[m][action[m] - 1] 
                    setup_loss[m] = self.setup_loss[m][action[m] - 1]
                machine_setup[m] = action[m]
                # 2. PRODUCTION
                self.production = self.machine_production_matrix[m][action[m] - 1] - setup_loss[m]
                inventory_level[action[m] - 1] += self.production
                if inventory_level[action[m] - 1] > self.max_inventory_level[action[m] - 1]:
                    inventory_level[action[m] - 1] = self.max_inventory_level[action[m] - 1]
            else:
                machine_setup[m] = 0

        # 3. SATIFING DEMAND
        for i in range(0, self.n_items):
            inventory_level[i] -= demand[i]
            if inventory_level[i] < 0:
                lost_sales[i] = - inventory_level[i] * self.lost_sales_costs[i]
                inventory_level[i] = 0
            # 4. HOLDING COSTS
            holding_costs[i] += inventory_level[i] * self.holding_costs[i]
        
        return {
            'setup_costs': sum(setup_costs),
            'lost_sales': sum(lost_sales),
            'holding_costs': sum(holding_costs),
        }


    def reset(self, random_start=False): 
        """
        Reset all environment variables important for the simulation.
            - Inventory
            - Setup
            - Demand_function
            - Current_step
        """
            
        # State variable:
        self.current_step = 1
        
        self.inventory_level = self.initial_inventory.copy()
        self.machine_setup =  self.machine_initial_setup.copy()
                
        if random_start:
            self.inventory_level = np.random.randint(
                low=0,
                high=np.array(self.max_inventory_level)+1,
                size=len(self.max_inventory_level)
            )
            
            for i in range(len(self.machine_setup)):
                feasible_pos = np.concatenate(
                    ([0], np.nonzero(self.machine_production_matrix[i])[0] + 1)
                )
                self.machine_setup[i] = int(
                    np.random.choice(
                        feasible_pos
                    )
                )
   
        # Monitoring variables
        self.total_cost = {
            "setup_costs": 0.0,
            "lost_sales": 0.0,
            "holding_costs": 0.0,
        }
        self.production = 0.0

        obs = self._next_observation()
        return obs


    def step(self, action):
        """
        Step method: Execute one time step within the environment

        Parameters
        ----------
        action : action given by the agent

        Returns
        -------
        obs : Observation of the state give the method _next_observation
        reward : Cost given by the _reward method
        done : returns True or False given by the _done method
        dict : possible information for control to environment monitoring

        """
        # TODO: ragionare bene sul tempo
        #if len(set(action)) != len(action):
        #    print("Error, at least two machines produce the same item")
        #    quit()


        self.total_cost = self._take_action(action, self.machine_setup, self.inventory_level, self.demand)
        
        reward = sum([ele for key, ele in self.total_cost.items()])
        
        self.current_step += 1
        done = self.current_step == self.T
        # demand at t + 1
        obs = self._next_observation()

        return obs, reward, done, self.total_cost


    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Time: {self.current_step}')
        print(f'\t production: {self.production}')
        print(f'\t inventory: {self.inventory_level}')
        print(f'\t setup: {self.machine_setup}')
        print(f'\t total_cost: {self.total_cost}')
        # print(f'\t demand + 1: {self.demand}')
