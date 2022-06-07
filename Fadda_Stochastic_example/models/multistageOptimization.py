# -*- coding: utf-8 -*-
import time
import numpy as np
import gurobipy as grb

class MultistageOptimization():
    """
    Standard version multistage problem.
    """
    def __init__(self, instance, scenario_tree):
        self.name = "multistage_optimization"
        self.instance = instance
        self.scenario_tree = scenario_tree

        # Sets
        self.items = range(instance.n_items)
        self.nodes = range(self.scenario_tree.n_nodes)
        self.machines = range(instance.n_machines)

        # Model
        self.model = grb.Model(self.name)

        # State 1 if machine m is able to produce item i in node n
        X = self.model.addVars(
            instance.n_items, instance.n_machines, self.scenario_tree.n_nodes,
            vtype=grb.GRB.BINARY,
            name='X'
        )

        # Setup 1 if a pay the setup related to item i in machine m
        D = self.model.addVars(
            instance.n_items, instance.n_machines, self.scenario_tree.n_nodes,
            vtype=grb.GRB.BINARY,
            name='D'
        )

        # Inventory
        I = self.model.addVars(
            instance.n_items, self.scenario_tree.n_nodes,
            vtype=grb.GRB.CONTINUOUS,
            lb=0.0,
            name='I'
        )
        
        # Lost Sales
        Z = self.model.addVars(
            instance.n_items, self.scenario_tree.n_nodes,
            vtype=grb.GRB.CONTINUOUS,
            lb=0.0,
            name='Z'
        )
    
        obj_func = grb.quicksum(
            self.scenario_tree.nodes[n]['prob'] * (instance.lost_sales_costs[i] * Z[i, n] + instance.holding_costs[i] * I[i, n])
            for i in self.items
            for n in self.nodes
        )
        obj_func += grb.quicksum(
            self.scenario_tree.nodes[n]['prob'] * (instance.setup_costs[m][i] * D[i, m, n])
            for i in self.items
            for m in self.machines
            for n in self.nodes
        )

        self.model.setObjective(obj_func, grb.GRB.MINIMIZE)
        
        # INITIAL INVENTORY
        self.model.addConstrs(
            (I[i, 0] == instance.initial_inventory[i] for i in self.items),
            name='initial_inventory'
        )

        self.model.addConstrs(
            (I[i, n] <= instance.max_inventory_level[i] for i in self.items for n in self.nodes),
            name='max_inventory'
        )

        # INITIAL STATE
        self.default_state = []
        for m in self.machines:
            self.default_state.append(instance.machine_initial_setup[m])
            if instance.machine_initial_setup[m] != 0:
                self.model.addConstr(
                    (X[instance.machine_initial_setup[m] - 1, m, 0] == 1),
                    name=f'initial_state_machine_{m}'
                )
            else:
                self.model.addConstrs(
                    (X[i, m, 0] == 0 for i in self.items),
                    name=f'initial_state_machine_{m}'
                )

        # EVOLUTION
        for n in range(1, self.scenario_tree.n_nodes):
            parent = list(self.scenario_tree.predecessors(n))[0]
            self.model.addConstrs(
                (I[i, n] - Z[i, n] == I[i, parent] + grb.quicksum( instance.machine_production_matrix[m][i] * X[i, m, n] - instance.setup_loss[m][i] * D[i, m, parent] for m in self.machines ) - scenario_tree.nodes[n]['obs'][i] for i in self.items),
                name=f'item_flow_{n}'
            )
  
        # Machine no multiple state    
        self.model.addConstrs(
            (grb.quicksum(X[i, m, n] for i in self.items) <= 1 for m in self.machines for n in self.nodes ),
            name=f"no_more_setting_machine_node"
        )
        
        if instance.n_machines > 1:
            # each item must be produced on at most one machine makes sense if there is only one fixture or die per item
            self.model.addConstrs(
                ( grb.quicksum( X[i, m, n] for m in self.machines) <= 1 for i in self.items for n in self.nodes ),
                name="item_no_more_than_one"
            )
            # EXTRA CONSTRAINTS:
            # avoid two machines in the same state in the future
            self.model.addConstrs(
                ( grb.quicksum( D[i, m, n] for m in self.machines) <= 1 for i in self.items for n in self.nodes ),
                name="item_no_more_than_one_D"
            )
        # avoid change to items with no production
        for i in self.items:
            for m in self.machines:
                if instance.machine_production_matrix[m][i] == 0:
                    self.model.addConstrs(
                        (D[i, m, n] == 0  for n in self.nodes),
                        name=f'no_change_in_forbidden_state_{n}'
                    )

        # LINK X D
        for n in range(1, self.scenario_tree.n_nodes):
            parent = list(self.scenario_tree.predecessors(n))[0]
            self.model.addConstrs(
                D[i, m, parent] >=  X[i, m, n] - X[i, m, parent]
                for i in self.items for m in self.machines
            )
        self.model.update()
        self.X = X
        self.D = D
        self.I = I
        self.Z = Z
        self.obj_func = obj_func


    def update_data(self, obs, new_scenario_tree):
        # 1. UPDATE INVENTORY
        for i in self.items:
            self.model.setAttr(
                "RHS",
                self.model.getConstrByName(f"initial_inventory[{i}]"),
                # self.instance.initial_inventory[i]
                obs['inventory_level'][i]
            )

        # 2. UPDATE STATE
        for m in self.machines:
            c = self.model.getConstrByName(f'initial_state_machine_{m}')
            if not c:
                # if there are more than 1 machine
                for i in self.items:
                    self.model.remove(
                        self.model.getConstrByName(f'initial_state_machine_{m}[{i}]')
                    )
            else:
                # if there is 1 machine
                self.model.remove(
                    c    
                )
            # if self.instance.machine_initial_setup[m] != 0:
            if obs['machine_setup'][m] != 0:
                self.model.addConstr(
                    # (self.X[self.instance.machine_initial_setup[m] - 1, m, 0] == 1),
                    (self.X[obs['machine_setup'][m] - 1, m, 0] == 1),
                    name=f'initial_state_machine_{m}'
                )
            else:
                self.model.addConstrs(
                    (self.X[i, m, 0] == 0 for i in self.items),
                    name=f'initial_state_machine_{m}'
                )

        # 3. UPDATE STATE DEMAND
        for i in self.items:
            for n in range(1, self.scenario_tree.n_nodes):
                self.model.setAttr(
                    "RHS",
                    self.model.getConstrByName(f"item_flow_{n}[{i}]"),
                    - new_scenario_tree.nodes[n]['obs'][i]
                )
        self.default_state = obs['machine_setup']
        self.model.update()

    def solve(
        self, time_limit=None,
        gap=None, verbose=False, debug_model=False
    ):

        if gap:
            self.model.setParam('MIPgap', gap)
        if time_limit:
            self.model.setParam(grb.GRB.Param.TimeLimit, time_limit)
        if verbose:
            self.model.setParam('OutputFlag', 1)
        else:
            self.model.setParam('OutputFlag', 0)
        self.model.setParam('LogFile', './logs/gurobi.log')
        if debug_model:
            self.model.write(f"./logs/{self.name}_{self.instance.current_step}.lp")
        # from datetime import datetime
        # self.model.write(f"./logs/RP_{ datetime.now() }.lp")

        start = time.time()
        self.model.optimize()
        end = time.time()
        comp_time = end - start

        if self.model.status == grb.GRB.Status.OPTIMAL:
            sol = [0] * self.instance.n_machines
            for m in self.machines:
                for i in self.items:
                    if round(self.D[i,m,0].X, 0) > 0.5:
                        sol[m] = i + 1 # 0 means idle                
                if sol[m] == 0:
                    # NB: the variable D has an interpretation in terms of changes
                    # D[i,m,0] = 0 means same setting or idle:
                    if self.default_state[m] == 0:
                        sol[m] = self.default_state[m]
                    else:
                        if sum([self.X[self.default_state[m] - 1,m,n].X for n in self.scenario_tree.successors(0)]) == 0:
                            sol[m] = 0
                        else:
                            sol[m] = self.default_state[m]

            return self.model.getObjective().getValue(), np.array(sol), comp_time
        else:
            print("MODEL INFEASIBLE OR UNBOUNDED")
            return -1, [], comp_time
