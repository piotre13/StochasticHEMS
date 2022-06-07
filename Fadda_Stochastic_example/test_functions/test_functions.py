import matplotlib.pyplot as plt
from tangled_up_in_unicode import age

def test_agent(env, agent, verbose=True):
    actions = []
    setup_costs = []
    lost_sales = []
    holding_costs = []

    # TEST
    done = False
    obs = env.reset()
    if verbose: env.render()
    while not done:
        if verbose: print(obs)
        action = agent.get_action(obs)
        if verbose: print(">>> ", action)
        obs, reward, done, info = env.step(action)
        actions.append(action)
        setup_costs.append(info['setup_costs'])
        lost_sales.append(info['lost_sales'])
        holding_costs.append(info['holding_costs'])
        if verbose: env.render()
    if verbose: print("******")
    if verbose: print("******")
    return actions, setup_costs, lost_sales, holding_costs


def plot_comparison(env, dict_results, col_dict={}):
    N_PLOT = 3
    if env.n_machines == 1:
        N_PLOT = 4

    plt.subplot(N_PLOT,1,1)
    for key in dict_results:
        plt.plot(dict_results[key]['setup_costs'], col_dict[key] if key in col_dict else "-b", label=key)
    plt.legend()
    plt.ylabel('setup')

    plt.subplot(N_PLOT,1,2)
    for key in dict_results:
        plt.plot(
            dict_results[key]['lost_sales'],
            col_dict[key] if key in col_dict else "-b",
            label=key
        )
    plt.legend()
    plt.ylabel('lost sales')
    
    plt.subplot(N_PLOT,1,3)
    for key in dict_results:
        plt.plot(
            dict_results[key]['holding_costs'],
            col_dict[key] if key in col_dict else "-b",
            label=key
        )
    plt.legend()
    plt.ylabel('holding costs')
    
    if env.n_machines == 1:
        plt.subplot(N_PLOT,1,4)
        for key in dict_results:
            plt.plot(
                dict_results[key]['actions'],
                col_dict[key] if key in col_dict else "-b",
                label=key
            )    
        plt.legend()
        plt.ylabel('action')
    plt.show()


def test_agents_and_plot(env, agents):
    dict_results = {}
    col_dict = {}
    for _ in range(10):
        for key, agent, color in agents:
            actions, setup_costs, lost_sales, holding_costs = test_agent(
                env, agent, verbose=False
            )
            col_dict[key] = color
            dict_results[key] = {
                'actions': actions,
                'setup_costs': setup_costs,
                'lost_sales': lost_sales,
                'holding_costs': holding_costs,
            }
        # plot_comparison(env, dict_results, col_dict)
        for key in dict_results:
            tot_cost = sum(dict_results[key]['setup_costs'])
            tot_cost += sum(dict_results[key]['lost_sales'])
            tot_cost += sum(dict_results[key]['holding_costs'])
            print(key, tot_cost)
