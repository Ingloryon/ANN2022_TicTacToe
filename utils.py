import math
import operator
import random
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
from tic_env import TictactoeEnv, OptimalPlayer
import matplotlib.pyplot as plt
import torch


def action_to_key(action,  state):
    if type(action) is tuple:
        action = action[0]*3 + action[1]
    return (action, grid_to_string(state))


def grid_to_string(grid):
    char_rep= {0:'-', 1: 'X',-1:'O'}
    return ''.join([char_rep[x] for x in grid.flatten()])


def plots_several_trainings(values, names, avg_step, nb_epoch):
    plt.figure(figsize=(20, 10))
    xs = range(0, nb_epoch, avg_step)
    for val, name in zip(values, names):
        plt.plot(xs, val, label=name, lw=2)
        
    plt.xlabel('Number of games played', fontsize= 20)
    plt.ylabel('Mean reward over {} games'.format(avg_step), fontsize = 20)
    plt.title('Evolution of mean reward (every {} games played) of the learner'.format(avg_step), fontsize = 20)
    plt.grid()
    plt.legend(loc=2)
    plt.show()
    
    
def get_other_player(player):
    """
    Get the other opponent player name
    :param player: the current player name
    :return: the opponent player name
    """
    return "X" if player == "O" else "O"
def grid_to_state(grid,  env, player):
    """
    Convert the numpy grid to a tensor according to the definition in the handout
    :param env: the current environement of the game
    :param player: our current learner
    """
    return torch.tensor([grid==env.player2value[player.player], grid==env.player2value[get_other_player(player.player)]],dtype=torch.float).unsqueeze(0)

def empty(grid):
        '''return all empty positions'''
        avail = []
        for i in range(9):
            pos = (int(i/3), i % 3)
            if grid[pos] == 0:
                avail.append(i)
        return avail
    

    
def plots_several_trainings_subfigures(values, names, avg_step, nb_epoch, nrows=3, ncols=2, mopt_mrng=False):
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(18, 14))
    xs = range(0, nb_epoch, avg_step)
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    i = 0
    for val, name in zip(values, names):
        ax = axs.flat[i]
        ax.set_yticks(np.arange(-1, 1, 0.1))
        if(mopt_mrng):
            ax.plot(xs, val[0], color='blue', lw=4, label="Against Optimal Player")
            ax.plot(xs, val[1], color='orange', lw=4, label="Against Random Player")       
        else:
            ax.plot(xs, val, color=default_colors[i%len(default_colors)], label=name, lw=2)
        ax.set(xlabel=name)
        ax.grid()
        ax.legend(loc=4)
        # ax.label_outer()
        i += 1

    fig.text(0.5, 0.04, 'Number of games played', ha='center', fontsize = 22)
    fig.text(0.04, 0.5, 'Mean reward over {} games'.format(avg_step), va='center', rotation='vertical', fontsize = 15)
    plt.suptitle('Evolution of mean reward (every {} games played) of the learner'.format(avg_step), fontsize = 15)
    plt.show()

    
# Source code: https://matplotlib.org/3.5.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def plot_game_heatmaps(states, q_vals, titles):
    
    fig, axs = plt.subplots(len(states)//3, 3, figsize=(16, 16))
    axis_labels = ['1', '2', '3']
    
    for i in range(0, len(states)):
        ax = axs[i]
        vals = np.zeros((3,3))
        state = states[i]
        for action in range(9):
            key = (action, state)
            vals[action//3, action%3]= q_vals.get(key, np.nan if state[action] != "-" else 0)

        im, cbar = heatmap(vals, axis_labels, axis_labels, ax=ax, cbarlabel="Q-values")
        
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(axis_labels)), labels=axis_labels)
        ax.set_yticks(np.arange(len(axis_labels)), labels=axis_labels)
        ax.set_title(titles[i], fontsize=15)

        # Loop over data dimensions and create text annotations.
        for i in range(len(axis_labels)):
            for j in range(len(axis_labels)):
                if vals[i,j]!=np.nan:
                    text = ax.text(j, i, "{:.2e}".format(vals[i, j]),
                                   ha="center", va="center", color="grey")

    fig.tight_layout()
    plt.show()
    

def get_max_Mopt_Mrng_for_epsilon(values_mopt_mrng, epsilon_opts, parameter, n_last_iter=8):
    max_Mopt = -2.0
    max_Mrnd = -2.0
    best_eps_opt = -1
    best_eps_rnd = -1

    for i, mopt_mrng in enumerate(values_mopt_mrng):
        m_opt = max(mopt_mrng[0][:-n_last_iter])
        m_rng = max(mopt_mrng[1][:-n_last_iter])

        if(m_opt > max_Mopt):
            max_Mopt = m_opt
            best_eps_opt = epsilon_opts[i]
        if(m_rng > max_Mrnd):
            max_Mrnd = m_rng
            best_eps_rnd = epsilon_opts[i]

    print('Maximal M_opt = {} and is achieved for {} = {}'.format(max_Mopt, parameter, best_eps_opt))
    print('Maximal M_rnd = {} and is achieved for {} = {}'.format(max_Mrnd, parameter, best_eps_rnd))
    return (max_Mrnd, max_Mopt), (best_eps_rnd, best_eps_opt)
    
    
def plot_game_heatmaps_deep_qlearning(states, agent, grids, turns ,titles):
    
    fig, axs = plt.subplots(len(states)//3, 3, figsize=(16, 16))
    axis_labels = ['1', '2', '3']
    
    for i in range(0, len(states)):
        env = TictactoeEnv()
        env.grid = grids[i]
        agent.player = turns[i]
        
        ax = axs[i]
        vals = np.zeros((3,3))
        state = states[i]
        
        for action in range(9):
            state = grid_to_state(env.grid,env, agent)
            q_vals = agent.model(state)
            vals[action//3, action%3]= q_vals[0][action]
            
        im, cbar = heatmap(vals, axis_labels, axis_labels, ax=ax, cbarlabel="Q-values")
        
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(axis_labels)), labels=axis_labels)
        ax.set_yticks(np.arange(len(axis_labels)), labels=axis_labels)
        ax.set_title(titles[i], fontsize=15)

        # Loop over data dimensions and create text annotations.
        for i in range(len(axis_labels)):
            for j in range(len(axis_labels)):
                if vals[i,j]!=np.nan:
                    text = ax.text(j, i, "{:.2e}".format(vals[i, j]),
                                   ha="center", va="center", color="grey")

    fig.tight_layout()
    plt.show()
    
def get_performance_table(qtraining_pol, deepqtraining_pol, qtraining_self, deepqtraining_self):
    results = []
    results.append(get_t_train_m_opt_m_rand(qtraining_pol.score_test_opt, qtraining_pol.score_test_rng, qtraining_pol.avg_step))
    results.append(get_t_train_m_opt_m_rand(deepqtraining_pol.score_test_opt, deepqtraining_pol.score_test_rng, deepqtraining_pol.AVG_STEP))
    results.append(get_t_train_m_opt_m_rand(qtraining_self.score_test_opt, qtraining_self.score_test_rng, qtraining_self.avg_step))
    results.append(get_t_train_m_opt_m_rand(deepqtraining_self.score_test_opt, deepqtraining_self.score_test_rng, deepqtraining_self.AVG_STEP))
   

    rcolors = plt.cm.BuPu(np.full(4, 0.1))
    ccolors = plt.cm.BuPu(np.full(3, 0.1))
    
    collabel=("$M_{RAND}$", "$M_{OPT}$", "$T_{train}$")
    rowlabel=("QLearning - Optimal policy", "DeepQLearning - Optimal policy","QLearning - Self learning","DeepQLearning - Self learning")
    plt.axis('off')
    
    the_table = plt.table(cellText=results,
                      rowLabels=rowlabel,
                      rowColours=rcolors,
                      rowLoc='right',
                      colColours=ccolors,
                      colLabels=collabel,
                      loc='center')
    the_table.scale(2, 2)
    the_table.set_fontsize(14)
    fig_background_color = 'skyblue'
    fig_border = 'steelblue'
    plt.figure(linewidth=2,
           edgecolor=fig_border,
           facecolor=fig_background_color
          )
    
    
    
    
def get_t_train_m_opt_m_rand(opt, rng, step):
    low_opt, high_opt = opt[0], np.mean(opt[-4])
    thresh_opt = 0.8*(-high_opt+low_opt)
    idx_opt = [y > thresh_opt for y in opt].index(True)
    
    low_rng, high_rng = rng[0], np.mean(rng[-4])
    thresh_rng = 0.8*(high_rng-low_rng) if high_rng > 0 else 0.8*(low_rng-high_rng)
    
    idx_rng = [y > thresh_rng for y in rng].index(True)
    return [high_rng,high_opt, max(idx_rng, idx_opt)*step] 
    
    
    
    
    