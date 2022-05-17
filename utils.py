import math
import operator
import random
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
from tic_env import TictactoeEnv, OptimalPlayer
import matplotlib.pyplot as plt


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
    

def get_max_Mopt_Mrng_for_epsilon(values_mopt_mrng, epsilon_opts, parameter):
    max_Mopt = -2.0
    max_Mrnd = -2.0
    best_eps_opt = -1
    best_eps_rnd = -1

    for i, mopt_mrng in enumerate(values_mopt_mrng):
        m_opt = max(mopt_mrng[0])
        m_rng = max(mopt_mrng[1])

        if(m_opt > max_Mopt):
            max_Mopt = m_opt
            best_eps_opt = epsilon_opts[i]
        if(m_rng > max_Mrnd):
            max_Mrnd = m_rng
            best_eps_rnd = epsilon_opts[i]

    print('Maximal M_opt = {} and is achieved for {} = {}'.format(max_Mopt, parameter, best_eps_opt))
    print('Maximal M_rnd = {} and is achieved for {} = {}'.format(max_Mrnd, parameter, best_eps_rnd))