import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatter
import json

myparams = {
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'Djvu Serif',
    'font.size': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'legend.fontsize':11
}
plt.rcParams.update(myparams)

def exp_smooth(arr, alpha = 0.99):
    ema = [arr[0]]
    for i in range(1, len(arr)):
        ema.append(alpha * ema[i-1] + (1 - alpha) * arr[i])
    return ema

def derivatives(arr):
    arr = np.array(arr)
    return arr[1:]-arr[:-1]

def der_mean(arr):
    arr = np.array(arr)
    sizes = np.arange(1, len(arr)+1) 
    cumsum = np.cumsum(arr)
    cum_avg = cumsum[:-1]/sizes[:-1] 
    return cum_avg[1:] - cum_avg[:-1]

def calc_abs_loss_derivatives(losses):
    cum_avg = np.cumsum(losses)/np.arange(1, len(losses)+1)
    abs_derivatives = np.abs(derivatives(cum_avg))
    return abs_derivatives

def visualize_and_save(config, bootstreped_size, begin_with = 10000):
    with open(f'{config.save_results_path}', 'r') as f:
        results = json.load(f)

    plt.figure()

    ys = dict()
    for d in results:
        param, losses = d['param'], d['losses']
        losses = np.array(losses)
        diffs_by_shuffle = []
        for _ in range(bootstreped_size):
            np.random.shuffle(losses)
            abs_der = calc_abs_loss_derivatives(losses)
            diffs_by_shuffle.append(abs_der)
        diffs_by_shuffle = np.array(diffs_by_shuffle)
        ys[param] = mean_diffs_by_shuffle = np.mean(diffs_by_shuffle, axis = 0)

    #########################
    for param in ys.keys():
        x = np.arange(len(ys[param]))[begin_with:]
        plt.plot(x, exp_smooth(ys[param], 0.995)[begin_with:], label = param)
    
    plt.yscale('log')

    plt.legend(title = config.legend_title)
    plt.title(config.dataset_name)
    plt.xlabel('$k$ (sample size)')
    plt.ylabel(r"$\left| \mathcal{L}_{k+1}(\hat{\boldsymbol{\theta}}) - \mathcal{L}_k(\hat{\boldsymbol{\theta}}) \right|$")

    plt.tight_layout()
    #########################

    plt.savefig(config.save_figs_path, bbox_inches='tight')
