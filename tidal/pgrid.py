#######################################################
# pgrid.py
# Julian C. Marohnic
# Created 8/10/23
#
# A small set of functions for loading and plotting sets of "data grids"
# produced by grid_call.py. Intended for use with output from tidal study.
#######################################################

import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

# Read a DataGrid csv file and process in preparation for plotting. The DataGrid class is defined in grid_call.py.
def load_csv_grid(filename):
    grid = pd.read_csv(filename, index_col=0)
    grid = grid.transpose()
    # Gross hack to fix weird numbering issue. E.g. '1.2' rendering as '1.200000000002'. Something to do with numpy linspace().
    #grid.rename(columns={grid.columns[1]: 1.2, grid.columns[6]: 1.7}, inplace=True)
    
    return grid

# Read a DataGrid pickle file and process in preparation for plotting. The DataGrid class is defined in grid_call.py.
def load_pickle_grid(filename):
    #grid = pd.read_pickle(filename, index_col=0)
    grid = pd.read_pickle(filename)
    grid = grid.transpose()
    # Gross hack to fix weird numbering issue. E.g. '1.2' rendering as '1.200000000002'. Something to do with numpy linspace().
    #grid.rename(columns={grid.columns[1]: 1.2, grid.columns[6]: 1.7}, inplace=True)
    
    return grid

# Intented to plot multiple data grids. big_size and med_size parameters are for plot text. 
def plot_grid(title, rows, columns, datagrids=[], labels=[], cmaps=[], big_size=32, med_size=28, xtick_freq=2):
    if len(datagrids) != len(labels):
        raise ValueError("Number of labels must match number of data grids.")

    # Find max and min values of the quantity in question.
    max_val = 0
    min_val = 1e100

    for grid in datagrids:
        current_max = np.nanmax(grid.values)
        current_min = np.nanmin(grid.values)
        
        if current_max > max_val:
            max_val = current_max

        if current_min < min_val:
            min_val = current_min
        
    fig, axes = plt.subplots(rows, columns, sharey=True)

    for i, grid in enumerate(datagrids):
        # Last plot/bottom row in a column must show x-axis labels in addition to y-axis labels. rows*columns gives total tiles, bottom row
        # should be last N tiles, where N is the number of *columns* plotted.
        if (rows*columns - i) <= columns:
            sb.heatmap(grid, vmin=min_val, vmax=max_val, linewidth=0.5, xticklabels=xtick_freq, ax=axes.flat[i], cmap=cmaps[i])
            axes.flat[i].xaxis.set_tick_params(labelsize=med_size)
        else:
            sb.heatmap(grid, vmin=min_val, vmax=max_val, linewidth=0.5, xticklabels=False, ax=axes.flat[i], cmap=cmaps[i])

        axes.flat[i].invert_yaxis()
        axes.flat[i].set_title(labels[i], fontsize=big_size)
        axes.flat[i].yaxis.set_tick_params(labelsize=med_size)

        # Only include colorbar for last columns of tiles? Not currently implemented.
        cbar = axes.flat[i].collections[0].colorbar
        cbar.ax.tick_params(labelsize=med_size)

        # Hide y-tick marks for all tiles with unlabeled y-axes.
        if i % columns != 0:
            axes.flat[i].tick_params(left=False)

    fig.text(0.45, 0.04, 'Closest approach distance (Earth radii)', ha='center', fontsize=med_size)
    fig.text(0.04, 0.5, 'Speed at infinity (km/s)', va='center', rotation='vertical', fontsize=med_size)
    fig.text(0.45, 0.95, title, ha='center', fontsize=big_size)

    plt.show()
