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
    
    return grid

# Read a DataGrid pickle file and process in preparation for plotting. The DataGrid class is defined in grid_call.py.
def load_pickle_grid(filename):
    #grid = pd.read_pickle(filename, index_col=0)
    grid = pd.read_pickle(filename)
    grid = grid.transpose()
    
    return grid

# Intented to plot multiple data grids. big_size and med_size parameters are for plot text. 
def plot_grid(title, rows, columns, datagrids=[], labels=[], cmaps=[], center=None, big_size=32, med_size=28, xtick_freq=2):
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

    # Colorbar centering options. We can specify nothing (just set colorbar based on limits), set center to zero, in which
    # case the value '0' will be set to the center of the colorbar, or 'True', which centers the midpoint between the extreme
    # values at the middle of the colorbar range.
    if center == 'True':
        center = (min_val + max_val)/2
    elif center == 0:
        center = 0
    else:
        center = None

    fig, axes = plt.subplots(rows, columns, sharey=True)

    for i, grid in enumerate(datagrids):
        # Last plot/bottom row in a column must show x-axis labels in addition to y-axis labels. rows*columns gives total tiles, bottom row
        # should be last N tiles, where N is the number of *columns* plotted.
        if (rows*columns - i) <= columns:
            sb.heatmap(grid, vmin=min_val, vmax=max_val, linewidth=0.5, xticklabels=xtick_freq, ax=axes.flat[i], cmap=cmaps[i], center=center)
            axes.flat[i].xaxis.set_tick_params(labelsize=med_size)
        else:
            sb.heatmap(grid, vmin=min_val, vmax=max_val, linewidth=0.5, xticklabels=False, ax=axes.flat[i], cmap=cmaps[i], center=center)

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

# Specialized function for plotting a six panel figure. Top three tiles are three plots of a given metric for tidal suites, bottom three
# are corresponding "differential" plots with the metric shown relative to the spherical case.
def combo_grid(title, grids, diffgrids, cmap, diffcmap, labels, difflabels, gridcenter=None, diffcenter=None, big_size=32, med_size=28, xtick_freq=2, grid_range=None, diffgrid_range=None):
    if grid_range == None:
        # Find max and min values across the grids.
        min_grid_val, max_grid_val = min_max_grid(grids)
    else:
        min_grid_val = grid_range[0]
        max_grid_val = grid_range[1]

    if diffgrid_range == None:
        # Find max and min values across the diffgrids.
        min_diff_val, max_diff_val = min_max_grid(diffgrids)
    else:
        min_diff_val = diffgrid_range[0]
        max_diff_val = diffgrid_range[1]

    # Colorbar centering options. We can specify nothing (just set colorbar based on limits), set center to zero, in which
    # case the value '0' will be set to the center of the colorbar, or 'True', which centers the midpoint between the extreme
    # values at the middle of the colorbar range. gridcenter applies to the standard grids, diffcenter applies to the differential grids.
    if gridcenter == 'True':
        gridcenter = (min_grid_val + max_grid_val)/2
    elif gridcenter == 0:
        gridcenter = 0
    else:
        gridcenter = None

    if diffcenter == 'True':
        diffcenter = (min_diff_val + max_diff_val)/2
    elif diffcenter == 0:
        diffcenter = 0
    else:
        diffcenter = None

    fig, axes = plt.subplots(2, 3, sharey=True)

    # Plot standard grids in the top row.
    for i, grid in enumerate(grids):
        sb.heatmap(grid, vmin=min_grid_val, vmax=max_grid_val, linewidth=0.5, xticklabels=False, ax=axes.flat[i], cmap=cmap, center=gridcenter)
        axes.flat[i].xaxis.set_tick_params(labelsize=med_size)

        axes.flat[i].invert_yaxis()
        axes.flat[i].set_title(labels[i], fontsize=big_size)
        axes.flat[i].yaxis.set_tick_params(labelsize=med_size)

        # Only include colorbar for last columns of tiles? Not currently implemented.
        cbar = axes.flat[i].collections[0].colorbar
        cbar.ax.tick_params(labelsize=med_size)

    # Plot differential grids in the bottom row.
    for j, diffgrid in enumerate(diffgrids):
        sb.heatmap(diffgrid, vmin=min_diff_val, vmax=max_diff_val, linewidth=0.5, xticklabels=xtick_freq, ax=axes.flat[3+j], cmap=diffcmap, center=diffcenter)
        axes.flat[3+j].xaxis.set_tick_params(labelsize=med_size)

        axes.flat[3+j].invert_yaxis()
        axes.flat[3+j].set_title(difflabels[j], fontsize=big_size)
        axes.flat[3+j].yaxis.set_tick_params(labelsize=med_size)
        
        # Only include colorbar for last columns of tiles? Not currently implemented.
        cbar = axes.flat[3+j].collections[0].colorbar
        cbar.ax.tick_params(labelsize=med_size)

    axes.flat[1] = False
    axes.flat[2] = False
    axes.flat[4] = False
    axes.flat[5] = False

    fig.text(0.5, 0.0, 'Closest approach distance (Earth radii)', ha='center', fontsize=med_size)
    fig.text(0.04, 0.5, 'Speed at infinity (km/s)', va='center', rotation='vertical', fontsize=med_size)
    fig.text(0.5, 0.95, title, ha='center', fontsize=big_size)

    #plt.show()
    return fig

# Similar to combo_grid(), but plots two different six panel figures. One with the low res version and one with the hi res.
# Color bar max and min values are calculated across all six regular tile plots and all six diff plots to facilitate comparison.
def dual_combo_grid(lowtitle, hititle, grids, diffgrids, cmap, diffcmap, labels, difflabels, gridcenter=None, diffcenter=None, big_size=32, med_size=28, xtick_freq=2, save=False):
    # Find max and min values across the grids.
    min_grid_val, max_grid_val = min_max_grid(grids)

    # Find max and min values across the diffgrids.
    min_diff_val, max_diff_val = min_max_grid(diffgrids)

    fig1 = combo_grid(title=lowtitle, grids=grids[:3], diffgrids=diffgrids[:3], cmap=cmap, diffcmap=diffcmap, labels=labels[:3], difflabels=difflabels[:3], gridcenter=gridcenter, diffcenter=diffcenter, big_size=big_size, med_size=med_size, xtick_freq=xtick_freq, grid_range=(min_grid_val, max_grid_val), diffgrid_range=(min_diff_val, max_diff_val))
    plt.show()

    fig2 = combo_grid(title=hititle, grids=grids[3:6], diffgrids=diffgrids[3:6], cmap=cmap, diffcmap=diffcmap, labels=labels[3:6], difflabels=difflabels[3:6], gridcenter=gridcenter, diffcenter=diffcenter, big_size=big_size, med_size=med_size, xtick_freq=xtick_freq, grid_range=(min_grid_val, max_grid_val), diffgrid_range=(min_diff_val, max_diff_val))
    plt.show()

def min_max_grid(grids):
    max_val = 0
    min_val = 1e100

    for grid in grids:
        current_max = np.nanmax(grid.values)
        current_min = np.nanmin(grid.values)

        if current_max > max_val:
            max_val = current_max

        if current_min < min_val:
            min_val = current_min

    return min_val, max_val
