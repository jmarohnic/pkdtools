##################################################################
# dregion.py
# Julian C. Marohnic
# Created: 11/5/23
#
# A set of functions for getting statistics on the "disruption
# region." Requires a previously calculated datagrid to operate
# on.
##################################################################

import numpy as np
import pandas as pd

from .pgrid import load_csv_grid

# Get the coordinates in q, v_inf space of the tidal encounters whose grid
# values exceed (or falls below, depending on sign) limit_val.
def get_coords(grid, limit_val, sign):
    # Establish empty list of (q, vinf) coordinates.
    coord_list = []
    columns = [column[:][1] for column in grid.items()]

    # Loop over columns
    for col in columns:
        # Column name is q value
        q = col.name
        # Loop over rows in each column, working top to bottom.
        for row in col[::-1].iteritems():
            # row here is a tuple. The 0th element is vinf. The 1st element is the datagrid value.
            vinf = int(row[0])
            val = row[1]

            # If we meet the condition, record this coordinate and move on to the next column.
            if sign == 'gt':
                if val > limit_val:
                    coord_list.append((q, vinf))
                    break
            elif sign == 'lt':
                if val < limit_val:
                    coord_list.append((q, vinf))
                    break
            else:
                raise ValueError("Error: valid sign arguments are 'lt' and 'gt'.")

    return coord_list

# Analyze the region of the grid defined by the list of coordinates given. func should
# accept a list of values. Usually will be something like average or median. E.g.,
# find the average value of grid on the grid cells in columns below the given points.
def analyze_region(grid, coords, func):
    val_list = []
    for coord in coords:
        q = coord[0]
        vinf = coord[1]
        # Pick out the column we want.
        col = grid.loc[:, str(q)]
        #col = grid.loc[:, q]

        # Loop through the rows in the current column. If the vinf value is less
        # than or equal to the value in our current coordinate, add the value to our list.
        # If not, break the current loop and move to the next coordinate.
        for row in col.iteritems():
            if int(row[0]) <= vinf:
                val_list.append(row[1])
            else:
                break

    return func(val_list)
