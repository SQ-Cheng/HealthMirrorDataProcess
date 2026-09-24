"""Shared subplot layouts for per-target experiment figures."""

import math


def target_grid_shape(panel_count):
    """Return rows/columns with stable layouts for the common 8/9-target runs."""
    if panel_count == 8:
        return 2, 4
    if panel_count == 9:
        return 3, 3
    columns = min(4, max(1, math.ceil(math.sqrt(panel_count))))
    return math.ceil(panel_count / columns), columns


def target_grid_figsize(rows, columns):
    return 4.6 * columns, 4.4 * rows
