# import modules
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import numpy as np


def print_elapsed(t0,task):
    print('Time spent on {} is {:.3f}s'.format(task,time.time()-t0))



# set up Nesta colours
nesta_colours= [[1, 184/255, 25/255],[1,0,65/255],[0,0,0],
    [1, 90/255,0],[155/255,0,195/255],[165/255, 148/255, 130/255],
[160/255,145/255,40/255],[196/255,176/255,0],
    [246/255,126/255,0],[200/255,40/255,146/255],[60/255,18/255,82/255]]

# combos, the lists are for:
# primaries, secondaries, bright combination, warm combination
# cool combination, neutral with accent colour combination,
# deep and accent colour combination
nesta_colours_combos = [[0,1,2,3,4,5],[0,6,7],[1,3,8],
                [4,9,10],[8,5],[1,11]]


def modify_legend(l = None, **kwargs):
    '''Note: this doesn't work '''
    
    import matplotlib as mpl

    if not l:
        l = plt.gca().legend_

    defaults = dict(
        loc = l._loc,
        numpoints = l.numpoints,
        markerscale = l.markerscale,
        scatterpoints = l.scatterpoints,
        scatteryoffsets = l._scatteryoffsets,
        prop = l.prop,
        # fontsize = None,
        borderpad = l.borderpad,
        labelspacing = l.labelspacing,
        handlelength = l.handlelength,
        handleheight = l.handleheight,
        handletextpad = l.handletextpad,
        borderaxespad = l.borderaxespad,
        columnspacing = l.columnspacing,
        ncol = l._ncol,
        mode = l._mode,
        fancybox = type(l.legendPatch.get_boxstyle())==mpl.patches.BoxStyle.Round,
        shadow = l.shadow,
        title = l.get_title().get_text() if l._legend_title_box.get_visible() else None,
        framealpha = l.get_frame().get_alpha(),
        bbox_to_anchor = l.get_bbox_to_anchor()._bbox,
        bbox_transform = l.get_bbox_to_anchor()._transform,
        frameon = l._drawFrame,
        handler_map = l._custom_handler_map,
    )

    if "fontsize" in kwargs and "prop" not in kwargs:
        defaults["prop"].set_size(kwargs["fontsize"])
    d = dict(defaults.items())
    d.update(kwargs.items())
    plt.legend(d) #**dict(defaults.items() + kwargs.items()))
