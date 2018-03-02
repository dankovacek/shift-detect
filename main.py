#   Regularized Linear Regression and Bias-Variance
#   Applied to Streamflow Data

try:
    from functools import lru_cache
except ImportError:
    # Python 2 does stdlib does not have lru_cache so let's just
    # create a dummy decorator to avoid crashing
    print("WARNING: Cache for this example is available on Python 3 only.")

    def lru_cache():
        def dec(f):
            def _(*args, **kws):
                return f(*args, **kws)
            return _
        return dec

import bokeh
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models import ColumnDataSource, Range1d, LinearAxis, Spacer, Band
from bokeh.models import LassoSelectTool, BoxSelectTool, Legend, LegendItem
from bokeh.models import Label, CDSView, GroupFilter, HoverTool
from bokeh.models import HBar
from bokeh.models.widgets import PreText, Select, Slider, DateRangeSlider, CheckboxGroup
from bokeh.plotting import figure
from bokeh.palettes import Spectral11, Viridis11, Paired12

from helper_functions import load_data
from helper_functions import linearRegCostFunction, getTestAndTrainingSets
from helper_functions import apply_poly, getIntialDataSets

# Define data folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

start_time = time.time()

# set up data sources for bokeh plotting
source = ColumnDataSource(data=dict())
source_static = ColumnDataSource(data=dict())

results_text = PreText(text="", width=250, height=90)

# initialize a colour pallette for plotting multiple lines
mypalette = Spectral11

VARS = {}
# global variables for series labels and constants


@lru_cache()
def get_data(s1, s2):
    df1 = load_data(VARS['stations'][s1]['file_path'], s1)
    df2 = load_data(VARS['stations'][s2]['file_path'], s2)
    # load all data and concatenate series for concurrent days
    all_data = pd.concat([df1, df2], axis=1, join='inner')

    all_data.reset_index(inplace=True)
    column_headers = all_data.columns.values

    s1_stage_header = [
        e for e in column_headers if 'Stage_' + VARS['s1_name'] in e]
    s2_stage_header = [
        e for e in column_headers if 'Stage_' + VARS['s2_name'] in e]

    print(all_data.head())

    all_data['stage_diff'] = all_data[s1_stage_header[0]] - \
        all_data[s2_stage_header[0]]

    all_data['normalized_stage_diff'] = all_data['normalized_stage_' +
                                                 VARS['s1_name']] - all_data['normalized_stage_' + VARS['s2_name']]

    return all_data


def update_time_range(attrname, old, new):
    new_range_start, new_range_end = [pd.to_datetime(
        e) for e in date_selector.value_as_datetime]

    VARS['time_range'] = (new_range_start, new_range_end)

    data = get_data_subset(VARS['time_range'])
    source.data = source.from_df(data)
    source_static.data = source.data


def get_data_subset(time_range):
    df = get_data(*VARS['stations'])
    df = df[(df['DateTime'] >= pd.to_datetime(time_range[0]))
            & (df['DateTime'] <= pd.to_datetime(time_range[1]))]

    return df


def update():

    files = []
    VARS['stations'] = {}
    for path in Path(DATA_DIR).glob('**/*.csv'):
        full_filename = str(path).split('/')[-1]
        data_type = full_filename.split('.')[0]
        interval = full_filename.split('.')[1].split('@')[0]
        station_name = full_filename.split('.')[1].split('@')[1]
        file_date = full_filename.split('.')[2]
        if station_name not in VARS['stations'].keys():
            VARS['stations'][station_name] = {}
            VARS['stations'][station_name]['file_date'] = file_date
            VARS['stations'][station_name]['interval'] = interval
            VARS['stations'][station_name]['data_type'] = data_type
            VARS['stations'][station_name]['file_path'] = path

    if len(VARS['stations'].keys()) == 2:
        s1_name = list(VARS['stations'].keys())[0]
        s2_name = list(VARS['stations'].keys())[1]
        s1 = VARS['stations'][s1_name]
        s2 = VARS['stations'][s2_name]
        VARS['s1_name'] = s1_name
        VARS['s2_name'] = s2_name

    df = get_data(*VARS['stations'])

    VARS['poly_order'] = 1
    VARS['global_end'] = pd.to_datetime(df['DateTime'].values[-1])
    VARS['global_start'] = pd.to_datetime(df['DateTime'].values[0])
    VARS['time_range'] = (VARS['global_start'], VARS['global_end'])

    source.data = source.from_df(df)
    source_static.data = source.data



#############################
# Initialize interactive widgets before calling first update
#############################
update()

date_selector = DateRangeSlider(title="Select Timeframe",
                                end=VARS['global_end'],
                                start=VARS['global_start'],
                                step=1,
                                value=(
                                    VARS['time_range'][0],
                                    VARS['time_range'][1],
                                )
                                )
date_selector.on_change('value', update_time_range)


##################################################################
# Bokeh Plots
##################################################################
TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset"

##########
# Stage Time Series
##########

hydrograph = figure(plot_width=1000, plot_height=350,
                    tools=TOOLS + ',box_zoom,hover', toolbar_location="above",
                    toolbar_sticky=False, x_axis_type="datetime")


hydrograph.title.text = 'Stage Difference at {} and {}'.format(
    VARS['s1_name'], VARS['s2_name'])
hydrograph.xaxis.axis_label = 'Date'
hydrograph.yaxis.axis_label = 'Difference in Stage [m]'
hydrograph.xaxis.axis_label_text_font_size = '12pt'
hydrograph.yaxis.axis_label_text_font_size = '12pt'
hydrograph.xaxis.major_label_text_font_size = '10pt'
hydrograph.yaxis.major_label_text_font_size = '10pt'


# TRAINING DATA SERIES (EXPLORATORY SET INPUT)
hydrograph.line('DateTime', 'stage_diff', line_width=2,
                color=mypalette[7], source=source)

# Normalized Stage Figure

normalized_stage_fig = figure(plot_width=800, plot_height=350,
                              tools=TOOLS + ',box_zoom,hover', toolbar_location="above",
                              toolbar_sticky=False, x_axis_type="datetime")

normalized_stage_fig.title.text = 'Normalized Stage'
normalized_stage_fig.xaxis.axis_label = 'Date'
normalized_stage_fig.yaxis.axis_label = 'Normalized Stage [-]'
normalized_stage_fig.xaxis.axis_label_text_font_size = '12pt'
normalized_stage_fig.yaxis.axis_label_text_font_size = '12pt'
normalized_stage_fig.xaxis.major_label_text_font_size = '10pt'
normalized_stage_fig.yaxis.major_label_text_font_size = '10pt'


# TRAINING DATA SERIES (EXPLORATORY SET INPUT)
normalized_stage_fig.line('DateTime', 'normalized_stage_{}'.format(VARS['s1_name']), line_width=2,
                          color=mypalette[7], source=source, legend=VARS['s1_name'])

normalized_stage_fig.line('DateTime', 'normalized_stage_{}'.format(VARS['s2_name']), line_width=2,
                          color=mypalette[8], source=source, legend=VARS['s2_name'])

# Normalized Stage Vs. Stage Difference

normalized_stage_correlation = figure(plot_width=400, plot_height=350,
                                      tools=TOOLS + ',box_zoom,hover', toolbar_location="above",
                                      toolbar_sticky=False)

normalized_stage_correlation.title.text = 'Normalized Stage at {} vs. Stage Difference'.format(
    VARS['s1_name'])
normalized_stage_correlation.xaxis.axis_label = 'Normalized Stage at {}'.format(
    VARS['s1_name'])
normalized_stage_correlation.yaxis.axis_label = 'Normalized Stage Difference [-]'
normalized_stage_correlation.xaxis.axis_label_text_font_size = '12pt'
normalized_stage_correlation.yaxis.axis_label_text_font_size = '12pt'
normalized_stage_correlation.xaxis.major_label_text_font_size = '10pt'
normalized_stage_correlation.yaxis.major_label_text_font_size = '10pt'


# TRAINING DATA SERIES (EXPLORATORY SET INPUT)
normalized_stage_correlation.circle('normalized_stage_{}'.format(VARS['s1_name']), 'normalized_stage_diff',
                                    color=mypalette[3], source=source, legend=VARS['s1_name'] + ' vs. difference')


# Normalized Stage Difference Figure

normalized_stage_diff_fig = figure(plot_width=800, plot_height=350,
                                   tools=TOOLS + ',box_zoom,hover', toolbar_location="above",
                                   toolbar_sticky=False, x_axis_type="datetime")

normalized_stage_diff_fig.title.text = 'Normalized Stage Difference'
normalized_stage_diff_fig.xaxis.axis_label = 'Date'
normalized_stage_diff_fig.yaxis.axis_label = 'Normalized Stage Difference [-]'
normalized_stage_diff_fig.xaxis.axis_label_text_font_size = '12pt'
normalized_stage_diff_fig.yaxis.axis_label_text_font_size = '12pt'
normalized_stage_diff_fig.xaxis.major_label_text_font_size = '10pt'
normalized_stage_diff_fig.yaxis.major_label_text_font_size = '10pt'


# TRAINING DATA SERIES (EXPLORATORY SET INPUT)
normalized_stage_diff_fig.line('DateTime', 'normalized_stage_diff', line_width=2,
                               color=mypalette[2], source=source, legend=VARS['s1_name'] + ' minus ' + VARS['s2_name'])

##################################################################
# Bokeh Interactive Widgets
##################################################################


##################################################################
# Layout of UI Elements
##################################################################

# results_box = column(siteSelector1,
#                      siteSelector2,
#                      widgetbox(poly_order_slider, width=250),
#                      flag_selector_group,
#                      results_text
#                      )

layout = column(
    # row(cp_training_fig),
    # row(widgetbox(training_set_date_selector, width=400),
    #     widgetbox(validation_set_date_selector, width=400),
    #     widgetbox(test_set_date_selector, width=400),
    #     ),
    row(normalized_stage_fig, normalized_stage_correlation),
    widgetbox(date_selector, width=900),
    row(normalized_stage_diff_fig)
)


curdoc().add_root(layout)
curdoc().title = "RC Shift Detection"
