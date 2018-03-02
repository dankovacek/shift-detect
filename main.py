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
from helper_functions import gradientDescent, featureNormalize, targetNormalize
from helper_functions import learningCurve, polyFeatures, estimateGaussian
from helper_functions import map_theta_to_features, multivariateGaussian
from helper_functions import apply_poly, getIntialDataSets
from helper_functions import SITES, Site

# Define data folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

start_time = time.time()

# set up data sources for bokeh plotting
source = ColumnDataSource(data=dict())
source_static = ColumnDataSource(data=dict())
cp_train_source = ColumnDataSource(data=dict())
val_source = ColumnDataSource(data=dict())
test_source = ColumnDataSource(data=dict())
efp_train_source = ColumnDataSource(data=dict())
efp_val_source = ColumnDataSource(data=dict())
efp_test_source = ColumnDataSource(data=dict())


cp_training_regression_source = ColumnDataSource(data=dict())
efp_training_regression_source = ColumnDataSource(data=dict())
equal_ur_source = ColumnDataSource(data=dict())
output_regression_source = ColumnDataSource(data=dict())
model_performance_source = ColumnDataSource(data=dict())
model_best_fit_source = ColumnDataSource(data=dict())
lag_source = ColumnDataSource(data=dict())

results_text = PreText(text="", width=250, height=90)

# initialize a colour pallette for plotting multiple lines
mypalette = Spectral11


def nix(val, lst):
    return [x for x in lst if x != val]


# global variables for series labels and constants
DEFAULT_SELECTIONS = ['lillooet', 'squamish', 'elaho', 'stave']
VARS = {}


@lru_cache()
def get_data(s1, s2):

    site1 = Site(SITES[s1])
    site1['classifier'] = 'proxy'
    site2 = Site(SITES[s2])
    site2['classifier'] = 'target'

    df1 = load_data(**site1)
    df2 = load_data(**site2)
    # load all data and concatenate series for concurrent days
    all_data = pd.concat([df1, df2], axis=1, join='inner')

    # add a 'day of year' column and normalize to be between 1 and 365
    all_data['day_of_year'] = all_data.index.dayofyear

    # set column for Xti and Xti+1
    # all_data['proxy_ur_t1'] = all_data['daily_ur_proxy'].shift()
    # all_data['target_ur_t1'] = all_data['daily_ur_target'].shift()

    # unit runoff
    ur_s1 = all_data['daily_ur_target']
    ur_s2 = all_data['daily_ur_proxy']

    # unit runoff ratio
    all_data['ur_ratio'] = ur_s1.divide(ur_s2).astype(float)

    all_data.reset_index(inplace=True)

    return all_data


def set_initial_dataset_labels(data):

    VARS['global_start'] = data['Date'].iloc[0] - pd.to_timedelta(1, unit='d')
    VARS['global_end'] = data['Date'].iloc[-1] + pd.to_timedelta(10, unit='d')

    record_start = VARS['global_start'].to_datetime()
    record_end = VARS['global_end'].to_datetime()

    timespan = record_end.year - record_start.year

    if timespan < 2:
        print('Insufficient Record for Analysis')

    elif timespan < 6:
        train_start = record_start + pd.to_timedelta(2, unit='d')
        train_end = train_start + pd.to_timedelta(6, unit='M')
        train_start, train_end = (
            record_start, record_start + pd.to_timedelta(6, unit='M'))
        val_start = train_end + pd.to_timedelta(1, unit='d')
        val_end = val_start + pd.to_timedelta(6, unit='M')
        test_start = val_end + pd.to_timedelta(1, unit='d')
        test_end = val_end + pd.to_timedelta(6, unit='M')
    else:
        test_start = record_end - pd.to_timedelta(12, unit='M')
        test_end = record_end - pd.to_timedelta(2, unit='d')
        val_end = test_start - pd.to_timedelta(1, unit='d')
        val_start = val_end - pd.to_timedelta(12, unit='M')
        train_end = val_start - pd.to_timedelta(6, unit='M')
        train_start = train_end - pd.to_timedelta(12, unit='M')

    VARS['sets'] = {}
    VARS['sets']['train'] = (train_start,  train_end)
    VARS['sets']['validation'] = (val_start, val_end)
    VARS['sets']['test'] = (test_start, test_end)

    #data.set_index('Date', inplace=True)

    data = update_dataset_labels(data)

    return data


def update_dataset_labels(df):
    start_relabel = time.time()
    train_start, train_end = VARS['sets']['train']
    val_start, val_end = VARS['sets']['validation']
    test_start, test_end = VARS['sets']['test']

    # add classification label to dataframe
    # need intermediate categories to be able to leave
    # space between series in the hydrograph

    c = pd.cut(
        df['Date'],
        [VARS['global_start'],
         train_start, train_end,
         val_start, val_end,
         test_start, test_end,
         VARS['global_end']],
        labels=['pre-record', 'train', 'inter_1',
                'validation', 'inter_2', 'test', 'future']
    )

    df.loc[:, 'set_name'] = c

    return df


########################################
#  Functions for OLS best-fit
########################################

def update_validation():
    regression_start = time.time()

    val_data = get_data_subset(VARS['sets']['validation'])

    model_best_fit_df = pd.DataFrame()

    p = VARS['poly_order']

    cp_label = VARS['cp_predicted_series_label']
    efp_label = VARS['efp_predicted_series_label']

    val_data.loc[:, cp_label] = [apply_poly(
        VARS['cp_training_fit'], e) for e in val_data['daily_ur_proxy']]

    val_data.loc[:, efp_label] = np.interp(x=val_data['daily_ur_proxy'].values,
                                           xp=val_data['daily_ur_proxy_sorted'].values,
                                           fp=val_data['daily_ur_target_sorted'])

    # prevent negative flow being calculated
    val_data.loc[:, cp_label] = val_data[cp_label].clip(lower=0)

    max_runoff = float(
        (val_data[cp_label].max() +
         val_data['daily_ur_target'].max()) / 2
    )

    max_proxy_runoff = val_data['daily_ur_proxy'].max()

    model_best_fit_df['best_fit_domain'] = np.linspace(
        0, max_proxy_runoff, 200)

    # use linear regression, not p for polynomial
    cp_model_fit = np.polyfit(
        val_data[cp_label],
        val_data['daily_ur_target'], 1)[:: -1]

    efp_model_fit = np.polyfit(
        val_data[efp_label],
        val_data['daily_ur_target'], 1)[:: -1]

    model_performance_source.data = model_performance_source.from_df(
        val_data)

    model_best_fit_df.loc[:, 'cp_best_fit_range'] = [
        apply_poly(cp_model_fit, e) for e in model_best_fit_df.loc[:, 'best_fit_domain']]

    model_best_fit_df.loc[:, 'efp_best_fit_range'] = [
        apply_poly(efp_model_fit, e) for e in model_best_fit_df.loc[:, 'best_fit_domain']]

    model_best_fit_source.data = model_best_fit_source.from_df(
        model_best_fit_df)

    cp_np_rsquared = np.corrcoef(np.array(val_data[cp_label].astype(float)),
                                 np.array(val_data['daily_ur_target'].astype(float)))[0, 1]**2

    efp_np_rsquared = np.corrcoef(np.array(val_data[efp_label].astype(float)),
                                  np.array(val_data['daily_ur_target'].astype(float)))[0, 1]**2

    results_text.text += "CP Val. R^2 = {:.2f}\n".format(cp_np_rsquared)
    results_text.text += "EFP Val. R^2 = {:.2f}".format(efp_np_rsquared)


def update_cp_training():
    """
    Update the training set regression and model output.
    """
    regression_start = time.time()
    model_best_fit_df = pd.DataFrame()

    train_data = get_data_subset(VARS['sets']['train'])

    p = VARS['poly_order']

    # find best fit curve equation for plotting on training fig
    current_fit = np.polyfit(
        train_data['daily_ur_proxy'],
        train_data['daily_ur_target'], p)[:: -1]

    VARS['cp_training_fit'] = current_fit

    max_proxy_runoff = train_data['daily_ur_proxy'].max()

    model_best_fit_df['best_fit_domain'] = np.linspace(
        0, max_proxy_runoff, 200)

    # regression plot of modeled and measured data
    # get the equation for best fit line

    model_best_fit_df.loc[:, 'cp_best_fit_range'] = [
        apply_poly(current_fit, e) for e in model_best_fit_df.loc[:, 'best_fit_domain']]

    # second check on training fit
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        train_data['daily_ur_proxy'],
        train_data['daily_ur_target'])

    np_rsquared = np.corrcoef(train_data['daily_ur_proxy'],
                              train_data['daily_ur_target'])[0, 1]**2

    # plot the best-fit curve over the ur_ratio vs. normalized day scatter plot
    model_best_fit_df['train_best_fit'] = [apply_poly(
        current_fit, day) for day in model_best_fit_df['best_fit_domain']]

    s1_da = SITES[VARS['proxy'].name]['DA']
    s2_da = SITES[VARS['target'].name]['DA']
    DA_ratio = s1_da / s2_da

    model_best_fit_df.loc[:, 'equal_ur'] = [DA_ratio *
                                            e for e in model_best_fit_df.loc[:, 'best_fit_domain']]

    cp_training_regression_source.data = cp_training_regression_source.from_df(
        model_best_fit_df)

    max_train_ur_proxy = train_data['daily_ur_proxy'].max()
    max_train_ur_target = train_data['daily_ur_target'].max()
    min_train_ur_proxy = train_data['daily_ur_proxy'].min()
    min_train_ur_target = train_data['daily_ur_target'].min()

    results_text.text = "Flags: E = estimated;\n B = icing conditions\n"
    results_text.text += "\nTraining R^2={:.2f}\n".format(
        np_rsquared)
    results_text.text += "For polynomial order (p={})\n".format(p)
    results_text.text += "Trainig R^2={:.2f}\n".format(
        r_value**2)

    # print('time to run update regression iteration = ',
    #       time.time() - regression_start)


def update_efp_training():
    """
    Update the training set regression and model output.
    """
    regression_start = time.time()
    model_best_fit_df = pd.DataFrame()

    train_data = get_data_subset(VARS['sets']['train'])

    p = VARS['poly_order']

    # find best fit curve equation for plotting on training fig
    current_fit = np.polyfit(
        train_data['daily_ur_proxy'],
        train_data['daily_ur_target'], p)[:: -1]

    VARS['efp_training_fit'] = current_fit

    max_proxy_runoff = train_data['daily_ur_proxy'].max()

    model_best_fit_df['best_fit_domain'] = np.linspace(
        0, max_proxy_runoff, 200)

    efp_training_regression_source.data = efp_training_regression_source.from_df(
        model_best_fit_df)

    # print('time to run update regression iteration = ',
    #       time.time() - regression_start)


def acf(series):
    """
    Returns coefficients for autocorrelation function.
    """
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) *
                   (data[h:] - mean)).sum() / float(n) / c0
        return round(acf_lag, 3)
    x = np.arange(n)  # Avoiding lag 0 calculation
    acf_coeffs = pd.Series(map(r, x)).round(decimals=3)
    acf_coeffs = acf_coeffs + 0
    return acf_coeffs


def significance(series):
    n = len(series)
    z95 = 1.959963984540054 / np.sqrt(n)
    z99 = 2.5758293035489004 / np.sqrt(n)
    return(z95, z99)


def bok_autocor(series):
    """
    Takes in a series, returns a figure object representing
    the autocorrelation.
    """
    x = pd.Series(range(1, len(series) + 1), dtype=float)
    z95, z99 = significance(series)
    y = acf(series)
    p = figure(title='Time Series Auto-Correlation', plot_width=1000,
               plot_height=500, x_axis_label="Lag", y_axis_label="Autocorrelation")
    p.line(x, z99, line_dash='dashed', line_color='grey')
    p.line(x, z95, line_color='grey')
    p.line(x, y=0.0, line_color='black')
    p.line(x, z99 * -1, line_dash='dashed', line_color='grey')
    p.line(x, z95 * -1, line_color='grey')
    p.line(x, y, line_width=2)
    return p


def update_poly_slider(attrname, old, new):
    VARS['poly_order'] = new
    update_cp_training()
    update_efp_training()
    update_validation()


def update_time_ranges():
    old_train_start, old_train_end = VARS['sets']['train']
    old_val_start, old_val_end = VARS['sets']['validation']
    old_test_start, old_test_end = VARS['sets']['test']

    new_train_start, new_train_end = [pd.to_datetime(
        e) for e in training_set_date_selector.value_as_datetime]
    new_val_start, new_val_end = [pd.to_datetime(
        e) for e in validation_set_date_selector.value_as_datetime]
    new_test_start, new_test_end = [pd.to_datetime(
        e) for e in test_set_date_selector.value_as_datetime]

    if new_train_end >= new_val_start:
        new_train_end = new_val_start - pd.to_timedelta(1, unit='d')
    if new_val_start <= new_train_end:
        new_val_start = new_train_end + pd.to_timedelta(1, unit='d')
    if new_val_end >= new_test_start:
        new_val_end = new_test_start - pd.to_timedelta(1, unit='d')
    if new_test_start <= new_val_end:
        new_test_start = new_val_end + pd.to_timedelta(1, unit='d')

    VARS['sets']['train'] = (new_train_start, new_train_end)
    VARS['sets']['validation'] = (new_val_start, new_val_end)
    VARS['sets']['test'] = (new_test_start, new_test_end)

    # reset extent of slider ranges to allow greater flexibility in range selection
    training_set_date_selector.start = VARS['global_start']
    training_set_date_selector.end = new_val_start - \
        pd.to_timedelta(1, unit='d')
    validation_set_date_selector.start = new_train_end + \
        pd.to_timedelta(1, unit='d')
    validation_set_date_selector.end = new_test_start - \
        pd.to_timedelta(1, unit='d')
    test_set_date_selector.start = new_val_end + pd.to_timedelta(1, unit='d')
    test_set_date_selector.end = VARS['global_end']


def get_data_subset(time_range):
    s1, s2 = siteSelector1.value, siteSelector2.value
    df = get_data(s1, s2)
    df = df[(df['Date'] >= pd.to_datetime(time_range[0]))
            & (df['Date'] <= pd.to_datetime(time_range[1]))]

    df['daily_ur_proxy_sorted'] = np.sort(df['daily_ur_proxy'].values)
    df['daily_ur_target_sorted'] = np.sort(df['daily_ur_target'].values)
    return df


def update_train_set(attrname, old, new):
    update_time_ranges()
    train_data = get_data_subset(VARS['sets']['train'])
    train_data = check_flags(train_data)
    cp_train_source.data = cp_train_source.from_df(train_data)
    efp_train_data = train_data.sort_values(by=['daily_ur_proxy'])
    efp_train_source.data = cp_train_source.from_df(efp_train_data)

    update_cp_training()
    update_efp_training()
    update_validation()


def update_val_set(attrname, old, new):
    update_time_ranges()

    val_data = get_data_subset(VARS['sets']['validation'])
    val_data = check_flags(val_data)

    val_source.data = val_source.from_df(val_data)

    update_cp_training()
    update_efp_training()
    update_validation()


def update_test_set(attrname, old, new):
    update_time_ranges()

    test_data = get_data_subset(VARS['sets']['test'])
    test_data = check_flags(test_data)

    test_source.data = test_source.from_df(test_data)

    update_cp_training()
    update_efp_training()
    update_validation()


def siteSelector1_change(attrname, old, new):
    siteSelector2.options = nix(new, DEFAULT_SELECTIONS)
    VARS['proxy'] = Site(SITES[new])
    update()


def siteSelector2_change(attrname, old, new):
    siteSelector1.options = nix(new, DEFAULT_SELECTIONS)
    VARS['target'] = Site(SITES[new])
    update()


def flag_selector_change(attrname, old, new):
    update_train_set(attrname, old, new)
    update_val_set(attrname, old, new)
    update_test_set(attrname, old, new)

    # something's busted here...
    # shouldn't be resetting date range on flag selector change

    # update_time_ranges()


def check_flags(df):
    # check flag filters and remove if selectors are untoggled
    active_flags = flag_selector_group.active

    if 0 not in active_flags:
        df = df[(df['proxy_flag'] != 'E') & (
            df['target_flag'] != 'E')]
    if 1 not in active_flags:
        df = df[(df['proxy_flag'] != 'B') & (
            df['target_flag'] != 'B')]
    return df


def update(selected=None):
    s1, s2 = siteSelector1.value, siteSelector2.value

    VARS['poly_order'] = 1
    VARS['proxy'] = Site(SITES[s1])
    VARS['target'] = Site(SITES[s2])
    VARS['cp_predicted_series_label'] = 'cp_predicted_target'
    VARS['efp_predicted_series_label'] = 'efp_predicted_target'

    df = get_data(s1, s2)

    VARS['len_dataset'] = len(df)

    VARS['max_ur'] = max(df['daily_ur_proxy'].max(),
                         df['daily_ur_target'].max())
    VARS['min_ur'] = max(df['daily_ur_proxy'].min(),
                         df['daily_ur_target'].min())

    df = set_initial_dataset_labels(df)

    df = check_flags(df)

    df['daily_ur_proxy_sorted'] = np.sort(df['daily_ur_proxy'].values)
    df['daily_ur_target_sorted'] = np.sort(df['daily_ur_target'].values)

    # df.reset_index(inplace=True)

    source.data = source.from_df(df)

    cp_train_source.data = cp_train_source.from_df(
        df[df['set_name'] == 'train'])
    val_source.data = val_source.from_df(df[df['set_name'] == 'validation'])
    test_source.data = test_source.from_df(df[df['set_name'] == 'test'])

    VARS['acf'] = bok_autocor(df['daily_ur_proxy'].values)

    update_cp_training()
    update_efp_training()
    update_validation()


#############################
# Initialize interactive widgets before calling first update
#############################


siteSelector1 = Select(title="Select Proxy Station:",
                       value='squamish', options=nix('lillooet', DEFAULT_SELECTIONS))
siteSelector2 = Select(title="Select Target Station:",
                       value='lillooet', options=nix('squamish', DEFAULT_SELECTIONS))

flag_selector_group = CheckboxGroup(
    labels=["Include E Flag", "Include B Flag"], active=[0, 1])

update()

##################################################################
# Bokeh Plots
##################################################################
TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset"

# test fig is the plot of daily UR ratio by day of the year
cp_training_fig = figure(plot_width=450, plot_height=400,
                         tools=TOOLS + ',box_zoom,hover')

cp_training_fig.title.text = 'Chronological Pairing (Training Set)'
cp_training_fig.xaxis.axis_label = 'Proxy Daily Avg. UR [L/s/km^2]'
cp_training_fig.yaxis.axis_label = 'Target Daily Avg. UR [L/s/km^2]'
cp_training_fig.xaxis.axis_label_text_font_size = '10pt'
cp_training_fig.yaxis.axis_label_text_font_size = '10pt'
cp_training_fig.xaxis.major_label_text_font_size = '10pt'
cp_training_fig.yaxis.major_label_text_font_size = '10pt'

cp_training_fig.scatter('daily_ur_proxy',
                        'daily_ur_target',
                        source=cp_train_source, legend="CP UR",
                        size=3, color=mypalette[7], alpha=0.4)

cp_training_fig.line('best_fit_domain', 'equal_ur', source=cp_training_regression_source, legend="Equal UR",
                     line_dash='dashed', line_color=mypalette[8], line_width=3)


cp_training_fig.line('best_fit_domain', 'train_best_fit', source=cp_training_regression_source,
                     legend='model', line_color=mypalette[2], line_width=3)


flag_view_proxyB = CDSView(source=cp_train_source, filters=[
    GroupFilter(column_name='proxy_flag', group='B')
])
flag_view_proxyE = CDSView(source=cp_train_source, filters=[
    GroupFilter(column_name='proxy_flag', group='E'),
])
flag_view_targetB = CDSView(source=cp_train_source, filters=[
    GroupFilter(column_name='target_flag', group='B'),
])
flag_view_targetE = CDSView(source=cp_train_source, filters=[
    GroupFilter(column_name='target_flag', group='E'),
])

cp_training_fig.circle('daily_ur_proxy',
                       'daily_ur_target',
                       legend="FLAG", size=3,
                       color='red', alpha=0.4,
                       source=cp_train_source, view=flag_view_proxyB)

cp_training_fig.circle('daily_ur_proxy',
                       'daily_ur_target',
                       size=3,
                       color='red', alpha=0.4,
                       source=cp_train_source, view=flag_view_proxyE)

cp_training_fig.circle('daily_ur_proxy',
                       'daily_ur_target',
                       size=3,
                       color='red', alpha=0.4,
                       source=cp_train_source, view=flag_view_targetE)

cp_training_fig.circle('daily_ur_proxy',
                       'daily_ur_target',
                       size=3,
                       color='red', alpha=0.4,
                       source=cp_train_source, view=flag_view_targetB)

cp_training_fig.select_one(HoverTool).tooltips = [
    ('Proxy Data Flag', '@proxy_flag'),
    ('Target Data Flag', '@target_flag'),
]

x_label = 'num_examples'
ytrain_label = 'training_error'
yval_error_label = 'val_error'

#######################################
# EFP Figure
# 3

# test fig is the plot of daily UR ratio by day of the year
efp_training_fig = figure(plot_width=450, plot_height=400,
                          tools=TOOLS + ',box_zoom,hover')

efp_training_fig.title.text = 'Empirical Frequency Pairing (Training Set)'
efp_training_fig.xaxis.axis_label = 'Proxy Daily Avg. UR [L/s/km^2]'
efp_training_fig.yaxis.axis_label = 'Target Daily Avg. UR [L/s/km^2]'
efp_training_fig.xaxis.axis_label_text_font_size = '10pt'
efp_training_fig.yaxis.axis_label_text_font_size = '10pt'
efp_training_fig.xaxis.major_label_text_font_size = '10pt'
efp_training_fig.yaxis.major_label_text_font_size = '10pt'


efp_training_fig.scatter('daily_ur_proxy_sorted',
                         'daily_ur_target_sorted',
                         source=cp_train_source, legend="EFP UR",
                         size=3, color=mypalette[0], alpha=0.8)

efp_training_fig.line('best_fit_domain', 'equal_ur', source=cp_training_regression_source, legend="Equal UR",
                      line_dash='dashed', line_color=mypalette[8], line_width=3)

###################################
# Model Output Figure (CV Regression)
###################################

model_output_fig = figure(
    plot_width=400, plot_height=350, tools=TOOLS + ',box_zoom')

model_output_fig.circle('daily_ur_target',
                        'cp_predicted_target',
                        source=model_performance_source, size=3,
                        legend="CP Model",
                        color=mypalette[2], selection_color="orange",
                        alpha=0.5, nonselection_alpha=0.01, selection_alpha=0.2)

model_output_fig.line('best_fit_domain', 'cp_best_fit_range', source=model_best_fit_source,
                      legend="CP Fit", color=mypalette[2], line_dash='dashed', line_width=3)

model_output_fig.circle('daily_ur_target',
                        'efp_predicted_target',
                        source=model_performance_source, size=3,
                        legend="EFP Model",
                        color=mypalette[7], selection_color="orange",
                        alpha=0.5, nonselection_alpha=0.01, selection_alpha=0.2)

model_output_fig.line('best_fit_domain', 'efp_best_fit_range', source=model_best_fit_source,
                      legend="EFP Fit", color=mypalette[7], line_dash='dashed', line_width=3)

model_output_fig.line('best_fit_domain', 'best_fit_domain', source=model_best_fit_source,
                      legend="Perfect Fit", color=mypalette[8], line_dash='dashed', line_width=3,
                      line_alpha=0.85)


model_output_fig.title.text = 'Model: Measured vs. Predicted {} Daily UR [L/s/km^2]'.format(
    VARS['target'].name)
model_output_fig.xaxis.axis_label = 'Measured'
model_output_fig.yaxis.axis_label = 'Predicted'
model_output_fig.xaxis.axis_label_text_font_size = '10pt'
model_output_fig.yaxis.axis_label_text_font_size = '10pt'
model_output_fig.xaxis.major_label_text_font_size = '10pt'
model_output_fig.yaxis.major_label_text_font_size = '10pt'

model_output_fig.legend.click_policy = "hide"

##########
# Daily UR Hydrograph
##########

hydrograph = figure(plot_width=1000, plot_height=350,
                    tools=TOOLS + ',box_zoom,hover', toolbar_location="above",
                    toolbar_sticky=False, x_axis_type="datetime")


hydrograph.title.text = 'Measured Target and Proxy by Test Set, with Model Output'
hydrograph.xaxis.axis_label = 'Date'
hydrograph.yaxis.axis_label = 'Daily Avg. UR [L/s/km^2]'
hydrograph.xaxis.axis_label_text_font_size = '10pt'
hydrograph.yaxis.axis_label_text_font_size = '10pt'
hydrograph.xaxis.major_label_text_font_size = '10pt'
hydrograph.yaxis.major_label_text_font_size = '10pt'


# TRAINING DATA SERIES (EXPLORATORY SET INPUT)
proxy_measured_train = hydrograph.line('Date', 'daily_ur_proxy', line_width=2,
                                       color=mypalette[7], source=cp_train_source)
target_measured_train = hydrograph.line('Date', 'daily_ur_target', line_width=2,
                                        color=mypalette[6], source=cp_train_source)

est_flag = hydrograph.line('Date', 'proxy_flag_val', color=mypalette[-1],
                           line_width=5, source=cp_train_source, view=flag_view_proxyE)
hydrograph.line('Date', 'proxy_flag_val', color=mypalette[-1],
                line_width=5, source=cp_train_source, view=flag_view_proxyB)
hydrograph.line('Date', 'target_flag_val', color=mypalette[-1],
                line_width=5, source=cp_train_source, view=flag_view_targetB)
hydrograph.line('Date', 'target_flag_val', color=mypalette[-1],
                line_width=5, source=cp_train_source, view=flag_view_targetE)

hydrograph.select_one(HoverTool).tooltips = [
    ('Proxy Data Flag', '@proxy_flag'),
    ('Target Data Flag', '@target_flag'),
]

proxy_measured_validate = hydrograph.line('Date', 'daily_ur_proxy', line_width=2,
                                          color=mypalette[3], source=val_source)  # , view=hydrograph_val_view)
target_measured_validate = hydrograph.line('Date', 'daily_ur_target', line_width=2,
                                           color=mypalette[2], source=val_source)  # , view=hydrograph_val_view)

proxy_measured_test = hydrograph.line('Date', 'daily_ur_proxy', line_width=2,
                                      color=mypalette[1], source=test_source)  # , view=hydrograph_test_view)
target_measured_test = hydrograph.line('Date', 'daily_ur_target', line_width=2,
                                       color=mypalette[0], source=test_source)  # , view=hydrograph_test_view)


# MODEL OUTPUT SERIES (Validation Set)
cp_output_validation = hydrograph.line('Date', VARS['cp_predicted_series_label'], line_width=2,
                                       color=mypalette[2], line_dash='dashed',
                                       source=model_performance_source)

efp_output_validation = hydrograph.line('Date', VARS['efp_predicted_series_label'], line_width=2,
                                        color=mypalette[3], line_dash='dotted',
                                        source=model_performance_source)

hydrograph_legend = Legend(items=[
    ("Flagged Data", [est_flag]),
    ("Proxy Training", [proxy_measured_train]),
    ("Target Training", [target_measured_train]),
    ("Proxy Validation", [proxy_measured_validate]),
    ("Target Validation", [target_measured_validate]),
    (VARS['cp_predicted_series_label'], [cp_output_validation]),
    (VARS['efp_predicted_series_label'], [efp_output_validation]),
    ("Proxy Test", [proxy_measured_test]),
    ("Target Test", [target_measured_test]),
], location=(0, -30), click_policy='hide')


hydrograph.add_layout(hydrograph_legend, 'right')


##################################################################
# Bokeh Interactive Widgets
##################################################################
siteSelector1.on_change('value', siteSelector1_change)
siteSelector2.on_change('value', siteSelector2_change)

flag_selector_group.on_change('active', flag_selector_change)

poly_order_slider = Slider(title="Polynomial Order",
                           start=1, end=7, step=1, value=1)

poly_order_slider.on_change('value', update_poly_slider)

training_set_date_selector = DateRangeSlider(title="Training Set",
                                             end=VARS['sets']['train'][1],
                                             start=VARS['global_start'],
                                             step=1,
                                             value=(
                                                 VARS['sets']['train'][0],
                                                 VARS['sets']['train'][1]),
                                             callback_policy='mouseup'
                                             )

validation_set_date_selector = DateRangeSlider(title="Validation Set",
                                               end=VARS['sets']['validation'][1],
                                               start=VARS['sets']['validation'][0],
                                               step=1,
                                               value=(
                                                   VARS['sets']['validation'][0],
                                                   VARS['sets']['validation'][1]),
                                               callback_policy='mouseup')

test_set_date_selector = DateRangeSlider(title="Test Set",
                                         end=VARS['global_end'],
                                         start=VARS['sets']['test'][0],
                                         step=1,
                                         value=(VARS['sets']['test'][0],
                                                VARS['sets']['test'][1]),
                                         callback_policy='mouseup')

training_set_date_selector.on_change('value', update_train_set)
validation_set_date_selector.on_change('value', update_val_set)
test_set_date_selector.on_change('value', update_test_set)


##################################################################
# Layout of UI Elements
##################################################################

results_box = column(siteSelector1,
                     siteSelector2,
                     widgetbox(poly_order_slider, width=250),
                     flag_selector_group,
                     results_text
                     )

layout = column(
    row(children=[cp_training_fig, efp_training_fig, results_box],
        sizing_mode='fixed'),
    row(widgetbox(training_set_date_selector, width=400),
        widgetbox(validation_set_date_selector, width=400),
        widgetbox(test_set_date_selector, width=400),
        ),
    row(model_output_fig, hydrograph)
)


curdoc().add_root(layout)
curdoc().title = "Time Series Streamflow Regression"
