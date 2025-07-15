# Imports for the project
from math import ceil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import seaborn as sns

# Setting up the placement of 19 electrodes like used in the experiment from published paper.
eegElectrodePlacements = {'FP1': [-0.03, 0.08],
                          'FP2': [0.03, 0.08],
                          'F7': [-0.073, 0.047],
                          'F3': [-0.04, 0.041],
                          'Fz': [0, 0.038],
                          'F4': [0.04, 0.041],
                          'F8': [0.073, 0.047],
                          'T3': [-0.085, 0],
                          'C3': [-0.045, 0],
                          'Cz': [0, 0],
                          'C4': [0.045, 0],
                          'T4': [0.085, 0],
                          'T5': [-0.073, -0.047],
                          'P3': [-0.04, -0.041],
                          'Pz': [0, -0.038],
                          'P4': [0.04, -0.041],
                          'T6': [0.07, -0.047],
                          'O1': [-0.03, -0.08],
                          'O2': [0.03, -0.08]}

# Creating a variable to transpose the eeg electrode values.
fictionalChannels = pd.DataFrame(eegElectrodePlacements).transpose()

# A for loop is used to develop a make_dig_montage which only allows for 3D coordinates.
for key in eegElectrodePlacements.keys():
    eegElectrodePlacements[key] += [0]

# Using mne package to create 2D plot of placed electrodes on scalp.
mneStandardMontage = mne.channels.make_dig_montage(eegElectrodePlacements)
#mneStandardMontage.plot()

# Creating a variable to initially read in dataset for project.
eegDatasetCleaned = pd.read_csv('eegDatasetCleanedForVisualisation.csv')

# Observing cleaned EEG dataset and prints the first 5 rows.
print('\n EEG DATA SET HEAD, \n----------------------------------------------------\n')
print(eegDatasetCleaned.head(5))

# Prints information about the EEG dataset.
# eegDatasetCleaned.info()

# Checks the shape of the dataset (prints number of rows and columns).
print('\n EEG DATA SET SHAPE,\n----------------------------------------------------\n')
print('EEG dataset shape: \n', eegDatasetCleaned.shape)

# Checks data for missing values and spits a sum of entries missing
print('\n EEG DATA SET CHECK FOR MISSING ENTRIES, \n----------------------------------------------------\n')
print('Number of missing entries: ', eegDatasetCleaned.isnull().sum())

# Counts the number of each disorder recorded and creates a plot to show the potential bias in the data.
disorderOccurrence = eegDatasetCleaned.groupby(['specific.disorder']).size()
print('\n EEG DISORDER OCCURRENCES COUNT, \n----------------------------------------------------\n')
print(disorderOccurrence)

# Plot used to visualise count of how many of each disorder is recorded.
sns.countplot(x='specific.disorder', data=eegDatasetCleaned)

# In the cleaned EEG dataset it is formatted as: specific disorder | abs power of PSD per band per channel | sep_col
# | functional connectivity data Below the data set is being reshaped to include the specific disorder and absolute
# power of each band to then be used for visuals
eegMissingDataCheck = eegDatasetCleaned.isna().sum()
separatorColumnInEEGDataset = eegMissingDataCheck[eegMissingDataCheck == eegDatasetCleaned.shape[0]].index[0]
eegDatasetWithDisorderAndAbsolutePowerOfBandPerChannel = eegDatasetCleaned.loc[:,
                                                         'specific.disorder':separatorColumnInEEGDataset].drop(
    separatorColumnInEEGDataset, axis=1)
print(
    '\n EEG DATA SET SHAPE WITH ONLY DISORDERS AND ABSOLUTE POWER BAND,\n----------------------------------------------------\n')
print('EEG new dataset shape for visualisation: \n', eegDatasetWithDisorderAndAbsolutePowerOfBandPerChannel.shape)


# Method below reformatted the data to become the band concatenated with the channel (reformat from XX.X.band.x.channel
# to band.channel). Method receives 1 parameter
def reformatName(name):
    band, _, channel = name[5:].split(sep='.')
    return f'{band}.{channel}'


# Variable stores vectorized reformat name input data.
reformatVector = np.vectorize(reformatName)

# Variable stores concatenated band.channel values as a vector received from reformatName method.
brainWaveBandWithChannel = np.concatenate((eegDatasetWithDisorderAndAbsolutePowerOfBandPerChannel.columns[:1],
                                           reformatVector(
                                               eegDatasetWithDisorderAndAbsolutePowerOfBandPerChannel.columns[1:])))
eegDatasetWithDisorderAndAbsolutePowerOfBandPerChannel.set_axis(brainWaveBandWithChannel, axis=1, inplace=True)
print('\n EEG DATA SET COLUMNS FOR VISUALISATION,\n----------------------------------------------------\n')
print('EEG dataset columns: \n', eegDatasetWithDisorderAndAbsolutePowerOfBandPerChannel.columns)

# Calculated mean powers per specific.disorder and transform form wide to long format:
meanPowersOfSpecifiedDisorder = eegDatasetWithDisorderAndAbsolutePowerOfBandPerChannel.groupby(
    'specific.disorder').mean().reset_index()
# The variable below contains an array of the 6 available brain wave bands recorded during the EEG data collection.
eegBrainWaveBands = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']
# Variable below stores conversion of data frame from wide to long.
meanPowersOfSpecifiedDisorder = pd.wide_to_long(meanPowersOfSpecifiedDisorder, eegBrainWaveBands, ['specific.disorder'],
                                                'channel', sep='.', suffix='\w+')


# The method below handle processed EEG data with the supporting package MNE.
# An if statement is used to determine if a mask has been preset and if so set shape column 0 with 1 values.
# To create the visualisation the mna.viz.plot_topomap is used along with the parameters -
# Levels - Receives a numpy array with the data values to plot on graph.
# Positions - Receives data for the channels with an array which provides the X and Y positions as set in the beginning.
# Axes - Determines the axes to plot the graph with.
# Fig - This parameter is used to create the colour bar.
# ch_names - Parameter uses data containing list of channel names.
# cmap - This is the colour map to apply with the data (Red is used for positive data), Spectral_r is default setting.
# cb_pos - This is the parameter that stores the list of float values which are used for coordinates of colour bar.
# cb_width - Handles the width of the colour bar.
# cb_height - Handles the height of the colour bar.
# Marker - Numpy.array of bool, shape (n_channels,) | None
# marker_style - This handles additional plotting styles for plotting significant sensors, it is defaulted to none.
# **kwargs - This parameter is used to take on any additional parameters set in the method.
def plotEEGSingleBandAndChannel(levels, positions, axes, fig, ch_names=None, cmap='Spectral_r', cb_pos=(0.9, 0.1),
                                cb_width=0.04, cb_height=0.9, marker=None, marker_style=None, **kwargs):
    if 'mask' not in kwargs:
        mask = np.ones(levels.shape[0], dtype='bool')
    else:
        mask = None
    im, cm = mne.viz.plot_topomap(levels, positions, axes=axes, names=ch_names,
                                  cmap=cmap, mask=mask, mask_params=marker_style, show=False, **kwargs)

    cbar_ax = fig.add_axes([cb_pos[0], cb_pos[1], cb_width, cb_height])
    clb = axes.figure.colorbar(im, cax=cbar_ax)
    return im, cm

# # This code is for the extraction power for one specified.disorder and one band.
# extractedPowerForOneDisorderAndBand = meanPowersOfSpecifiedDisorder.loc['schizophrenia', 'gamma']
# # Use the assert method to ensure that channels are in correct order for indexing.
# assert (extractedPowerForOneDisorderAndBand.index == fictionalChannels.index).all()
# # The plot below shows the concentrated area for this particular disorder along with its channel.
# fig, ax = plt.subplots()
# plot_eeg(extractedPowerForOneDisorderAndBand, fictionalChannels.to_numpy(), ax, fig,
#          marker_style={'markersize': 4, 'markerfacecolor': 'black'})


# The method below handle processed EEG data with the supporting package MNE.
# To create the visualisation the mna.viz.plot_topomap is used along with the parameters -
# dataset - This parameter contains the dataset that is to be visualised.
# Channels - Parameter takes in the channels names along with their X & Y positions.
# gwidth - Sets the width of a single topology map.
# gheight - Sets the height of a single topology map.
# wspace - Sets the space between sub plots.
# marker_style - This handles additional plotting styles for plotting significant sensors, it is defaulted to none.
# band_ordered - Contains the List of EEG bands/brain wave frequencies. They must be in 1 or more columns of dataset.
# With mine set to none all columns of dataset are used in other of columns.
# conditions_ordered - This parameter controls the order left to right the resulting complex figure.
# (With mine being set to none all indecies are used).
# band_lables - This parameter contains a list of custom EEG bands and must be same length as band_ordered.
# In my case with value set to none the column names of the dataset are used instead.
# condition_labels - List of custom labels for conditions and if none the original induce names of the dataset are used.
# **kwargs - This parameter is used to take on any additional parameters set in the method.
def plotAllBandsAndChannels(dataset, channels, gwidth=2, gheight=1.5, wspace=0,
                            marker_style={'markersize': 2, 'markerfacecolor': 'black'},
                            band_ordered=None, conditions_ordered=None, band_labels=None,
                            condition_labels=None, **kwargs):
    if band_ordered is None:
        band_ordered = dataset.columns
    if conditions_ordered is None:
        conditions_ordered = dataset.index.get_level_values(0).unique()
    if band_labels is None:
        band_labels = band_ordered
    if condition_labels is None:
        condition_labels = conditions_ordered
    # Number of rows to display in the visual image
    numberOfRows = len(band_ordered)
    # Number of columns to display in the visual image
    numberOfColumns = len(conditions_ordered)
    # The code below develops the figure with variables for width gwidth and gheight per graph
    customFigureGraph = plt.figure(constrained_layout=True, figsize=(gwidth * numberOfColumns, gheight * numberOfRows))
    # The code below creates a variable that develops the individual figures for each band and channel.
    subCustomFigureGraph = customFigureGraph.subfigures(numberOfRows, numberOfColumns, wspace=wspace)
    # The for loop below works its way through the bands, disorders, levels and axes to create each graph with the
    # appropriate information retrieved from the dataset.
    for ind, subFigure in np.ndenumerate(subCustomFigureGraph):
        i, j = ind
        # The variable below selects the band which is the column name in the specified mean.
        band = band_ordered[i]
        # This variable stores the disorder from the row name in the specified disorder.
        disorder = conditions_ordered[j]
        # Variable below selects band levels to match the disorder.
        levels = dataset.loc[disorder, band]
        # This develops the axes for the sub plots.
        ax = subFigure.subplots()
        # Method takes in required parameters to plot single eeg band with its accompanied channel.
        plotEEGSingleBandAndChannel(levels, channels.to_numpy(), ax, subFigure, marker_style=marker_style, **kwargs)
        # The code below creates annotations for the bands to understand the plots better.
        if j == 0:
            ax.set_ylabel(band_labels[i])
        # annotate disorder (if needed)
        if i == 0:
            subFigure.suptitle(condition_labels[j], y=1.3)
    return customFigureGraph, subCustomFigureGraph


# Variable contains an array with a list of all the disorders found in the EEG dataset.
listOfDisorders = ['healthyControl',
         'schizophrenia',
         'acuteStressDisorder',
         'adjustmentDisorder',
         'alcoholUseDisorder',
         'behavioralAddictionDisorder',
         'bipolarDisorder',
         'depressiveDisorder',
         'obsessiveCompulsiveDisorder',
         'panicDisorder',
         'posttraumaticStressDisorder',
         'socialAnxietyDisorder']

#
# conds_labs = [x.replace('disorder', '') for x in listOfDisorders]
# # print('eher',conds_labs)
# Code below calls the method to create the plots for the mean powers of specified diorders with the set channels
# from the dataset.
#plotAllBandsAndChannels(meanPowersOfSpecifiedDisorder, fictionalChannels, conditions_ordered=listOfDisorders, condition_labels=listOfDisorders)

# Plot show displays the visuals of the graph above to observer.
plt.show()

# Credited code for preprocessed visualisation:
# https://www.kaggle.com/code/lazygene/visualising-pre-processed-eeg-data

# Other sourced references used to develop the project:
# https://www.geeksforgeeks.org/pandas-groupby-count-occurrences-in-column/
# https://mne.discourse.group/t/pip-installation-modulenotfounderror-mne-is-not-a-package/3550
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
