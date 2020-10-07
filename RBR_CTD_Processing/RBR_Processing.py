
"""
author: Lu Guan
date: Oct. 06, 2020
about: This script is for processing RBR CTD data and producing .ctd files in IOS Header format.

"""

import sys
import os
import pyrsktools
import itertools
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from copy import copy, deepcopy
from scipy import signal
import gsw
import xarray as xr
from matplotlib import pyplot as plt
import glob
from datetime import datetime
from datetime import timedelta
from decimal import Decimal


#----------------------- step 1. Export profile data to .csv files from .rsk files----------------------------------------------

def EXPORT_FILES(dest_dir, file, year, cruise_number):
    """
    Read in a rsk file and output in csv format
    Inputs:
        - folder, file, year, cruise_number: rsk-format file containing raw RBR data
    Outputs:
        - csv file: csv files containing the profile data
    """
    # function to export .csv files from .rsk file.
    filename = str(dest_dir) + str(file) #full path and name of .rsk file
    rsk = pyrsktools.open(filename) # load up an RSK

    #check the number of profiles
    n_profiles = len(list(rsk.profiles())) # get the number of profiles recorded

    ctd_data = pd.DataFrame() # set an empty pandas dataframe to store data

    #export .csv file for each profile
    for i in range(0, n_profiles, 1):

        downcast = list(itertools.islice(rsk.casts(pyrsktools.Region.CAST_DOWN), i, i+1))[0].npsamples() #separate samples for each downcast file
        downcast_dataframe = pd.DataFrame(data=downcast, columns=downcast.dtype.names) # convert data into pandas data frame

        upcast = list(itertools.islice(rsk.casts(pyrsktools.Region.CAST_UP), i, i+1))[0].npsamples()
        upcast_dataframe = pd.DataFrame(data=upcast, columns=upcast.dtype.names)


        column_names = list(downcast.dtype.names) #read the column names
        column_names[0] = 'Time(yyyy-mm-dd HH:MM:ss.FFF)' #update time
        for j in range(1, 11, 1):       # update the column names
            column_names[j]= column_names[j][0: -3] + "(" + list(rsk.channels.items())[j-1][1][4] + ")"

        downcast_dataframe.columns = column_names # update column names in downcast data frame
        downcast_dataframe["Cast_direction"] = "d"  # add a column for cast direction
        downcast_dataframe["Event"] = i+1 # add a column for event number
        upcast_dataframe.columns = column_names
        upcast_dataframe["Cast_direction"] = "u"
        upcast_dataframe["Event"] = i+1
        downcast_name = filename.split("/")[-1][0:-4].upper() + "_profile" + str(i+1).zfill(4) + "_DOWNCAST.csv" #downcast file name
        upcast_name = filename.split("/")[-1][0:-4].upper() + "_profile" + str(i+1).zfill(4) + "_UPCAST.csv" #upcast file name
        profile_name = filename.split("/")[-1][0:-4].upper() + "_profile" + str(i+1).zfill(4) + ".csv" # profile name
        ctd_data_name = str(year) + "-" + str(cruise_number) + "_CTD_DATA.csv" # all data file name

        profile_data = pd.concat([downcast_dataframe, upcast_dataframe]) #combine downcast and upcast into one profile
        ctd_data = ctd_data.append(profile_data, ignore_index=True) #combine profiles into one file


        #downcast_dataframe.to_csv(folder + downcast_name)
        #upcast_dataframe.to_csv(folder + upcast_name)
        profile_data.to_csv(dest_dir + profile_name) #export each profile

    ctd_data.to_csv(dest_dir + ctd_data_name) #export all data in one .csv file


#Example: run function to export files

EXPORT_FILES(dest_dir = '/home/guanl/Desktop/Projects/RBR/Processing/2019-107/', file = '066024_20190823_0915_CTD_Data.rsk', year = 2019, cruise_number=107)

#----------------------------   Step 2. Create Metadata dictionay    ---------------------------------------------------
def CREATE_META_DICT(dest_dir, file, year, cruise_number):
    """
     Read in a csv file and output a metadata dictionary
     Inputs:
         - folder, file, year, cruise_number: rsk-format file containing raw RBR data & csv file containing metadata
     Outputs:
         - metadata dictionary
     """
    meta_dict = {}
    # function to export .csv files from .rsk file.
    rsk_filename = str(dest_dir) + str(file) #full path and name of .rsk file
    rsk = pyrsktools.open(rsk_filename) # load up an RSK

    header_input_name = str(year) + '-' + str(cruise_number) + '_header-merge.csv'
    header_input_filename = dest_dir + header_input_name
    header = pd.read_csv(header_input_filename, header=0)

    csv_input_name = str(year) + '-' + str(cruise_number) + '_METADATA.csv'
    csv_input_filename = dest_dir + csv_input_name

    meta_csv = pd.read_csv(csv_input_filename)

    meta_dict['number_of_profiles'] = len(list(rsk.profiles()))
    meta_dict['Processing_Start_time'] = datetime.now()
    meta_dict['Instrument_information'] = rsk.instrument
    meta_dict['RSK_filename'] = rsk.name
    meta_dict['Channels'] = list(rsk.channels.keys())
    meta_dict['Channel_details'] = list(rsk.channels.items())
    meta_dict['Data_description'] = meta_csv['Value'][meta_csv['Name'] == 'Data_description'].values[0]
    meta_dict['Final_file_type'] = meta_csv['Value'][meta_csv['Name'] == 'Final_file_type'].values[0]
    meta_dict['Number_of_channels'] = meta_csv['Value'][meta_csv['Name'] == 'Number_of_channels'].values[0]
    meta_dict['Mission'] = meta_csv['Value'][meta_csv['Name'] == 'Mission'].values[0]
    meta_dict['Agency'] = meta_csv['Value'][meta_csv['Name'] == 'Agency'].values[0]
    meta_dict['Country'] = meta_csv['Value'][meta_csv['Name'] == 'Country'].values[0]
    meta_dict['Project'] = meta_csv['Value'][meta_csv['Name'] == 'Project'].values[0]
    meta_dict['Scientist'] = meta_csv['Value'][meta_csv['Name'] == 'Scientist'].values[0]
    meta_dict['Platform'] = meta_csv['Value'][meta_csv['Name'] == 'Platform'].values[0]
    meta_dict['Instrument_Model'] = meta_csv['Value'][meta_csv['Name'] == 'Instrument_Model'].values[0]
    meta_dict['Serial_number'] = meta_csv['Value'][meta_csv['Name'] == 'Serial_number'].values[0]
    meta_dict['Instrument_type'] = meta_csv['Value'][meta_csv['Name'] == 'Instrument_type'].values[0]
    meta_dict['Location'] = header

    return meta_dict
    
#Example:
metadata = CREATE_META_DICT(dest_dir = '/home/guanl/Desktop/Projects/RBR/Processing/2019-107/', file = '066024_20190823_0915_CTD_Data.rsk', year = 2019, cruise_number=107)

#-------------------------------  step 3. Add 6 line headers to CTD_DATA.csv file--------------------------------------------------
# Prepare data file with six line header for further applications in IOS Shell

def ADD_6LINEHEADER(dest_dir, year, cruise_number):
    """
     Read in a csv file and output in csv format for IOSShell
     Inputs:
         - folder, file, year, cruise: csv-format file containing raw RBR CTD data exported from rsk file
     Outputs:
         - csv file: csv files containing 6-header line for IOSShell
     """
    # Add six-line header to the .csv file.
    # This file could be used for data processing via IOSShell
    input_name = str(year) + '-' + str(cruise_number) + '_CTD_DATA.csv'
    output_name = str(year) + "-" + str(cruise_number) + '_CTD_DATA-6linehdr.csv'
    input_filename = dest_dir + input_name
    ctd_data = pd.read_csv(input_filename, header=0)
    ctd_data['Time(yyyy-mm-dd HH:MM:ss.FFF)'] = ctd_data['Time(yyyy-mm-dd HH:MM:ss.FFF)'].str[:19]
    ctd_data['Date'] = pd.to_datetime(ctd_data['Time(yyyy-mm-dd HH:MM:ss.FFF)']) #add new column of Date
    ctd_data['Date'] = [d.date() for d in ctd_data['Date']]
    ctd_data['TIME:UTC'] = pd.to_datetime(ctd_data['Time(yyyy-mm-dd HH:MM:ss.FFF)']) #add new column of time
    ctd_data['TIME:UTC'] = [d.time() for d in ctd_data['TIME:UTC']]
    ctd_data['Date'] = pd.to_datetime(ctd_data['Date'], format='%Y-%m-%d').dt.strftime('%d/%m/%Y')

    ctd_data = ctd_data.drop(ctd_data.columns[[0, 1]], 1)
    cols = list(ctd_data.columns)
    cols = cols[0:2] + cols[3:10] + [cols[2]] + cols[10:]
    ctd_data=ctd_data[cols]
    columns = ctd_data.columns.tolist()

    column_names = dict.fromkeys(ctd_data.columns, '') #set empty column names
    ctd_data = ctd_data.rename(columns=column_names)   #remove column names

    #write header infomation into a dataframe
    channel = ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N', 'Y', 'Y','Y']
    index = ['Conductivity', 'Temperature', 'Fluorescence', 'Oxygen:Dissolved:Saturation', 'Pressure', 'Depth', 'Salinity:CTD', ' ', ' ', 'Pressure_Air', 'Cast_direction', 'Event_number', 'Date', 'TIME:UTC']
    unit = ['mS/cm', 'deg C(ITS90)', 'mg/m^3', '%', 'decibar', 'meters', 'PSS-78', ' ', ' ', 'decibar', 'n/a', 'n/a', 'n/a', 'n/a']
    input_format = ['R4', 'R4', 'R4', 'R4', 'R4', 'R4', 'R4', ' ', ' ', 'R4', ' ', 'I4', 'D:dd/mm/YYYY', 'T:HH:MM:SS']
    output_format = ['R4:F11.4', 'R4:F9.4', 'R4:F8.3', 'R4:F11.4', 'R4:F7.1', 'R4:F7.1', 'R4:F9.4', ' ', ' ','R4:F7.1', ' ', 'I:I4', 'D:YYYY/mm/dd', 'T:HH:MM:SS']
    na_value = ['-99', '-99', '-99', '-99', '-99', '-99', '-99', '-99', '', '-99', '', '', '', '']
    header = pd.DataFrame([channel, index, unit, input_format, output_format, na_value])
    column_names_header = dict.fromkeys(header.columns, '') #set empty column names
    header = header.rename(columns=column_names_header)

    ctd_data_header = header.append(ctd_data)
    ctd_data_header.to_csv(dest_dir + output_name)


#Example
ADD_6LINEHEADER(dest_dir = '/home/guanl/Desktop/Projects/RBR/Processing/2019-107/', year = 2019, cruise_number = 107)


#--------------------------------  step 4. Plot and check profile locations  -----------------------------------------------------
#define function to plot and check the location of CTD


def Plot_Track_Location(dest_dir, year, cruise_number, left_lon, right_lon, bot_lat, top_lat):
    """
     Read in a csv file and output a map
     Inputs:
         - folder, year, cruise: csv file containing raw RBR data
     Outputs:
         - A map showing sampling locations
     """
    input_name = str(year) + '-' + str(cruise_number) + '_header-merge.csv'
    input_filename = dest_dir + input_name
    header = pd.read_csv(input_filename, header=0)
    header['lat_degree'] = header['LOC:LATITUDE'].str[:2].astype(int)
    header['lat_min'] = header['LOC:LATITUDE'].str[3:10].astype(float)
    header['lat'] = header['lat_degree'] + header['lat_min']/60
    header['lon_degree'] = header['LOC:LONGITUDE'].str[:3].astype(int)
    header['lon_min'] = header['LOC:LONGITUDE'].str[4:12].astype(float)
    header['lon'] = 0 - (header['lon_degree'] + header['lon_min']/60)
    event = header['LOC:STATION'].astype(str)

    lon = header['lon'].tolist()
    lat = header['lat'].tolist()
    event = event.tolist()

    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat,
                projection='lcc',
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

    x, y = m(lon, lat)

    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    m.drawcoastlines(linewidth=0.2)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')
    m.drawrivers()

    m.scatter(x, y, marker='D', color='m', s=4)
    #m.plot(x, y, marker='D', color='m', markersize=4)
    for event, xpt, ypt in zip(event, x, y):
        plt.text(xpt, ypt, event)

    parallels = np.arange(bot_lat, top_lat, 0.2)  # parallels = np.arange(48., 54, 0.2), parallels = np.linspace(bot_lat, top_lat, 10)
    m.drawparallels(parallels, labels=[True, False, True, False])  # draw parallel lat lines
    meridians = np.arange(left_lon, right_lon, 0.4)
    m.drawmeridians(meridians, labels=[False, False, False, True])
    plt.show()

#Example: run function to plot the locations of RBR casts
Plot_Track_Location(dest_dir='/home/guanl/Desktop/Projects/RBR/Processing/2019-107/', year = 2019, cruise_number= 107, left_lon=-128.5, right_lon=-126.5, bot_lat=50.5, top_lat=51.5)




#------------------------------------------------- Step 5.  create variable dictionaries  ------------------------------------------------------

def CREATE_CAST_VARIABLES(year, cruise_number, dest_dir):
    """
     Read in a csv file and output data dictionaries to hold profile data
     Inputs:
         - folder, year, cruise: csv file containing raw RBR data
     Outputs:
         - three dictionaries containing casts, downcasts and upcasts
     """
    input_name = str(year) + "-" + str(cruise_number) + '_CTD_DATA-6linehdr.csv'
    input_filename = dest_dir + input_name
    ctd_data = pd.read_csv(input_filename, header=0)
    ctd_data = ctd_data.rename(columns=ctd_data.iloc[1]) #assign the second row as column names
    ctd_data = ctd_data.rename(columns={'Oxygen:Dissolved:Saturation': 'Oxygen', 'Salinity:CTD': 'Salinity', 'TIME:UTC': 'TIME'})
    ctd = ctd_data.iloc[6:]
    ctd = ctd.copy()
    cols = ctd.columns[0:8]
    ctd[cols] = ctd[cols].apply(pd.to_numeric, errors='coerce', axis=1)

    n = ctd['Event_number'].nunique()

    var_holder = {}
    for i in range(1, n+1, 1):
        var_holder['cast' + str(i)] = ctd.loc[(ctd['Event_number'] == str(i))]
    #var_holder['Processing_history'] = ""

    var_holder_d = {}
    for i in range(1, n+1, 1):
        var_holder_d['cast' + str(i)] = ctd.loc[(ctd['Event_number'] == str(i)) & (ctd['Cast_direction'] == 'd')]
    #var_holder_d['Processing_history'] = ""

    var_holder_u = {}
    for i in range(1, n+1, 1):
        var_holder_u['cast' + str(i)] = ctd.loc[(ctd['Event_number'] == str(i)) & (ctd['Cast_direction'] == 'u')]
    #var_holder_u['Processing_history'] = ""

    return var_holder, var_holder_d, var_holder_u


# Example
cast, cast_d, cast_u = CREATE_CAST_VARIABLES(year = 2019 , cruise_number = 107, dest_dir = '/home/guanl/Desktop/Projects/RBR/Processing/2019-107/')

#--------------------------------------   plot data from all profiles  -----------------------------------------------------
#plot salinity of all cast together
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Salinity, cast_d['cast1'].Pressure, color='blue', label='cast1')
ax.plot(cast_u['cast1'].Salinity, cast_u['cast1'].Pressure, '--', color='blue', label='cast1')
ax.plot(cast_d['cast2'].Salinity, cast_d['cast2'].Pressure, color='red', label='cast2')
ax.plot(cast_u['cast2'].Salinity, cast_u['cast2'].Pressure, '--', color='red', label='cast2')
ax.plot(cast_d['cast3'].Salinity, cast_d['cast3'].Pressure, color='green', label='cast3')
ax.plot(cast_u['cast3'].Salinity, cast_u['cast3'].Pressure, '--', color='green', label='cast3')
ax.plot(cast_d['cast4'].Salinity, cast_d['cast4'].Pressure, color='black', label='cast4')
ax.plot(cast_u['cast4'].Salinity, cast_u['cast4'].Pressure, '--', color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Salinity')
ax.set_ylabel('Pressure (decibar)')
ax.legend()


#plot temperature of all cast together
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Temperature, cast_d['cast1'].Pressure, color='blue', label='cast1')
ax.plot(cast_u['cast1'].Temperature, cast_u['cast1'].Pressure, '--', color='blue', label='cast1')
ax.plot(cast_d['cast2'].Temperature, cast_d['cast2'].Pressure, color='red', label='cast2')
ax.plot(cast_u['cast2'].Temperature, cast_u['cast2'].Pressure, '--', color='red', label='cast2')
ax.plot(cast_d['cast3'].Temperature, cast_d['cast3'].Pressure, color='green', label='cast3')
ax.plot(cast_u['cast3'].Temperature, cast_u['cast3'].Pressure, '--', color='green', label='cast3')
ax.plot(cast_d['cast4'].Temperature, cast_d['cast4'].Pressure, color='black', label='cast4')
ax.plot(cast_u['cast4'].Temperature, cast_u['cast4'].Pressure, '--', color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Temperature(C)')
ax.set_ylabel('Pressure (decibar)')
ax.legend()


#plot Conductivity of all cast together
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Conductivity, cast_d['cast1'].Pressure, color='blue', label='cast1')
ax.plot(cast_d['cast2'].Conductivity[1:], cast_d['cast2'].Pressure[1:], color='red', label='cast2')
ax.plot(cast_u['cast2'].Conductivity[1:], cast_u['cast2'].Pressure[1:], '--', color='red', label='cast2')
ax.plot(cast_d['cast3'].Conductivity, cast_d['cast3'].Pressure, color='green', label='cast3')
ax.plot(cast_u['cast3'].Conductivity, cast_u['cast3'].Pressure, '--', color='green', label='cast3')
ax.plot(cast_d['cast4'].Conductivity, cast_d['cast4'].Pressure, color='black', label='cast4')
ax.plot(cast_u['cast4'].Conductivity, cast_u['cast4'].Pressure, '--', color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Conductivity (S/cm)')
ax.set_ylabel('Pressure (decibar)')
ax.legend()


#plot Oxygen of all cast together
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Oxygen, cast_d['cast1'].Pressure, color='blue', label='cast1')
ax.plot(cast_u['cast1'].Oxygen, cast_u['cast1'].Pressure, '--', color='blue', label='cast1')
ax.plot(cast_d['cast2'].Oxygen, cast_d['cast2'].Pressure, color='red', label='cast2')
ax.plot(cast_u['cast2'].Oxygen, cast_u['cast2'].Pressure, '--', color='red', label='cast2')
ax.plot(cast_d['cast3'].Oxygen, cast_d['cast3'].Pressure, color='green', label='cast3')
ax.plot(cast_u['cast3'].Oxygen, cast_u['cast3'].Pressure, '--', color='green', label='cast3')
ax.plot(cast_d['cast4'].Oxygen, cast_d['cast4'].Pressure, color='black', label='cast4')
ax.plot(cast_u['cast4'].Oxygen, cast_u['cast4'].Pressure, '--', color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Oxygen Saturation (%)')   # Check unit here
ax.set_ylabel('Pressure (decibar)')
ax.legend()

#plot Fluorescence of all cast together
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Fluorescence, cast_d['cast1'].Pressure, color='blue', label='cast1')
ax.plot(cast_u['cast1'].Fluorescence, cast_u['cast1'].Pressure, '--', color='blue', label='cast1')
ax.plot(cast_d['cast2'].Fluorescence, cast_d['cast2'].Pressure, color='red', label='cast2')
ax.plot(cast_u['cast2'].Fluorescence, cast_u['cast2'].Pressure, '--', color='red', label='cast2')
ax.plot(cast_d['cast3'].Fluorescence, cast_d['cast3'].Pressure, color='green', label='cast3')
ax.plot(cast_u['cast3'].Fluorescence, cast_u['cast3'].Pressure, '--', color='green', label='cast3')
ax.plot(cast_d['cast4'].Fluorescence, cast_d['cast4'].Pressure, color='black', label='cast4')
ax.plot(cast_u['cast4'].Fluorescence, cast_u['cast4'].Pressure, '--', color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Fluorescence (ug/L)')   # Check unit here
ax.set_ylabel('Pressure (decibar)')
ax.legend()

# TS Plot
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Temperature, cast_d['cast1'].Salinity, '.', color='blue')
ax.plot(cast_d['cast2'].Temperature, cast_d['cast2'].Salinity, '.', color='red')
ax.plot(cast_d['cast3'].Temperature, cast_d['cast3'].Salinity, '.', color='green')
ax.plot(cast_d['cast4'].Temperature, cast_d['cast4'].Salinity, '.', color='black')
ax.set_xlabel('Salinity')
ax.set_ylabel('Temperature (C)')
ax.set_title('T-S Plot')
ax.legend()


#--------------------------------------  Plot individual profile ------------------------------------------------------------------------
# T, C, O, S, F for each profile in one plot
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True)
#Temperature
ax1.plot(cast_d['cast1'].Temperature, cast_d['cast1'].Pressure, color='red', label='cast_down')
ax1.plot(cast_u['cast1'].Temperature, cast_u['cast1'].Pressure, '--', color='red', label='cast_up')
ax1.set_ylabel('Pressure(decibar)')
ax1.set_ylim(ax1.get_ylim()[::-1])
ax1.set_xlabel('Temperature(C)')
ax1.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('top')
ax1.legend()


#Conductivity
ax2.plot(cast_d['cast1'].Conductivity, cast_d['cast1'].Pressure, color='yellow', label='cast_down')
ax2.plot(cast_u['cast1'].Conductivity, cast_u['cast1'].Pressure, '--', color='yellow', label='cast1_up')
ax2.set_ylabel('Pressure(decibar)')
ax2.set_ylim(ax1.get_ylim()[::-1])
ax2.set_xlabel('Conductivity (S/cm)')
ax2.xaxis.set_label_position('top')
ax2.xaxis.set_ticks_position('top')
ax2.legend()


#Oxygen
ax3.plot(cast_d['cast1'].Oxygen, cast_d['cast1'].Pressure, color='black', label='cast_down')
ax3.plot(cast_u['cast1'].Oxygen, cast_u['cast1'].Pressure, '--', color='black', label='cast_up')
ax3.set_ylabel('Pressure(decibar)')
ax3.set_ylim(ax1.get_ylim()[::-1])
ax3.set_xlabel('Oxygen Saturation (%)')
ax3.xaxis.set_label_position('top')
ax3.xaxis.set_ticks_position('top')
ax3.legend()

#Salinity
ax4.plot(cast_d['cast1'].Salinity, cast_d['cast1'].Pressure, color='blue', label='cast_down')
ax4.plot(cast_u['cast1'].Salinity, cast_u['cast1'].Pressure, '--', color='blue', label='cast_up')
ax4.set_ylabel('Pressure(decibar)')
ax4.set_ylim(ax1.get_ylim()[::-1])
ax4.set_xlabel('Salinity')
ax4.xaxis.set_label_position('top')
ax4.xaxis.set_ticks_position('top')
ax4.legend()

#Fluorescence
ax5.plot(cast_d['cast1'].Fluorescence, cast_d['cast1'].Pressure, color='green', label='cast_down')
ax5.plot(cast_u['cast1'].Fluorescence, cast_u['cast1'].Pressure, '--', color='green', label='cast1_up')
ax5.set_ylabel('Pressure(decibar)')
ax5.set_ylim(ax1.get_ylim()[::-1])
ax5.set_xlabel('Fluoresence(ug/L)')
ax5.xaxis.set_label_position('top')
ax5.xaxis.set_ticks_position('top')
ax5.legend()

#--------------------------------   Plot by Index and by Profile--------------------------------------------------------------------------------------------
# separate plot for T, C, O, S, F of each profile
#Temperature
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Temperature, cast_d['cast1'].Pressure, color='red', label='cast_down')
ax.plot(cast_u['cast1'].Temperature, cast_u['cast1'].Pressure, '--', color='red', label='cast_up')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Temperature(C)')
ax.set_ylabel('Pressure (decibar)')
ax.legend()

#Salinity
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Salinity, cast_d['cast1'].Pressure, color='blue', label='cast1')
#ax.plot(cast_u['cast1'].Salinity, cast_u['cast1'].Pressure, '--', color='blue', label='cast1')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Salinity')
ax.set_ylabel('Pressure (decibar)')
ax.legend()

#Conductivity
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Conductivity, cast_d['cast1'].Pressure, color='yellow', label='cast1')
#ax.plot(cast_u['cast1'].Conductivity, cast_u['cast1'].Pressure, '--', color='yellow', label='cast1')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Conductivity (S/cm)')
ax.set_ylabel('Pressure (decibar)')
ax.legend()


#Oxygen
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Oxygen, cast_d['cast1'].Pressure, color='black', label='cast1')
ax.plot(cast_u['cast1'].Oxygen, cast_u['cast1'].Pressure, '--', color='black', label='cast1')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Oxygen Saturation (%)')   # Check unit here
ax.set_ylabel('Pressure (decibar)')
ax.legend()

#Fluoresence
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Fluorescence, cast_d['cast1'].Pressure, color='green', label='cast1')
#ax.plot(cast_u['cast1'].Fluorescence, cast_u['cast1'].Pressure, '--', color='green', label='cast1')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Fluorescence (ug/L)')   # Check unit here
ax.set_ylabel('Pressure (decibar)')
ax.legend()


#---------------------------   step 6. Plot and Check for zero-order hold   --------------------------------------------------

def PLOT_PRESSURE_DIFF(dest_dir, year, cruise_number):
    """
     Read in a csv file and output a plot to check zero-order holds
     Inputs:
         - folder, year, cruise: csv file containing raw RBR data
     Outputs:
         - a plot showing the time derivative of raw pressure
     """

    input_name = str(year) + "-" + str(cruise_number) + '_CTD_DATA-6linehdr.csv'
    input_filename = dest_dir + input_name
    ctd_data = pd.read_csv(input_filename, header=0)
    ctd_data = ctd_data.rename(columns=ctd_data.iloc[1]) #assign the second row as column names
    ctd_data = ctd_data.rename(columns={'Oxygen:Dissolved:Saturation': 'Oxygen', 'Salinity:CTD': 'Salinity', 'TIME:UTC': 'TIME'})
    ctd = ctd_data.iloc[6:]
    ctd.index = np.arange(0, len(ctd))
    cols = ctd.columns[0:7]
    ctd[cols] = ctd[cols].apply(pd.to_numeric, errors='coerce', axis=1)
    pressure = ctd.Pressure
    pressure_lag = ctd.Pressure[1:]
    pressure_lag.index = np.arange(0, len(pressure_lag))
    pressure_diff = pressure_lag - pressure

    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    plt.plot(pressure_diff, color='blue', label='Pressure_diff')
    plt.ylabel('Pressure (decibar)')
    plt.xlabel('Scans')
    plt.legend()
    plt.show()

#Example
PLOT_PRESSURE_DIFF(dest_dir = '/home/guanl/Desktop/Projects/RBR/Processing/2019-107/', year=2019, cruise_number=107)

#----------------------------   step 7. CALIB: Pressure/Depth correction --------------------------------------------------------------------

#plot Conductivity vs Pressure  & Conductivity vs Depth for the first 10 records to check the need of pressure correction
# check the value of conductivity when sensor is above sea surface
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Conductivity[0:10], cast_d['cast1'].Pressure[0:10], color='blue', label='cast1')
ax.plot(cast_d['cast2'].Conductivity[0:10], cast_d['cast2'].Pressure[0:10], color='red', label='cast2')
ax.plot(cast_d['cast3'].Conductivity[0:10], cast_d['cast3'].Pressure[0:10], color='green', label='cast3')
ax.plot(cast_d['cast4'].Conductivity[0:10], cast_d['cast4'].Pressure[0:10], color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Conductivity (S/cm)')
ax.set_ylabel('Pressure (decibar)')
ax.legend()

fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Conductivity[0:10], cast_d['cast1'].Depth[0:10], color='blue', label='cast1')
ax.plot(cast_d['cast2'].Conductivity[0:10], cast_d['cast2'].Depth[0:10], color='red', label='cast2')
ax.plot(cast_d['cast3'].Conductivity[0:10], cast_d['cast3'].Depth[0:10], color='green', label='cast3')
ax.plot(cast_d['cast4'].Conductivity[0:10], cast_d['cast4'].Depth[0:10], color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Conductivity (S/cm)')
ax.set_ylabel('Depth (m)')
ax.legend()


# check the figure and add coorection_value (0.1 for 2019-107) to pressure and depth.

def CALIB(var, var_downcast, var_upcast, correction_value, metadata_dict):
    """
     Correct pressure and depth data
     Inputs:
         - cast, downcast, upcast and metadata dictionaries
     Outputs:
         - cast, downcast, upcast and metadata dictionaries after pressure correction
     """
    n = len(var.keys())
    var1 = deepcopy(var)
    var2 = deepcopy(var_downcast)
    var3 = deepcopy(var_upcast)
    for i in range(1, n+1, 1):
        var1['cast' + str(i)].Pressure = var1['cast' + str(i)].Pressure + correction_value
        var1['cast' + str(i)].Depth = var1['cast' + str(i)].Depth + correction_value
        var2['cast' + str(i)].Pressure = var2['cast' + str(i)].Pressure + correction_value
        var2['cast' + str(i)].Depth = var2['cast' + str(i)].Depth + correction_value
        var3['cast' + str(i)].Pressure = var3['cast' + str(i)].Pressure + correction_value
        var3['cast' + str(i)].Depth = var3['cast' + str(i)].Depth + correction_value

    metadata_dict['Processing_history'] = '-CALIB parameters:|' \
                                          ' Calibration type = Correct|' \
                                          ' Calibrations applied:|' \
                                          ' Pressure (decibar) = {}'.format(str(correction_value)) + '|' \
                                          ' Depth (meters) = {}'.format(str(correction_value)) + '|'
    metadata_dict['CALIB_Time'] = datetime.now()

    return var1, var2, var3

#Example:
#output variables: cast_pc, cast_d_pc, cast_u_pc. pc = pressure correction
cast_pc, cast_d_pc, cast_u_pc = CALIB(cast, cast_d, cast_u, correction_value = 0.1, metadata_dict = metadata)

#------------------------------  Step 8: Data Despiking  --------------------------------------------------------------------------
# plot profiles to look for spikes




#------------------------------  Step 9: CLIP  -----------------------------------------------------------------------------------
#Remove poor/bad data from surface and bottom
#Plot profiles to determine the data needs to be removed.

fig, ax = plt.subplots()
ax.plot(cast_pc['cast1'].TIME, cast_pc['cast1'].Pressure, color='blue', label='cast1')
#ax.plot(cast_pc['cast2'].TIME, cast_pc['cast2'].Pressure, color='red', label='cast2')
#ax.plot(cast_pc['cast3'].TIME, cast_pc['cast3'].Pressure, color='green', label='cast3')
#ax.plot(cast_pc['cast4'].TIME, cast_pc['cast4'].Pressure, color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Time')
ax.set_ylabel('Pressure (decibar)')
ax.legend()



#check the downcast values at the surface
fig, ax = plt.subplots()
#ax.plot(cast_d_pc['cast1'].TIME[0:37], cast_d_pc['cast1'].Pressure[0:37], color='blue', label='cast1') # remove 0:39 for cast1 at surface
#ax.plot(cast_d_pc['cast2'].TIME[0:173], cast_d_pc['cast2'].Pressure[0:173], color='red', label='cast2') # remove 0:178 for cast2 at surface
#ax.plot(cast_d_pc['cast3'].TIME[0:20], cast_d_pc['cast3'].Pressure[0:20], color='green', label='cast3')
ax.plot(cast_d_pc['cast4'].TIME[0:63], cast_d_pc['cast4'].Pressure[0:63], color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Time')
ax.set_ylabel('Pressure (decibar)')
ax.legend()



# check the downcast values at bottom

fig, ax = plt.subplots()
#ax.plot(cast_d_pc['cast1'].TIME[-100:-1], cast_d_pc['cast1'].Pressure[-1:-1], color='blue', label='cast1') # remove 0 for profile 1
#ax.plot(cast_d_pc['cast2'].TIME[-100:-1], cast_d_pc['cast2'].Pressure[-1:-1], color='red', label='cast2') # remove 0 for profile 2
#ax.plot(cast_d_pc['cast3'].TIME[-10:-1], cast_d_pc['cast3'].Pressure[-10:-1], color='green', label='cast3') # remove 10 for profile 3
#ax.plot(cast_d_pc['cast4'].TIME[-60:-1], cast_d_pc['cast4'].Pressure[-60:-1], color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Time')
ax.set_ylabel('Pressure (decibar)')
ax.legend()



# check the upcast values at surface
fig, ax = plt.subplots()
#ax.plot(cast_u_pc['cast1'].TIME[0:70], cast_u_pc['cast1'].Pressure[0:70], color='blue', label='cast1') # remove 0: 70  for profile 1 upcast at bottom
#ax.plot(cast_u_pc['cast2'].TIME[0:215], cast_u_pc['cast2'].Pressure[0:215], color='red', label='cast2')
#ax.plot(cast_u_pc['cast3'].TIME[0:80], cast_u_pc['cast3'].Pressure[0:80], color='green', label='cast3')
ax.plot(cast_u_pc['cast4'].TIME[168:], cast_u_pc['cast4'].Pressure[168:], color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Time')
ax.set_ylabel('Pressure (decibar)')
ax.legend()



# check the upcast values at surface
fig, ax = plt.subplots()
#ax.plot(cast_u_pc['cast1'].TIME[-25:-1], cast_u_pc['cast1'].Pressure[-25:-1], color='blue', label='cast1')
#ax.plot(cast_u_pc['cast2'].TIME[-42:-1], cast_u_pc['cast2'].Pressure[-42:-1], color='red', label='cast2')
#ax.plot(cast_u_pc['cast3'].TIME[169:-55], cast_u_pc['cast3'].Pressure[169:-55], color='green', label='cast3')
ax.plot(cast_u_pc['cast4'].TIME[:-49], cast_u_pc['cast4'].Pressure[:-49], color='black', label='cast4')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Time')
ax.set_ylabel('Pressure (decibar)')
ax.legend()

# input variables: cast_d_pc, cast_u_pc
# output variables: cast_d_clip, cast_u_clip

def CLIP_DOWNCAST(var, metadata_dict, cast_number, cut_at_start, cut_at_end):
    """
     CLIP the unstable measurement from sea surface and bottom
     Inputs:
         - Downcast, metadata dictionary, cast_number,
     Outputs:
         - Downcast after removing records near surface and bottom
     """
    #var = deepcopy(var_downcast)
    if cut_at_end == 0:
        measurement_number = var['cast' + str(cast_number)].shape[0]
        var['cast' + str(cast_number)] = var['cast' + str(cast_number)][cut_at_start:]
    else:
        measurement_number = var['cast' + str(cast_number)].shape[0]
        var['cast' + str(cast_number)] = var['cast' + str(cast_number)][cut_at_start:cut_at_end]

    metadata_dict['Processing_history'] += '-CLIP_downcast{}'.format(str(cast_number)) + ': First Record = {}'.format(str(cut_at_start)) + ', Last Record = {}'.format(str(measurement_number + cut_at_end)) + '|'
    metadata_dict['CLIP_D_Time' + str(cast_number)] = datetime.now()


def CLIP_UPCAST(var, metadata_dict, cast_number, cut_at_start, cut_at_end):
    """
     CLIP the unstable measurement from sea surface and bottom
     Inputs:
         - Upcast, metadata dictionary, cast_number,
     Outputs:
         - Upcast after removing records near surface and bottom
     """
    #var = deepcopy(var)
    if cut_at_end == 0:
        measurement_number = var['cast' + str(cast_number)].shape[0]
        var['cast' + str(cast_number)] = var['cast' + str(cast_number)][cut_at_start:]
    else:
        measurement_number = var['cast' + str(cast_number)].shape[0]
        var['cast' + str(cast_number)] = var['cast' + str(cast_number)][cut_at_start:cut_at_end]

    metadata_dict['Processing_history'] += '-CLIP_upcast{}'.format(str(cast_number)) + ': First Record = {}'.format(str(cut_at_start)) + ', Last Record = {}'.format(str(measurement_number + cut_at_end)) + '|'
    metadata_dict['CLIP_U_Time' + str(cast_number)] = datetime.now()

#Example: 
#Downcast: cast1[37:], cast2[173:], cast3[20:-10], cast4[63:-60]
#Upcast: cast1[70:-25], cast2[215:-42], cast3[80:-55], cast4[168:-49]
cast_d_clip = deepcopy(cast_d_pc)
cast_u_clip = deepcopy(cast_u_pc)
CLIP_DOWNCAST(cast_d_clip , metadata_dict = metadata, cast_number = 2,  cut_at_start = 173,  cut_at_end = 0)
CLIP_UPCAST(cast_u_clip , metadata_dict = metadata, cast_number = 1,  cut_at_start = 70,  cut_at_end = -25)



#Plot to check the profiles after clip by cast
fig, ax = plt.subplots()
ax.plot(cast_d_clip['cast1'].TIME, cast_d_clip['cast1'].Pressure, color='blue', label='cast1')
ax.plot(cast_u_clip['cast1'].TIME, cast_u_clip['cast1'].Pressure, color='blue', label='cast1')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Time')
ax.set_ylabel('Pressure (decibar)')
ax.legend()



#------------------------------  Step 10: Filter  -----------------------------------------------------------------------------------
#apply a moving average FIR filter (a simple low pass )

#def filter(x, n):# n -  filter size, 9 suggest by RBR manual, choose the smallest one which can do the job
#    b = (np.ones(n))/n #numerator co-effs of filter transfer function
#    #b = repeat(1.0/n, n)
#    a = np.ones(1)  #denominator co-effs of filter transfer function
#    #y = signal.convolve(x,b) #filter output using convolution
#    #y = signal.lfilter(b, a, x) #filter output using lfilter function
#    y = signal.filtfilt(b, a, x)  # Apply a digital filter forward and backward to a signal.
#    return y


def FILTER(var_downcast, var_upcast, window_width, sample_rate, time_constant, filter_type, metadata_dict):
    """
     Filter the profile data using a low pass filter: moving average
     Inputs:
         - downcast and upcast data dictionaries
     Outputs:
         - two dictionaries containing downcast and upcast profiles after applying filter
     """
    # filter type: 0 - FIR, 1 - moving average
    cast_number = len(var_downcast.keys())
    if filter_type == 0:
        Wn = (1.0/time_constant)/(sample_rate*2)
        b, a = signal.butter(2, Wn, "low")
        filter_name = "FIR"
    elif filter_type == 1:
        b = (np.ones(window_width)) / window_width  # numerator co-effs of filter transfer function
        a = np.ones(1)  # denominator co-effs of filter transfer function
        filter_name = "Moving average filter"

    var1 = deepcopy(var_downcast)
    var2 = deepcopy(var_upcast)
    for i in range(1, cast_number+1, 1):
        var1['cast' + str(i)].Temperature = signal.filtfilt(b, a, var1['cast' + str(i)].Temperature)
        var1['cast' + str(i)].Conductivity = signal.filtfilt(b, a, var1['cast' + str(i)].Conductivity)
        #var1['cast' + str(i)].Pressure = signal.filtfilt(b, a, var1['cast' + str(i)].Pressure)
        var1['cast' + str(i)].Fluorescence = signal.filtfilt(b, a, var1['cast' + str(i)].Fluorescence)
        var2['cast' + str(i)].Temperature = signal.filtfilt(b, a, var2['cast' + str(i)].Temperature)
        var2['cast' + str(i)].Conductivity = signal.filtfilt(b, a, var2['cast' + str(i)].Conductivity)
        #var2['cast' + str(i)].Pressure = signal.filtfilt(b, a, var2['cast' + str(i)].Pressure)
        var2['cast' + str(i)].Fluorescence = signal.filtfilt(b, a, var2['cast' + str(i)].Fluorescence)
    metadata_dict['Processing_history'] += '-FILTER parameters:|' \
                                           ' ' + filter_name + ' was used.|' \
                                           ' Filter width = {}'.format(str(window_width)) + '.|' \
                                           ' The following channel(s) were filtered.|' \
                                           ' Pressure|' \
                                           ' Temperature|' \
                                           ' Conductivity|'
    metadata_dict['FILTER_Time'] = datetime.now()

    return var1, var2

#Example:
cast_d_filtered, cast_u_filtered = FILTER(cast_d_clip, cast_u_clip, window_width = 4, sample_rate = 6, time_constant = 1/6, filter_type = 1, metadata_dict = metadata) # n = 5 should be good.


#plot to check values before and after filtering
fig, ax = plt.subplots()
ax.plot(cast_d_clip['cast3'].Conductivity, cast_d_clip['cast3'].Pressure, color='blue', label='Pre-filtering')
ax.plot(cast_d_filtered['cast3'].Conductivity, cast_d_filtered['cast3'].Pressure, '--', color='red', label='Post-filtering')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel(' ')
ax.set_ylabel('Pressure (decibar)')
ax.legend()


#------------------------------  Step 11: Shift conductivity and recalculate salinity ----------------------------------------------------
#input variable: cast_d_filtered, cast_u_filtered, try 2-3s (12-18 scans)

def SHIFT_CONDUCTIVITY(var_downcast, var_upcast, shifted_scan_number, metadata_dict): # n: number of scans shifted. +: delay; -: advance
    """
     Delay the conductivity signal, and recalculate salinity
     Inputs:
         - downcast and upcast data dictionaries, metadata dictionary
     Outputs:
         - two dictionaries containing downcast and upcast profiles
     """
    cast_number = len(var_downcast.keys())
    var1 = deepcopy(var_downcast)
    var2 = deepcopy(var_upcast)
    for i in range(1, cast_number+1, 1):
        index_1 = var1['cast' + str(i)].Conductivity.index[0]
        v1 = var1['cast' + str(i)].Conductivity[index_1]
        index_2 = var2['cast' + str(i)].Conductivity.index[0]
        v2 = var2['cast' + str(i)].Conductivity[index_2]
        # shift C for n scans
        var1['cast' + str(i)].Conductivity = var1['cast' + str(i)].Conductivity.shift(periods=shifted_scan_number, fill_value = v1)
        #calculates SP from C using the PSS-78 algorithm (2 < SP < 42)
        var1['cast' + str(i)].Salinity = gsw.SP_from_C(var1['cast' + str(i)].Conductivity, var1['cast' + str(i)].Temperature, var1['cast' + str(i)].Pressure)
        var2['cast' + str(i)].Conductivity = var2['cast' + str(i)].Conductivity.shift(periods=shifted_scan_number, fill_value = v2)
        var2['cast' + str(i)].Salinity = gsw.SP_from_C(var2['cast' + str(i)].Conductivity, var2['cast' + str(i)].Temperature, var2['cast' + str(i)].Pressure)
    metadata_dict['Processing_history'] += '-SHIFT parameters:|' \
                                           ' Shift Channel: Conductivity|' \
                                           ' # of Records to Delay (-ve for Advance):|' \
                                           ' Shift = {}'.format(str(shifted_scan_number)) + '|' \
                                           ' Salinity was recalculated after shift|'
    metadata_dict['SHIFT_Conductivity_Time'] = datetime.now()

    return var1, var2

#Example:
cast_d_shift_c, cast_u_shift_c = SHIFT_CONDUCTIVITY(cast_d_filtered, cast_u_filtered, shifted_scan_number = 2, metadata_dict = metadata)  # delay conductivity data by 2 scans

# Plot Salinity and T-S for post-shift check
# Salinity
fig, ax = plt.subplots()
ax.plot(cast_d_filtered['cast1'].Salinity, cast_d_filtered['cast1'].Pressure, color='blue', label='Pre-shift')
#ax.plot(cast_u_filtered['cast1'].Salinity, cast_u_filtered['cast1'].Pressure, '--', color='blue', label='Pre-shift')
ax.plot(cast_d_shift_c['cast1'].Salinity, cast_d_shift_c['cast1'].Pressure, color='red', label='Post-shift')
#ax.plot(cast_u_shift_c['cast1'].Salinity, cast_u_shift_c['cast1'].Pressure, '--', color='red', label='Post-shift')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel(' ')
ax.set_ylabel('Pressure (decibar)')
ax.legend()


# TS Plot
fig, ax = plt.subplots()
ax.plot(cast_d_filtered['cast4'].Temperature, cast_d_filtered['cast4'].Salinity, color='blue', label='Pre-shift')
#ax.plot(cast_u_filtered['cast1'].Temperature, cast_u_filtered['cast1'].Salinity, '--', color='blue', label='Pre-shift')
ax.plot(cast_d_shift_c['cast4'].Temperature, cast_d_shift_c['cast4'].Salinity, color='red', label='Post-shift')
#ax.plot(cast_u_shift_c['cast1'].Temperature, cast_u_shift_c['cast1'].Salinity, '--', color='red', label='Post-shift')
ax.set_xlabel('Salinity')
ax.set_ylabel('Temperature (C)')
ax.set_title('T-S Plot')
ax.legend()

#------------------------------    Step 12: Shift Oxygen   ----------------------------------------------------
#input variable: cast_d_filtered, cast_u_filtered, try 2-3s (12-18 scans)

def SHIFT_OXYGEN(var_downcast, var_upcast, shifted_scan_number, metadata_dict): # n: number of scans shifted. +: delay; -: advance
    """
     Advance oxygen data by 2-3s
     Inputs:
         - downcast and upcast data dictionaries, metadata dictionary
     Outputs:
         - two dictionaries containing downcast and upcast profiles
     """
    cast_number = len(var_downcast.keys())
    var1 = deepcopy(var_downcast)
    var2 = deepcopy(var_upcast)
    for i in range(1, cast_number+1, 1):
        index_1 = var1['cast' + str(i)].Oxygen.index[-1]
        v1 = var1['cast' + str(i)].Oxygen[index_1]
        index_2 = var2['cast' + str(i)].Oxygen.index[-1]
        v2 = var2['cast' + str(i)].Oxygen[index_2]
        # shift C for n scans
        var1['cast' + str(i)].Oxygen = var1['cast' + str(i)].Oxygen.shift(periods=shifted_scan_number, fill_value = v1)
        var2['cast' + str(i)].Oxygen = var2['cast' + str(i)].Oxygen.shift(periods=shifted_scan_number, fill_value = v2)
    metadata_dict['Processing_history'] += '-SHIFT parameters:|' \
                                           ' Shift Channel: Oxygen:Dissolved:Saturation|' \
                                           ' # of Records to Delay (-ve for Advance):|' \
                                           ' Shift = {}'.format(str(shifted_scan_number)) + '|'
    metadata_dict['SHIFT_Oxygen_Time'] = datetime.now()

    return var1, var2

#Example:
cast_d_shift_o, cast_u_shift_o = SHIFT_OXYGEN(cast_d_shift_c, cast_u_shift_c, shifted_scan_number = -11, metadata_dict = metadata)  # advance oxygen data by 11 scans

#plot T-O2 before and after shift to check the results
# T-O2 Plot to check whether the shift bring the downcas and upcast together
fig, ax = plt.subplots()
ax.plot(cast_d_shift_c['cast2'].Temperature, cast_d_shift_c['cast2'].Oxygen, color='blue', label='Pre-shift')
ax.plot(cast_u_shift_c['cast2'].Temperature, cast_u_shift_c['cast2'].Oxygen, '--', color='blue', label='Pre-shift')
ax.plot(cast_d_shift_o['cast2'].Temperature, cast_d_shift_o['cast2'].Oxygen, color='red', label='Post-shift')
ax.plot(cast_u_shift_o['cast2'].Temperature, cast_u_shift_o['cast2'].Oxygen, '--', color='red', label='Post-shift')
ax.set_xlabel('Oxygen Saturation (%)')
ax.set_ylabel('Temperature (C)')
ax.set_title('T-S Plot')
ax.legend()


#------------------------------    Step 13: Delete (swells/slow drop)  ----------------------------------------------------
#correct for the wake effect, remove the pressure reversal

def DELETE_PRESSURE_REVERSAL(var_downcast, var_upcast, metadata_dict):
    """
     Detect and delete pressure reversal
     Inputs:
         - downcast and upcast data dictionaries, metadata dictionary
     Outputs:
         - two dictionaries containing downcast and upcast profiles
     """
    cast_number = len(var_downcast.keys())
    var1 = deepcopy(var_downcast)
    var2 = deepcopy(var_upcast)
    for i in range(1, cast_number+1, 1):
        press = var1['cast' + str(i)].Pressure.values
        ref = press[0]
        inversions = np.diff(np.r_[press, press[-1]]) < 0
        mask = np.zeros_like(inversions)
        for k, p in enumerate(inversions):
            if p:
                ref = press[k]
                cut = press[k+1:] < ref
                mask[k + 1:][cut] = True
        var1['cast' + str(i)][mask] = np.NaN

    for i in range(1, cast_number+1, 1):
        press = var2['cast' + str(i)].Pressure.values
        ref = press[0]
        inversions = np.diff(np.r_[press, press[-1]]) > 0
        mask = np.zeros_like(inversions)
        for k, p in enumerate(inversions):
            if p:
                ref = press[k]
                cut = press[k+1:] > ref
                mask[k + 1:][cut] = True
        var2['cast' + str(i)][mask] = np.NaN
    metadata_dict['Processing_history'] += '-DELETE_PRESSURE_REVERSAL parameters:|' \
                                           ' Remove pressure reversals|'
    metadata_dict['DELETE_PRESSURE_REVERSAL_Time'] = datetime.now()

    return var1, var2

#Example:
cast_d_wakeeffect, cast_u_wakeeffect = DELETE_PRESSURE_REVERSAL(cast_d_shift_o, cast_u_shift_o, metadata_dict = metadata)

# Plot Salinity and T-S to check the index after shift
# Salinity
fig, ax = plt.subplots()
ax.plot(cast_d_shift_o['cast1'].Salinity, cast_d_shift_o['cast1'].Pressure, color='blue', label='Pre-shift')
#ax.plot(cast_u_filtered['cast1'].Salinity, cast_u_filtered['cast1'].Pressure, '--', color='blue', label='Pre-shift')
ax.plot(cast_d_wakeeffect['cast1'].Salinity, cast_d_wakeeffect['cast1'].Pressure, color='red', label='Post-shift')
#ax.plot(cast_u_wakeeffect['cast1'].Salinity, cast_u_wakeeffect['cast1'].Pressure, '--', color='red', label='Post-shift')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel(' ')
ax.set_ylabel('Pressure (decibar)')
ax.legend()


# TS Plot
fig, ax = plt.subplots()
ax.plot(cast_d_shift_o['cast1'].Temperature, cast_d_shift_o['cast1'].Salinity, color='blue', label='Pre-shift')
#ax.plot(cast_u_filtered['cast1'].Temperature, cast_u_filtered['cast1'].Salinity, '--', color='blue', label='Pre-shift')
ax.plot(cast_d_wakeeffect['cast1'].Temperature, cast_d_wakeeffect['cast1'].Salinity, color='red', label='Post-shift')
#ax.plot(cast_u_wakeeffect['cast1'].Temperature, cast_u_wakeeffect['cast1'].Salinity, '--', color='red', label='Post-shift')
ax.set_xlabel('Salinity')
ax.set_ylabel('Temperature (C)')
ax.set_title('T-S Plot')
ax.legend()


#-------------------------  Plot processed profiles    ------------------------------------------------------------
#plot salinity of all cast together
fig, ax = plt.subplots()
ax.plot(cast_d['cast3'].Salinity, cast_d['cast3'].Pressure, color='blue', label='cast1')
#ax.plot(cast_u['cast1'].Salinity, cast_u['cast1'].Pressure, '--', color='blue', label='cast1')
ax.plot(cast_d_wakeeffect['cast3'].Salinity, cast_d_wakeeffect['cast3'].Pressure, color='red', label='cast1')
#ax.plot(cast_u_wakeeffect['cast1'].Salinity, cast_u_wakeeffect['cast1'].Pressure, '--', color='red', label='cast1')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Salinity')
ax.set_ylabel('Pressure (decibar)')
ax.legend()


#plot temperature of all cast together
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Temperature, cast_d['cast1'].Pressure, color='blue', label='cast1')
#ax.plot(cast_u['cast1'].Temperature, cast_u['cast1'].Pressure, '--', color='blue', label='cast1')
ax.plot(cast_d_wakeeffect['cast1'].Temperature, cast_d_wakeeffect['cast1'].Pressure, color='red', label='cast1')
#ax.plot(cast_u_wakeeffect['cast1'].Temperature, cast_u_wakeeffect['cast1'].Pressure, '--', color='red', label='cast1')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Temperature(C)')
ax.set_ylabel('Pressure (decibar)')
ax.legend()


#plot Conductivity of all cast together
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Conductivity, cast_d['cast1'].Pressure, color='blue', label='cast1')
#ax.plot(cast_u['cast2'].Conductivity[1:], cast_u['cast2'].Pressure[1:], '--', color='red', label='cast2')
ax.plot(cast_d_wakeeffect['cast1'].Conductivity, cast_d_wakeeffect['cast1'].Pressure, color='red', label='cast1')
#ax.plot(cast_u_wakeeffect['cast1'].Conductivity, cast_u_wakeeffect['cast1'].Pressure, color='red', label='cast1')
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Conductivity (S/cm)')
ax.set_ylabel('Pressure (decibar)')
ax.legend()


#plot Oxygen of all cast together
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Oxygen, cast_d['cast1'].Pressure, color='blue', label='cast1')
#ax.plot(cast_u['cast1'].Oxygen, cast_u['cast1'].Pressure, '--', color='blue', label='cast1')
ax.plot(cast_d_wakeeffect['cast1'].Oxygen, cast_d_wakeeffect['cast1'].Pressure, color='red', label='cast1')
#ax.plot(cast_d_wakeeffect['cast1'].Oxygen, cast_d_wakeeffect['cast1'].Pressure, color='red', label='cast1')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Oxygen Saturation (%)')   # Check unit here
ax.set_ylabel('Pressure (decibar)')
ax.legend()

#plot Fluorescence of all cast together
fig, ax = plt.subplots()
ax.plot(cast_d['cast1'].Fluorescence, cast_d['cast1'].Pressure, color='blue', label='cast1')
ax.plot(cast_u['cast1'].Fluorescence, cast_u['cast1'].Pressure, '--', color='blue', label='cast1')
ax.plot(cast_d_wakeeffect['cast1'].Fluorescence, cast_d_wakeeffect['cast1'].Pressure, color='red', label='cast1')
ax.plot(cast_u_wakeeffect['cast1'].Fluorescence, cast_u_wakeeffect['cast1'].Pressure, color='red', label='cast1')
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Fluorescence (ug/L)')   # Check unit here
ax.set_ylabel('Pressure (decibar)')
ax.legend()


#------------------------------    Step 14: BINAVE: bin averages  ----------------------------------------------------
#input variables: cast_d_wakeeffect, cast_u_wakeeffect
def BINAVE(var_downcast, var_upcast, interval, metadata_dict):
    """
     Bin average the profiles
     Note: Bin width and spacing are both universally chosen to be 1m in coastal waters
     Inputs:
         - downcast and upcast data dictionaries, metadata dictionary
     Outputs:
         - two dictionaries containing downcast and upcast profiles
     """
    cast_number = len(var_downcast.keys())
    var1 = deepcopy(var_downcast)
    var2 = deepcopy(var_upcast)
    for i in range(1, cast_number+1, 1):
        start_d = np.floor(np.nanmin(var1['cast' + str(i)].Pressure.values))
        stop_d = np.ceil(np.nanmax(var1['cast' + str(i)].Pressure.values))
        new_press_d = np.arange(start_d - 0.5, stop_d+1.5, interval)
        binned_d = pd.cut(var1['cast' + str(i)].Pressure, bins = new_press_d)
        obs_count_d = var1['cast' + str(i)].groupby(binned_d).size()
        var1['cast' + str(i)] = var1['cast' + str(i)].groupby(binned_d).mean()
        var1['cast' + str(i)]['Observation_counts'] = obs_count_d

        start_u = np.ceil(np.nanmax(var2['cast' + str(i)].Pressure.values))
        stop_u = np.floor(np.nanmin(var2['cast' + str(i)].Pressure.values))
        new_press_u = np.arange(start_u+0.5, stop_u-1.5, -interval)
        binned_u = pd.cut(var2['cast' + str(i)].Pressure, bins=new_press_u[::-1])
        obs_count_u = var2['cast' + str(i)].groupby(binned_u).size()
        var2['cast' + str(i)] = var2['cast' + str(i)].groupby(binned_u).mean()
        var2['cast' + str(i)] = var2['cast' + str(i)].sort_values('Depth', ascending=False)
        var2['cast' + str(i)]['Observation_counts'] = obs_count_u
    metadata_dict['Processing_history'] += '-BINAVE parameters:' \
                                           ' Bin channel = Pressure|' \
                                           ' Averaging interval = 1.00|' \
                                           ' Minimum bin value = 0.000|' \
                                           ' Average value were used|' \
                                           ' Interpolated values were NOT used for empty bins|' \
                                           ' Channel NUMBER_OF_BIN_RECORDS was added to file|'
    metadata_dict['BINAVE_Time'] = datetime.now()
    return var1, var2
    
#Example:
cast_d_binned, cast_u_binned = BINAVE(cast_d_wakeeffect, cast_u_wakeeffect, interval = 1, metadata_dict = metadata)



#------------------------------    Step 15: Final edits  ----------------------------------------------------
#input variables: cast_d_binned, cast_u_binned

def FINAL_EDIT(var_cast, metadata_dict):
    """
     Final editing the profiles: edit header information, correct the unit of conductivity
     Inputs:
         - downcast and upcast data dictionaries, metadata dictionary
     Outputs:
         - two dictionaries containing downcast and upcast profiles
     """
    cast_number = len(var_cast.keys())
    var = deepcopy(var_cast)
    col_list = ['Pressure', 'Depth', 'Temperature', 'Salinity', 'Fluorescence', 'Oxygen', 'Conductivity', 'Observation_counts']
    for i in range(1, cast_number+1, 1):
        var['cast' + str(i)] = var['cast' + str(i)].reset_index(drop=True) # drop index column
        var['cast' + str(i)] = var['cast' + str(i)][col_list]  # select columns
        var['cast' + str(i)].Conductivity = var['cast' + str(i)].Conductivity * 0.1 # convert Conductivity to S/m

        var['cast' + str(i)].Pressure = var['cast' + str(i)].Pressure.apply('{:,.1f}'.format)
        var['cast' + str(i)].Depth = var['cast' + str(i)].Depth.apply('{:,.1f}'.format)
        var['cast' + str(i)].Temperature = var['cast' + str(i)].Temperature.apply('{:,.4f}'.format)
        var['cast' + str(i)].Salinity = var['cast' + str(i)].Salinity.apply('{:,.4f}'.format)
        var['cast' + str(i)].Fluorescence = var['cast' + str(i)].Fluorescence.apply('{:,.3f}'.format)
        var['cast' + str(i)].Oxygen = var['cast' + str(i)].Oxygen.apply('{:,.2f}'.format)
        var['cast' + str(i)].Conductivity = var['cast' + str(i)].Conductivity.apply('{:,.5f}'.format)
        var['cast' + str(i)].columns = ['Pressure', 'Depth', 'Temperature', 'Salinity', 'Fluorescence:URU',
                                         'Oxygen:Dissolved:Satuation:RBR', 'Conductivity', 'Number_of_bin_records']
    metadata_dict['Processing_history'] += '-Remove Channels:|' \
                                           ' The following CHANNEL(S) were removed:|' \
                                           ' Date|' \
                                           ' TIME:UTC|' \
                                           '-CALIB parameters:|' \
                                           ' Calobration type = Correct|' \
                                           ' Calibration applied:|' \
                                           ' Conductivity (S/m) = 0.1* Conductivity (mS/cm)|'
    metadata_dict['FINALEDIT_Time'] = datetime.now()
    return var

#Example:
cast_d_final = FINAL_EDIT(cast_d_binned, metadata_dict=metadata)


#----------------------------   Prepare .ctd files with IOS Header File   ---------------------------------------------------------------------

# define function to write file section
def write_file(cast_number, cast_original, cast_final, metadata_dict):
    """
     Bin average the profiles
     Inputs:
         - cast_number, cast_original = cast, cast_final = cast_d_final, metadata_dict = metadata
     Outputs:
         - two dictionaries containing downcast and upcast profiles
     """
    start_time = pd.to_datetime(cast_original['cast' + str(cast_number)].Date.values[0] + ' ' + cast_original['cast' + str(cast_number)].TIME.values[0]).strftime("%Y/%m/%d %H:%M:%S.%f")[0:-3]
    end_time = pd.to_datetime(cast_original['cast' + str(cast_number)].Date.values[-1] + ' ' + cast_original['cast' + str(cast_number)].TIME.values[-1]).strftime("%Y/%m/%d %H:%M:%S.%f")[0:-3]
    number_of_records = str(cast_final['cast' + str(cast_number)].shape[0])  # number of ensumbles
    data_description = metadata_dict['Data_description']
    number_of_channels = str(cast_final['cast' + str(cast_number)].shape[1])
    nan = -99
    file_type = "ASCII"

    print("*FILE")
    print("    " + '{:20}'.format('START TIME') + ": UTC " + start_time)
    print("    " + '{:20}'.format('END TIME') + ": UTC " + end_time)
    print("    " + '{:20}'.format('NUMBER OF RECORDS') + ": " + number_of_records)
    print("    " + '{:20}'.format('DATA DESCRIPTION') + ": " + data_description)
    print("    " + '{:20}'.format('FILE TYPE') + ": " + file_type)
    print("    " + '{:20}'.format('NUMBER OF CHANNELS') + ": " + number_of_channels)
    print()
    print('{:>20}'.format('$TABLE: CHANNELS'))
    print('    ' + '! No Name                               Units            Minimum          Maximum')
    print('    ' + '!--- ---------------------------------  ---------------  ---------------  ---------------')

    print('{:>8}'.format('1') + " " + '{:35}'.format(list(cast_final['cast' + str(cast_number)].columns)[0]) + '{:17}'.format(
        "decibar") + '{:17}'.format(str(np.nanmin(cast_final['cast' + str(cast_number)].Pressure.astype(np.float))))+ '{:17}'.format(str(np.nanmax(cast_final['cast' + str(cast_number)].Pressure.astype(np.float)))))

    print('{:>8}'.format('2') + " " + '{:35}'.format(list(cast_final['cast' + str(cast_number)].columns)[1]) + '{:17}'.format(
        "meters") + '{:17}'.format(str(np.nanmin(cast_final['cast' + str(cast_number)].Depth.astype(np.float)))) + '{:17}'.format(str(np.nanmax(cast_final['cast' + str(cast_number)].Depth.astype(np.float)))))

    print('{:>8}'.format('3') + " " + '{:35}'.format(list(cast_final['cast' + str(cast_number)].columns)[2]) + '{:17}'.format(
        "'deg C(ITS90)'") + '{:17}'.format(str(np.nanmin(cast_final['cast' + str(cast_number)].Temperature.astype(np.float)))) + '{:17}'.format(str(np.nanmax(cast_final['cast' + str(cast_number)].Temperature.astype(np.float)))))

    print('{:>8}'.format('4') + " " + '{:35}'.format(list(cast_final['cast' + str(cast_number)].columns)[3]) + '{:17}'.format(
        "PSS-78") + '{:17}'.format(str(np.nanmin(cast_final['cast' + str(cast_number)].Salinity.astype(np.float)))) + '{:17}'.format(str(float('%.04f'%np.nanmax(cast_final['cast' + str(cast_number)].Salinity.astype(np.float))))))

    print('{:>8}'.format('4') + " " + '{:35}'.format(list(cast_final['cast' + str(cast_number)].columns)[4]) + '{:17}'.format(
        "mg/m^3") + '{:17}'.format(str(np.nanmin(cast_final['cast' + str(cast_number)]['Fluorescence:URU'].astype(np.float)))) + '{:17}'.format(str(float('%.03f'%np.nanmax(cast_final['cast' + str(cast_number)]['Fluorescence:URU'].astype(np.float))))))

    print('{:>8}'.format('5') + " " + '{:35}'.format(list(cast_final['cast' + str(cast_number)].columns)[5]) + '{:17}'.format(
        "%") + '{:17}'.format(str(np.nanmin(cast_final['cast' + str(cast_number)]['Oxygen:Dissolved:Satuation:RBR'].astype(np.float)))) + '{:17}'.format(str(float('%.04f'%np.nanmax(cast_final['cast' + str(cast_number)]['Oxygen:Dissolved:Satuation:RBR'].astype(np.float))))))

    print('{:>8}'.format('6') + " " + '{:35}'.format(list(cast_final['cast' + str(cast_number)].columns)[6]) + '{:17}'.format(
        "S/m") + '{:17}'.format(str(np.nanmin(cast_final['cast' + str(cast_number)].Conductivity.astype(np.float)))) + '{:17}'.format(str(float('%.05f'%np.nanmax(cast_final['cast' + str(cast_number)].Conductivity.astype(np.float))))))

    print('{:>8}'.format('7') + " " + '{:35}'.format(list(cast_final['cast' + str(cast_number)].columns)[7]) + '{:17}'.format(
        "n/a") + '{:17}'.format(str(np.nanmin(cast_final['cast' + str(cast_number)]['Number_of_bin_records'].astype(np.float)))) + '{:17}'.format(str(np.nanmax(cast_final['cast' + str(cast_number)]['Number_of_bin_records'].astype(np.float)))))

    # Add in table of Channel summary
    print('{:>8}'.format('$END'))
    print()
    print('{:>26}'.format('$TABLE: CHANNEL DETAILS'))
    print('    ' + '! No  Pad            Start  Width  Format      Type  Decimal_Places')
    print('    ' + '!---  -------------  -----  -----  ----------  ----  --------------')
    # print('{:>8}'.format('1') + "  " + '{:15}'.format("' '") + '{:7}'.format(' ') + '{:7}'.format("' '") + '{:22}'.format('YYYY-MM-DDThh:mm:ssZ') + '{:6}'.format('D, T') + '{:14}'.format("' '"))
    print(
        '{:>8}'.format('1') + "  " + '{:15}'.format(str(nan)) + '{:7}'.format("' '") + '{:7}'.format(str(7)) + '{:12}'.format(
            'F') + '{:6}'.format('R4') + '{:3}'.format(1))
    print(
        '{:>8}'.format('2') + "  " + '{:15}'.format(str(nan)) + '{:7}'.format("' '") + '{:7}'.format("' '") + '{:12}'.format(
            'F7.1') + '{:6}'.format('R4') + '{:3}'.format("' '"))
    print(
        '{:>8}'.format('3') + "  " + '{:15}'.format(str(nan)) + '{:7}'.format("' '") + '{:7}'.format(str(9)) + '{:12}'.format(
            'F') + '{:6}'.format('R4') + '{:3}'.format(4))
    print(
        '{:>8}'.format('4') + "  " + '{:15}'.format(str(nan)) + '{:7}'.format("' '") + '{:7}'.format("' '") + '{:12}'.format(
            'F9.4') + '{:6}'.format('R4') + '{:3}'.format("' '"))
    print(
        '{:>8}'.format('5') + "  " + '{:15}'.format(str(nan)) + '{:7}'.format("' '") + '{:7}'.format(str(8)) + '{:12}'.format(
            'F') + '{:6}'.format('R4') + '{:3}'.format(3))
    print(
        '{:>8}'.format('6') + "  " + '{:15}'.format(str(nan)) + '{:7}'.format("' '") + '{:7}'.format(str(8)) + '{:12}'.format(
            'F') + '{:6}'.format('R4') + '{:3}'.format(2))
    print(
        '{:>8}'.format('7') + "  " + '{:15}'.format(str(nan)) + '{:7}'.format("' '") + '{:7}'.format(str(10)) + '{:12}'.format(
            'F') + '{:6}'.format('R4') + '{:3}'.format(5))
    print(
        '{:>8}'.format('8') + "  " + '{:15}'.format("' '") + '{:7}'.format("' '") + '{:7}'.format(str(5)) + '{:12}'.format(
            'I') + '{:6}'.format('I') + '{:3}'.format(0))
    # Add in table of Channel detail summary
    print('{:>8}'.format('$END'))
    print()


# define function to write administation section
def write_admin(metadata_dict):
    mission = metadata_dict["Mission"]
    agency = metadata_dict["Agency"]
    country = metadata_dict["Country"]
    project = metadata_dict["Project"]
    scientist = metadata_dict["Scientist"]
    platform = metadata_dict["Platform"]
    print("*ADMINISTRATION")
    print("    " + '{:20}'.format('AGENCY') + ": " + agency)
    print("    " + '{:20}'.format('COUNTRY') + ": " + country)
    print("    " + '{:20}'.format('PROJECT') + ": " + project)
    print("    " + '{:20}'.format('SCIENTIST') + ": " + scientist)
    print("    " + '{:20}'.format('PLATFORM ') + ": " + platform)
    print()


def write_location(cast_number, metadata_dict):
    """
     write location part in IOS header file
     Inputs:
         - cast_number, metadata_list
     Outputs:
         - part of txt file
     """
    station_number = metadata_dict['Location']['LOC:STATION'].tolist()
    event_number = metadata_dict['Location']['LOC:Event Number'].tolist()
    lon = metadata_dict['Location']['LOC:LONGITUDE'].tolist()
    lat = metadata_dict['Location']['LOC:LATITUDE'].tolist()
    print("*LOCATION")
    print("    " + '{:20}'.format('STATION') + ": " + str(station_number[cast_number-1]))
    print("    " + '{:20}'.format('LATITUDE') + ": " + lat[cast_number-1][0:10] + "   " + lat[cast_number-1][-14:-1] + ")")
    print("    " + '{:20}'.format('LONGITUDE') + ": " + lon[cast_number-1])
    print()



# define function to write instrument info
def write_instrument(metadata_dict):
    model = metadata_dict['Instrument_Model']
    serial_number = f'{0:0}' + metadata_dict['Serial_number']
    data_description = metadata_dict['Data_description']
    instrument_type = metadata_dict['Instrument_type']
    print("*INSTRUMENT")
    print("    MODEL               : " + model)
    print("    SERIAL NUMBER       : " + serial_number)
    print("    DATA DESCRIPTION    : " + data_description + "                               ! custom item")
    print("    INSTRUMENT TYPE     : " + instrument_type + "                           ! custom item")
    print()


# define function to write raw info
def write_history(cast_original, cast_clip, cast_filtered, cast_shift_c, cast_shift_o, cast_wakeeffect, cast_binned, cast_final, metadata_dict, cast_number):
    print("*HISTORY")
    print()
    print("    TABLES: PROGRAMS")
    print("    !   Name      Vers    Date        Time      Recs In    Recs Out")
    print("    !   --------  ------  ----------  --------  ---------  ---------")
    print("        CALIB     " + '{:8}'.format(str(1.0))
          + '{:12}'.format(metadata['CALIB_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[0])
          + '{:10}'.format(metadata['CALIB_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[1])
          + '{:>9}'.format(str(cast_original['cast' + str(cast_number)].shape[0]))
          + '{:>11}'.format(str(cast_original['cast' + str(cast_number)].shape[0])))
    print("        CLIP      " + '{:8}'.format(str(1.0))
          + '{:12}'.format(metadata['CLIP_D_Time' + str(cast_number)].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[0])
          + '{:10}'.format(metadata['CLIP_D_Time' + str(cast_number)].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[1])
          + '{:>9}'.format(str(cast_original['cast' + str(cast_number)].shape[0]))
          + '{:>11}'.format(str(cast_clip['cast' + str(cast_number)].shape[0])))
    print("        FILTER    " + '{:8}'.format(str(1.0))
          + '{:12}'.format(metadata['FILTER_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[0])
          + '{:10}'.format(metadata['FILTER_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[1])
          + '{:>9}'.format(str(cast_clip['cast' + str(cast_number)].shape[0]))
          + '{:>11}'.format(str(cast_filtered['cast' + str(cast_number)].shape[0])))
    print("        SHIFT     " + '{:8}'.format(str(1.0))
          + '{:12}'.format(metadata['SHIFT_Conductivity_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[0])
          + '{:10}'.format(metadata['SHIFT_Conductivity_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[1])
          + '{:>9}'.format(str(cast_filtered['cast' + str(cast_number)].shape[0]))
          + '{:>11}'.format(str(cast_shift_c['cast' + str(cast_number)].shape[0])))
    print("        SHIFT     " + '{:8}'.format(str(1.0))
          + '{:12}'.format(metadata['SHIFT_Oxygen_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[0])
          + '{:10}'.format(metadata['SHIFT_Oxygen_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[1])
          + '{:>9}'.format(str(cast_shift_c['cast' + str(cast_number)].shape[0]))
          + '{:>11}'.format(str(cast_shift_o['cast' + str(cast_number)].shape[0])))
    print("        DELETE    " + '{:8}'.format(str(1.0))
          + '{:12}'.format(metadata['DELETE_PRESSURE_REVERSAL_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[0])
          + '{:10}'.format(metadata['DELETE_PRESSURE_REVERSAL_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[1])
          + '{:>9}'.format(str(cast_shift_o['cast' + str(cast_number)].shape[0]))
          + '{:>11}'.format(str(cast_wakeeffect['cast' + str(cast_number)].shape[0]-list(cast_wakeeffect['cast' + str(cast_number)].isna().sum())[0])))
    print("        BINAVE    " + '{:8}'.format(str(1.0))
          + '{:12}'.format(metadata['BINAVE_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[0])
          + '{:10}'.format(metadata['BINAVE_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[1])
          + '{:>9}'.format(str(cast_wakeeffect['cast' + str(cast_number)].shape[0]))
          + '{:>11}'.format(str(cast_binned['cast' + str(cast_number)].shape[0])))
    print("        EDIT      " + '{:8}'.format(str(1.0))
          + '{:12}'.format(metadata['FINALEDIT_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[0])
          + '{:10}'.format(metadata['FINALEDIT_Time'].strftime("%Y/%m/%d %H:%M:%S.%f")[0:-7].split(" ")[1])
          + '{:>9}'.format(str(cast_binned['cast' + str(cast_number)].shape[0]))
          + '{:>11}'.format(str(cast_final['cast' + str(cast_number)].shape[0])))

    print("    $END")
    print(" $REMARKS")

    list_number = len(metadata['Processing_history'].split("|"))
    for i in range(0, list_number, 1):
        print("     " + metadata['Processing_history'].split("|")[i])
    print("$END")
    print()

def write_comments(metadata_dict):
    cruise_ID = metadata["Mission"]
    print("*COMMENTS")
    print("    " + "-"*85)
    print()
    print("    Data Processing Notes:")
    print("    " + "-"*22)
    print("       " + "No calibration sampling was available")
    print("       " + "For details on the processing see document: " + cruise_ID + "_Processing_Report.doc")
    print("!--1--- --2--- ---3---- ---4---- ---5--- ---6--- ----7---- -8--")
    print("!Pressu Depth  Temperat Salinity Fluores Oxygen: Conductiv Numb")
    print("!re            ure               cence:  Dissolv ity       er_o")
    print("!                                URU     ed:               ~bin")
    print("!                                        Satuati           _rec")
    print("!                                        on:RBR            ords")
    print("!------ ------ -------- -------- ------- ------- --------- ----")
    print("*END OF HEADER")

def write_data(cast_data, cast_number):
    for i in range(len(cast_data['cast' + str(cast_number)])):
        #print(cast_data['cast' + str(cast_number)]['Pressure'][i] + cast_data['cast' + str(cast_number)]['Depth'][i] + "  ")
        print('{:>7}'.format(cast_data['cast' + str(cast_number)].Pressure[i]) + " "
            + '{:>6}'.format(cast_data['cast' + str(cast_number)].Depth[i]) + " "
            + '{:>8}'.format(cast_data['cast' + str(cast_number)].Temperature[i]) + " "
            + '{:>8}'.format(cast_data['cast' + str(cast_number)].Salinity[i]) + " "
            + '{:>7}'.format(cast_data['cast' + str(cast_number)]['Fluorescence:URU'][i]) + " "
            + '{:>7}'.format(cast_data['cast' + str(cast_number)]['Oxygen:Dissolved:Satuation:RBR'][i]) + " "
            + '{:>9}'.format(cast_data['cast' + str(cast_number)]['Conductivity'][i]) + " "
            + '{:>4}'.format(cast_data['cast' + str(cast_number)]['Number_of_bin_records'][i]) + " ")


def main_header(dest_dir, n_cast, meta_data, cast_d, cast_d_clip, cast_d_filtered, cast_d_shift_c, cast_d_shift_o, cast_d_wakeeffect, cast_d_binned, cast_d_final):
    f_name = dest_dir.split("/")[-2]
    f_output = f_name.split("_")[0] + '-' + f'{n_cast:04}' + ".ctd"
    output = dest_dir + "CTD/" + f_output
    newnc_dir = './{}CTD/'.format(dest_dir)
    if not os.path.exists(newnc_dir):
        os.makedirs(newnc_dir)
    # Start
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S.%f")[0:-4]
    IOS_string = '*IOS HEADER VERSION 2.0      2020/03/01 2020/04/15 PYTHON'

    orig_stdout = sys.stdout
    file_handle = open(output, 'wt')
    try:
        sys.stdout = file_handle
        print("*" + dt_string)
        print(IOS_string)
        print() # print("\n") pring("\n" * 40)
        write_file(cast_number=n_cast, cast_original=cast, cast_final=cast_d_final, metadata_dict=meta_data)
        write_admin(metadata_dict=meta_data)
        write_location(cast_number=n_cast, metadata_dict=metadata)
        write_instrument(metadata_dict=meta_data)
        write_history(cast_original=cast_d, cast_clip=cast_d_clip, cast_filtered=cast_d_filtered,
                      cast_shift_c=cast_d_shift_c, cast_shift_o=cast_d_shift_o, cast_wakeeffect=cast_d_wakeeffect,
                      cast_binned=cast_d_binned, cast_final=cast_d_final, metadata_dict=meta_data, cast_number=n_cast)
        write_comments(metadata_dict=meta_data)
        write_data(cast_data=cast_d_final, cast_number=n_cast)
        sys.stdout.flush() #Recommended by Tom
    finally:
        sys.stdout = orig_stdout
    return os.path.abspath(output)

#Example:
main_header(dest_dir = '/home/guanl/Desktop/Projects/RBR/Processing/2019-107/', n_cast = 1,
            meta_data = metadata, cast_d = cast_d, cast_d_clip = cast_d_clip, cast_d_filtered = cast_d_filtered,
            cast_d_shift_c = cast_d_shift_c, cast_d_shift_o = cast_d_shift_o,
            cast_d_wakeeffect = cast_d_wakeeffect, cast_d_binned = cast_d_binned, cast_d_final = cast_d_final)






