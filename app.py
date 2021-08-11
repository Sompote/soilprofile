import os
from flask import Flask, render_template, request
import base64
import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from scipy import stats
from matplotlib.backends.backend_agg import FigureCanvasAgg

import statistics as stat
import math
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.optimize import curve_fit


matplotlib.use('Agg')

app = Flask(__name__, template_folder='./Templates')


UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pd.options.display.max_columns = 20
pd.options.display.max_rows = None
pd.options.mode.chained_assignment = None

url = "/Users/sompoteyouwai/PycharmProjects/soil_layer/BH-2.xlsx"  # @param {type:"string"}

df = pd.read_excel(url, skiprows=2, usecols='A, B, C, D, E, G, H, I, J, K, L')


###################define function and read data####################
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def func(x, a, b):
    return a + np.sqrt(b * x)


def dist_point_to_line(x1, y1, x2, y2, x3, y3):
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    distance = np.abs((-m * x3 + y3 - c)) / np.sqrt(m ** 2 + 1)
    return distance



def linear_order_clustering1(km_labels, outlier_tolerance=1):  ######ก๊อปมาครับ ฟังก์ชันนี้
    '''Expects clustering outputs as an array/list'''
    global lastcluster
    prev_label = km_labels[0]  # keeps track of last seen item's real cluster
    cluster = 1  # like a counter for our new linear clustering outputs
    result = [cluster]  # initialize first entry
    for i, label in enumerate(km_labels[1:]):
        if prev_label == label:
            # just written for clarity of control flow,
            # do nothing special here
            pass
        else:  # current cluster label did not match previous label
            # check if previous cluster label reappears
            # on the right of current cluster label position
            # (aka current non-matching cluster is sandwiched
            # within a reasonable tolerance)
            if (outlier_tolerance and prev_label in km_labels[i + 1: i + 2 + outlier_tolerance]):
                label = prev_label  # if so, overwrite current label
            else:
                cluster += 1  # its genuinely a new cluster
        result.append(cluster)
        prev_label = label
    lastcluster = result[-1]
    return result


def linear_order_clustering2(km_labels, outlier_tolerance=1):  ######ก๊อปมาครับ ฟังก์ชันนี้
    '''Expects clustering outputs as an array/list'''
    global lastcluster, lastcluster2
    prev_label = km_labels[0]  # keeps track of last seen item's real cluster
    cluster = lastcluster + 1  # like a counter for our new linear clustering outputs
    result = [cluster]  # initialize first entry
    for i, label in enumerate(km_labels[1:]):
        if prev_label == label:
            # just written for clarity of control flow,
            # do nothing special here
            pass
        else:  # current cluster label did not match previous label
            # check if previous cluster label reappears
            # on the right of current cluster label position
            # (aka current non-matching cluster is sandwiched
            # within a reasonable tolerance)
            if (outlier_tolerance and prev_label in km_labels[i + 1: i + 2 + outlier_tolerance]):
                label = prev_label  # if so, overwrite current label
            else:
                cluster += 1  # its genuinely a new cluster
        result.append(cluster)
        prev_label = label
    lastcluster2 = result[-1]
    return result

def soil_layer(df):
    global lastcluster, lastcluster2
    ################################################
    ################################################
    ##################Curve fitting for Ncorr vs Phi#####################
    xData = np.array([5, 10, 17, 24, 30, 38, 46, 56, 70, 90])
    yData = np.array([28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
    yp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    popt, pcov = curve_fit(func, xData, yData)
    ybar = np.mean(yData)
    i = 0
    while i < 10:
        yp[i] = func(xData[i], *popt)
        i = i + 1
    totvar = np.sum((yData - ybar) ** 2)
    expvar = np.sum((yp - ybar) ** 2)
    R2 = expvar / totvar
    xFit = np.arange(5, 90, 0.1)
    ########################################################

    ################separate SS and ST and defining soil type in each row##################
    ndf = df
    ndf2 = ndf[ndf['SampleNo'].str.match('ST')]
    ndf = ndf[ndf['SampleNo'].str.match('SS')]
    if ndf.empty == False:
        # ndf['#200 (%)'] = ndf['#200 (%)'].fillna(method = 'ffill', limit = 1)
        ndf['#200 (%)'] = ndf['#200 (%)'].fillna(100)
        ndf.loc[(ndf['#200 (%)'] >= 50), 'Soil type'] = 0  # clay
        ndf.loc[(ndf['#200 (%)'] < 50), 'Soil type'] = 3  # sand
    else:
        pass
    if ndf2.empty == False:
        # ndf2['#200 (%)'] = ndf2['#200 (%)'].fillna(method = 'ffill', limit = 1)
        ndf2['#200 (%)'] = ndf2['#200 (%)'].fillna(100)
        ndf2.loc[(ndf2['#200 (%)'] >= 50), 'Soil type'] = 0  # clay
        ndf2.loc[(ndf2['#200 (%)'] < 50), 'Soil type'] = 3  # sand
    else:
        pass
    ###############################################

    ################Number of layer calculation (Elbow method)####################
    # For ST
    if ndf2.empty == False:
        ndf2['y (t/cu.m.)'].fillna(method='ffill', inplace=True)
        ndf2['y (t/cu.m.)'].fillna(method='bfill', inplace=True)
        ndf2['Wn %'].fillna(method='ffill', inplace=True)
        ndf2['Wn %'].fillna(method='bfill', inplace=True)
        ndf2['Su (t/sq.m.)'].fillna(method='ffill', inplace=True)
        ndf2['Su (t/sq.m.)'].fillna(method='bfill', inplace=True)

        ndf2.N.fillna(0, inplace=True)
        ndf2['Sumean'] = ndf2['Su (t/sq.m.)'].mean()  ######
        ndf2['STD_Su'] = stat.stdev(ndf2['Su (t/sq.m.)'])  #############
        ndf2['STDED_Su'] = (ndf2['Su (t/sq.m.)'] - ndf2.Sumean) / ndf2.STD_Su  ##########
        ndf2['Wnmean'] = ndf2['Wn %'].mean()  ######
        ndf2['STD_Wn'] = stat.stdev(ndf2['Wn %'])  #############
        ndf2['STDED_Wn'] = (ndf2['Wn %'] - ndf2['Wnmean']) / ndf2.STD_Wn  ##########
        ndf2['y (t/cu.m.)mean'] = ndf2['y (t/cu.m.)'].mean()  ######
        ndf2['STD_y (t/cu.m.)'] = stat.stdev(ndf2['y (t/cu.m.)'])  #############
        ndf2['STDED_y (t/cu.m.)'] = (ndf2['y (t/cu.m.)'] - ndf2['y (t/cu.m.)mean']) / ndf2[
            'STD_y (t/cu.m.)']  ##########
        ndf2['Tomean'] = ndf2['To'].mean()  ######
        ndf2['STD_To'] = stat.stdev(ndf2['To'])  #############
        ndf2['STDED_To'] = (ndf2['To'] - ndf2['Tomean']) / ndf2.STD_To  ##########
        Y = ndf2[['STDED_To', 'STDED_y (t/cu.m.)', 'STDED_Wn', 'STDED_Su', 'Soil type']]
        index = ndf2.index
        number_of_rows = len(index)
        y3 = []
        dist = []
        if 1 <= number_of_rows <= 5:
            ST_Layers = 1
        elif 5 < number_of_rows <= 12:
            K = range(1, number_of_rows + 1)
            for k in range(1, 1 + number_of_rows):
                km = KMeans(n_clusters=k, n_init=100, random_state=100)
                km.fit(Y)
                y3.append(km.inertia_)
            for k in range(1, 1 + number_of_rows):
                dist.append(dist_point_to_line(K[0], y3[0], K[-2], y3[K[-2]], K[k - 1], y3[k - 1]))
            ST_Layers = 1 + dist.index(max(dist))
        else:
            K = range(1, math.floor(number_of_rows / 5) + 10)
            for k in range(1, math.floor(number_of_rows / 5) + 10):
                km = KMeans(n_clusters=k, n_init=100, random_state=100)
                km.fit(Y)
                y3.append(km.inertia_)
            for k in range(1, math.floor(number_of_rows / 5) + 10):
                dist.append(dist_point_to_line(K[0], y3[0], K[-2], y3[K[-2]], K[k - 1], y3[k - 1]))
            ST_Layers = 1 + dist.index(max(dist))
    else:
        pass
    ###
    # For SS
    if ndf.empty == False:
        ndf['y (t/cu.m.)'].fillna(method='ffill', inplace=True)
        ndf['y (t/cu.m.)'].fillna(method='bfill', inplace=True)
        ndf['Wn %'].fillna(method='ffill', inplace=True)
        ndf['Wn %'].fillna(method='bfill', inplace=True)
        ndf.N.fillna(0, inplace=True)
        ndf['Nmean'] = ndf.N.mean()  ######
        ndf['STD_N'] = stat.stdev(ndf.N)  #############
        ndf['STDED_N'] = (ndf.N - ndf.Nmean) / ndf.STD_N  ##########
        ndf['Wnmean'] = ndf['Wn %'].mean()  ######
        ndf['STD_Wn'] = stat.stdev(ndf['Wn %'])  #############
        ndf['STDED_Wn'] = (ndf['Wn %'] - ndf['Wnmean']) / ndf.STD_Wn  ##########
        ndf['y (t/cu.m.)mean'] = ndf['y (t/cu.m.)'].mean()  ######
        ndf['STD_y (t/cu.m.)'] = stat.stdev(ndf['y (t/cu.m.)'])  #############
        ndf['STDED_y (t/cu.m.)'] = (ndf['y (t/cu.m.)'] - ndf['y (t/cu.m.)mean']) / ndf['STD_y (t/cu.m.)']  ##########
        ndf['Tomean'] = ndf['To'].mean()  ######
        ndf['STD_To'] = stat.stdev(ndf['To'])  #############
        ndf['STDED_To'] = (ndf['To'] - ndf['Tomean']) / ndf.STD_To  ##########
        X = ndf[['STDED_To', 'STDED_y (t/cu.m.)', 'STDED_Wn', 'STDED_N', 'Soil type']]
        index = ndf.index
        number_of_rows = len(index)
        y3 = []
        dist = []
        if 1 <= number_of_rows <= 5:
            SS_Layers = 1
        elif 5 < number_of_rows <= 12:
            K = range(1, number_of_rows + 1)
            for k in range(1, 1 + number_of_rows):
                km = KMeans(n_clusters=k, n_init=100, random_state=100)
                km.fit(X)
                y3.append(km.inertia_)
            for k in range(1, 1 + number_of_rows):
                dist.append(dist_point_to_line(K[0], y3[0], K[-2], y3[K[-2]], K[k - 1], y3[k - 1]))
            SS_Layers = 1 + dist.index(max(dist))
            plt.plot(K, y3)
        else:
            K = range(1, math.floor(number_of_rows / 5) + 10)
            for k in range(1, math.floor(number_of_rows / 5) + 10):
                km = KMeans(n_clusters=k, n_init=100, random_state=100)
                km.fit(X)
                y3.append(km.inertia_)
            for k in range(1, math.floor(number_of_rows / 5) + 10):
                dist.append(dist_point_to_line(K[0], y3[0], K[-2], y3[K[-2]], K[k - 1], y3[k - 1]))
            SS_Layers = 1 + dist.index(max(dist))
    else:
        pass
    ############################clustering data ST######################
    if ndf2.empty == False:
        model1 = KMeans(n_clusters=ST_Layers, n_init=100, random_state=100)
        y = model1.fit_predict(ndf2[['STDED_Wn', 'STDED_To', 'STDED_y (t/cu.m.)', 'Soil type']])
        ndf2['PseudoGroup'] = y
        model2 = KMeans(n_clusters=ST_Layers, n_init=100)
        x = model2.fit_predict(ndf2[['PseudoGroup', 'To', 'Soil type']])
        result1 = linear_order_clustering1(model2.labels_)
        ndf2['Group'] = result1
    else:
        pass
    ###################################################
    #################clustering data SS##################
    if ndf.empty == False and ndf2.empty == False:  # ndf=SS data, ndf2 = ST data
        model1 = KMeans(n_clusters=SS_Layers, n_init=100, random_state=0)
        y = model1.fit_predict(ndf[['STDED_N', 'STDED_Wn', 'STDED_To', 'STDED_y (t/cu.m.)', 'Soil type']])
        ndf['PseudoGroup'] = y
        model2 = KMeans(n_clusters=SS_Layers, n_init=100, random_state=0)
        x = model2.fit_predict(ndf[['PseudoGroup', 'To', 'Soil type']])
        result2 = linear_order_clustering2(model2.labels_)
        ndf['Group'] = result2
    elif ndf.empty == False and ndf2.empty == True:
        lastcluster = 0
        model1 = KMeans(n_clusters=SS_Layers, n_init=100, random_state=0)
        y = model1.fit_predict(ndf[['STDED_N', 'STDED_Wn', 'STDED_To', 'STDED_y (t/cu.m.)', 'Soil type']])
        ndf['PseudoGroup'] = y
        model2 = KMeans(n_clusters=SS_Layers, n_init=100, random_state=0)
        x = model2.fit_predict(ndf[['PseudoGroup', 'To', 'Soil type']])
        result2 = linear_order_clustering2(model2.labels_)
        ndf['Group'] = result2
    else:
        pass
    ###########################################
    # Combining the group that has one row with another group
    # val, val2 = number of group that has one row
    if ndf.empty == False:
        ndf.reset_index(inplace=True)
        stats = pd.DataFrame
        ndf['count'] = ndf.groupby('Group')['Group'].transform('count')
        if 1 in ndf['count'].values:
            stats = ndf.groupby('count').size().reset_index(name='countss')
            val = stats.at[0, 'countss']
            ndf.loc[(ndf['count'] == 1), 'Group'] = ndf.Group - 1
        else:
            val = 0
    else:
        pass
    if ndf2.empty == False:
        ndf2.reset_index(inplace=True)
        ndf2['count'] = ndf2.groupby('Group')['Group'].transform('count')
        if 1 in ndf2['count'].values:
            stats = ndf2.groupby('count').size().reset_index(name='countst')
            val2 = stats.at[0, 'countst']
            ndf2.loc[(ndf2['count'] == 1), 'Group'] = ndf2.Group - 1
        else:
            val2 = 0
    else:
        pass
    #############defining soil type in each layer####################
    # For ST#
    if ndf2.empty == False:
        ndf2['Soil type'] = ndf2.groupby(['Group'])['Soil type'].transform('mean')  #
    else:
        pass
    # For SS#
    if ndf.empty == False:
        ndf['Soil type'] = ndf.groupby(['Group'])['Soil type'].transform('mean')  #
    else:
        pass
    #################################################################
    ###########Parameters Calculation######################
    # For ST#
    if ndf2.empty == False:
        ndf2['Avg. y (t/cu.m.)'] = np.nan
        ndf2['Avg. Su(t/sq.m.)'] = np.nan
        ndf2['Avg. y (t/cu.m.)'] = ndf2.groupby(['Group'])['y (t/cu.m.)'].transform('mean')  #
        ndf2['Avg. Su(t/sq.m.)'] = ndf2.groupby(['Group'])['Su (t/sq.m.)'].transform('mean')  #
        ndf2['Wn %'] = ndf2.groupby(['Group'])['Wn %'].transform('mean')  #
    else:
        pass
    # For SS#
    if ndf.empty == False:
        ndf['Avg. y (t/cu.m.)'] = np.nan
        ndf['Avg. y (t/cu.m.)'] = ndf.groupby(['Group'])['y (t/cu.m.)'].transform('mean')  #
        ndf['Avg. N'] = np.nan
        ndf['Avg. N'] = ndf.groupby(['Group'])['N'].transform('mean')
        ndf.loc[(ndf['Soil type'] < 1.5), 'Ncorr'] = ndf['Avg. N']  # clay
        ndf.loc[(ndf['Soil type'] >= 1.5), 'Ncorr'] = 7.5 + 0.5 * ndf['Avg. N']  # sand
        ndf.loc[(ndf['Soil type'] < 1.5), 'Su from Ncorr'] = ndf['Ncorr'] * 5  # clay
        ndf.loc[(ndf['Soil type'] >= 1.5), 'Phi from Ncorr'] = func(ndf['Ncorr'], *popt)  # sand
        ndf['Wn %'] = ndf.groupby(['Group'])['Wn %'].transform('mean')  #
        ndf['Su from Ncorr'].fillna(0, inplace=True)
        ndf['Phi from Ncorr'].fillna(0, inplace=True)
    else:
        pass
    ###############################################
    ################plotting soil profile####################
    # Plot the line and ticks
    fig = plt.figure(figsize=(10, 10))
    # For ST
    if ndf2.empty == False:
        To = ndf2.groupby(['Group'], sort=False)['To'].max() * -1
        st_array = To.to_numpy()
        for i in range(lastcluster):
            plt.axhline(y=st_array[i], color='r', linestyle='-')
    else:
        pass
    # For SS
    if ndf.empty == False and ndf2.empty == False:
        To1 = ndf.groupby(['Group'], sort=False)['To'].max() * -1
        ss_array = To1.to_numpy()
        for i in range(lastcluster2 - lastcluster - 1):
            plt.axhline(y=ss_array[i], color='r', linestyle='-')
    elif ndf.empty == False and ndf2.empty == True:
        To1 = ndf.groupby(['Group'], sort=False)['To'].max() * -1
        ss_array = To1.to_numpy()
        for i in range(lastcluster2):
            plt.axhline(y=ss_array[i], color='r', linestyle='-')
    else:
        pass
    if ndf2.empty == False and ndf.empty == False:
        plt.yticks(np.concatenate((st_array, ss_array)))
        plt.ylim(ss_array[-1], 0)
    elif ndf2.empty == False and ndf.empty == True:
        plt.yticks((st_array))
        plt.ylim(st_array[-1], 0)
    else:
        plt.yticks((ss_array))
        plt.ylim(ss_array[-1], 0)
    plt.xlim(0, 6)
    plt.xticks([])
    ax = plt.gca()
    yticks = ax.yaxis.get_major_ticks()

    # Write soil type and parameters in each layer
    if ndf2.empty == False and ndf.empty == False:
        ST_Params = ndf2.groupby(['Group'], sort=False)['To', 'Wn %', 'Avg. Su(t/sq.m.)', 'Avg. y (t/cu.m.)'].mean()
        ST_Params['Tomax'] = ndf2.groupby(['Group'], sort=False)['To'].max()
        ST_Params['Avg. Su(t/sq.m.)'] = ST_Params['Avg. Su(t/sq.m.)'] * 9.81
        SS_Params = ndf.groupby(['Group'], sort=False)[
            'To', 'Wn %', 'Su from Ncorr', 'Phi from Ncorr', 'Avg. y (t/cu.m.)', 'N'].mean()
        SS_Params['Tomax'] = ndf.groupby(['Group'], sort=False)['To'].max()
        np.round(ST_Params, 2)
        np.round(SS_Params, 2)
        data = [{"low": 0.0000000001, "high": 12.5, "name": "Very soft clay"},
                {"low": 12.5, "high": 25, "name": "Soft clay"},
                {"low": 25, "high": 50, "name": "Medium clay"},
                {"low": 50, "high": 100, "name": "Stiff clay"},
                {"low": 100, "high": 200, "name": "Very stiff clay"},
                {"low": 200, "high": 20000, "name": "Hard clay"}, ]

        data2 = [{"low": 0, "high": 5, "name": "Very loose sand"},
                 {"low": 5, "high": 10, "name": "Loose sand"},
                 {"low": 10, "high": 30, "name": "Medium sand"},
                 {"low": 30, "high": 100000, "name": "Dense sand"}, ]

        myDF = pd.DataFrame(data)
        mySeries = pd.Series(ST_Params['Avg. Su(t/sq.m.)'])
        bins = list(myDF["high"])
        bins.insert(0, 0)
        ST_Params['Soil Type'] = pd.cut(mySeries, bins, labels=myDF["name"])
        ST_Params = ST_Params.values.tolist()
        number_of_layer = lastcluster - val2
        # Code below is for combining adjacent group that has similar properties
        for i in range(number_of_layer - 1):
            if i < number_of_layer - 1:
                if ST_Params[i][5] == ST_Params[i + 1][5]:
                    if ST_Params[i][2] - 7 <= ST_Params[i + 1][2] <= ST_Params[i][2] + 7:
                        if ST_Params[i][3] - 0.1 <= ST_Params[i + 1][3] <= ST_Params[i][3] + 0.1:
                            ST_Params[i][0] = (ST_Params[i][0] + ST_Params[i + 1][0]) / 2
                            ST_Params[i][1] = (ST_Params[i][1] + ST_Params[i + 1][1]) / 2
                            ST_Params[i][2] = (ST_Params[i][2] + ST_Params[i + 1][2]) / 2
                            ST_Params[i][3] = (ST_Params[i][3] + ST_Params[i + 1][3]) / 2
                            ST_Params[i][4] = ST_Params[i + 1][4]
                            yticks[i].label1.set_visible(False)
                            ax.lines[i].remove()
                            del ST_Params[i + 1]
                            number_of_layer = len(ST_Params)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            else:
                pass
        ############################################################
        myDF2 = pd.DataFrame(data2)
        mySeries3 = pd.Series(SS_Params['N'])
        bins3 = list(myDF2["high"])
        bins3.insert(0, 0)
        SS_Params['Soil Type'] = pd.cut(mySeries3, bins3, labels=myDF2["name"])
        mySeries2 = pd.Series(SS_Params['Su from Ncorr'])
        bins2 = list(myDF["high"])
        bins2.insert(0, 0)
        SS_Params['Soil Type2'] = pd.cut(mySeries2, bins2, labels=myDF["name"])
        SS_Params = SS_Params.ffill(axis=1)
        SS_Params = SS_Params.values.tolist()
        number_of_layer2 = lastcluster2 - lastcluster - val
        for i in range(lastcluster2 - lastcluster - 1):
            if i < lastcluster2 - lastcluster - 1 - val:
                if SS_Params[i][8] == SS_Params[i + 1][8] and 'sand' in str(SS_Params[i][8]):
                    if SS_Params[i][3] - 1 <= SS_Params[i + 1][3] <= SS_Params[i][3] + 1:
                        if SS_Params[i][4] - 0.1 <= SS_Params[i + 1][4] <= SS_Params[i][4] + 0.1:
                            SS_Params[i][0] = (SS_Params[i][0] + SS_Params[i + 1][0]) / 2
                            SS_Params[i][3] = (SS_Params[i][3] + SS_Params[i + 1][3]) / 2
                            SS_Params[i][4] = (SS_Params[i][4] + SS_Params[i + 1][4]) / 2
                            SS_Params[i][6] = SS_Params[i + 1][6]
                            yticks[i + lastcluster].label1.set_visible(False)
                            ax.lines[i + lastcluster].remove()
                            del SS_Params[i + 1]
                            number_of_layer2 = len(SS_Params)
                            lastcluster2 = lastcluster2 - 1
                        else:
                            pass
                    else:
                        pass
                else:
                    if SS_Params[i][2] - 20 <= SS_Params[i + 1][2] <= SS_Params[i][2] + 20:
                        if SS_Params[i][4] - 0.1 <= SS_Params[i + 1][4] <= SS_Params[i][4] + 0.1:
                            SS_Params[i][0] = (SS_Params[i][0] + SS_Params[i + 1][0]) / 2
                            SS_Params[i][2] = (SS_Params[i][2] + SS_Params[i + 1][2]) / 2
                            SS_Params[i][4] = (SS_Params[i][4] + SS_Params[i + 1][4]) / 2
                            yticks[i + lastcluster].label1.set_visible(False)
                            ax.lines[i + lastcluster].remove()
                            del SS_Params[i + 1]
                            number_of_layer2 = len(SS_Params)
                            lastcluster2 = lastcluster2 - 1
                        else:
                            pass
                    else:
                        pass
            else:
                pass
        for i in range(number_of_layer):
            plt.text((1.5), -1 * ST_Params[i][0], 'Su (kPa)  %.2f' % (ST_Params[i][2]))
            plt.text((3), -1 * ST_Params[i][0], 'y (t/cu.m.) %.2f' % (ST_Params[i][3]))
            plt.text(0.5, -1 * ST_Params[i][0], ST_Params[i][5])
        for i in range(number_of_layer2):
            plt.text((1.5), -1 * SS_Params[i][0], 'Su (kPa)  %.2f' % (SS_Params[i][2]))
            plt.text((3), -1 * SS_Params[i][0], 'Phi %.2f degree' % (SS_Params[i][3]))
            plt.text((4.5), -1 * SS_Params[i][0], 'y (t/cu.m.) %.2f' % (SS_Params[i][4]))
            plt.text(0.5, -1 * SS_Params[i][0], SS_Params[i][8])

    elif ndf2.empty == False and ndf.empty == True:
        ST_Params = ndf2.groupby(['Group'], sort=False)['To', 'Wn %', 'Avg. Su(t/sq.m.)', 'Avg. y (t/cu.m.)'].mean()
        ST_Params['Avg. Su(t/sq.m.)'] = ST_Params['Avg. Su(t/sq.m.)'] * 9.81
        ST_Params['Tomax'] = ndf2.groupby(['Group'], sort=False)['To'].max()
        np.round(ST_Params, 2)
        data = [{"low": 0, "high": 12.5, "name": "Very soft clay"},
                {"low": 12.5, "high": 25, "name": "Soft clay"},
                {"low": 25, "high": 50, "name": "Medium clay"},
                {"low": 50, "high": 100, "name": "Stiff clay"},
                {"low": 100, "high": 200, "name": "Very stiff clay"},
                {"low": 200, "high": 20000, "name": "Hard clay"}, ]
        myDF = pd.DataFrame(data)
        mySeries = pd.Series(ST_Params['Avg. Su(t/sq.m.)'])
        bins = list(myDF["high"])
        bins.insert(0, 0)
        ST_Params['Soil Type'] = pd.cut(mySeries, bins, labels=myDF["name"])
        ST_Params = ST_Params.values.tolist()
        number_of_layer = lastcluster - val2
        for i in range(number_of_layer - 1):
            if i < number_of_layer - 1:
                if ST_Params[i][5] == ST_Params[i + 1][5]:
                    if ST_Params[i][2] - 7 <= ST_Params[i + 1][2] <= ST_Params[i][2] + 7:
                        if ST_Params[i][3] - 0.1 <= ST_Params[i + 1][3] <= ST_Params[i][3] + 0.1:
                            ST_Params[i][0] = (ST_Params[i][0] + ST_Params[i + 1][0]) / 2
                            ST_Params[i][1] = (ST_Params[i][1] + ST_Params[i + 1][1]) / 2
                            ST_Params[i][2] = (ST_Params[i][2] + ST_Params[i + 1][2]) / 2
                            ST_Params[i][3] = (ST_Params[i][3] + ST_Params[i + 1][3]) / 2
                            ST_Params[i][4] = ST_Params[i + 1][4]
                            yticks[i].label1.set_visible(False)
                            ax.lines[i].remove()
                            del ST_Params[i + 1]
                            number_of_layer = len(ST_Params)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            else:
                pass
        for i in range(lastcluster):
            plt.text((1.5), -1 * ST_Params[i][0], 'Su (kPa)  %.2f' % (ST_Params[i][2]))
            plt.text((3), -1 * ST_Params[i][0], 'y (t/cu.m.) %.2f' % (ST_Params[i][3]))
            plt.text(0.5, -1 * ST_Params[i][0], ST_Params[i][4])
    else:
        SS_Params = ndf.groupby(['Group'], sort=False)[
            'To', 'Wn %', 'Su from Ncorr', 'Phi from Ncorr', 'Avg. y (t/cu.m.)', 'N'].mean()
        SS_Params['Tomax'] = ndf.groupby(['Group'], sort=False)['To'].max()
        np.round(SS_Params, 2)
        data = [{"low": 0.0000000001, "high": 12.5, "name": "Very soft clay"},
                {"low": 12.5, "high": 25, "name": "Soft clay"},
                {"low": 25, "high": 50, "name": "Medium clay"},
                {"low": 50, "high": 100, "name": "Stiff clay"},
                {"low": 100, "high": 200, "name": "Very stiff clay"},
                {"low": 200, "high": 20000, "name": "Hard clay"}, ]

        data2 = [{"low": 0, "high": 5, "name": "Very loose sand"},
                 {"low": 5, "high": 10, "name": "Loose sand"},
                 {"low": 10, "high": 30, "name": "Medium sand"},
                 {"low": 30, "high": 100000, "name": "Dense sand"}, ]
        myDF2 = pd.DataFrame(data2)
        mySeries3 = pd.Series(SS_Params['N'])
        bins3 = list(myDF2["high"])
        bins3.insert(0, 0)
        SS_Params['Soil Type'] = pd.cut(mySeries3, bins3, labels=myDF2["name"])
        myDF = pd.DataFrame(data)
        mySeries2 = pd.Series(SS_Params['Su from Ncorr'])
        bins2 = list(myDF["high"])
        bins2.insert(0, 0)
        SS_Params['Soil Type2'] = pd.cut(mySeries2, bins2, labels=myDF["name"])
        SS_Params = SS_Params.ffill(axis=1)
        SS_Params = SS_Params.values.tolist()
        number_of_layer2 = lastcluster2 - val
        for i in range(lastcluster2):
            if i < lastcluster2 - 1:
                if SS_Params[i][8] == SS_Params[i + 1][8] and 'sand' in str(SS_Params[i][8]):
                    if SS_Params[i][3] - 1 <= SS_Params[i + 1][3] <= SS_Params[i][3] + 1:
                        if SS_Params[i][4] - 0.1 <= SS_Params[i + 1][4] <= SS_Params[i][4] + 0.1:
                            SS_Params[i][0] = (SS_Params[i][0] + SS_Params[i + 1][0]) / 2
                            SS_Params[i][3] = (SS_Params[i][3] + SS_Params[i + 1][3]) / 2
                            SS_Params[i][4] = (SS_Params[i][4] + SS_Params[i + 1][4]) / 2
                            SS_Params[i][6] = SS_Params[i + 1][6]
                            yticks[i].label1.set_visible(False)
                            ax.lines[i].remove()
                            del SS_Params[i + 1]
                            number_of_layer2 = len(SS_Params)
                            lastcluster2 = lastcluster2 - 1
                        else:
                            pass
                    else:
                        pass
                else:
                    if SS_Params[i][2] - 20 <= SS_Params[i + 1][2] <= SS_Params[i][2] + 20:
                        if SS_Params[i][4] - 0.1 <= SS_Params[i + 1][4] <= SS_Params[i][4] + 0.1:
                            SS_Params[i][0] = (SS_Params[i][0] + SS_Params[i + 1][0]) / 2
                            SS_Params[i][2] = (SS_Params[i][2] + SS_Params[i + 1][2]) / 2
                            SS_Params[i][4] = (SS_Params[i][4] + SS_Params[i + 1][4]) / 2
                            SS_Params[i][6] = SS_Params[i + 1][6]
                            yticks[i].label1.set_visible(False)
                            ax.lines[i].remove()
                            del SS_Params[i + 1]
                            number_of_layer2 = len(SS_Params)
                            lastcluster2 = lastcluster2 - 1
                        else:
                            pass
                    else:
                        pass
            else:
                pass
        for i in range(lastcluster2):
            plt.text((1.5), -1 * SS_Params[i][0], 'Su (kPa)  %.2f' % (SS_Params[i][2]))
            plt.text((3), -1 * SS_Params[i][0], 'Phi %.2f degree' % (SS_Params[i][3]))
            plt.text((4.5), -1 * SS_Params[i][0], 'y (t/cu.m.) %.2f' % (SS_Params[i][4]))
            plt.text(0.5, -1 * SS_Params[i][0], SS_Params[i][8])
            ############################################
    ############################################
    # If we haven't already shown or saved the plot, then we need to
    # convert it to an OpenCV image/numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # convert canvas to image
    graph_image = np.array(fig.canvas.get_renderer()._renderer)

    # it still is rgb, convert to opencv's default bgr
    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
    return graph_image


#flask GUI


@app.route("/")
def start_page():
    print("Start")
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']

    # Save file
    #filename = 'static/' + file.filename
    #file.save(filename)

    # Read image
    #image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    df = pd.read_excel(file.read(), skiprows=2, usecols='A, B, C, D, E, G, K, L, M, N')
    image_output=soil_layer(df)


    # In memory
    image_content = cv2.imencode('.jpg', image_output)[1].tostring()
    encoded_image = base64.encodebytes(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')



    return render_template('home.html', image_to_show=to_send,init=True)



if __name__ == "__main__":
    # Only for debugging while developing
    app.run(debug=True)
