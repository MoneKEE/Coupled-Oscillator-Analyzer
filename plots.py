from matplotlib import colors
from matplotlib.colors import Colormap
from numpy.random.mtrand import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import time
from datetime import datetime as dt
import models
import misc

def showplots(df1,pks,obv,refresh,figcols,caller='dump'):
    colors = ['b', 'g', 'r', 'c', 'm', 'y','b', 'g', 'r', 'c', 'm', 'y']


    with plt.style.context(style='seaborn'):

        #LINE SUBLPLOTS
        plt.ion()
        gx=20;gy=30
        s1x=0; s1y=0

        s1rs=4; s1cs=10
        s2rs=4; s2cs=10
        s3rs=4; s3cs=8
        s4rs = 10

        dates_idx = df1.index
        plt.figure('Analysis Dashboard')
        #SECTION 1: ROWS:3 COLUMNS:10 DV PLOTS
        ax1 = plt.subplot2grid((gx,gy),(s1x,s1y),rowspan=s1rs,colspan=s1cs)
        ax1.plot(dates_idx,df1[f'd{obv[0]}t_o'],color='b',label=f'd{obv[0]}t_o')
        ax1.legend(loc='upper right')

        ax2 = plt.subplot2grid((gx,gy),(s1x+s1rs+1,s1y),rowspan=s1rs,colspan=s1cs)
        ax2.plot(dates_idx,df1[f'd{obv[0]}t_oo'],color='g',label=f'd{obv[0]}t_oo')
        ax2.legend(loc='upper right')

        #SECTION 2: ROWS:3 COLUMNS:10 DC PLOTS
        ax4 = plt.subplot2grid((gx,gy),(s1x,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax4.plot(dates_idx,df1[f'd{obv[1]}t_o'],color='b',label=f'd{obv[1]}t_o')
        ax4.legend(loc='upper right')

        ax5 = plt.subplot2grid((gx,gy),(s1x+s2rs+1,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax5.plot(dates_idx,df1[f'd{obv[1]}t_oo'],color='g',label=f'd{obv[1]}t_oo')
        ax5.legend(loc='upper right')

        #SECTION 3: ROWS:3 COLUMNS:5 SCATTER PLOTS
        ax7 = plt.subplot2grid((gx,gy),(s1x,(s1y+s1cs+1)*2),rowspan=s3rs,colspan=s3cs)
        ax7.plot(df1[{f'd{obv[0]}t_o'}],df1[f'd{obv[1]}t_o'],color='b',linestyle='None',marker='o',label=f'{obv[0]}P0o-{obv[1]}P0o')
        ax7.legend(loc='upper right')
 
        ax8 = plt.subplot2grid((gx,gy),(s1x+s3rs+1,(s1y+s1cs+1)*2),rowspan=s3rs,colspan=s3cs)
        ax8.plot(df1[{f'd{obv[0]}t_oo'}],df1[{f'd{obv[1]}t_oo'}],color='g',linestyle='None',marker='o',label=f'{obv[0]}P0oo-{obv[1]}P0oo')
        ax8.legend(loc='upper right')


        #SECTION 4: FOURIER PLOTS
        ax10 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y),rowspan=s4rs,colspan=s1cs)
        ax10.plot(dates_idx,df1[f'{obv[0]}_f'],color='c',linestyle='dotted',label=f'{obv[0]}_f')
        ax10.legend(loc='upper right')
        ax11 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s4rs,colspan=s1cs)
        ax11.plot(dates_idx,df1[f'{obv[1]}_f'],color='c',linestyle='dotted',label=f'{obv[1]}_f')
        ax11.legend(loc='upper right')

        #SECTION 5: PHASE ANGLE PLOTS
        ax12 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),(s1y+s1cs+1)*2),rowspan=s4rs,colspan=s1cs,projection='polar')
        ax12.plot(df1.v_theta,[1 for x in range(len(df1))],label='v_theta',linestyle='None',marker='^')
        ax12.plot(df1.c_theta,[1 for x in range(len(df1))],label='c_theta',color='r',linestyle='None',marker='o')
        ax12.legend(loc='center')

        #SECTION 6: MAGNITUDE SPECTRUMS
        plt.figure('Spectrums')
        ax13 = plt.subplot2grid((1,2),(0,0),rowspan=1,colspan=1)
        ax13.magnitude_spectrum(df1['v_w'],color='b',label='v_w')
        ax13.legend(loc='upper right')

        ax14 = plt.subplot2grid((1,2),(0,1),rowspan=1,colspan=1)
        ax14.magnitude_spectrum(df1['c_w'],color='g' ,label='c_w')
        ax14.legend(loc='upper right')

        if caller == 'dump':
            try:
                while True:
                    plt.pause(500)
            except KeyboardInterrupt:
                pass
        else:
            plt.pause(refresh)
            plt.clf()

###################################################
#TEST BED
###################################################
# import test
# data_o = test.testbed(start=dt(2018,1,1),stop=dt(2018,2,1))
# showplots(data_o)


