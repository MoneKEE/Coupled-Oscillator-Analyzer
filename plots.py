from matplotlib import colors
from matplotlib.colors import Colormap
from numpy.random.mtrand import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import time
from datetime import datetime as dt
import datacapture as dc
import models

def showplots(df1,caller='dump'):
    legend = ['dv1-dc1','dv2-dc2','dv3-dc3']
    fig2cols = ['dc1','dv1','dc2','dv2','dc3','dv3']
    fig1cols = [['dv1','dc1'],['dv2','dc2'],['dv3','dc3']]
    colors = ['b', 'g', 'r', 'c', 'm', 'y','b', 'g', 'r', 'c', 'm', 'y']

    plt.ion()

    with plt.style.context(style='seaborn'):

        #LINE SUBLPLOTS
        plt.figure(1)
        rn = len(fig1cols)
        gx=20;gy=30
        s1x=0; s1y=0

        s1rs=4; s1cs=10
        s2rs=4; s2cs=10
        s3rs=4; s3cs=8

        #SECTION 1: ROWS:3 COLUMNS:10 DV PLOTS
        ax1 = plt.subplot2grid((gx,gy),(s1x,s1y),rowspan=s1rs,colspan=s1cs)
        ax1.plot(mdate.date2num(df1.index),df1[fig1cols[0][0]],color='r',label=fig1cols[0][0])
        ax1.legend()

        ax2 = plt.subplot2grid((gx,gy),(s1x+s1rs+1,s1y),rowspan=s1rs,colspan=s1cs)
        ax2.plot(mdate.date2num(df1.index),df1[fig1cols[1][0]],color='g',label=fig1cols[1][0])
        ax2.legend()

        ax3 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y),rowspan=s1rs,colspan=s1cs)
        ax3.plot(mdate.date2num(df1.index),df1[fig1cols[2][0]],color='b',label=fig1cols[2][0])
        ax3.legend()

        #SECTION 2: ROWS:3 COLUMNS:10 DC PLOTS
        ax4 = plt.subplot2grid((gx,gy),(s1x,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax4.plot(mdate.date2num(df1.index),df1[fig1cols[0][1]],color='r',label=fig1cols[0][1])
        ax4.legend()

        ax5 = plt.subplot2grid((gx,gy),(s1x+s2rs+1,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax5.plot(mdate.date2num(df1.index),df1[fig1cols[1][1]],color='g',label=fig1cols[1][1])
        ax5.legend()

        ax6 = plt.subplot2grid((gx,gy),(2*(s1x+s2rs+1),s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax6.plot(mdate.date2num(df1.index),df1[fig1cols[2][1]],color='b',label=fig1cols[2][1])
        ax6.legend()

        #SECTION 3: ROWS:3 COLUMNS:5 SCATTER PLOTS
        ax7 = plt.subplot2grid((gx,gy),(s1x,(s1y+s1cs+1)*2),rowspan=s3rs,colspan=s3cs)
        ax7.plot(df1[fig2cols[0]],df1[fig2cols[-1]],color='r',linestyle='None',marker='o',label=legend[0])
        ax7.legend(loc='upper right')
 
        
        ax8 = plt.subplot2grid((gx,gy),(s1x+s3rs+1,(s1y+s1cs+1)*2),rowspan=s3rs,colspan=s3cs)
        ax8.plot(df1[fig2cols[1]],df1[fig2cols[-2]],color='g',linestyle='None',marker='o',label=legend[1])
        ax8.legend(loc='upper right')


        ax9 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),(s1y+s1cs+1)*2),rowspan=s3rs,colspan=s3cs)
        ax9.plot(df1[fig2cols[2]],df1[fig2cols[-3]],color='b',linestyle='None',marker='o',label=legend[2])
        ax9.legend(loc='upper right')

        #SECTION 4: FOURIER PLOTS
        ax10 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y),rowspan=s1rs,colspan=s1cs)
        ax10.plot(mdate.date2num(df1.index),df1['dv1_ifft'],color='c',linestyle='None',marker='o',label='dv1_ifft')

        ax10 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s1rs,colspan=s1cs)
        ax10.plot(mdate.date2num(df1.index),df1['dc1_ifft'],color='y',linestyle='None',marker='o',label='dc1_ifft')


        plt.draw()

        if caller == 'dump':
            try:
                while True:
                    plt.pause(500)
            except KeyboardInterrupt:
                pass
        else:
            plt.pause(0.5)
            plt.clf()

###################################################
#TEST BED
###################################################
# import test

# data_o = test.testbed(start=dt(2018,1,1),stop=dt(2018,2,1))
# showplots(data_o)

