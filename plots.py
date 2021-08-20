from matplotlib import colors, ticker
from matplotlib.colors import Colormap
from numpy.random.mtrand import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import time
from datetime import datetime as dt
import models
import misc

def showplots(df1,obv,refresh,Fs,k1,k2,pks_v,pks_c,figcols,caller='dump',asset='ETH-USD'):
    colors = ['b', 'g', 'r', 'c', 'm', 'y','b', 'g', 'r', 'c', 'm', 'y']

    with plt.style.context(style='seaborn'):

        #LINE SUBLPLOTS
        plt.ion()
        gx=20;gy=30
        s1x=0; s1y=0

        s1rs=4; s1cs=10
        s2rs=4; s2cs=10
        s3rs=4; s3cs=8
        s4rs = 5
        s5rs = 8

        T = len(df1)/Fs
        df = round(1/T,3)
        dw = round((2*np.pi)/T,3)
        ny = round((dw*len(df1))/2,3)

        dates_idx = dates.date2num(df1.index)
        plt.figure(f'Analysis Dashboard - Asset:{asset} |N ={len(df1)} |T={T} |Fs={Fs} |F={df} |W={dw} |Fn={ny}')
        #SECTION 1: ROWS:3 COLUMNS:10 V signal PLOTS
        df_v3 = df1[[f'd{obv[0]}3t_ooo','ma1','m1']]
        df_v2 = df1[[f'd{obv[0]}2t_oo','p1','c1']]


        ax1 = plt.subplot2grid((gx,gy),(s1x,s1y),rowspan=s1rs,colspan=s1cs)
        ax1.plot_date(dates_idx,df1['dv1t_o'],xdate=True,linestyle='-',fmt='',color='c',label='dv1t_o')
        ax1.legend(loc='upper right')

        ax2 = plt.subplot2grid((gx,gy),(s1x+s1rs+1,s1y),rowspan=s1rs,colspan=s1cs)
        ax2.plot_date(dates_idx,df_v2,xdate=True,linestyle='-',fmt='',label=df_v2.columns)
        ax2.legend(loc='upper right')

        ax3 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y),rowspan=s1rs,colspan=s1cs)
        ax3.plot_date(dates_idx,df_v3,xdate=True,linestyle='-',fmt='',label=df_v3.columns)
        ax3.legend(loc='upper right')

        #SECTION 2: ROWS:3 COLUMNS:10 C signal PLOTS
        df_c3 = df1[[f'd{obv[1]}3t_ooo','ma2','m2']]
        df_c2 = df1[[f'd{obv[1]}2t_oo','p2','c2']]

        ax4 = plt.subplot2grid((gx,gy),(s1x,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax4.plot_date(dates_idx,df1[f'd{obv[1]}1t_o'],color='c',xdate=True,linestyle='-',fmt='',label=f'd{obv[1]}1t_o')
        ax4.legend(loc='upper right')

        ax5 = plt.subplot2grid((gx,gy),(s1x+s2rs+1,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax5.plot_date(dates_idx,df_c2,xdate=True,linestyle='-',fmt='',label=df_c2.columns)
        ax5.legend(loc='upper right')

        ax6 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s1rs,colspan=s1cs)
        ax6.plot_date(dates_idx,df_c3,label=df_c3.columns,xdate=True,linestyle='-',fmt='')
        ax6.legend(loc='upper right')

        #SECTION 3: ROWS:3 COLUMNS:5 SCATTER PLOTS
        ax7 = plt.subplot2grid((gx,gy),(s1x,(s1y+s1cs+1)*2),rowspan=s3rs,colspan=s3cs)
        ax7.plot(df1[{f'd{obv[0]}1t_o'}],df1[f'd{obv[1]}1t_o'],color='c',linestyle='None',marker='o',label=f'd{obv[0]}1t_o-d{obv[1]}1t_o')
        ax7.legend(loc='upper right')
 
        ax8 = plt.subplot2grid((gx,gy),(s1x+s3rs+1,(s1y+s1cs+1)*2),rowspan=s3rs,colspan=s3cs)
        ax8.plot(df1[{f'd{obv[0]}2t_oo'}],df1[{f'd{obv[1]}2t_oo'}],color='g',linestyle='None',marker='o',label=f'd{obv[0]}2t_oo-d{obv[1]}2t_oo')
        ax8.legend(loc='upper right')


        #SECTION 4: MAG AND ANGLE PLOTS
        ax10 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y),rowspan=s4rs,colspan=s1cs)
        ax10.magnitude_spectrum(df1.v_fft[pks_v],color='c',Fs=Fs,label=['v_fft'])
        ax10.magnitude_spectrum(df1.c_fft[pks_c],color='g',Fs=Fs,label=['c_fft'])
       # ax10.magnitude_spectrum(df1['c_fft'],color='g',Fs=Fs,label=['c_fft'])
        ax10.legend(loc='upper right')

        ax11 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s4rs,colspan=s1cs)
        ax11.phase_spectrum(df1['v_fft'],color='c',Fs=Fs,label=['v_fft'])
        ax11.phase_spectrum(df1['c_fft'],color='g',Fs=Fs,label=['c_fft'])
        ax11.legend(loc='upper right')

        #SECTION 5: PHASE ANGLE PLOTS
        ax12 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1)+1,(s1y+s1cs+1)*2),rowspan=s5rs,colspan=s1cs,projection='polar',)
        # ax12.plot(np.rad2deg(df1['vf_w'][np.abs(df1['vf_rad']) > 2*df1.vf_rad.std()]),df1['vf_rad'][np.abs(df1['vf_rad']) > 2*df1.vf_rad.std()],color='c',label='vf_w',marker='^',markersize=6,linestyle='none') 
        # ax12.plot(np.rad2deg(df1['cf_w'][np.abs(df1['cf_rad']) > 2*df1.cf_rad.std()]),df1['cf_rad'][np.abs(df1['cf_rad']) > 2*df1.cf_rad.std()],color='g',label='cf_w',marker='o',markersize=6,linestyle='none')
        ax12.plot(np.rad2deg(df1['vf_theta'][pks_v]),df1['vf_rad'][pks_v],color='c',label='vf_w',marker='^',markersize=6,linestyle='none') 
        ax12.plot(np.rad2deg(df1['cf_theta'][pks_c]),df1['cf_rad'][pks_c],color='g',label='cf_w',marker='o',markersize=6,linestyle='none')
        ax12.legend(loc='center')

        # #SECTION 6: MAGNITUDE SPECTRUMS
        # plt.figure('Spectrums')
        # ax13 = plt.subplot2grid((1,2),(0,0),rowspan=1,colspan=1)
        # ax13.magnitude_spectrum(df1['_r'],Fs=Fs,color='b',label='_r')
        # ax13.legend(loc='upper right')

        # ax14 = plt.subplot2grid((1,2),(0,1),rowspan=1,colspan=1)
        # ax14.magnitude_spectrum(df1['c_r'],Fs=Fs,color='g' ,label='c_r')
        # ax14.legend(loc='upper right')

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


