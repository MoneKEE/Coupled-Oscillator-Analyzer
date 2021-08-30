from numpy.random.mtrand import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import time
from datetime import datetime as dt
import models
import misc

def showplots(df1,obv,refresh,Fs,m,pks_v,pks_c,figcols,caller='dump',asset='ETH-USD'):
    colors = ['b', 'g', 'r', 'c', 'm', 'y','b', 'g', 'r', 'c', 'm', 'y']

    with plt.style.context(style='seaborn'):

        #LINE SUBLPLOTS
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

        plt.ion()

        dates_idx = dates.date2num(df1.index)
        if len(plt.get_fignums()) == 0:
            plt.figure(f'Analysis Dashboard - Asset:{asset} |N ={len(df1)} |T={T} |Fs={Fs} |F={df} |W={dw} |Fn={ny}')
        else:
            plt.figure(1)
            
        #SECTION 1: ROWS:3 COLUMNS:10 V signal PLOTS
        df_v3 = df1[['ma1']]
        df_v2 = df1[['a1k1']]

        ax1 = plt.subplot2grid((gx,gy),(s1x,s1y),rowspan=s1rs,colspan=s1cs)
        ax1.plot_date(dates_idx,df1['dv1t_o'],xdate=True,linestyle='-',fmt='',label='dv1t_o')
        ax1.plot_date(dates_idx,df1.k1,xdate=True,linestyle='dotted',fmt='',linewidth=0.5,label='k1')
        ax1.legend(loc='upper right')


        ax2 = plt.subplot2grid((gx,gy),(s1x+s1rs+1,s1y),rowspan=s1rs,colspan=s1cs)
        ax2.plot_date(dates_idx,df1[f'd{obv[0]}2t_oo'],xdate=True,linestyle='-',fmt='',label=f'd{obv[0]}2t_oo')
        ax2.plot_date(dates_idx,df_v2,xdate=True,linestyle='dotted',fmt='',linewidth=0.5,label='a1k1')
        ax2.legend(loc='upper right')

        ax3 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y),rowspan=s1rs,colspan=s1cs)
        ax3.plot_date(dates_idx,df1[f'd{obv[0]}3t_ooo'],xdate=True,linestyle='-',fmt='',label=f'd{obv[0]}3t_ooo')
        ax3.plot_date(dates_idx,df_v3,xdate=True,linestyle='dotted',fmt='',linewidth=0.5,label='ma1')
        ax3.legend(loc='upper right')
        
        #SECTION 2: ROWS:3 COLUMNS:10 C signal PLOTS
        df_c3 = df1[['ma2']]
        df_c2 = df1[['a2k2']]

        ax4 = plt.subplot2grid((gx,gy),(s1x,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax4.plot_date(dates_idx,df1[f'd{obv[1]}1t_o'],xdate=True,linestyle='-',fmt='',label=f'd{obv[1]}1t_o')
        ax4.plot_date(dates_idx,df1.k2,xdate=True,linestyle='dotted',fmt='',linewidth=0.5,label='k2')
        ax4.legend(loc='upper right')

        ax5 = plt.subplot2grid((gx,gy),(s1x+s2rs+1,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax5.plot_date(dates_idx,df1[f'd{obv[1]}2t_oo'],xdate=True,linestyle='-',fmt='',label=f'd{obv[1]}2t_oo')
        ax5.plot_date(dates_idx,df_c2,xdate=True,linestyle='dotted',fmt='',linewidth=0.5,label='a2k2')
        ax5.legend(loc='upper right')
       
        ax6 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s1rs,colspan=s1cs)
        ax6.plot_date(dates_idx,df1[f'd{obv[1]}3t_ooo'],xdate=True,linestyle='-',fmt='',label=f'd{obv[1]}3t_ooo')
        ax6.plot_date(dates_idx,df_c3,label='ma2',xdate=True,linestyle='dotted',linewidth=0.5,fmt='')
        ax6.legend(loc='upper right')

        #SECTION 3: ROWS:3 COLUMNS:5 SCATTER PLOTS
        ax7 = plt.subplot2grid((gx,gy),(s1x,(s1y+s1cs+1)*2),rowspan=s3rs,colspan=s3cs)
        ax7.plot(df1[{f'd{obv[0]}1t_o'}],df1[f'd{obv[1]}1t_o'],color='m',linestyle='None',marker='o',label=f'd{obv[0]}1t_o-d{obv[1]}1t_o')
        ax7.plot(df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'd{obv[0]}1t_o')],df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'd{obv[1]}1t_o')],color='b',linestyle='None',marker='o')
        ax7.plot(df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'd{obv[0]}1t_o')],df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'd{obv[1]}1t_o')],color='c',linestyle='None',marker='o')
        ax7.plot(df1.iloc[-1,df1.columns.get_loc(f'd{obv[0]}1t_o')],df1.iloc[-1,df1.columns.get_loc(f'd{obv[1]}1t_o')],color='g',linestyle='None',marker='o')
        ax7.plot(df1.dv1t_o.mean(),df1.dc1t_o.mean(),color='r',linestyle='None',marker='o')
        ax7.legend(loc='upper right')

        ax8 = plt.subplot2grid((gx,gy),(s1x+s3rs+1,(s1y+s1cs+1)*2),rowspan=s3rs,colspan=s3cs)
        ax8.plot(df1[{f'd{obv[0]}2t_oo'}],df1[{f'd{obv[1]}2t_oo'}],color='m',linestyle='None',marker='o',label=f'd{obv[0]}2t_oo-d{obv[1]}2t_oo')
        ax8.plot(df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'd{obv[0]}2t_oo')],df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'd{obv[1]}2t_oo')],color='b',linestyle='None',marker='o')
        ax8.plot(df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'd{obv[0]}2t_oo')],df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'd{obv[1]}2t_oo')],color='c',linestyle='None',marker='o')
        ax8.plot(df1.iloc[-1,df1.columns.get_loc(f'd{obv[0]}2t_oo')],df1.iloc[-1,df1.columns.get_loc(f'd{obv[1]}2t_oo')],color='g',linestyle='None',marker='o')
        ax8.plot(df1.dv2t_oo.mean(),df1.dc2t_oo.mean(),color='r',linestyle='None',marker='o')
        ax8.legend(loc='upper right')

        #SECTION 4: MAG AND ENERGY PLOTS
        ax10 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y),rowspan=s4rs,colspan=s1cs)
        ax10.magnitude_spectrum(df1.vf_t,Fs=Fs,linestyle='none',markersize='5',marker='o',label='vf_t')
        ax10.magnitude_spectrum(df1.cf_t,Fs=Fs,linestyle='none',markersize='5',marker='o',label='cf_t')
        ax10.legend(loc='upper left')
    
        ax11 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s4rs,colspan=s1cs)
        ax11.plot_date(dates_idx,df1[['TE1']],label='TE1',xdate=True,linestyle='dotted',linewidth=1,fmt='')
        ax11.plot_date(dates_idx,df1[['TE2']],label='TE2',xdate=True,linestyle='dotted',linewidth=1,fmt='')
        ax11.plot_date(dates_idx,df1[['idposc']],label='idposc',xdate=True,linestyle='dotted',linewidth=1,fmt='')
        ax11.legend(loc='upper left')

        #SECTION 5: PHASE ANGLE PLOTS
        ax12 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1)+1,(s1y+s1cs+1)*2),rowspan=s5rs,colspan=s1cs,projection='polar',)
        # ax12.plot(np.rad2deg(df1['vf_w'][np.abs(df1['vf_rad']) > 2*df1.vf_rad.std()]),df1['vf_rad'][np.abs(df1['vf_rad']) > 2*df1.vf_rad.std()],color='c',label='vf_w',marker='^',markersize=6,linestyle='none') 
        # ax12.plot(np.rad2deg(df1['cf_w'][np.abs(df1['cf_rad']) > 2*df1.cf_rad.std()]),df1['cf_rad'][np.abs(df1['cf_rad']) > 2*df1.cf_rad.std()],color='g',label='cf_w',marker='o',markersize=6,linestyle='none')
        ax12.plot(np.rad2deg(np.arctan(df1.dc1t_o/df1.dv1t_o)),np.sqrt(df1.dc1t_o**2+df1.dv1t_o**2),label='Px',marker='^',markersize=6,linestyle='none')
        ax12.plot(np.rad2deg(np.arctan(df1.ma2/df1.ma1)),np.sqrt(df1.ma1**2+df1.ma2**2),label='Fx',marker='^',markersize=6,linestyle='none')
        ax12.plot(np.rad2deg(np.arctan(df1.a2k2/df1.a1k1)),np.sqrt(df1.a1k1**2+df1.a2k2**2),label='AKx',marker='^',markersize=6,linestyle='none') 
        #ax12.plot(np.rad2deg(df1['w2_n']),df1['dc1t_o'],label='c1-w2_n',marker='o',markersize=6,linestyle='none')
        ax12.legend(loc='upper left')

        if caller == 'dump':
            pass
        else:
            plt.pause(refresh)
            plt.clf()



