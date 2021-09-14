from numpy.random.mtrand import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates

def showplots(df1,obv,Fs,refresh=0.5,caller='dump',asset='ETH-USD'):
    pd.plotting.register_matplotlib_converters()

    with plt.style.context(style='dark_background'):
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

        ttl = f'Analysis Dashboard - Asset:{asset} |N ={len(df1)} |T={T} |Fs={Fs} |F={df} |W={dw} |Fn={ny}'

        plt.ion()

        dates_idx = dates.date2num(df1.index)
        if len(plt.get_fignums()) == 0:
            fig = plt.figure(ttl)
        else:
            plt.figure(1)
            
        #SECTION 1: ROWS:3 COLUMNS:10 V signal PLOTS
        df_v3 = df1[['ft1','ma1','ff1']]
        df_v2 = df1['dr1']

        from dateutil import rrule
        import matplotlib.ticker as tck

        ax1 = plt.subplot2grid((gx,gy),(s1x,s1y),rowspan=s1rs,colspan=s1cs)
        ax1.plot_date(dates_idx,df1['dv1t_o'],xdate=False,linestyle='-',label='dv1t_o',marker='')
        ax1.plot_date(dates_idx,df1.x1_pr,xdate=False,linestyle='dotted',marker='',linewidth=0.8,label='x1_pr')
        ax1.plot_date(dates_idx,df1.x1_gn,xdate=False,linestyle='dotted',marker='',linewidth=0.8,label='x1_gn')
        ax1.legend(loc='lower left')

        ax2 = plt.subplot2grid((gx,gy),(s1x+s1rs+1,s1y),rowspan=s1rs,colspan=s1cs)
        ax2.plot_date(dates_idx,df1[f'd{obv[0]}2t_oo'],xdate=False,marker='',linestyle='-',label=f'd{obv[0]}2t_oo')
        ax2.plot_date(dates_idx,df_v2,xdate=False,marker='',linestyle='dotted',linewidth=0.8,label='dr1')
        ax2.legend(loc='lower left')

        ax3 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y),rowspan=s1rs,colspan=s1cs)
        ax3.plot_date(dates_idx,df1[f'd{obv[0]}3t_ooo'],xdate=False,marker='',linestyle='-',label=f'd{obv[0]}3t_ooo')
        ax3.plot_date(dates_idx,df_v3,label=df_v3.columns,xdate=False,marker='',linestyle='dotted',linewidth=0.8)
        ax3.legend(loc='lower left')
        
        #SECTION 2: ROWS:3 COLUMNS:10 C signal PLOTS
        df_c3 = df1[['ft2','ma2','ff2']]
        df_c2 = df1['dr2']

        ax4 = plt.subplot2grid((gx,gy),(s1x,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax4.plot_date(dates_idx,df1[f'd{obv[1]}1t_o'],marker='',xdate=False,linestyle='-',label=f'd{obv[1]}1t_o')
        ax4.plot_date(dates_idx,df1.x2_pr,xdate=False,marker='',linestyle='dotted',linewidth=0.8,label='x2_pr')
        ax4.plot_date(dates_idx,df1.x2_gn,xdate=False,marker='',linestyle='dotted',linewidth=0.8,label='x2_gn')
        ax4.legend(loc='lower left')

        ax5 = plt.subplot2grid((gx,gy),(s1x+s2rs+1,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax5.plot_date(dates_idx,df1[f'd{obv[1]}2t_oo'],marker='',xdate=False,linestyle='-',label=f'd{obv[1]}2t_oo')
        ax5.plot_date(dates_idx,df_c2,xdate=False,marker='',linestyle='dotted',linewidth=0.8,label='dr2')
        ax5.legend(loc='lower left')
       
        ax6 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s1rs,colspan=s1cs)
        ax6.plot_date(dates_idx,df1[f'd{obv[1]}3t_ooo'],marker='',xdate=False,linestyle='-',label=f'd{obv[1]}3t_ooo')
        ax6.plot_date(dates_idx,df_c3,label=df_c3.columns,marker='',xdate=False,linestyle='dotted',linewidth=0.8)
        ax6.legend(loc='lower left')

        #SECTION 3: ROWS:3 COLUMNS:5 SCATTER PLOTS
        ax7 = plt.subplot2grid((gx,gy),(s1x,(s1y+s1cs+1)*2),rowspan=s3rs+5,colspan=s3cs)
        ax7.plot(df1[{f'd{obv[0]}1t_o'}],df1[f'd{obv[1]}1t_o'],color='m',linestyle='None',marker='o',label=f'd{obv[0]}1t_o-d{obv[1]}1t_o')
        ax7.plot(df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'd{obv[0]}1t_o')],df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'd{obv[1]}1t_o')],color='b',linestyle='None',marker='o')
        ax7.plot(df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'd{obv[0]}1t_o')],df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'd{obv[1]}1t_o')],color='c',linestyle='None',marker='o')
        ax7.plot(df1.dv1t_o.mean(),df1.dc1t_o.mean(),color='r',linestyle='None',marker='o')
        ax7.plot(df1.dv1t_o[-1],df1.dc1t_o[-1],color='g',linestyle='None',marker='o')
        ax7.plot(df1.ft1[-1],df1.ft2[-1],linestyle='None',marker='o')
        ax7.plot(df1.ma1[-1],df1.ma2[-1],linestyle='None',marker='o')
        ax7.plot(df1.dr1[-1],df1.dr2[-1],linestyle='None',marker='o')
        ax7.plot(df1.q1[-1],df1.q2[-1],linestyle='None',marker='o')
        ax7.plot(df1.wrk1[-1],df1.wrk2[-1],linestyle='None',marker='o')
        ax7.plot(df1.rf1[-1],df1.rf2[-1],linestyle='None',marker='o')

        ax7.annotate('Cm',xy=(df1.dv1t_o.mean(),df1.dc1t_o.mean()),xytext=(df1.dv1t_o.mean(),df1.dc1t_o.mean()),fontweight='bold',xycoords='data',textcoords='data')
        ax7.annotate('Px',xy=(df1.dv1t_o[-1],df1.dc1t_o[-1]),xytext=(df1.dv1t_o[-1],df1.dc1t_o[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Fx',xy=(df1.ft1[-1],df1.ft2[-1]),xytext=(df1.ft1[-1],df1.ft2[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('MAx',xy=(df1.ma1[-1],df1.ma2[-1]),xytext=(df1.ma1[-1],df1.ma2[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Drx',xy=(df1.dr1[-1],df1.dr2[-1]),xytext=(df1.dr1[-1],df1.dr2[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Qx',xy=(df1.q1[-1],df1.q2[-1]),xytext=(df1.q1[-1],df1.q2[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Wkx',xy=(df1.wrk1[-1],df1.wrk2[-1]),xytext=(df1.wrk1[-1],df1.wrk2[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Rfx',xy=(df1.rf1[-1],df1.rf2[-1]),xytext=(df1.rf1[-1],df1.rf2[-1]),xycoords='data',textcoords='data',fontweight='bold')

        #SECTION 4: MAG AND ENERGY PLOTS
        ax10 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y),rowspan=s4rs,colspan=s1cs)
        ax10.magnitude_spectrum(df1.dv1t_o,Fs=Fs,linestyle='none',markersize='5',marker='o',label='vt')
        ax10.magnitude_spectrum(df1.dc1t_o,Fs=Fs,linestyle='none',markersize='5',marker='o',label='ct')
        ax10.legend(loc='upper left')
    
        ax11 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s4rs,colspan=s1cs)
        ax11.step(dates_idx,df1.TE1,label='TE1',linewidth=1,linestyle='dotted')
        ax11.step(dates_idx,df1.TE2,label='TE2',linewidth=1,linestyle='dotted')
        ax11.step(dates_idx,df1.idpos1,label='pos1',linewidth=0.8,linestyle='dotted', alpha=0.8)
        ax11.step(dates_idx,df1.idpos2,label='pos2',linewidth=0.8,linestyle='dotted', alpha=0.8)
        ax11.step(dates_idx,df1.Ftm,label='Fx',linewidth=1,linestyle='dotted')
        ax11.step(dates_idx,df1.Qfm,label='Qx',linewidth=1,linestyle='dotted')
        ax11.step(dates_idx,df1.Drm,label='Drx',linewidth=1,linestyle='dotted')
        ax11.step(dates_idx,df1.Wkm,label='Wkx',linewidth=1,linestyle='dotted')
        ax11.legend(loc='lower left')

        #SECTION 5: PHASE ANGLE PLOTS
        ax12 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1)+1,(s1y+s1cs+1)*2),rowspan=s5rs,colspan=s1cs,projection='polar',)
        ax12.plot(df1.Pxa[-1],df1.Pxm[-1],label='Px',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Maa,df1.Mam,label='MAx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Dra,df1.Drm,label='Drx',marker='o',markersize=6,linestyle='none') 
        ax12.plot(df1.Fta,df1.Ftm,label='Fx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Qfa,df1.Qfm,label='Qx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Wka,df1.Wkm,label='Wkx',marker='o',markersize=6,linestyle='none')

        ax12.annotate('Fx',xy=(df1.Fta[-1],df1.Ftm[-1]),xytext=(df1.Fta[-1],df1.Ftm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Qx',xy=(df1.Qfa[-1],df1.Qfm[-1]),xytext=(df1.Qfa[-1],df1.Qfm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Px',xy=(df1.Pxa[-1],df1.Pxm[-1]),xytext=(df1.Pxa[-1],df1.Pxm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('MAx',xy=(df1.Maa[-1],df1.Mam[-1]),xytext=(df1.Maa[-1],df1.Mam[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Drx',xy=(df1.Dra[-1],df1.Drm[-1]),xytext=(df1.Dra[-1],df1.Drm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Wkx',xy=(df1.Wka[-1],df1.Wkm[-1]),xytext=(df1.Wka[-1],df1.Wkm[-1]),xycoords='data',textcoords='data',fontweight='bold')

        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.25, wspace=0.25)
        plt.figure(1).suptitle(ttl)

        if caller == 'dump':
            plt.show()
        else:
            plt.pause(refresh)
            plt.clf()



