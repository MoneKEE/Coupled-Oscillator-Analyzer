from numpy.random.mtrand import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates

def showplots(df1,obv,F,refresh=0.5,caller='dump',asset='ETH-USD'):
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

        N = len(df1)
        Fs = F
        sr = 1/Fs
        T = N*sr
        df = 1/T
        dw = 2*np.pi*df
        ny = Fs/2

        ttl = f'Analysis Dashboard - Asset:{asset} |N ={len(df1)} |T={round(T,3)} |Fs={round(Fs,3)} |F={round(df,3)} |W={round(dw,3)} |Fn={round(ny,3)}'

        plt.ion()

        dates_idx = dates.date2num(df1.index)
        if len(plt.get_fignums()) == 0:
            fig = plt.figure(ttl)
        else:
            plt.figure(1)

        x1 = df1.x1nm
        x2 = df1.x2nm

        dotx1 = df1.d1dotx1nm
        dotx2 = df1.d1dotx2nm

        ddotx1 = df1.d2dotx1nm
        ddotx2 = df1.d2dotx2nm

        wndtr = len(df1)//10        
            
        #SECTION 1: ROWS:3 COLUMNS:10 V signal PLOTS
        df_v3 = df1[['ft1nm','ma1nm','ff1nm']]
        df_v2 = df1[['dr1nm','lmda1nm','q1nm','resf1nm']]

        ax1 = plt.subplot2grid((gx,gy),(s1x,s1y),rowspan=s1rs,colspan=s1cs,ylim=(-1,1))
        ax1.plot_date(dates_idx[-wndtr:],x1[-wndtr:],xdate=False,linestyle='-',label='x1',marker='')
        # ax1.step(dates_idx[-wndtr:],df1.idpos1[-wndtr:],label='pos1',linewidth=0.8,linestyle='dotted',marker='')
        ax1.plot_date(dates_idx[-wndtr:],df1.x1prnm[-wndtr:],xdate=False,linestyle='dotted',marker='',linewidth=0.8,label='x1prnm')
        ax1.plot_date(dates_idx[-wndtr:],df1.x1gnnm[-wndtr:],xdate=False,linestyle='dotted',marker='',linewidth=0.8,label='x1gnnm')
        ax1.legend(loc='lower left')

        ax2 = plt.subplot2grid((gx,gy),(s1x+s1rs+1,s1y),rowspan=s1rs,colspan=s1cs,ylim=(-1,1))
        ax2.plot_date(dates_idx[-wndtr:],dotx1[-wndtr:],xdate=False,marker='',linestyle='-',label=f'dotx1')
        ax2.plot_date(dates_idx[-wndtr:],df_v2[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=0.8,label=df_v2.columns)
        ax2.legend(loc='lower left')

        ax3 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y),rowspan=s1rs,colspan=s1cs,ylim=(-1,1))
        ax3.plot_date(dates_idx[-wndtr:],ddotx1[-wndtr:],xdate=False,marker='',linestyle='-',label=f'ddotx1')
        ax3.plot_date(dates_idx[-wndtr:],df_v3[-wndtr:],label=df_v3.columns,xdate=False,marker='',linestyle='dotted',linewidth=0.8)
        ax3.legend(loc='lower left')
        
        #SECTION 2: ROWS:3 COLUMNS:10 C signal PLOTS
        df_c3 = df1[['ft2nm','ma2nm','ff2nm']]
        df_c2 = df1[['dr2nm','lmda2nm','q2nm','resf2nm']]

        ax4 = plt.subplot2grid((gx,gy),(s1x,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs,ylim=(-1,1))
        ax4.plot_date(dates_idx[-wndtr:],x2[-wndtr:],marker='',xdate=False,linestyle='-',label=f'x2')
        # ax4.step(dates_idx[-wndtr:],df1.idpos1[-wndtr:],label='pos1',linewidth=0.8,linestyle='dotted',marker='')
        ax4.plot_date(dates_idx[-wndtr:],df1.x2prnm[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=0.8,label='x2prnm')
        ax4.plot_date(dates_idx[-wndtr:],df1.x2gnnm[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=0.8,label='x2gnnm')
        ax4.legend(loc='lower left')

        ax5 = plt.subplot2grid((gx,gy),(s1x+s2rs+1,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs,ylim=(-1,1))
        ax5.plot_date(dates_idx[-wndtr:],dotx2[-wndtr:],marker='',xdate=False,linestyle='-',label=f'dotx2')
        ax5.plot_date(dates_idx[-wndtr:],df_c2[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=0.8,label=df_c2.columns)
        ax5.legend(loc='lower left')
       
        ax6 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s1rs,colspan=s1cs,ylim=(-1,1))
        ax6.plot_date(dates_idx[-wndtr:],ddotx2[-wndtr:],marker='',xdate=False,linestyle='-',label=f'ddotx2')
        ax6.plot_date(dates_idx[-wndtr:],df_c3[-wndtr:],label=df_c3.columns,marker='',xdate=False,linestyle='dotted',linewidth=0.8)
        ax6.legend(loc='lower left')

        #SECTION 3: ROWS:3 COLUMNS:5 SCATTER PLOTS
        ax7 = plt.subplot2grid((gx,gy),(s1x,(s1y+s1cs+1)*2),rowspan=s3rs+5,colspan=s3cs,ylim=(-1,1),xlim=(-1,1))
        ax7.plot(x1,x2,color='m',linestyle='None',marker='o',label=f'x1-x2')
        ax7.plot(df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'x1nm')],df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'x2nm')],color='b',linestyle='None',marker='o')
        ax7.plot(df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'x1nm')],df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'x2nm')],color='c',linestyle='None',marker='o')
        ax7.plot(df1.x1nm.mean(),df1.x2nm.mean(),color='r',linestyle='None',marker='o')
        ax7.plot(df1.x1nm[-1],df1.x2nm[-1],color='g',linestyle='None',marker='o')
        ax7.plot(df1.x2gnnm[-1],df1.x2gnnm[-1],color='g',linestyle='None',marker='o')
        ax7.plot(df1.x1prnm[-1],df1.x2prnm[-1],color='g',linestyle='None',marker='o')
        ax7.plot(df1.ft1nm[-1],df1.ft2nm[-1],linestyle='None',marker='o')
        ax7.plot(df1.ma1nm[-1],df1.ma2nm[-1],linestyle='None',marker='o')
        ax7.plot(df1.ff1nm[-1],df1.ff2nm[-1],linestyle='None',marker='o')
        ax7.plot(df1.dr1nm[-1],df1.dr2nm[-1],linestyle='None',marker='o')
        ax7.plot(df1.q1nm[-1],df1.q2nm[-1],linestyle='None',marker='o')
        ax7.plot(df1.x1wrknm[-1],df1.x2wrknm[-1],linestyle='None',marker='o')
        ax7.plot(df1.resf1nm[-1],df1.resf2nm[-1],linestyle='None',marker='o')

        ax7.annotate('Cm',xy=(df1.x1nm.mean(),df1.x2nm.mean()),xytext=(df1.x1nm.mean(),df1.x2nm.mean()),fontweight='bold',xycoords='data',textcoords='data')
        ax7.annotate('Px',xy=(df1.x1nm[-1],df1.x2nm[-1]),xytext=(df1.x1nm[-1],df1.x2nm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Xg',xy=(df1.x2gnnm[-1],df1.x2gnnm[-1]),xytext=(df1.x2gnnm[-1],df1.x2gnnm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Xp',xy=(df1.x1prnm[-1],df1.x2prnm[-1]),xytext=(df1.x1prnm[-1],df1.x2prnm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Fx',xy=(df1.ft1nm[-1],df1.ft2nm[-1]),xytext=(df1.ft1nm[-1],df1.ft2nm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Max',xy=(df1.ma1nm[-1],df1.ma2nm[-1]),xytext=(df1.ma1nm[-1],df1.ma2nm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Ffx',xy=(df1.ff1nm[-1],df1.ff2nm[-1]),xytext=(df1.ff1nm[-1],df1.ff2nm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Drx',xy=(df1.dr1nm[-1],df1.dr2nm[-1]),xytext=(df1.dr1nm[-1],df1.dr2nm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Qx',xy=(df1.q1nm[-1],df1.q2nm[-1]),xytext=(df1.q1nm[-1],df1.q2nm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Wkx',xy=(df1.x1wrknm[-1],df1.x2wrknm[-1]),xytext=(df1.x1wrknm[-1],df1.x2wrknm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Rfx',xy=(df1.resf1nm[-1],df1.resf2nm[-1]),xytext=(df1.resf1nm[-1],df1.resf2nm[-1]),xycoords='data',textcoords='data',fontweight='bold')

        #SECTION 4: MAG AND ENERGY PLOTS
        ax10 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y),rowspan=s4rs,colspan=s1cs)
        ax10.magnitude_spectrum(df1.x1nm,Fs=Fs,linestyle='none',markersize='5',marker='o',label='vt')
        ax10.magnitude_spectrum(df1.x2nm,Fs=Fs,linestyle='none',markersize='5',marker='o',label='ct')
        ax10.legend(loc='upper left')
    
        ax11 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s4rs,colspan=s1cs,ylim=(-1,1))
        # ax11.step(dates_idx[-wndtr:],df1.x1te[-wndtr:],label='x1te',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.x2te[-wndtr:],label='x2te',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.Kr1[-wndtr:],label='Kr1',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.Kr2[-wndtr:],label='Kr2',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.Pr1[-wndtr:],label='Pr1',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.Pr2[-wndtr:],label='Pr2',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.idpos1[-wndtr:],label='pos1',linewidth=1,linestyle='dotted')
        ax11.step(dates_idx[-wndtr:],df1.Pwtnm[-wndtr:],label='Pwr',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.x2prnmt[-wndtr:],label='x2prnmt',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.x1grt[-wndtr:],label='x1grt',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.x2grt[-wndtr:],label='x2grt',linewidth=1,linestyle='dotted')
        ax11.step(dates_idx[-wndtr:],df1.resf1nm[-wndtr:],label='resf1nm',linewidth=1,linestyle='dotted')
        ax11.step(dates_idx[-wndtr:],df1.resf2nm[-wndtr:],label='resf2nm',linewidth=1,linestyle='dotted')
        ax11.step(dates_idx[-wndtr:],df1.Trqnm[-wndtr:],label='Torq',linewidth=1,linestyle='dotted')
        ax11.step(dates_idx[-wndtr:],df1.Rstnm[-wndtr:],label='Imped',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.Qfm[-wndtr:],label='Qx',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.Drm[-wndtr:],label='Drx',linewidth=1,linestyle='dotted')
        # ax11.step(dates_idx[-wndtr:],df1.Wkt[-wndtr:],label='Wkx',linewidth=1,linestyle='dotted')
        ax11.legend(loc='lower left')

        #SECTION 5: PHASE ANGLE PLOTS
        ax12 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1)+1,(s1y+s1cs+1)*2),rowspan=s5rs,colspan=s1cs,projection='polar')
        ax12.set_rscale('symlog')

        ax12.plot(df1.Qfa,df1.Qfm,label='Qx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Dra,df1.Drm,label='Drx',marker='o',markersize=6,linestyle='none') 
        ax12.plot(df1.Wka,df1.Wkm,label='Wkx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Fta,df1.Ftm,label='Fx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Xga,df1.Xgm,label='Xg',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Xpa,df1.Xpm,label='Xp',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Maa,df1.Mam,label='Max',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Ffa,df1.Ffm,label='Ffx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Pwa,df1.Pwm,label='Pwx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Pxa,df1.Pxm,label='Px',marker='o',markersize=6,linestyle='none')

        ax12.annotate('Fx',xy=(df1.Fta[-1],df1.Ftm[-1]),xytext=(df1.Fta[-1],df1.Ftm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Max',xy=(df1.Maa[-1],df1.Mam[-1]),xytext=(df1.Maa[-1],df1.Mam[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Ffx',xy=(df1.Fta[-1],df1.Ftm[-1]),xytext=(df1.Fta[-1],df1.Ftm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Qx',xy=(df1.Qfa[-1],df1.Qfm[-1]),xytext=(df1.Qfa[-1],df1.Qfm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Pwx',xy=(df1.Pwa[-1],df1.Pwm[-1]),xytext=(df1.Pwa[-1],df1.Pwm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Px',xy=(df1.Pxa[-1],df1.Pxm[-1]),xytext=(df1.Pxa[-1],df1.Pxm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Xg',xy=(df1.Xga[-1],df1.Xgm[-1]),xytext=(df1.Xga[-1],df1.Xgm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Xp',xy=(df1.Xpa[-1],df1.Xpm[-1]),xytext=(df1.Xpa[-1],df1.Xpm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Drx',xy=(df1.Dra[-1],df1.Drm[-1]),xytext=(df1.Dra[-1],df1.Drm[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Wkx',xy=(df1.Wka[-1],df1.Wkm[-1]),xytext=(df1.Wka[-1],df1.Wkm[-1]),xycoords='data',textcoords='data',fontweight='bold')

        # ax12.legend(loc='upper left')
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.25, wspace=0.25)
        plt.figure(1).suptitle((f'x1| w1n:{df1.w1n[0].round(3)} |w1:{df1.w1[-1].round(3)} |dr1nm:{df1.dr1nm[-1].round(3)} |q1nm:{df1.q1nm[-1].round(3)} |resf1nm:{df1.resf1nm[-1].round(3)}   x2| w2n:{df1.w2n[-1].round(3)} |w2:{df1.w2[-1].round(3)} |dr2nm:{df1.dr2nm[-1].round(3)} |q2nm:{df1.q2nm[-1].round(3)} |resf2nm:{df1.resf2nm[-1].round(3)}'))
        

        if caller == 'dump':
            plt.show()
        else:
            plt.pause(refresh)
            plt.clf()



