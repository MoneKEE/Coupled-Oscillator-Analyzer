from numpy.random.mtrand import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates

def showplots(df1,obv,F,m,refresh=0.5,caller='dump',asset='ETH-USD'):
    pd.plotting.register_matplotlib_converters()

    with plt.style.context(style='dark_background'):
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
        T = N/Fs
        df = 1/T
        dw = 2*np.pi*df
        ny = Fs/2

        wndtr = round(2048*T)  

        ttl = f'Analysis Dashboard - Asset:{asset} |N={len(df1)} |wnd={wndtr} |T={round(T,3)} |Fs={round(Fs,3)} |F={round(df,3)} |W={round(dw,3)} |Fn={round(ny,3)} |m={m}'

        plt.ion()

        dates_idx = dates.date2num(df1.index)
        if len(plt.get_fignums()) == 0:
            fig = plt.figure(ttl)
        else:
            plt.figure(1)

        x1 = df1.x1
        x2 = df1.x2

        dotx1 = df1.dotx1
        dotx2 = df1.dotx2

        ddotx1 = df1.ddotx1
        ddotx2 = df1.ddotx2
            
        #SECTION 1: ROWS:3 COLUMNS:10 V signal PLOTS
        df_v3 = df1[['ft1','ma1','ff1','pwr1','wrk1','te1']]
        df_v2 = df1[['dr1','lmda1','q1','rf1','psr1']]

        ax1 = plt.subplot2grid((gx,gy),(s1x,s1y),rowspan=s1rs,colspan=s1cs)
        # ax1.step(dates_idx[-wndtr:],df1.pos[-wndtr:],label='pos1',linewidth=0.5,linestyle='dotted',marker='')
        ax1.plot_date(dates_idx[-wndtr:],x1[-wndtr:],xdate=False,linewidth=1,linestyle='-',label='x1',marker='')
        ax1.plot_date(dates_idx[-wndtr:],df1.xg1[-wndtr:],xdate=False,linestyle='dotted',marker='',linewidth=1,label='xg1')
        ax1.plot_date(dates_idx[-wndtr:],df1.xp1[-wndtr:],xdate=False,linestyle='dotted',marker='',linewidth=1,label='xp1')
        ax1.legend(loc='lower left')

        ax2 = plt.subplot2grid((gx,gy),(s1x+s1rs+1,s1y),rowspan=s1rs,colspan=s1cs)
        ax2.plot_date(dates_idx[-wndtr:],dotx1[-wndtr:],xdate=False,marker='',linestyle='-',label=f'dotx1',linewidth=1)
        ax2.plot_date(dates_idx[-wndtr:],df_v2[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=1,label=df_v2.columns)
        ax2.legend(loc='lower left')

        ax3 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y),rowspan=s1rs,colspan=s1cs)
        ax3.plot_date(dates_idx[-wndtr:],df_v3[-wndtr:],label=df_v3.columns,xdate=False,marker='',linestyle='dotted',linewidth=1)
        ax3.legend(loc='lower left')
        
        #SECTION 2: ROWS:3 COLUMNS:10 C signal PLOTS
        df_c3 = df1[['ft2','ma2','ff2','pwr2','wrk2','te2']]
        df_c2 = df1[['dr2','lmda2','q2','rf2','psr2']]

        ax4 = plt.subplot2grid((gx,gy),(s1x,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        # ax4.step(dates_idx[-wndtr:],df1.pos[-wndtr:],label='pos1',linewidth=0.5,linestyle='dotted',marker='')
        ax4.plot_date(dates_idx[-wndtr:],x2[-wndtr:],marker='',xdate=False,linestyle='-',label=f'x2',linewidth=1)
        
        ax4.plot_date(dates_idx[-wndtr:],df1.xg2[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=1,label='xg2')
        ax4.plot_date(dates_idx[-wndtr:],df1.xp2[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=1,label='xp2')
        ax4.legend(loc='lower left')

        ax5 = plt.subplot2grid((gx,gy),(s1x+s2rs+1,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax5.plot_date(dates_idx[-wndtr:],dotx2[-wndtr:],marker='',xdate=False,linestyle='-',label=f'dotx2',linewidth=1)
        ax5.plot_date(dates_idx[-wndtr:],df_c2[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=1,label=df_c2.columns)
        ax5.legend(loc='lower left')
       
        ax6 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s1rs,colspan=s1cs)
        ax6.plot_date(dates_idx[-wndtr:],df_c3[-wndtr:],label=df_c3.columns,marker='',xdate=False,linestyle='dotted',linewidth=1)
        ax6.legend(loc='lower left')

        #SECTION 3: ROWS:3 COLUMNS:5 SCATTER PLOTS
        ax7 = plt.subplot2grid((gx,gy),(s1x,(s1y+s1cs+1)*2),rowspan=s3rs+5,colspan=s3cs)
        ax7.plot(x1,x2,color='m',linestyle='None',marker='o',label=f'x1-x2')
        ax7.plot(df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'x1')],df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'x2')],color='b',linestyle='None',marker='o')
        ax7.plot(df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'x1')],df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'x2')],color='c',linestyle='None',marker='o')
        ax7.plot(x1.mean(),x2.mean(),color='r',linestyle='None',marker='o')
        ax7.plot(x1.iloc[-1],x2.iloc[-1],color='g',linestyle='None',marker='o')
        ax7.plot(df1.xg2.iloc[-1],df1.xg2.iloc[-1],linestyle='None',marker='o')
        ax7.plot(df1.xp1.iloc[-1],df1.xp2.iloc[-1],linestyle='None',marker='o')
        ax7.plot(df1.ft1.iloc[-1],df1.ft2.iloc[-1],linestyle='None',marker='o')
        ax7.plot(df1.ff1.iloc[-1],df1.ff2.iloc[-1],linestyle='None',marker='o')

       # ax7.annotate('Cm',xy=(x1.mean(),x2.mean()),xytext=(x1.mean(),x2.mean()),fontweight='bold',xycoords='data',textcoords='data')
        ax7.annotate('Px',xy=(x1.iloc[-1],x2.iloc[-1]),xytext=(x1.iloc[-1],x2.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Xg',xy=(df1.xg2.iloc[-1],df1.xg2.iloc[-1]),xytext=(df1.xg2.iloc[-1],df1.xg2.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Xp',xy=(df1.xp1.iloc[-1],df1.xp2.iloc[-1]),xytext=(df1.xp1.iloc[-1],df1.xp2.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Ft',xy=(df1.ft1.iloc[-1],df1.ft2.iloc[-1]),xytext=(df1.ft1.iloc[-1],df1.ft2.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Ff',xy=(df1.ff1.iloc[-1],df1.ff2.iloc[-1]),xytext=(df1.ff1.iloc[-1],df1.ff2.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        
        #SECTION 4: MAG AND ENERGY PLOTS
        ax10 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y),rowspan=s4rs,colspan=s1cs)
        ax10.magnitude_spectrum(x1,Fs=Fs,scale='dB',linestyle='none',markersize='5',marker='o',label='vt')
        ax10.magnitude_spectrum(x2,Fs=Fs,scale='dB',linestyle='none',markersize='5',marker='o',label='ct')
        ax10.legend(loc='upper left')
    
        ax11 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s4rs,colspan=s1cs)
        ax11.plot_date(dates_idx[-wndtr:],df1.Pwm[-wndtr:],xdate=False,label='Pwm',linewidth=1.0,marker='',linestyle='dotted')
        ax11.plot_date(dates_idx[-wndtr:],df1.Trq[-wndtr:],xdate=False,label='Trq',linewidth=1.0,marker='',linestyle='dotted')
        ax11.plot_date(dates_idx[-wndtr:],df1.Pxm[-wndtr:],xdate=False,label='Pxm',linewidth=1.0,marker='',linestyle='dotted')
        ax11.plot_date(dates_idx[-wndtr:],df1.Mam[-wndtr:],xdate=False,label='Mam',linewidth=1.0,marker='',linestyle='dotted')
        ax11.plot_date(dates_idx[-wndtr:],df1.Ftm[-wndtr:],xdate=False,label='Ftm',linewidth=1.0,marker='',linestyle='dotted')
        ax11.plot_date(dates_idx[-wndtr:],df1.Wkm[-wndtr:],xdate=False,label='Wkm',linewidth=1.0,marker='',linestyle='dotted')
        ax11.plot_date(dates_idx[-wndtr:],df1.Sen[-wndtr:],xdate=False,label='Sen',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.pwr1[-wndtr:],label='pwr1',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.pwr2[-wndtr:],label='pwr2',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.wrk1[-wndtr:],label='wrk1',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.wrk2[-wndtr:],label='wrk2',linewidth=1.0,marker='',linestyle='dotted')
        ax11.legend(loc='lower left')

        #SECTION 5: PHASE ANGLE PLOTS
        ax12 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1)+1,(s1y+s1cs+1)*2),rowspan=s5rs,colspan=s1cs,projection='polar')
        ax12.plot(df1.Fta,df1.Ftm,label='Fx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Ffa,df1.Ffm,label='Ffx',marker='o',markersize=6,linestyle='none')
        
        ax12.plot(df1.Pwa,df1.Pwm,label='Pwx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Maa,df1.Mam,label='Max',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Wka,df1.Wkm,label='Wkx',marker='o',markersize=6,linestyle='none')

        ax12.annotate('Fx',xy=(df1.Fta.iloc[-1],df1.Ftm.iloc[-1]),xytext=(df1.Fta.iloc[-1],df1.Ftm.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Max',xy=(df1.Maa.iloc[-1],df1.Mam.iloc[-1]),xytext=(df1.Maa.iloc[-1],df1.Mam.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Ffx',xy=(df1.Fta.iloc[-1],df1.Ftm.iloc[-1]),xytext=(df1.Fta.iloc[-1],df1.Ftm.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Pwx',xy=(df1.Pwa.iloc[-1],df1.Pwm.iloc[-1]),xytext=(df1.Pwa.iloc[-1],df1.Pwm.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Wkx',xy=(df1.Wka.iloc[-1],df1.Wkm.iloc[-1]),xytext=(df1.Wka.iloc[-1],df1.Wkm.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.set_rscale('symlog')
        ax12.legend(loc='upper left')

        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.25, wspace=0.25)
        plt.figure(1).suptitle((f'x1| w1n:{df1.w1o[0].round(3)} |w1:{df1.w1.iloc[-1].round(3)} |dr1:{df1.dr1.iloc[-1].round(3)} |q1:{df1.q1.iloc[-1].round(3)} |rf1:{df1.rf1.iloc[-1].round(3)}   x2| w2n:{df1.w2o.iloc[-1].round(3)} |w2:{df1.w2.iloc[-1].round(3)} |dr2:{df1.dr2.iloc[-1].round(3)} |q2:{df1.q2.iloc[-1].round(3)} |rf2:{df1.rf2.iloc[-1].round(3)}'))
        

        if caller == 'dump':
            plt.show()
        else:
            plt.pause(refresh)
            plt.clf()


def showplots2(df1,obv,F,m,refresh=0.5,caller='dump',asset='ETH-USD'):
    pd.plotting.register_matplotlib_converters()

    with plt.style.context(style='fast'):
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
        T = N/Fs
        df = 1/T
        dw = 2*np.pi*df
        ny = Fs/2

        wndtr = round(1024*T)  

        ttl = f'Analysis Dashboard - Asset:{asset} |N={len(df1)} |wnd={wndtr} |T={round(T,3)} |Fs={round(Fs,3)} |F={round(df,3)} |W={round(dw,3)} |Fn={round(ny,3)} |m={m}'

        plt.ion()

        dates_idx = dates.date2num(df1.index)
        if len(plt.get_fignums()) == 0:
            fig = plt.figure(ttl)
        else:
            plt.figure(1)

        x1 = df1.x1
        x2 = df1.x2

        dotx1 = df1.dotx1
        dotx2 = df1.dotx2

        ddotx1 = df1.ddotx1
        ddotx2 = df1.ddotx2
            
        #SECTION 1: ROWS:3 COLUMNS:10 V signal PLOTS
        df_v3 = df1[['ft1','ma1','ff1','pwr1','wrk1','te1']]
        df_v2 = df1[['dr1','lmda1','q1','rf1','psr1']]

        ax1 = plt.subplot2grid((gx,gy),(s1x,s1y),rowspan=s1rs,colspan=s1cs,ylim=(-1,1))
        # ax1.step(dates_idx[-wndtr:],df1.pos[-wndtr:],label='pos1',linewidth=0.5,linestyle='dotted',marker='')
        ax1.plot_date(dates_idx[-wndtr:],x1[-wndtr:],xdate=False,linestyle='-',label='x1',marker='')
        ax1.plot_date(dates_idx[-wndtr:],df1.xg1[-wndtr:],xdate=False,linestyle='dotted',marker='',linewidth=0.8,label='xg1')
        ax1.plot_date(dates_idx[-wndtr:],df1.xp1[-wndtr:],xdate=False,linestyle='dotted',marker='',linewidth=0.8,label='xp1')
        ax1.legend(loc='lower left')

        ax2 = plt.subplot2grid((gx,gy),(s1x+s1rs+1,s1y),rowspan=s1rs,colspan=s1cs)
        ax2.plot_date(dates_idx[-wndtr:],dotx1[-wndtr:],xdate=False,marker='',linestyle='-',label=f'dotx1')
        ax2.plot_date(dates_idx[-wndtr:],df_v2[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=0.8,label=df_v2.columns)
        ax2.legend(loc='lower left')

        ax3 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y),rowspan=s1rs,colspan=s1cs)
        # ax3.plot_date(dates_idx[-wndtr:],ddotx1[-wndtr:],xdate=False,marker='',linestyle='-',label=f'ddotx1')
        ax3.plot_date(dates_idx[-wndtr:],df_v3[-wndtr:],label=df_v3.columns,xdate=False,marker='',linestyle='dotted',linewidth=0.8)
        ax3.legend(loc='lower left')
        
        #SECTION 2: ROWS:3 COLUMNS:10 C signal PLOTS
        df_c3 = df1[['ft2','ma2','ff2','pwr2','wrk2','te2']]
        df_c2 = df1[['dr2','lmda2','q2','rf2','psr2']]

        ax4 = plt.subplot2grid((gx,gy),(s1x,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs,ylim=(-1,1))
        # ax4.step(dates_idx[-wndtr:],df1.pos[-wndtr:],label='pos1',linewidth=0.5,linestyle='dotted',marker='')
        ax4.plot_date(dates_idx[-wndtr:],x2[-wndtr:],marker='',xdate=False,linestyle='-',label=f'x2')
        
        ax4.plot_date(dates_idx[-wndtr:],df1.xg2[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=0.8,label='xg2')
        ax4.plot_date(dates_idx[-wndtr:],df1.xp2[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=0.8,label='xp2')
        ax4.legend(loc='lower left')

        ax5 = plt.subplot2grid((gx,gy),(s1x+s2rs+1,s1y+s1cs+1),rowspan=s2rs,colspan=s2cs)
        ax5.plot_date(dates_idx[-wndtr:],dotx2[-wndtr:],marker='',xdate=False,linestyle='-',label=f'dotx2')
        ax5.plot_date(dates_idx[-wndtr:],df_c2[-wndtr:],xdate=False,marker='',linestyle='dotted',linewidth=0.8,label=df_c2.columns)
        ax5.legend(loc='lower left')
       
        ax6 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s1rs,colspan=s1cs)
        # ax6.plot_date(dates_idx[-wndtr:],ddotx2[-wndtr:],marker='',xdate=False,linestyle='-',label=f'ddotx2')
        ax6.plot_date(dates_idx[-wndtr:],df_c3[-wndtr:],label=df_c3.columns,marker='',xdate=False,linestyle='dotted',linewidth=0.8)
        ax6.legend(loc='lower left')

        #SECTION 3: ROWS:3 COLUMNS:5 SCATTER PLOTS
        ax7 = plt.subplot2grid((gx,gy),(s1x,(s1y+s1cs+1)*2),rowspan=s3rs+5,colspan=s3cs,ylim=(-1.0,1.0),xlim=(-1.0,1.0))
        ax7.plot(x1,x2,color='m',linestyle='None',marker='o',label=f'x1-x2')
        ax7.plot(df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'x1')],df1.iloc[int(len(df1)/2):-1,df1.columns.get_loc(f'x2')],color='b',linestyle='None',marker='o')
        ax7.plot(df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'x1')],df1.iloc[-len(df1)//6:,df1.columns.get_loc(f'x2')],color='c',linestyle='None',marker='o')
        ax7.plot(x1.mean(),x2.mean(),color='r',linestyle='None',marker='o')
        ax7.plot(x1.iloc[-1],x2.iloc[-1],color='g',linestyle='None',marker='o')
        ax7.plot(df1.xg2.iloc[-1],df1.xg2.iloc[-1],linestyle='None',marker='o')
        ax7.plot(df1.xp1.iloc[-1],df1.xp2.iloc[-1],linestyle='None',marker='o')
        ax7.plot(df1.ft1.iloc[-1],df1.ft2.iloc[-1],linestyle='None',marker='o')
        ax7.plot(df1.ff1.iloc[-1],df1.ff2.iloc[-1],linestyle='None',marker='o')

       # ax7.annotate('Cm',xy=(x1.mean(),x2.mean()),xytext=(x1.mean(),x2.mean()),fontweight='bold',xycoords='data',textcoords='data')
        ax7.annotate('Px',xy=(x1.iloc[-1],x2.iloc[-1]),xytext=(x1.iloc[-1],x2.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Xg',xy=(df1.xg2.iloc[-1],df1.xg2.iloc[-1]),xytext=(df1.xg2.iloc[-1],df1.xg2.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Xp',xy=(df1.xp1.iloc[-1],df1.xp2.iloc[-1]),xytext=(df1.xp1.iloc[-1],df1.xp2.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Ft',xy=(df1.ft1.iloc[-1],df1.ft2.iloc[-1]),xytext=(df1.ft1.iloc[-1],df1.ft2.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax7.annotate('Ff',xy=(df1.ff1.iloc[-1],df1.ff2.iloc[-1]),xytext=(df1.ff1.iloc[-1],df1.ff2.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        
        #SECTION 4: MAG AND ENERGY PLOTS
        ax10 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y),rowspan=s4rs,colspan=s1cs)
        ax10.magnitude_spectrum(x1,Fs=Fs,scale='dB',linestyle='none',markersize='5',marker='o',label='vt')
        ax10.magnitude_spectrum(x2,Fs=Fs,scale='dB',linestyle='none',markersize='5',marker='o',label='ct')
        ax10.legend(loc='upper left')
    
        ax11 = plt.subplot2grid((gx,gy),(3*(s1x+s1rs+1),s1y+s1cs+1),rowspan=s4rs,colspan=s1cs,ylim=(-0.5,0.5))
        ax11.plot_date(dates_idx[-wndtr:],df1.Trq[-wndtr:],label='Trq',linestyle='-',marker='')
        # ax11.plot_date(dates_idx[-wndtr:],df1.te2[-wndtr:],label='te2',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.ke1[-wndtr:],label='ke1',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.ke2[-wndtr:],label='ke2',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.pe1[-wndtr:],label='pe1',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.pe2[-wndtr:],label='pe2',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.pwr1[-wndtr:],label='pwr1',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.pwr2[-wndtr:],label='pwr2',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.wrk1[-wndtr:],label='wrk1',linewidth=1.0,marker='',linestyle='dotted')
        # ax11.plot_date(dates_idx[-wndtr:],df1.wrk2[-wndtr:],label='wrk2',linewidth=1.0,marker='',linestyle='dotted')
        ax11.legend(loc='lower left')

        #SECTION 5: PHASE ANGLE PLOTS
        ax12 = plt.subplot2grid((gx,gy),(2*(s1x+s1rs+1)+1,(s1y+s1cs+1)*2),rowspan=s5rs,colspan=s1cs,projection='polar')
        ax12.plot(df1.Fta,df1.Ftm,label='Fx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Ffa,df1.Ffm,label='Ffx',marker='o',markersize=6,linestyle='none')
        
        ax12.plot(df1.Pwa,df1.Pwm,label='Pwx',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Maa,df1.Mam,label='Max',marker='o',markersize=6,linestyle='none')
        ax12.plot(df1.Wka,df1.Wkm,label='Wkx',marker='o',markersize=6,linestyle='none')

        ax12.annotate('Fx',xy=(df1.Fta.iloc[-1],df1.Ftm.iloc[-1]),xytext=(df1.Fta.iloc[-1],df1.Ftm.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Max',xy=(df1.Maa.iloc[-1],df1.Mam.iloc[-1]),xytext=(df1.Maa.iloc[-1],df1.Mam.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Ffx',xy=(df1.Fta.iloc[-1],df1.Ftm.iloc[-1]),xytext=(df1.Fta.iloc[-1],df1.Ftm.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Pwx',xy=(df1.Pwa.iloc[-1],df1.Pwm.iloc[-1]),xytext=(df1.Pwa.iloc[-1],df1.Pwm.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')
        ax12.annotate('Wkx',xy=(df1.Wka.iloc[-1],df1.Wkm.iloc[-1]),xytext=(df1.Wka.iloc[-1],df1.Wkm.iloc[-1]),xycoords='data',textcoords='data',fontweight='bold')

        ax12.legend(loc='upper left')
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.25, wspace=0.25)
        plt.figure(1).suptitle((f'x1| w1n:{df1.w1o[0].round(3)} |w1:{df1.w1.iloc[-1].round(3)} |dr1:{df1.dr1.iloc[-1].round(3)} |q1:{df1.q1.iloc[-1].round(3)} |rf1:{df1.rf1.iloc[-1].round(3)}   x2| w2n:{df1.w2o.iloc[-1].round(3)} |w2:{df1.w2.iloc[-1].round(3)} |dr2:{df1.dr2.iloc[-1].round(3)} |q2:{df1.q2.iloc[-1].round(3)} |rf2:{df1.rf2.iloc[-1].round(3)}'))
        

        if caller == 'dump':
            plt.show()
        else:
            plt.pause(refresh)
            plt.clf()



