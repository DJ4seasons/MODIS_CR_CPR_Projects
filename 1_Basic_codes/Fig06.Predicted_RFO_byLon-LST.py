"""
# 
# Read predicted_CPR_nums (hourly) 
# and draw RFO in Lon-LST domain for select CPR(s)
# 
# Daeho Jin, 2026.01.08
#
"""
 
import numpy as np
import sys
from datetime import timedelta, date
from netCDF4 import Dataset, num2date

from math import ceil
lat_deg2y = lambda lat,lat0,dlat: ceil((lat-lat0)/dlat)

def get_km(cr_params):
    '''
    Return total number of clusters (km) depending on cr_params
    '''
    prwt_set= [0,1,7]
    try:
        ind= prwt_set.index(cr_params['prwt'])
    except:
        print('prwt should be 0, 1, or 7, but now prwt=',cr_params['prwt'])
        sys.exit()
    if cr_params['rg']==15:
        km_set= [14,16,19]
        return km_set[ind]
    elif cr_params['rg']==50:
        km_set= [15,20,22]
        return km_set[ind]
    else:
        print('rg should be 15 or 50, but now rg=',cr_params['rg'])
        sys.exit()

def main(cr_params, tgt_cr=1):
    ###-- Parameters and defalut values
    ###-------------------------------------
    rg, nelemp, prwt= cr_params['rg'], cr_params['nelemp'], cr_params['prwt']
    nelemc= 42
    nelem=nelemc+nelemp
    km= get_km(cr_params)   # total number of clusters  
    p_letter= 'P' if prwt>0 else ''
    prset_nm = f'Cld{nelemc}+Pr{nelemp}x{prwt}' if prwt>0 else f'Cld{nelemc}'
    rg_nm= f'{rg}S-{rg}N'
    
    ## Read CR_nums and CR_nums_predicted
    tgt_lats= [-rg,rg]  ## It is limited to the original domain, although data were extended.
    tgt_dates= [date(2014,6,1),date(2019,5,31)]
    tgt_date_names= '-'.join([dd.strftime('%Y.%m') for dd in tgt_dates])
    ndy= (tgt_dates[1]-tgt_dates[0]).days+1
    nt_per_day= 24  ## Hourly data

    mdnm_prd= 'MODIS_t+a_C{}R_predicted_hourly.{}_{}'.format(p_letter,rg_nm,prset_nm)
    indir= f'../{mdnm_prd}/'
    iyr, eyr= tgt_dates[0].year, tgt_dates[1].year
    crnums=[]
    for yy in range(iyr,eyr+1,1):
        infn_prd= indir+f'{mdnm_prd}.{yy}.nc'
    
        #-- Open netCDF file #1
        fid= Dataset(infn_prd,'r')    
    
        #-- Read dimension info
        times= fid.variables['time']
        time_units = times.units
        t1= num2date(times[0], units = times.units, calendar=times.calendar,
                          only_use_cftime_datetimes=True,)
        t2= num2date(times[-1], units = times.units, calendar=times.calendar,
                          only_use_cftime_datetimes=True,)
        idate= date(t1.year, t1.month, t1.day)
        edate= date(t2.year, t2.month, t2.day)  
        ndays= (edate-idate).days+1      
        if tgt_dates[1]<edate:
            edy_idx= (tgt_dates[1]-idate).days+1
        else:
            edy_idx= None
        if idate<tgt_dates[0]:
            idy_idx= (tgt_dates[0]-idate).days
        else:
            idy_idx= 0
                      
        if yy==iyr:    
            lons= fid.variables['lon'][:]; nlon= len(lons)
            lon0, dlon= lons[0], (lons[-1]-lons[0])/(nlon-1)
            lats= fid.variables['lat'][:]; nlat0= len(lats)
            lat0, dlat= lats[0], (lats[-1]-lats[0])/(nlat0-1)
            lat_idx= [lat_deg2y(lt,lat0,dlat) for lt in tgt_lats]
            print(lat0,dlat,lat_idx) #; sys.exit()
            lats= lats[lat_idx[0]:lat_idx[1]]
        
        #-- Read CR-nums
        vn= 'Predicted_CRnum' 
        crnums1= fid.variables[vn][:].filled(-9)
        crnums1= crnums1.reshape([ndays,nt_per_day,nlat0,nlon])[idy_idx:edy_idx,:,lat_idx[0]:lat_idx[1],:]
        print(yy,crnums1.shape)
        crnums.append(crnums1)
        fid.close()
    crnums= np.concatenate(crnums,axis=0)
    print(crnums.shape)
    
    #-- Check if tgt_cr is single number or multiple
    if isinstance(tgt_cr, int):
        tcridx= crnums==tgt_cr    
        tcr_nm= f'C{p_letter}R{tgt_cr}'
    elif isinstance(tgt_cr, (tuple, list)):
        tcridx= crnums==tgt_cr[0]    
        tcr_nm= f'C{p_letter}R'+'+'.join([str(v) for v in tgt_cr])
        for cr1 in tgt_cr[1:]:
            tcridx= np.logical_or(tcridx, crnums==cr1)
    else:
        print('Check type of tgt_cr variable',type(tgt_cr))
        sys.exit()
    crnums=0        
    
    #-- Rearange by Lon and LST
    locs= np.where(tcridx)
    t_loc,lon_loc= locs[1],locs[3]
    #print(len(locs),tt.shape,lons.shape)

    #-- Transform lon_idx to lon in degree
    lon_loc= lon_loc*dlon+lon0
    lon_loc[lon_loc<0]+=nlon

    #-- Calculate Local Standard Time (LST) from longitude info
    lst= lon_loc/360*24+t_loc
    lst[lst>=24]-=24

    #-- Bins to build 2D histogram by Lon and LST
    binx= np.arange(0,361,10)
    biny= np.arange(0,24.1,1)
                        
    ###---- Plot
    suptit1= f'{tcr_nm} Distribution in Lon-LST domain'
    suptit2= 'C{}R in {}, {}, k={} [{}]'.format(
            p_letter,rg_nm,prset_nm,km,tgt_date_names)
    outdir = './Pics/'
    outfn = outdir+"Fig.RFO_Lon-LST."+mdnm_prd+f".{tgt_date_names}_{tcr_nm}.png"
    
    pic_data= dict(
        data= (lon_loc,lst), bins= (binx,biny),
        suptit1=suptit1, suptit2=suptit2, outfn=outfn,
    )
    plot_main(pic_data)

    return

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,AutoMinorLocator,FuncFormatter,FixedLocator

def plot_main(pdata):
    ## Parameters and variables
    lon_loc,lst= pdata['data']
    binx,biny= pdata['bins']
    
    #-- Build 2D histogram by Lon and LST
    X,Y=np.meshgrid(binx,biny) ### Boundary grid prepared for pcolormesh
    H, xedges, yedges = np.histogram2d(lon_loc, lst, bins=(binx,biny))
    tot_pop= H.sum()
    H=(H/tot_pop*100.).T  ### Normalized. Now it is in percent(%). Transpose is necessary.
    print(X.shape,H.shape,H.min(),H.max()) ### Check dimension and values
        
    cmax= int(H.max()*20)/20 # About 0.25
    tt= np.round(np.arange(0,cmax+0.001,0.05),2)

    ##-- 
    fig=plt.figure()
    fig.set_size_inches(8,6.5)  ## (xsize,ysize)

    ### Page Title
    fig.suptitle(pdata['suptit1'],fontsize=17,y=1.01,va='bottom',stretch='semi-condensed')

    ## definitions of axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.007

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ax1 = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax1)
    ax_histy = fig.add_axes(rect_histy, sharey=ax1)
    ax_histx.set_title(pdata['suptit2'],fontsize=13, stretch='semi-condensed')
    
    ## Main 2D histogram
    props = dict(edgecolor='none',alpha=0.8,vmin=0.,vmax=cmax,cmap='viridis')
    pic1= ax1.pcolormesh(X,Y,H,**props)

    ax1.set_xlabel('Longitude(deg)',fontsize=12)
    ax1.set_ylabel('Local Solar Time',fontsize=12)
    ax1.tick_params(axis='both',labelsize=10)
    ax1.xaxis.set_major_locator(MultipleLocator(60))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_major_locator(MultipleLocator(6))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(color='0.4',lw=0.8,ls=':')
    ax1.set_xlim([-15,375])
    ax1.set_ylim([-1,25])

    ly0= height/30
    loc1= [left,bottom-ly0*4.5,width,ly0]
    cb= draw_colorbar(fig,pic1,loc1,tt,extend='max')
    cb.set_label('Normalized Fraction(%)',fontsize=12)

    ## Top and right histograms
    arrx,bx,histx= ax_histx.hist(lon_loc,bins=binx)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.yaxis.set_ticks_position('both')
    ax_histx.grid(color='0.4',lw=0.8,ls=':')
    func= lambda x, pos: "{:.0f}".format(x/tot_pop*100)
    ax_histx.yaxis.set_major_formatter(FuncFormatter(func))
    ax_histx.yaxis.set_major_locator(MultipleLocator(tot_pop*0.02))
    ax_histx.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_histx.set_ylabel('Norm. Frac.(%)',fontsize=12) 
        
    arry,by,histy= ax_histy.hist(lst,bins=biny, orientation='horizontal')
    ax_histy.tick_params(axis="y", labelleft=False, labelright=True)
    ax_histy.yaxis.set_ticks_position('both')
    ax_histy.grid(color='0.4',lw=0.8,ls=':')
    ax_histy.xaxis.set_major_formatter(FuncFormatter(func))
    ax_histy.xaxis.set_major_locator(MultipleLocator(tot_pop*0.02))
    ax_histy.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_histy.set_xlabel('Norm. Frac.(%)',fontsize=12)

    ###--- Save or Show
    #plt.show()
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150)
    print(pdata['outfn'])

    return

def draw_colorbar(fig,pic1,loc,tt,ft=10,extend='both'):    
    tt2=[str(x)+'%' for x in tt]

    cb_ax = fig.add_axes(loc)  ##<= (left,bottom,width,height)
    if loc[2]<loc[3]:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='vertical',ticks=tt,extend=extend)
        cb.ax.set_yticklabels(tt2,size=ft,stretch='condensed')
    else:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='horizontal',ticks=tt,extend=extend)
        cb.ax.set_xticklabels(tt2,size=ft,stretch='condensed')
    return cb

if __name__=="__main__":    
    cr_params= dict(
        rg= 50,      # domain max latitude, 15 or 50
        nelemp= 6,   # dimension of pr_hist, fixed to 6
        prwt= 7,     # weight of pr_hist. Prediction is only available with 7
    )
    tgt_cr= [1,2] #1 #
    main(cr_params,tgt_cr)


