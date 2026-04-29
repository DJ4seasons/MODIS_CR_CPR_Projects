"""
#
# Read CR_nums and display mean RFO map for C(P)Rs
#
# Daeho Jin, 2026.01.07
# 
"""

import numpy as np
import sys
from netCDF4 import Dataset, num2date
from datetime import timedelta, date

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

def main(cr_params, sat_nm='TAmean'):
    ###-- Parameters and defalut values
    ###-------------------------------------
    rg, nelemp, prwt= cr_params['rg'], cr_params['nelemp'], cr_params['prwt']
    nelemc= 42
    nelem=nelemc+nelemp
    km= get_km(cr_params)   # total number of clusters  
    p_letter= 'P' if prwt>0 else ''
    prset_nm = f'Cld{nelemc}+Pr{nelemp}x{prwt}' if prwt>0 else f'Cld{nelemc}'
    rg_nm= f'{rg}S-{rg}N'
    
    indir= './Data/'
    mdnm= 'MODIS_t+a_C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)
    infn= indir+f'{mdnm}.nc'
    
    ## Read CR_nums and calculate RFOs
    tgt_lats= [-rg,rg]  ## It is limited to the original domain, although data were extended.
    tgt_dates= [date(2014,6,1),date(2019,5,31)]
    tgt_date_names= '-'.join([dd.strftime('%Y.%m') for dd in tgt_dates])
    ndy= (tgt_dates[1]-tgt_dates[0]).days+1
    
    #-- Open netCDF file
    fid= Dataset(infn,'r')    
    
    #-- Read dimension info
    times= fid.variables['time']
    time_units = times.units
    times= num2date(times[:], units = times.units, calendar=times.calendar,
                      only_use_cftime_datetimes=True,)
    date_range= [date(t1.year, t1.month, t1.day) for t1 in [times[0],times[-1]]]
    itidx= (tgt_dates[0]-date_range[0]).days
    
    lons= fid.variables['lon'][:]
    lats= fid.variables['lat']
    lat0, dlat= lats[0], (lats[-1]-lats[0])/(len(lats)-1)
    lat_idx= [lat_deg2y(lt,lat0,dlat) for lt in tgt_lats]
    print(lat0,dlat,lat_idx) #; sys.exit()
    lats= lats[lat_idx[0]:lat_idx[1]]
    
    #-- Centroid info
    ctd_cld= fid.variables['Centroid_cloud_part'][:]
    ctd_cf = np.sum(ctd_cld,axis=(1,2))*100.    
    
    #-- Read CR-nums
    vn= 'CRnum_on_map_TAmean' if sat_nm=='TAmean' else 'CRnum_on_map_{}'.format(sat_nm.title())
    crnums= fid.variables[vn]
    crnums= crnums[itidx:itidx+ndy,lat_idx[0]:lat_idx[1],:]
    print(crnums.shape)
    fid.close()
    
    #-- Calculate RFO
    tgt_crs= np.arange(1,km+1,1,dtype=int)
    map0=[]
    for cr in tgt_crs:  
        if prwt==0 and cr==km:
            idx1= crnums>km*10  ## Considering sub-regime numbering
        else:
            idx1 = crnums==cr
        map0.append(idx1.mean(axis=0))  ## All time mean
    
    map0= np.asarray(map0)*100.  ## Now in percent

    #-- Global mean RFO; latitude weights are not considered here.
    grfo=map0.mean(axis=(1,2))
    print(grfo)


    ###---- Plot
    suptit= "C{}R Mean RFO in {}, {}, k={}, [{}, {}]".format(p_letter,rg_nm,prset_nm,km,sat_nm,tgt_date_names)
    outdir = './Pics/'
    outfn = outdir+"Fig.RFO_"+mdnm+f".{sat_nm}_{tgt_date_names}.png"
    xy= np.meshgrid(lons,lats)
    
    pic_data= dict(
        rfo_maps= map0, grfo= grfo,
        tgt_crs= tgt_crs, ctd_cf= ctd_cf, 
        xy=xy, p_letter= p_letter,
        suptit=suptit, outfn=outfn,
    )
    if rg==15:
        plot_map15(pic_data)
    elif rg==50:
        plot_map50(pic_data)

    return

import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator, FuncFormatter
import cartopy.crs as ccrs

def plot_map15(pdata):
    ## Parameters and variables
    rfos= pdata['rfo_maps']
    grfo= pdata['grfo']
    tgt_crs= pdata['tgt_crs']
    ctd_cf= pdata['ctd_cf']
    xy= pdata['xy']
    p_letter= pdata['p_letter']
    lat_max= rfos.shape[1]//2
    
    ##---
    fig = plt.figure()
    fig.set_size_inches(12,15)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=20,y=0.975,va='bottom')

    lf=0.05;rf=0.95
    bf=0.12;tf=0.94
    gapx=0.02; npnx=2
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy=0.04; npny=10
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    ix=lf; iy=tf

    cm = plt.get_cmap('magma_r').resampled(80) 
    cmnew = cm(np.arange(80)) 
    cmnew = np.concatenate((np.array([1,1,1,1]).reshape([1,-1]),cmnew[:-1,:])) 
    newcm = cls.LinearSegmentedColormap.from_list("newCMR",cmnew)
    newcm.set_under("white")

    lon_ext= [20,360+20]
    lc= (lon_ext[0]+lon_ext[1])/2
    map_proj= ccrs.PlateCarree(central_longitude=lc)
    data_crs= ccrs.PlateCarree()
    props_pc = dict(cmap=newcm,vmin=0.,vmax=50,alpha=0.9,shading='nearest',transform=data_crs)
    cb_xt= range(0,51,10)
        
    for k,cr in enumerate(tgt_crs):
        crnm= 'C{}R{}'.format(p_letter,cr)
        ax= fig.add_axes([ix,iy-ly,lx,ly],projection=map_proj)
        ax.set_extent(lon_ext+[-lat_max*1.03,lat_max*1.03],data_crs)

        ocf= ctd_cf[cr-1]
        rfo= grfo[cr-1]
        amap= rfos[cr-1]
        
        # add a title.
        rfo_text= 'RFO={:.1f}%'.format(rfo) if rfo>9.995 else 'RFO={:.2f}%'.format(rfo)
        subtit='{} [CF={:.1f}%, {}]'.format(crnm,ocf,rfo_text)
        print(subtit)
        ax.set_title(subtit,fontsize=15,x=0.0,ha='left',stretch='semi-condensed') #
        cs=ax.pcolormesh(*xy,amap,**props_pc)
        
        ll= True if ix==lf or k==0 else False
        lr= True if ix+lx+gapx>rf else False
        label_idx=[ll,lr,False,True]
        map_common(ax,lat_max,label_idx)
        
        ix= ix+lx+gapx
        if ix>rf:
            iy=iy-ly-gapy
            ix=lf

    ##-- Draw Color Bar
    ix_cb,iy_cb= ix, iy
    ly_cb= gapy/4
    if ix_cb==lf:
        iy_cb+= gapy/3
        loc1= [0.2,iy_cb-ly_cb,0.6,ly_cb]
    else:
        iy_cb-= ly_cb
        loc1= [ix_cb,iy_cb-ly_cb,lx,ly_cb]
    cb1=draw_colorbar(fig,cs,loc1,cb_xt,ft=12,extend='max')
    cb1.ax.set_xlabel('RFO',fontsize=14)

    ##-- Show or Save
    #plt.show()
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150)
    print(pdata['outfn'])
    return

def plot_map50(pdata):
    ## Parameters and variables
    rfos= pdata['rfo_maps']
    grfo= pdata['grfo']
    tgt_crs= pdata['tgt_crs']
    ctd_cf= pdata['ctd_cf']
    xy= pdata['xy']
    p_letter= pdata['p_letter']
    lat_max= rfos.shape[1]//2
    
    ##---
    fig = plt.figure()
    fig.set_size_inches(13,16)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=20,y=0.975,va='bottom')

    lf=0.05;rf=0.95
    bf=0.04;tf=0.944
    gapx=0.015; npnx=3
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy=0.042; npny=8
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    ix=lf; iy=tf

    cm = plt.get_cmap('magma_r').resampled(80) 
    cmnew = cm(np.arange(80)) 
    cmnew = np.concatenate((np.array([1,1,1,1]).reshape([1,-1]),cmnew[:-1,:])) 
    newcm = cls.LinearSegmentedColormap.from_list("newCMR",cmnew)
    newcm.set_under("white")

    lon_ext= [20,360+20]
    lc= (lon_ext[0]+lon_ext[1])/2
    map_proj= ccrs.PlateCarree(central_longitude=lc)
    data_crs= ccrs.PlateCarree()
    props_pc = dict(cmap=newcm,vmin=0.,vmax=50,alpha=0.9,shading='nearest',transform=data_crs)
    cb_xt= range(0,51,10)
        
    for k,cr in enumerate(tgt_crs):
        crnm= 'C{}R{}'.format(p_letter,cr)
        ax= fig.add_axes([ix,iy-ly,lx,ly],projection=map_proj)
        ax.set_extent(lon_ext+[-lat_max*1.03,lat_max*1.03],data_crs)
        
        ocf= ctd_cf[cr-1]
        rfo= grfo[cr-1]
        amap= rfos[cr-1]
        
        # add a title.
        rfo_text= 'RFO={:.1f}%'.format(rfo) if rfo>9.995 else 'RFO={:.2f}%'.format(rfo)
        subtit='{} [CF={:.1f}%, {}]'.format(crnm,ocf,rfo_text)
        print(subtit)
        ax.set_title(subtit,fontsize=15,x=0.0,ha='left',stretch='semi-condensed') #
        cs=ax.pcolormesh(*xy,amap,**props_pc)
        
        ll= True if ix==lf or k==0 else False
        lr= True if ix+lx+gapx>rf else False
        label_idx=[ll,lr,False,True]
        map_common(ax,lat_max,label_idx)
        
        ix= ix+lx+gapx
        if ix>rf:
            iy=iy-ly-gapy
            ix=lf

    ##-- Draw Color Bar
    ix_cb,iy_cb= ix, iy
    ly_cb= gapy/4
    if ix_cb==lf:
        iy_cb+= gapy/3
        loc1= [0.2,iy_cb-ly_cb,0.6,ly_cb]
    else:
        iy_cb-= ly_cb
        loc1= [ix_cb,iy_cb-ly_cb,lx,ly_cb]
    cb1=draw_colorbar(fig,cs,loc1,cb_xt,ft=12,extend='max')
    cb1.ax.set_xlabel('RFO',fontsize=14)

    ##-- Show or Save
    #plt.show()
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150)
    print(pdata['outfn'])
    return

def map_common(ax,lat_max,label_idx=[True,True,False,True]):

    if lat_max<=30:
        #ax.set_extent([0.,360,-16,16],ccrs.PlateCarree())
        ax.set_yticks([-15,0,15])
    else:
        #ax.set_extent([0.,360,-52,52],ccrs.PlateCarree())
        ax.set_yticks([-50,-25,0,25,50])

    ax.tick_params(axis='y',labelright=False,labelleft=False,labelsize=11)
    ax.tick_params(direction='in',left=True,right=True,top=True,bottom=True,)
    
    ax.coastlines(color='silver',linewidth=1.)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = label_idx[2]
    gl.left_labels = label_idx[0]
    gl.right_labels = label_idx[1]
    gl.bottom_labels = label_idx[3]

    gl.xlocator = MultipleLocator(60) 

    if lat_max>30:
        gl.ylocator = MultipleLocator(25)
    else:
        gl.ylocator = MultipleLocator(15)
        
    gl.xlabel_style = {'size': 11, 'color': 'k'}
    gl.ylabel_style = {'size': 11, 'color': 'k'}

    ax.set_aspect('auto')
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

###-------------------------------------

if __name__=="__main__":    
    cr_params= dict(
        rg= 15,      # domain max latitude, 15 or 50
        nelemp= 6,   # dimension of pr_hist, fixed to 6
        prwt= 0,     # weight of pr_hist, 0, 1, or 7
    )
    sat_nm= 'TAmean' # 'terra', 'aqua', or 'TAmean'
    main(cr_params,sat_nm)
    sys.exit()
