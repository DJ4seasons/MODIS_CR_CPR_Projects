"""
#
# Read CPR_nums and predicted_CPR_nums (daily)  
# and draw a confusion matrix
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
    
    indir= '../'
    mdnm= 'MODIS_t+a_C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)
    infn= indir+f'{mdnm}.nc'
    
    mdnm_prd= 'MODIS_t+a_C{}R_predicted_daily.{}_{}'.format(p_letter,rg_nm,prset_nm)
    infn_prd= indir+f'{mdnm_prd}.nc'
    
    ## Read CR_nums and CR_nums_predicted
    tgt_lats= [-rg,rg]  ## It is limited to the original domain, although data were extended.
    tgt_dates= [date(2014,6,1),date(2019,5,31)]
    tgt_date_names= '-'.join([dd.strftime('%Y.%m') for dd in tgt_dates])
    ndy= (tgt_dates[1]-tgt_dates[0]).days+1
    
    #-- Open netCDF file #1
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
    ctd_cld= fid.variables['Centroid_cloud_part'][:].filled(-999.)
    ctd_phist= fid.variables['Centroid_precipitation_part'][:].filled(-999.)
    ctd_cf = np.sum(ctd_cld,axis=(1,2))*100.    
    ctd_pf = np.sum(ctd_phist,axis=1)*100.
    
    #-- Select CRs having notable precip
    tgt_cpr= np.where(ctd_pf>=10.)[0]  ## Only consider regimes having notable precip
    np.set_printoptions(precision=1,suppress=True)
    print("CF: ",ctd_cf)
    print("PF: ",ctd_pf)
    print("Target CPR: ",[x+1 for x in tgt_cpr])
    tgt_cpr= np.concatenate((tgt_cpr,[98,]))+1  ## 99: other regimes with weak precip.
    km2= len(tgt_cpr)
    no_tgt_cpr= [cpr for cpr in range(1,km+1,1) if cpr not in tgt_cpr]
    print("Out of Target CPR: ", no_tgt_cpr) 
    
    #-- Read CR-nums
    vn= 'CRnum_on_map_TAmean' if sat_nm=='TAmean' else 'CRnum_on_map_{}'.format(sat_nm.title())
    crnums= fid.variables[vn]
    crnums= crnums[itidx:itidx+ndy,lat_idx[0]:lat_idx[1],:]
    print(crnums.shape)
    fid.close()
    
    #-- Open netCDF file #2
    fid2= Dataset(infn_prd,'r')   
    
    #-- Now its dimension is assumed as identical to original CR_nums
    vn= 'Predicted_CRnum_TAmean' if sat_nm=='TAmean' else 'Predicted_CRnum_{}'.format(sat_nm.title())
    crnums_prd= fid2.variables[vn]
    crnums_prd= crnums_prd[itidx:itidx+ndy,lat_idx[0]:lat_idx[1],:]
    print(crnums_prd.shape)
    fid2.close()
    
    ## Check consistency
    #-- Check if tgt_cprs is identical
    cpr_prd= np.unique(crnums_prd[:5,:,:])
    if cpr_prd[0]==-1: cpr_prd= cpr_prd[1:]
    if not np.array_equal(tgt_cpr,cpr_prd):
        print("Target CPR Mismatch: ",tgt_cpr,cpr_prd)
        sys.exit()
    
    #-- Replace other regime numbers to 99
    for notcpr in no_tgt_cpr:
        crnums= np.where(crnums==notcpr,99,crnums)
    print(np.unique(crnums[:5,:,:]))

    #-- Build confusion matrix
    conf_mtx=np.zeros([km2,km2],dtype=float)
    cr_bin= np.concatenate((tgt_cpr-0.5,[tgt_cpr[-1]+0.5,]))
    print(cr_bin) 

    for k,cpr in enumerate(tgt_cpr):
        idx1= crnums==cpr
        hist1= np.histogram(crnums_prd[idx1],bins=cr_bin)[0]
        conf_mtx[k,:]= hist1/hist1.sum()*100.
        print(k,cpr,np.sort(conf_mtx[k,:])[-3:])


    ###---- Plot
    suptit1= 'Original vs. "By Precip-only" Assignment'
    suptit2= 'C{}R in {}, {}, k={} [{}, {}]'.format(
            p_letter,rg_nm,prset_nm,km,sat_nm,tgt_date_names)
    outdir = './Pics/'
    outfn = outdir+"Fig.Confusion_Matrix."+mdnm+f".{sat_nm}_{tgt_date_names}.png"
    
    pic_data= dict(
        conf_mtx= conf_mtx,
        tgt_crs= tgt_cpr, 
        suptit1=suptit1, suptit2=suptit2, outfn=outfn,
    )
    plot_main(pic_data)

    return

import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator
from itertools import repeat

def plot_main(pdata):
    ## Parameters and variables
    conf_mtx= pdata['conf_mtx']
    tgt_crs= pdata['tgt_crs']
    km2= len(tgt_crs)
    
    ##-- 
    fig = plt.figure()
    fig.set_size_inches(7.5,11)    ## (lx,ly)
    plt.suptitle(pdata['suptit1'],fontsize=17,y=0.98,va='bottom')
    
    lf=0.05;rf=0.95
    bf=0.04;tf=0.94
    gapx=0.05; npnx=1
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy=0.01; npny=20
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    ix=lf; iy=tf

    cmnew= plt.get_cmap('CMRmap_r').resampled(100)(np.arange(100))
    cmnew = np.concatenate((np.array([1,1,1,1]).reshape([1,-1]),cmnew[4:96,:])) 
    newcm = cls.LinearSegmentedColormap.from_list("newCMR",cmnew)
    newcm.set_under("white")
    props = dict(cmap=newcm,alpha=0.9,vmin=0.0,vmax=100) 

    for i in range(km2):
        ax1= fig.add_axes([ix,iy-ly,lx,ly])
        pic1= ax1.imshow(conf_mtx[i,:].reshape([1,km2]),**props)
        subtit= pdata['suptit2']
        if i==0:
            pic_common(ax1,i,tgt_crs,subtit=subtit)
        elif i==km2-1:
            pic_common(ax1,i,tgt_crs,xtlab=True)
        else:
            pic_common(ax1,i,tgt_crs)

        write_val(ax1,conf_mtx[i,:],i,np.arange(km2),repeat(0),crt0=9.5,crt1=89.5,ft=11)    
        iy=iy-ly-gapy

    iy=iy-ly*1.2
    tt=np.arange(0,101,20)
    loc1= [ix,iy-ly/3,lx,ly/3]
    cb1= draw_colorbar(fig,pic1,loc1,tt,ft=10,extend='neither')

    ###-----------------------------------
    ### Show or Save
    #plt.show()
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150)
    print(pdata['outfn'])

    return

def pic_common(ax1,ind,tgt_cpr,xtlab=False,subtit=[]):
    if len(subtit)>0:
        ax1.set_title(subtit,fontsize=14,va='bottom',stretch='semi-condensed') #,ha='left',x=0.0)
    km= len(tgt_cpr)
    ax1.axis([-0.5,km-0.5,-0.5,+0.5])
    ax1.set_xticks(np.arange(km))
    if xtlab==True:
        xticklabs=["CPR{}".format(cpr) if cpr<30 else "Others" for cpr in tgt_cpr ]
        ax1.set_xlabel("Assigned by Precip-only",fontsize=13,labelpad=4)
        ax1.set_ylabel("Original",fontsize=13,y=3.5,ha='left',labelpad=0)
    else:
        xticklabs=[]
    ax1.set_xticklabels(xticklabs)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.set_yticks([0,])
    ax1.set_yticklabels(["CPR{}".format(cpr) if cpr<30 else "Others" for cpr in tgt_cpr[ind:ind+1] ])
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(which='minor',axis='both')
    ax1.set_aspect('auto')
    ax1.tick_params(labelsize=10)
    
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

def write_val(ax,values,ind,xloc,yloc,crt0=0,crt1=50,ft=10,ha='center',va='center'):
    """
    Show values on designated location if val>crt.
    Input values, xloc, and yloc should be of same dimension
    """
    ### Show data values
    for i,(val,xl,yl) in enumerate(zip(values,xloc,yloc)):
        pctxt='{:.0f}%'.format(val)
        if i==ind:
            props=dict(stretch='semi-condensed',fontsize=ft,fontweight='bold')
        else:
            props=dict(stretch='semi-condensed',fontsize=ft)
        if val>crt1:
            ax.text(xl,yl,pctxt,ha=ha,va=va,color='0.8',**props)
        elif val>crt0: # Write only for large numbers
            ax.text(xl,yl,pctxt,ha=ha,va=va,**props)    

    return

if __name__=="__main__":    
    cr_params= dict(
        rg= 15,      # domain max latitude, 15 or 50
        nelemp= 6,   # dimension of pr_hist, fixed to 6
        prwt= 7,     # weight of pr_hist. Prediction is only available with 7
    )
    sat_nm= 'TAmean' # 'terra', 'aqua', or 'TAmean'
    main(cr_params,sat_nm)

