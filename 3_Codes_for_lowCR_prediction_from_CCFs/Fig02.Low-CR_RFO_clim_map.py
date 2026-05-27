'''
Seasonal climatology of Low-CR group RFOs and a map of all RFOs > rfo_crt
2002.09-2024.08 (22 years)

By Daeho Jin
2026.03.24
---

It requires "MODIS_t+a_CR_set.50S-50N_Cld42.nc", which can be downloaded from 
https://zenodo.org/records/18356023

'''

import numpy as np
import sys
import os.path
from datetime import timedelta, date
from netCDF4 import Dataset, num2date
import common_functions as cf

def main():
    ###--- Parameters
    ###-------------------------------------
    rg, nelemp, prwt, km= 50, 0, 0, 15
    nelemc= 42
    nelem=nelemc+nelemp

    p_letter= 'P' if prwt>0 else ''
    prset_nm = f'Cld{nelemc}+Pr{nelemp}x{prwt}' if prwt>0 else f'Cld{nelemc}'
    rg_nm= f'{rg}S-{rg}N'

    indir= './Your_directory/'
    mdnm= 'MODIS_t+a_C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)
    infn= indir+f'{mdnm}.nc'
    mdnm2= 'C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)

    ## Parameters for CR_nums
    sat_nm= 'TAmean'
    max_lat= 64
    tgt_lats= [-max_lat,max_lat]  ## Data is available from 65S to 65N
    tgt_dates= [date(2002,9,1),date(2024,8,31)]
    tgt_date_names= '-'.join([dd.strftime('%Y.%m') for dd in tgt_dates])
    ndy= (tgt_dates[1]-tgt_dates[0]).days+1
    nmon= cf.get_tot_months(*tgt_dates)
    nmon_yr= 12
    nyr= nmon//12
    sn_names= ['All','SON','DJF','MAM','JJA']
    sn_mon_idx=[]
    sn_nms=[]
    all_months= np.asarray([dd.month for dd in cf.yield_date_range(*tgt_dates)])
    for sn_idx in [2,4]: # DJF and JJA
        sn_mons= np.arange(3,dtype=int)+sn_idx*3+6 if sn_idx>0 else np.arange(12,dtype=int)+9
        sn_mons[sn_mons>12]-=12
        sn_idx_bool=np.isin(all_months,sn_mons)
        sn_mon_idx.append(sn_idx_bool)
        print(sn_names[sn_idx],sn_mons,sn_idx_bool.sum())
        sn_nms.append(sn_names[sn_idx])

        
    tgt_cr_groups= [
        ('L1_tk',(11,13)), ('L2_tk',(10,12)),                    
        ('L_tn',(14,)),    ('S-Clr',(153,0)),
    ]
    tgt_crs= [item[1] for item in tgt_cr_groups]
    cr_name= [item[0] for item in tgt_cr_groups]
    ncr= len(tgt_cr_groups)
    
    ## Read LO Mask
    wpct= cf.get_Water_Pct(tgt_lats+[-180,180])
    print(wpct.shape)
    print(wpct.min(), wpct.max())
    lomask0= wpct>= 90  ## Ocean only
    
    
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
    lat_idx= [cf.lat_deg2y(lt,lat0,dlat) for lt in tgt_lats]
    print(lat0,dlat,lat_idx) #; sys.exit()
    lats= lats[lat_idx[0]:lat_idx[1]]
    nlat,nlon= len(lats), len(lons)
    lat_weight= cf.apply_lat_weight(np.ones([nlat,nlon,]),nlat,nlon,lats,geodetic=True)
    xy= np.meshgrid(lons,lats)

    #-- Read CR-nums
    ## CR-num data
    crnums= fid.variables[f'CRnum_on_map_{sat_nm}']
    crnums= crnums[itidx:itidx+ndy,lat_idx[0]:lat_idx[1],:]
    print(crnums.shape)
    fid.close()    
    
    ## Seasonal filtering
    mrfo_all=[]
    for sn_idx in sn_mon_idx:        
        by_tcr=[]
        for tgt_cr in tgt_crs:
            idx_all=False
            for tcr in tgt_cr:
                idx= crnums[sn_idx,:]==tcr                    
                idx_all= np.logical_or(idx_all,idx)
            by_tcr.append(idx_all.mean(axis=0))

        tmp_rfos= np.asarray(by_tcr)*100 ## Now in %  #[ncr,nlat,nlon]
        mrfo_all.append(tmp_rfos)

    ## Check results
    for sn_nm,rfos in zip(sn_nms,mrfo_all):
        print(sn_nm,rfos.shape, rfos.min(), rfos.max(),rfos[:,4::12,:].mean(axis=(1,2))) #; sys.exit() #[ncr,nmon,nlat,nlon]
    
    ### RFO clim and intersection
    for i,rfos in enumerate(mrfo_all):
        rfos_clim= np.ma.masked_array(rfos,mask=np.tile(np.logical_not(lomask0),(ncr,1)))
        clim_crt= 10

        rfos_common= (rfos_clim>=clim_crt).sum(axis=0)==ncr
        #print(rfos_common.sum()) 
        mrfo_all[i]=[rfos_clim,rfos_common]

    ### For Figure    
    rg_nm= '{a}S-{a}N_Ocean'.format(a=max_lat)    
    suptit= 'Mean RFO of Low Cloud Groups [{}-{}]'.format(*tgt_date_names)
    outdir= './Pics/'       
    outfn= outdir+'Fig02.CR_RFO_clim_map.{}.{}_{}.png'.format(
         rg_nm,tgt_date_names,'+'.join(sn_nms))
    pic_data= dict(rfos_clim=[item[0] for item in mrfo_all],
                   rfos_common=[item[1] for item in mrfo_all],
                   cld_names= cr_name, sn_names=sn_nms,
                   xy=xy,lw=lat_weight,clim_crt=clim_crt,
                   suptit=suptit, outfn=outfn, )
    plot_main0(pic_data)

    return

import matplotlib as mpl
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
import cartopy.crs as ccrs
from cartopy.feature import LAND
def plot_main0(pdata):
    rfos_clim = pdata['rfos_clim']
    rfos_common= pdata['rfos_common']
    cld_names= pdata['cld_names']
    xy,lw= pdata['xy'], pdata['lw']
    clim_crt= pdata['clim_crt']
    sn_names= pdata['sn_names']
    
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'
    
    ###---
    fig=plt.figure()
    fig.set_size_inches(8.5,6)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=17,y=0.98,va='bottom',stretch='semi-condensed') 
    ncol,nrow=2,2
    lf,rf,bf,tf=0.02,0.98,0.3,0.92
    gapx, npnx=0.054,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.095,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    
    ix=lf; iy=tf

    ## Map setting and props
    lon_ext= [20,360+20]
    cm = (lon_ext[0]+lon_ext[1])/2 #180
    map_proj= ccrs.PlateCarree(central_longitude=cm)
    data_crs= ccrs.PlateCarree()

    vmin,vmax= clim_crt,90
    ccb= range(vmin,vmax+1,15)
    n_ccb= len(ccb)+1
    cm= plt.get_cmap('plasma_r').resampled(n_ccb)(np.arange(n_ccb))
    newcm= cls.ListedColormap(cm[1:-1,:]).with_extremes(over=cm[-1,:],under=[1.,1.,1.,1.]) 
    norm = cls.BoundaryNorm(ccb, newcm.N)

    props_mesh= dict(cmap=newcm,alpha=0.86,norm=norm,transform=data_crs)
    props_contour= dict(alpha=0.7,colors='0.1',linewidths=1,transform=data_crs)
    mpl.rcParams["hatch.color"]='0.4'
    
    ### Plot maps    
    ai=0    
    for ii,crnm in enumerate(cld_names):
        for jj,snm in enumerate(sn_names):
            tgt_boxes= get_tgt_boxes_12d(snm)  
            amap= rfos_clim[jj][ii,:]
            ax1=fig.add_axes([ix,iy-ly,lx,ly], projection=map_proj)
            ax1.set_extent(lon_ext+[-64,64],data_crs)
            
            pic1= ax1.pcolormesh(*xy,amap,shading='nearest',**props_mesh)
            pic2= ax1.contour(*xy,amap,ccb[0::2],**props_contour)
            ax1.clabel(pic2,ccb[0::2],inline=True,fontsize=8)
            
            subtit= "({}) {} in {}".format(abc[ai],crnm,snm); ai+=1
            ax1.set_title(subtit,fontsize=13,x=0,ha='left')
        
            ix+= lx+gapx
            if ix+gapx>rf:
                right_label=True
                ix=lf
                iy-= ly+gapy
            else:
                right_label=False
            map_common(ax1,data_crs,right_label=right_label,lon_ext=lon_ext)
            ax1.add_feature(LAND,facecolor='0.85') #'#bfbfbf')
            draw_box(ax1,tgt_boxes,data_crs,ls='-',c='c')

            mtxt= 'Mean= {:.1f}%'.format(np.average(amap.compressed(),weights=lw[~amap.mask]))

    
    if True:        
        hh=0.025 
        loc0= [0.2,iy+gapy*0.15,0.6,hh]
        tt= ccb 
        tt2= [f'{v}' for v in tt]
        cb0 =draw_colorbar(fig,pic1,loc0,ft=10,extend='both',tt=tt,tt2=tt2)
        cb0.ax.set_xlabel('RFO (%)',fontsize=11,labelpad=0) 

    iy-= gapy*1
    
    ## Intersection map
    if True:
        for jj,snm in enumerate(sn_names):
            tgt_boxes= get_tgt_boxes_12d(snm)  
            amap= rfos_common[jj]
                
            ax2=fig.add_axes([ix,iy-ly,lx,ly], projection=map_proj)    
            ax2.set_extent(lon_ext+[-64,64],data_crs)
            pmap= np.ma.masked_less(amap,0.5)
            pic2= ax2.pcolor(*xy,pmap,hatch='xxxxx',alpha=0.,transform=data_crs)
            map_common(ax2,data_crs,right_label=False,lon_ext=lon_ext)
            ax2.add_feature(LAND,facecolor='0.85') #'#bfbfbf')
            draw_box(ax2,tgt_boxes,data_crs,ls='-',c='c')
            subtit= "({}) All RFOs \u2265 {}% in {}".format(abc[ai],clim_crt,snm); ai+=1
            ax2.set_title(subtit,fontsize=13,x=0,ha='left')
        
            ix+= lx+gapx    

    ###---
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150) 
    #plt.show()
    print(pdata['outfn'])
    return


def map_common(ax,data_crs,right_label=False,lon_ext=[0,360]):

    ax.coastlines(color='silver',linewidth=1.)
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
    label_idx=[False,False,False,True] #[Left,Right,Top,Bottom]
    gl.top_labels = label_idx[2]
    gl.left_labels = label_idx[0]
    gl.right_labels = label_idx[1]
    gl.bottom_labels = label_idx[3]
    gl.ylocator = MultipleLocator(30)
    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}

    ax.set_aspect('auto')

    for lt in range(-60,61,30):
        ax.text(lon_ext[0]+0.01,lt,cf.lat_formatter(lt,0)+' ',ha='right',va='center',fontsize=10,c='k',transform=data_crs)
        if right_label:
            ax.text(lon_ext[1]-0.01,lt,' '+cf.lat_formatter(lt,0),ha='left',va='center',fontsize=10,c='k',transform=data_crs)
    
    return

def get_tgt_boxes_12d(sn_idx):
    tgt_boxes_JJA= [
        ('Peruvian',(-22,-10,-100,-88)), 
        ('Namibian',(-22,-10,-10,2)), 
        ('Californian',(18,30,-146,-134)), 
    ]

    tgt_boxes_DJF= [
        ('Peruvian',(-30,-18,-90,-78)), 
        ('Namibian',(-26,-14,-6,6)), 
        ('Australian',(-36,-24,94,106)), 
    ]
    if isinstance(sn_idx,int):
        if sn_idx==2:
            return tgt_boxes_DJF
        elif sn_idx==4:
            return tgt_boxes_JJA
        else:
            return tgt_boxes_tk
    elif isinstance(sn_idx,str):
        if sn_idx.lower()=='djf':
            return tgt_boxes_DJF
        elif sn_idx.lower()=='jja':
            return tgt_boxes_JJA
        else:
            return tgt_boxes_tk
    else:
        print('Check argument',sn_idx)
        sys.exit()
    

def draw_box(ax1,tgt_boxes,data_crs,ls='-',c='0.5'):
    alp=1
    for abox in tgt_boxes:
        y1,y2,x1,x2= abox[1]

        ax1.plot([x1,x1],[y1,y2],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
        ax1.plot([x2,x2],[y1,y2],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
        ax1.plot([x1,x2],[y1,y1],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
        ax1.plot([x1,x2],[y2,y2],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
    return

def draw_colorbar(fig,pic1,loc,ft=10,extend='both',tt=[],tt2=[]): 

    cb_ax = fig.add_axes(loc)  ##<= (left,bottom,width,height)
    if loc[2]<loc[3]:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='vertical',ticks=tt,extend=extend)
        cb.ax.set_yticklabels(tt2,size=ft,stretch='condensed')
    else:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='horizontal',ticks=tt,extend=extend)
        cb.ax.set_xticklabels(tt2,size=ft,stretch='condensed')
    cb_ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    return cb


if __name__=="__main__":
    main()

