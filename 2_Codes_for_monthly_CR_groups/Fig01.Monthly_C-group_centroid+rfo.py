"""
Read CR-group NC file and draw a figure showing centroids (left) and RFO maps (right)

Clear-sky regime here is defined with relaxed criterion of 5%

Apply geodetic weight + month_day weight for mean RFO
month_day weight for centroid

By Daeho Jin
2025.11.13

Caution: NetCDF file should be downloaded from https://zenodo.org/records/17831744
"""

import numpy as np
import sys
import os.path
from datetime import timedelta, date
from netCDF4 import Dataset, num2date
import common_functions as cf

def main():
    ###-- Parameters and defalut values
    ###-------------------------------------
    sat_nm= 'TA_mean'
    rg,nelemp,prwt,km= 50,0,0,15 
    rg_set= dict(rg=rg,nelemp=nelemp,prwt=prwt,km=km)
    tgt_cr,subk= km,3
    nelemc,nelem= 42,42+nelemp
    p_letter= 'P' if prwt>0 else ''
    prset_nm = f'Cld{nelemc}+Pr{nelemp}x{prwt}' if prwt>0 else f'Cld{nelemc}'
    rg_nm= f'{rg}S-{rg}N'
    mdnm = 'MODIS_t+a_C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)

    cr_names= ['H1_tk','H2_tk','H_tn','Mid','L1_tk','L2_tk','L_tn','S-Clr','Clr (CF<5%)']
    ncr= len(cr_names)


    ### Open netCDF file
    indir= './Data/'
    infn= indir+'Monthly_Composite_Histogram+RFO_map.by{}CRgroups.nc'.format(ncr)
    fid= Dataset(infn,'r')

    ##-- Read dimension info
    times= fid.variables['time']
    time_units = times.units
    times= num2date(times[:], units = times.units, calendar=times.calendar,
                      only_use_cftime_datetimes=True,)
    date_range= [date(t1.year, t1.month, t1.day) for t1 in [times[0],times[-1]]]
    tgt_dates= date_range
    tgt_date_names= '-'.join([dd.strftime('%Y.%m') for dd in tgt_dates])
    nmon= cf.get_tot_months(*tgt_dates)
    nyr= nmon//12
    month_days= np.asarray(cf.get_month_days(tgt_dates))
    
    lats= fid.variables['lat'][:]
    lons= fid.variables['lon'][:]
    resol=np.round(lons[1]-lons[0],0).astype(int)  ## Now in 4-deg resolution

    ##-- Geodetic weight is available at 1-deg resolution
    nlat,nlon= lats.shape[0], lons.shape[0]
    lats_1deg= np.arange(nlat*resol)+lats[0]-resol/2+0.5
    nlat1,nlon1= nlat*resol, nlon*resol
    lat_weight= cf.apply_lat_weight(np.ones([nlat1,nlon1,]),nlat1,nlon1,lats_1deg,geodetic=True)
    ltw= lat_weight.reshape([nlat,resol,nlon,resol]).sum(axis=1)[:,:,0]
    print(ltw.shape,ltw[0,0],ltw[nlat//2,0])
    xy= np.meshgrid(lons,lats)

    ##-- Read data and average them
    mrfo= fid.variables['mRFO'][:]
    jh= fid.variables['Monthly_CF_Joint_Hist']
    ncr,nmon,nlat,nlon,nctp,ntau= jh.shape
    
    rfo_maps= np.average(mrfo,weights=month_days,axis=1)*100.  ## Now in %
    cent_all, grfo_all=[],[]
    for icr in range(ncr):
        wt1= (mrfo[icr]*month_days[:,None,None]).reshape(-1)  
        cent1= jh[icr,:].reshape([nmon*nlat*nlon,nctp*ntau])
        cent1= np.ma.average(cent1,weights=wt1,axis=0)*100.  ## Now in %
        cent_all.append(cent1)
        grfo_all.append(np.ma.average(rfo_maps[icr],weights=ltw))
    cent_all= np.asarray(cent_all).reshape([ncr,nctp,ntau])
    print('Total CF:',cent_all.sum(axis=(1,2))) 
    grfo_all= np.asarray(grfo_all)
    print('Domain Mean RFO:',grfo_all)

    
    ### Read LO Mask (if needed)
    indir= './Data/'
    infn= indir+'PctWater.dat'
    lat_idx= [90+np.round(lats_1deg[0]-0.5,0).astype(int),90+np.round(lats_1deg[-1]+0.5,0).astype(int)]
    lomask0= cf.bin_file_read2mtx(infn).reshape([180,360])[lat_idx[0]:lat_idx[1],:]
    lomask0= lomask0 >= 90  ## Ocean only
    

    ###---- Plot
    suptit= f"MODIS_C6.1 {sat_nm} Cloud Group: Mean Histogram (left) and RFO Map (right)"
    outdir = './Pics/'
    outfn = outdir+ f"Fig01.Monthly_C-group8_ctd+rfo.{sat_nm}.png"
    pic_data= dict(
        cent_all= cent_all, rfo_maps= rfo_maps, grfo_all= grfo_all,
        cr_names= cr_names, xy= xy, lat_max= lats[-1]+resol/2,
        suptit=suptit, outfn=outfn,
    )
    plot_main(pic_data)
    return
        
###-------------------------------------
#import matplotlib as mpl
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator, FuncFormatter
import cartopy.crs as ccrs

def plot_main(pdata):
    ## Parameters and variables
    rfo_maps= pdata['rfo_maps']
    grfo_all= pdata['grfo_all']
    cr_names= pdata['cr_names']
    cent_all= pdata['cent_all']
    ctd_cf= cent_all.sum(axis=(1,2))
    xy= pdata['xy']
    lat_max= pdata['lat_max']
    ncr= len(cr_names)
    
    ###-- Plotting basics
    fig= plt.figure()
    fig.set_size_inches(14.5,16)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=20,y=0.98,va='bottom')
    
    lf=0.05;rf=0.95
    bf=0.05;tf=0.95
    gapx0,lx0=0.04, 0.13 #;
    npnx=2

    gapy0=0.043; npny=6.5
    ly0=(tf-bf-gapy0*(npny-1))/float(npny)

    gapx1=0.018
    lx1=(rf-lf-npnx*(gapx0+lx0)-gapx1*(npnx-1))/float(npnx)

    ix0,ix1, iy= lf,lf+2*(lx0+gapx0),tf

    ## Draw centroids first
    ix=ix0
    for l,(crnm,vv) in enumerate(zip(cr_names,cent_all)):
        ax1= fig.add_axes([ix,iy-ly0,lx0,ly0])
        vv1= vv[::-1,:]
        pic1= cent_show(ax1,vv1)
        cent_show_common(ax1,crnm,vv1.sum())
    
        if ix==ix0:
            ax1.set_ylabel('Pressure (hPa)',fontsize=13,labelpad=0)
    
        ix+=(lx0+gapx0)
        
        if ix>ix0+lx0*2: # or l==2:
            ix=ix0
            iy-=ly0+gapy0

        if l>=len(cent_all)-npnx:
            ax1.set_xlabel('Optical Thickness',fontsize=13)

    ## Colorbar for centroids
    if True:
        tt=[0.1,0.3,1,3,10,30]
        tt2=[str(x)+'%' for x in tt]
        hh= ly0/10
        #loc1= [ix,iy-ly0*0.25,lx0*2+gapx1,hh]
        loc1= [ix,iy-ly0,hh,ly0]
        cb1=cf.draw_colorbar(fig,pic1,loc1,tt,tt2,ft=12)
        cb1.ax.set_ylabel('Cloud Fraction',fontsize=14) #,rotation=-90,va='bottom') #,labelpad=0)
        cb1.ax.minorticks_off()
        

    ## Draw RFO maps second
    cm = plt.get_cmap('magma_r',80)       
    cmnew = cm(np.arange(80)) 
    cmnew = np.concatenate((np.array([1,1,1,1]).reshape([1,-1]),cmnew[:-1,:])) 
    newcm = cls.LinearSegmentedColormap.from_list("newCMR",cmnew)
    newcm.set_under("white")

    lon_ext= [20,360+20]
    cm= (lon_ext[0]+lon_ext[1])/2   
    data_crs= ccrs.PlateCarree()
    props_pc = dict(cmap=newcm,alpha=0.9,transform=data_crs,vmin=0.,vmax=60,shading='nearest')
    lat_max1= lat_max+1
    ix,iy= ix1,tf
    for l,(crnm,amap,grfo) in enumerate(zip(cr_names,rfo_maps,grfo_all)):
        ax=fig.add_axes([ix,iy-ly0,lx1,ly0],projection=ccrs.PlateCarree(central_longitude=cm))
        ax.set_extent(lon_ext+[-lat_max1,lat_max1],data_crs)
        rfo_text= 'RFO= {:.1f}%'.format(grfo) #if rfo>9.995 else 'RFO={:.2f}%'.format(grfo)
        subtit='{} [{}]'.format(crnm,rfo_text)
        print(subtit)
        ax.set_title(subtit,fontsize=14,stretch='condensed') #x=0.0,ha='left',
        cs=ax.pcolormesh(*xy,amap,**props_pc)

        if ix==ix1:
            ll,lr= True,False
        else:
            ll,lr= False,True
        ax.set_yticks([-60,-30,0,30,60])
        ax.tick_params(axis='y',labelright=lr,labelleft=ll,labelsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(cf.lat_formatter))

        ax.tick_params(direction='in',left=True,right=True,top=True,bottom=True,)
    
        label_idx=[False,False,False,True]
        map_common(ax,label_idx)

        ix+=(lx1+gapx1)
        if l==ncr-1:
            loc1= [ix,iy-hh/2,lx1,hh] if ix==ix1 else [ix,iy-ly0*0.25,lx1,hh]
            tt=range(0,61,10)
            tt2=[str(x)+'%' for x in tt]
            cb1=cf.draw_colorbar(fig,cs,loc1,tt,tt2,ft=12,extend='max')
            cb1.ax.set_xlabel('RFO',fontsize=14) #,rotation=-90,va='bottom')
        
        if ix>ix1+lx1*2: 
            ix=ix1
            iy-=ly0+gapy0
        


    ###--- Save
    fnout = pdata['outfn']
    ### Show or Save
    plt.savefig(fnout,bbox_inches='tight',dpi=150)
    #plt.show()

    print(fnout)
    return

def cent_show(ax1,ctd):     

    nx = 6 #TAU (Optical Thickness)
    ny = 7 #CTP
    if len(ctd.reshape(-1)) != nx*ny:
        print("Error: centroid data size is bad:",ctd.shape)
        sys.exit()

    cm = plt.get_cmap('jet',512)       
    cmnew = cm(np.arange(512))
    cmnew = cmnew[72:,:]
    newcm = cls.LinearSegmentedColormap.from_list("newJET",cmnew)
    newcm.set_under('white')

    props = dict(norm=cls.LogNorm(vmin=0.1,vmax=30),cmap=newcm,alpha=0.9)

    pic1=ax1.imshow(ctd,interpolation='nearest',aspect=0.84,**props)

    ### Axis Control
    xlabs=['0','1.3','3.6','9.4','23','60','150']
    ylabs=[1100,800,680,560,440,310,180,0]

    ax1.set_xlim(-0.5,5.5)
    ax1.set_ylim(-0.5,6.5)
    ax1.set_xticks(np.arange(nx+1)-0.5)
    ax1.set_xticklabels(xlabs)

    ax1.set_yticks(np.arange(ny+1)-0.5)
    ax1.set_yticklabels(ylabs)        
        
    for j in range(7):
        for i in range(6):
            if abs(ctd[j,i])>4.5:
                ax1.annotate("%.0f" %(ctd[j,i]),xy=(i,j),ha='center',va='center',stretch='semi-condensed',fontsize=11)
    return pic1

def cent_show_common(ax1,crnm,cf,km=0):

    ### add a title.
    subtit= "{} [CF={:.1f}%]".format(crnm,cf); print(subtit)
    ax1.set_title(subtit,fontsize=14,stretch='condensed') #x=0.0,ha='left',

    ### Draw Guide Line
    ax1.axvline(x=1.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axvline(x=3.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axhline(y=1.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axhline(y=3.5,linewidth=0.7,color='k',linestyle=':')

    ### Ticks
    ax1.tick_params(axis='both',which='major',labelsize=11)
    ax1.tick_params(left=True,right=True)
    return

def map_common(ax,label_idx=[True,True,False,True]):
    ax.set_extent([0.,360,-61,61],ccrs.PlateCarree())

    ax.coastlines(color='silver',linewidth=1.)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = label_idx[2]
    gl.left_labels = label_idx[0]
    gl.right_labels = label_idx[1]
    gl.bottom_labels = label_idx[3]

    gl.xlocator = MultipleLocator(60) 
    gl.ylocator = MultipleLocator(30)
    gl.xlabel_style = {'size': 11, 'color': 'k'}
    gl.ylabel_style = {'size': 11, 'color': 'k'}
    ax.set_aspect('auto')
    return

if __name__=="__main__":
    main()
