"""
OSR contribution by mean RFO changes in the form of seasonal lat-zone

Geodetic weight + mon_days weight

By Daeho Jin
2025.11.14
"""

import numpy as np
import sys
import os.path
from datetime import timedelta, date
from netCDF4 import Dataset, num2date
import common_functions as cf

def main(rad_idx):
    ###--- Parameters
    mdnm0= 'EBAF_4.2.1' 
    rad_names= ['SW','LW']
    rad_name= rad_names[rad_idx]+'_outgoing'
    rad_name_tit= 'O{}R'.format(rad_names[rad_idx][0])

    sat_nm= 'TAmean'
    rg,nelemp,prwt,km= 50,0,0,15 
    rg_set= dict(rg=rg,nelemp=nelemp,prwt=prwt,km=km)
    tgt_cr,subk= km,3
    nelemc,nelem= 42,42+nelemp
    p_letter= 'P' if prwt>0 else ''
    prset_nm = f'Cld{nelemc}+Pr{nelemp}x{prwt}' if prwt>0 else f'Cld{nelemc}'
    rg_nm= f'{rg}S-{rg}N'
    mdnm = 'MODIS_t+a_C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)

    cr_names= ['H1_tk','H2_tk','H_tn','Mid','L1_tk','L2_tk','L_tn','S-Clr','Clear']
    ncr= len(cr_names)

    tgt_dates= (date(2002,9,1),date(2024,8,31)) 
    tgt_date_names= '-'.join([dd.strftime('%Y.%m') for dd in tgt_dates])
    
    ### Open netCDF file for RFO data
    indir= '/Users/djin1/Documents/CLD_Work/Data_Obs/MODIS_c61/Composite/' #'./Data/'
    infn= indir+'Monthly_Composite_Histogram+RFO_map.by{}CRgroups.nc'.format(ncr)
    fid= Dataset(infn,'r')

    ##-- Read dimension info
    times= fid.variables['time']
    time_units = times.units
    times= num2date(times[:], units = times.units, calendar=times.calendar,
                      only_use_cftime_datetimes=True,)
    date_range= [date(t1.year, t1.month, t1.day) for t1 in [times[0],times[-1]]]
    
    imon,nmon= cf.get_tot_months(date_range[0],tgt_dates[0])-1, cf.get_tot_months(*tgt_dates)
    nmon_yr,nyr= 12, nmon//12
    mon_days= np.asarray(cf.get_month_days(tgt_dates))
    mon_days_clim_norm= np.asarray(mon_days).reshape(-1,12)[:4,:].mean(axis=0)
    mon_days_clim_norm= mon_days_clim_norm/mon_days_clim_norm.sum()
    #xt= cf.yield_monthly_date_range(*tgt_dates,mdelta=1)
    
    lats= fid.variables['lat'][:]
    lons= fid.variables['lon'][:]
    resol=np.rint(lons[1]-lons[0]).astype(int)  ## Now in 4-deg resolution
    max_lat= lats[-1]+resol/2
    tgt_latlon1, tgt_rg_name1= [-max_lat,max_lat,-180,180], '{a}S-{a}N'.format(a=max_lat)
    
    ##-- Geodetic weight is available at 1-deg resolution
    nlat,nlon= lats.shape[0], lons.shape[0]
    lats_1deg= np.arange(nlat*resol)+lats[0]-resol/2+0.5
    nlat1,nlon1= nlat*resol, nlon*resol
    lat_weight= cf.apply_lat_weight(np.ones([nlat1,nlon1,]),nlat1,nlon1,lats_1deg,geodetic=True)
    ltw= lat_weight.reshape([nlat,resol,nlon,resol]).sum(axis=1)[:,:,0]
    print(ltw.shape,ltw[0,0],ltw[nlat//2,0])
    xy= np.meshgrid(lons,lats)

    ##-- Set Lat-Zone
    latzone_loc= [-60,-28,0,28,60]
    lz_iloc= (np.asarray(latzone_loc)+max_lat).astype(int)//resol
    print(lz_iloc)
    nzone= len(lz_iloc)-1
    
    ##-- Two periods    
    dr0,dr1= (date(2002,9,1),date(2012,8,31)),(date(2014,9,1),date(2024,8,31))
    imon1,nmon1= cf.get_tot_months(tgt_dates[0],dr0[0])-1,cf.get_tot_months(*dr0)
    iyr1,nyr1= imon1//nmon_yr, nmon1//nmon_yr
    pd1_date_name= 'PD1: {}-{}'.format(*[dd.strftime('%Y.%m') for dd in dr0])
    imon2,nmon2= cf.get_tot_months(tgt_dates[0],dr1[0])-1,cf.get_tot_months(*dr1)
    iyr2,nyr2= imon2//nmon_yr, nmon2//nmon_yr
    pd2_date_name= 'PD2: {}-{}'.format(*[dd.strftime('%Y.%m') for dd in dr1])
    print(imon1,iyr1,nmon1,nyr1)
    print(imon2,iyr2,nmon2,nyr2) #; sys.exit()

    ##-- Read rfo data
    rfos= fid.variables['mRFO'][:,imon:imon+nmon,:].reshape([ncr,nyr,nmon_yr,nlat,nlon])
        
    ##-- RFO clim 
    #rfos_clim_all= rfos.mean(axis=1)
    rfos_clim_pd1= rfos[:,iyr1:iyr1+nyr1,:,:,:].mean(axis=1)
    rfos_clim_pd2= rfos[:,iyr2:iyr2+nyr2,:,:,:].mean(axis=1)
    rfos= None

    cdiff_lz= lat_zone_mean(rfos_clim_pd2-rfos_clim_pd1,lz_iloc,ltw)
    print(cdiff_lz.shape) # [nzone, ncr,nmon_yr]
    cdiff_lz= (cdiff_lz*mon_days_clim_norm[None,None,:]).reshape([nzone,ncr,4,3]).sum(axis=-1)
    cdiff_lz= cdiff_lz/mon_days_clim_norm.reshape([4,3]).sum(axis=-1)*100 ## Now in %

    
    ### Read RFO kernel
    indir= './Data/'
    LM_fn= indir+f'Linear_Model_Coef.4Outgoing_Radiation_w{ncr}CRgroups.vEBAF_regression.nc'
    fid= Dataset(LM_fn,'r')
    rad_clr_name= ['Insol','OLR_clr']

    #for rad_idx,rad_name in enumerate(rad_names):
    if True:
        half_kc,half_clr=[],[]
        for half_idx in range(2):
            vn= f'O{rad_name[0].upper()}R_PD{half_idx}'
            kc= fid.variables[vn][:]
            print(kc.shape, kc.min(), kc.max(),kc.mean(axis=(0,1,2)))
            rad_clr_vn= f'{rad_clr_name[rad_idx]}_clim_PD{half_idx}'
            rad_clr= fid.variables[rad_clr_vn][:]
            
            half_kc.append(kc)
            half_clr.append(rad_clr)
        
    ### Build R from RFO_clim
    ### kc: [nmon_yr,nlat,nlon,ncr]
    rad_built_set=[]    
    for kc1 in half_kc:
        by_clim=[]        
        for j,rc in enumerate([rfos_clim_pd1,rfos_clim_pd2]):
            rad_compo=[]
            for icr in range(ncr):
                rad_tmp2= kc1[:,:,:,icr]*rc[icr,:]

                ## De-normalize
                rad_tmp2*= half_clr[j]

                rad_tmp2= lat_zone_mean(rad_tmp2,lz_iloc,ltw)
                rad_tmp2_sn= (rad_tmp2*mon_days_clim_norm[None,:]).reshape([nzone,4,3]).sum(axis=-1)
                rad_tmp2_sn= rad_tmp2_sn/mon_days_clim_norm.reshape([4,3]).sum(axis=-1)
                rad_compo.append(rad_tmp2_sn)                
                
            by_clim.append(rad_compo)
        rad_built_set.append(by_clim)
        
    rad_built_set= np.asarray(rad_built_set)
    print(rad_built_set.shape) #[n_model, n_rfo, ncr+1,n_latzone,n_season]

    ### By RFO change:
    rad_built_set= rad_built_set.mean(axis=0)
    rad_built_diff= rad_built_set[1]-rad_built_set[0]  #[ncr+1,n_latzone,n_season]

    ### RFO diff:
    cdiff_lz= cdiff_lz.swapaxes(0,1)  #[ncr,n_latzone,n_season]
    
    ### For Figure
    sn_names= ['SON','DJF','MAM','JJA']
    for isn in [[0,2],[1,3]]:
        sn_name= '+'.join([sn_names[i] for i in isn])
        suptit= '{} Contribution by Mean RFO Diff. vs. RFO Diff.\n(Weighted by Area Fraction)'.format(rad_name_tit) #, tgt_rg_name1
        outdir= './Pics/'       
        outfn= outdir+'Fig05.{}_Contribution_byRFO_vs_RFOdiff.{}_vs_{}.LatZone.{}_{}.png'.format(rad_name_tit,*[val.split()[1] for val in [pd1_date_name,pd2_date_name]],mdnm0,sn_name)

        #pn_tit= ['By Model Diff.','By RFO Diff.',] #'Combined']
        pic_data= dict(rad_built_set1= [rad_built_diff[:,:,i] for i in isn],
                       rfo_set1= [cdiff_lz[:,:,i] for i in isn],
                       sn_names= [sn_names[i] for i in isn], latzone_loc=latzone_loc,
                       cr_group_names=cr_names,
                       suptit=suptit, outfn=outfn,)
            
        plot_main(pic_data)
                                                         
    return

def lat_zone_mean(arr1,lz_iloc,lw):
    by_lat=[]
    for lt0,lt1 in zip(lz_iloc[:-1],lz_iloc[1:]):
        if arr1.ndim==3:
            mtmp1= (arr1[:,lt0:lt1,:]*lw[None,lt0:lt1,:]).sum(axis=(1,2))
        elif arr1.ndim==4:
            mtmp1= (arr1[:,:,lt0:lt1,:]*lw[None,None,lt0:lt1,:]).sum(axis=(2,3))
        by_lat.append(mtmp1)
    by_lat= np.asarray(by_lat)/lw.sum()
    return by_lat 
 
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
from matplotlib.dates import DateFormatter, YearLocator
import matplotlib.patches as mpatches

def plot_main(pdata):
    #rad_ref= pdata['rad_ref']
    rad_built_set1= pdata['rad_built_set1']
    rfo_set1= pdata['rfo_set1']    
    sn_names= pdata['sn_names']
    lz_loc= pdata['latzone_loc']
    cr_names= pdata['cr_group_names']
    
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'
    rad_name= pdata['suptit'].split()[0]
    ###---
    fig=plt.figure()
    fig.set_size_inches(10,9.5)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=16,y=0.975,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    ncol,nrow=2,2
    lf,rf,bf,tf=0.06,0.92,0.34,0.94
    gapx, npnx=0.08,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.08,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    
    ix=lf; iy=tf    

    unit0= r'$Wm^{-2}$'
    #unit1= unit0+r'$yr^{-1}$'
    ai=ci=0
    #cc= ['C0','C1','C8','C6','C3','0.5']
    cc= ['C{}'.format(val) for val in [3,5,0,4,1,2,6,8,9]]
    
    #xtlabs= cr_group_names
    nv,n_lat= rad_built_set1[0].shape
    wd=0.76*28
    yloc0= np.array([(v0+v1)/2 for v0,v1 in zip(lz_loc[:-1],lz_loc[1:])]) #np.arange(-50,53,20)
    #yloc1, yloc2= yloc0+wd/2, yloc0-wd/2
    yloc1=yloc0
    props0= dict(height=wd,alpha=0.8)
    #props1= dict(height=wd,alpha=0.8)
    props_t= dict(fontsize=10,va='center',ha='center',stretch='condensed')
    
    ###--- panel1
    #axes=[]
    #xmin,xmax= 999,-999
    if rad_name=='OSR':
        xlim1= [-0.86,0.76]
        xlint1= 0.2
        vcrt= 0.145
    elif rad_name=='OLR':
        xlim1= [-2.1,2.1]
        xlint1= 1.
        vcrt= 0.495
    xlim2= [-0.76,0.76]
    patches=[]
    ai=0
    for ip,sn in enumerate(sn_names):
        ax2= fig.add_axes([ix,iy-ly,lx,ly])
        diff2= rfo_set1[ip]
        for jj in range(n_lat):
            bm1=bp1=0
            for ii in range(nv-1,-1,-1):
                if diff2[ii,jj]>=0:
                    pic1= ax2.barh([yloc1[jj],],diff2[ii,jj],left=bp1,color=cc[ii],**props0)
                    bp1+=diff2[ii,jj]
                else:
                    pic1= ax2.barh([yloc1[jj],],diff2[ii,jj],left=bm1,color=cc[ii],**props0)
                    bm1+=diff2[ii,jj]
                    
                val,yl= diff2[ii,jj],yloc1[jj]
                xl= bp1-val/2 if val>=0 else bm1-val/2
                if abs(val)>0.145:
                    ax2.text(xl,yl,'{:.2f}'.format(val),c='0.1',**props_t) #,stretch='condensed'
                elif abs(val)>0.095:                  
                    ax2.text(xl,yl,'{:.2f}'.format(val),c='0.1',**props_t)

                
        subtit= '({}) {}: RFO Difference'.format(abc[ai],sn); ai+=1
        print(subtit)
        ax2.set_title(subtit,fontsize=13,x=0.,ha='left') #,stretch='semi-condensed'

        ax2.set_xlim(xlim2)
        ax2.xaxis.set_major_locator(MultipleLocator(0.3))
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.set_ylim([-60.01,60.01])        
        ax2.yaxis.set_major_locator(FixedLocator(lz_loc))
        ax2.yaxis.set_major_formatter(FuncFormatter(cf.lat_formatter))
        ax2.set_xlabel('RFO Diff. (%)',fontsize=11)
        ax2.set_ylabel('Latitude',fontsize=11)
        ax2.tick_params(labelsize=10,direction='in')
        ax2.yaxis.set_ticks_position('both')
        ax2.axvline(x=0,c='0.15',ls='--',lw=1.2)
        if False: #ip==ncol-1:
            ax2.legend(patches[::-1],cr_names,
                       loc='upper left', bbox_to_anchor=[1.02,1.],
                       fontsize=11,framealpha=0.6,borderaxespad=0.)
        #axes.append(ax1)
        ix+= lx+gapx

        ax1= fig.add_axes([ix,iy-ly,lx,ly])
        diff1= rad_built_set1[ip]
        for jj in range(n_lat):
            bm1=bp1=0
            for ii in range(nv-1,-1,-1):
                if diff1[ii,jj]>=0:
                    pic1= ax1.barh([yloc1[jj],],diff1[ii,jj],left=bp1,color=cc[ii],**props0)
                    bp1+=diff1[ii,jj]
                else:
                    pic1= ax1.barh([yloc1[jj],],diff1[ii,jj],left=bm1,color=cc[ii],**props0)
                    bm1+=diff1[ii,jj]
                    
                val,yl= diff1[ii,jj],yloc1[jj]
                xl= bp1-val/2 if val>=0 else bm1-val/2
                if abs(val)>vcrt:
                    #ax1.text(xl,yl,'{:.1f}'.format(val),c='0.1',weight='bold',**props_t)
                    ax1.text(xl,yl,'{:.2f}'.format(val),c='0.1',**props_t) #

                if ip==0 and jj==0:
                    patch1= mpatches.Patch(color=cc[ii],alpha=0.8)
                    patches.append(patch1)
        subtit= '({}) {}: Contribution'.format(abc[ai],sn); ai+=1
        print(subtit)
        ax1.set_title(subtit,fontsize=13,x=0.,ha='left') #,stretch='semi-condensed'

        ax1.set_xlim(xlim1)
        ax1.xaxis.set_major_locator(MultipleLocator(xlint1))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax1.set_ylim([-60.01,60.01])        
        ax1.yaxis.set_major_locator(FixedLocator(lz_loc))
        ax1.yaxis.set_major_formatter(FuncFormatter(cf.lat_formatter))        
        ax1.set_xlabel('Rad. Diff. ({})'.format(unit0),fontsize=11)
        ax1.set_ylabel('Latitude',fontsize=11)
        ax1.tick_params(labelsize=10,direction='in')
        ax1.yaxis.set_ticks_position('both')
        ax1.axvline(x=0,c='0.15',ls='--',lw=1.2)
        ax1.legend(patches[::-1],cr_names,
                   loc='upper left', bbox_to_anchor=[1.02,1.],
                   fontsize=11,framealpha=0.6,borderaxespad=0.)
            
        if True: #ix+gapx>rf:
            iy -= ly+gapy
            ix=lf

    ###---
    print(pdata['outfn'])
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150) #
    #plt.show()
    
    return

def bar_x_locator(width,data_dim=[1,10]):
    """
    Depending on width and number of bars,
    return bar location on x axis
    Input width: (0,1) range
    Input data_dim: [# of vars, # of bins]
    Output locs: list of 1-D array(s)
    """
    xx=np.arange(data_dim[1])
    shifter= -width/2*(data_dim[0]-1)
    locs=[]
    for x1 in range(data_dim[0]):
        locs.append(xx+(shifter+width*x1))
    return locs


if __name__=="__main__":
    # rad_names= ['SW','LW']
    rad_idx= 1
    main(rad_idx)
