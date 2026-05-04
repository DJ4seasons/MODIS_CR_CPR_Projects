"""
Compare albedo distribution between various methods

Geodetic weight + mon_days weight

By Daeho Jin
2025.11.18
"""

import numpy as np
import sys
import os.path
from datetime import timedelta, date
from netCDF4 import Dataset, num2date
import common_functions as cf

def main(rad_idx):
    ###--- Parameters
    rad_names= ['SW','LW']
    rad_name= rad_names[rad_idx]+'_outgoing'
    rad_name_tit= 'O{}R'.format(rad_names[rad_idx][0])

    rfo_crt=5.
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
    indir= './Data/'
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
    max_lat= (lats[-1]+resol/2).astype(int)
    tgt_latlon1, tgt_rg_name1= [-max_lat,max_lat,-180,180], '{a}S-{a}N'.format(a=max_lat)
    
    ##-- Geodetic weight is available at 1-deg resolution
    nlat,nlon= lats.shape[0], lons.shape[0]
    lats_1deg= np.arange(nlat*resol)+lats[0]-resol/2+0.5
    nlat1,nlon1= nlat*resol, nlon*resol
    lat_weight= cf.apply_lat_weight(np.ones([nlat1,nlon1,]),nlat1,nlon1,lats_1deg,geodetic=True)
    ltw= lat_weight.reshape([nlat,resol,nlon,resol]).sum(axis=1)[:,:,0]
    print(ltw.shape,ltw[0,0],ltw[nlat//2,0])
    xy= np.meshgrid(lons,lats)
    
    ##-- Two periods    
    dr0,dr1= (date(2002,9,1),date(2012,8,31)),(date(2014,9,1),date(2024,8,31))
    imon1,nmon1= cf.get_tot_months(tgt_dates[0],dr0[0])-1,cf.get_tot_months(*dr0)
    iyr1,nyr1= imon1//nmon_yr, nmon1//nmon_yr
    dr0_name= '-'.join([d.strftime('%Y.%m') for d in dr0])
    imon2,nmon2= cf.get_tot_months(tgt_dates[0],dr1[0])-1,cf.get_tot_months(*dr1)
    iyr2,nyr2= imon2//nmon_yr, nmon2//nmon_yr
    dr1_name= '-'.join([d.strftime('%Y.%m') for d in dr1])
    print(imon1,iyr1,nmon1,nyr1)
    print(imon2,iyr2,nmon2,nyr2) #; sys.exit()

    ##-- Read rfo data
    rfos= fid.variables['mRFO'][:,imon:imon+nmon,:].reshape([ncr,nyr,nmon_yr,nlat,nlon])
            
    ### Screend out by clim_RFO
    cr_idx=[]
    for icr in range(ncr):
        clim_rfo= rfos[icr,:].mean(axis=0)
        cr_idx.append(clim_rfo>rfo_crt/100)
        print(icr,cr_idx[-1].mean())


    ### Read RFO kernel
    indir= './Data/'
    md_vers= ['vEBAF_regression','vFBCT_composite','vFBCT_regression']    
    kc_all=[]
    for ridx,md_ver in enumerate(md_vers):
        LM_fn= indir+f'Linear_Model_Coef.4Outgoing_Radiation_w{ncr}CRgroups.{md_ver}.nc'
        fid= Dataset(LM_fn,'r')
    
        half_kc=[]
        for half_idx in range(2):
            vn= f'O{rad_name[0].upper()}R_PD{half_idx}'
            kc= fid.variables[vn][:]
            print(kc.shape, kc.min(), kc.max(),kc.mean(axis=(0,1,2)))            
            half_kc.append(kc)
        kc_all.append(half_kc)
    
            
    ### Collecting slope info
    ### kc: [nmon_yr,nlat,nlon,ncr+1]
    slope_all=[]
    for ik in range(2):
        slope_half=[]
        for ridx,half_kc in enumerate(kc_all):
            kc1= half_kc[ik]
            slope_byCR=[]
            for icr in range(ncr):
                slope_byCR.append(kc1[:,:,:,icr][cr_idx[icr]])
            slope_half.append(slope_byCR)
        slope_all.append(slope_half)

    ## Print stat of slope info
    for r,md_ver in enumerate(md_vers):
        print(r+1,md_ver)
        for icr in range(ncr):
            d0= slope_all[0][r][icr]
            d1= slope_all[1][r][icr]
            print(icr,np.median(d1)-np.median(d0), d1.mean()-d0.mean())
            

    ### For Figure
    rg_name_tit= '{a}\u00B0S\u2013{a}\u00B0N'.format(a=max_lat)
    suptit= 'Slope Comparison by CR groups [{} in {}, {}>{:.0f}%]'.format(rad_name_tit,rg_name_tit,r'$RFO_{clim}$',rfo_crt)
    outdir= './Pics/'       
    outfn= outdir+'Fig09.regr_coeff_comparison_byCRgroup.{}_comp_vs_regr.{}.png'.format(tgt_date_names,rad_name_tit)

    mv_names= [mv[:ind] for mv,ind in zip(md_vers,[-6,-5,-6])]
    pic_data= dict(albedo_all= slope_all, mv_names= mv_names,
                   pn_tit= ['Period1 ({})'.format(dr0_name),'Period2 ({})'.format(dr1_name)],
                   cr_group_names=cr_names,
                   suptit=suptit, outfn=outfn,)
    plot_main(pic_data)
                                                         
    return

import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
from matplotlib.dates import DateFormatter, YearLocator

def plot_main(pdata):
    albedo_all= pdata['albedo_all']
    pn_tit= pdata['pn_tit']
    mv_names= pdata['mv_names']
    cr_group_names= pdata['cr_group_names']
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'

    ###---
    fig=plt.figure()
    fig.set_size_inches(7.6,9.)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=16,y=0.975,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    ncol,nrow=1,3.5
    lf,rf,bf,tf=0.05,0.95,0.1,0.925
    gapx, npnx=0.06,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.08,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    
    ix=lf; iy=tf    

    ai=0
    wd= 0.26
    nv,nc= len(albedo_all[0]), len(albedo_all[0][0])
    xlocs= bar_x_locator(wd,data_dim= [nv,nc])
    props= dict(showfliers=False,widths=wd,showmeans=False,medianprops=dict(color='k')) #,meanprops=meanprops
    cc= ['C0','C1','C2']
    xtlabs= cr_group_names
    ###--- panel1
    for i,tit1 in enumerate(pn_tit):
        data1= albedo_all[i]
        ax1= fig.add_axes([ix,iy-ly,lx,ly])
        
        for k,lab in enumerate(mv_names):
            box1= ax1.boxplot(data1[k],whis=[5,95],positions=xlocs[k],patch_artist=True,**props)
            for bp in box1['boxes']:
                bp.set_facecolor(cc[k])
                bp.set_alpha(0.85)
            ax1.plot([],[],color=cc[k],marker='s',linestyle='None',markersize=10,label=lab)
            
        subtit= '({}) {} '.format(abc[ai],tit1); ai+=1
        ax1.set_title(subtit,fontsize=12,x=0,ha='left')

        ax1.set_xticks(range(nc))
        ax1.set_xticklabels(xtlabs)
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))    
        #ax1.xaxis.set_major_formatter(DateFormatter("%Y\n%b"))
        ax1.yaxis.set_major_locator(MultipleLocator(0.2))    
        ax1.set_ylabel('Slope Coef.',fontsize=10)
        ax1.grid(axis='y',ls=':',c='0.7',lw=1)
        ax1.tick_params(labelsize=9)
        ax1.yaxis.set_ticks_position('both')

        ax1.legend(loc='upper left',fontsize=10,framealpha=0.9, borderaxespad=0.,bbox_to_anchor=(1.02,1.))

        for k in range(nc-1):
            ax1.axvline(x=k+0.5,ls='--',lw=1.2,c='0.7',alpha=0.9)
        ax1.set_xlim([-0.6,nc-0.4])
        
        ix+= lx+gapx
        if ix+lx>rf:
            ix=lf
            iy-= ly+gapy
    
        
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
    rad_idx= 0
    main(rad_idx)

