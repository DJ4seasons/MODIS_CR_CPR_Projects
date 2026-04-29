"""
EBAF v4.2.1 vs. Predicted TOA outgoing SW/LW

Geodetic weight + mon_days weight are applied

By Daeho Jin
2025.11.12
"""

import numpy as np
import sys
import os.path
from datetime import timedelta, date
from netCDF4 import Dataset, num2date
import common_functions as cf

def main():
    ###--- Parameters
    mdnm0= 'EBAF_4.2.1' 
    rad_names= ['SW','LW']
    rad_names= [val+'_outgoing' for val in rad_names]
    rad_names_tit= ['Outgoing Shortwave Radiation','Outgoing Longwave Radiation']

    sat_nm= 'TAmean'
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
    tgt_dates= (date(2002,9,1),date(2024,8,31)) 
    tgt_date_names= '-'.join([dd.strftime('%Y.%m') for dd in tgt_dates])
    imon,nmon= cf.get_tot_months(date_range[0],tgt_dates[0])-1, cf.get_tot_months(*tgt_dates)
    nmon_yr,nyr= 12, nmon//12
    mon_days= np.asarray(cf.get_month_days(tgt_dates))
    xt= cf.yield_monthly_date_range(*tgt_dates,mdelta=1)
    
    lats= fid.variables['lat'][:]
    lons= fid.variables['lon'][:]
    resol=np.round(lons[1]-lons[0],0).astype(int)  ## Now in 4-deg resolution
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

    ##-- Read rfo data
    rfos= fid.variables['mRFO'][:,imon:imon+nmon,:].reshape([ncr,nyr,nmon_yr,nlat,nlon])

    ##-- Two periods    
    dr0,dr1= (date(2002,9,1),date(2012,8,31)),(date(2014,9,1),date(2024,8,31))
    imon1,nmon1= cf.get_tot_months(tgt_dates[0],dr0[0])-1,cf.get_tot_months(*dr0)
    iyr1,nyr1= imon1//nmon_yr, nmon1//nmon_yr
    imon2,nmon2= cf.get_tot_months(tgt_dates[0],dr1[0])-1,cf.get_tot_months(*dr1)
    iyr2,nyr2= imon2//nmon_yr, nmon2//nmon_yr
        
    
    ### Read RFO kernel
    indir= './Data/'
    LM_fn= indir+f'Linear_Model_Coef.4Outgoing_Radiation_w{ncr}CRgroups.vEBAF_regression.nc'
    fid= Dataset(LM_fn,'r')
    rad_clr_name= ['Insol','OLR_clr']
    kc_all, rad_clr_all= [],[]
    for rad_idx,rad_name in enumerate(rad_names):
        half_kc,half_clr=[],[]
        for half_idx in range(2):
            vn= f'O{rad_name[0].upper()}R_PD{half_idx}'
            kc= fid.variables[vn][:]
            print(kc.shape, kc.min(), kc.max(),kc.mean(axis=(0,1,2)))
            rad_clr_vn= f'{rad_clr_name[rad_idx]}_clim_PD{half_idx}'
            rad_clr= fid.variables[rad_clr_vn][:]
            
            half_kc.append(kc)
            half_clr.append(rad_clr)
        kc_all.append(half_kc)
        rad_clr_all.append(half_clr)
        
    ### Build R from RFO
    ### kc: [nmon_yr,nlat,nlon,ncr]
    rad_built_all=[]
    for rad_idx,rad_name in enumerate(rad_names):
        kc2= kc_all[rad_idx]
        rad_built_half=[]
        for ik,kc1 in enumerate(kc2):
            rad_built=0.
            for icr in range(ncr):
                rad_built+= kc1[None,:,:,:,icr].filled(0.)*rfos[icr,:]

            ## Denormalize
            rad_built= rad_built*rad_clr_all[rad_idx][ik][None,:,:,:]

            ## semi-global mean
            rbuilt1_gm= np.ma.average(rad_built.reshape([nmon,nlat*nlon]),weights=ltw.reshape(-1),axis=1)
            print(rbuilt1_gm[:120].mean())
            rad_built_half.append(rbuilt1_gm)
        rad_built_all.append(rad_built_half)

    ### Semi-global mean of CERES reference
    var_names= ['toa_sw_all_mon','toa_lw_all_mon',]
    rad_ref_all=[]
    for rad_idx,rad_name in enumerate(var_names):
        rref1= cf.get_NRB_TOA_monthly(rad_name,tgt_dates,tgt_latlon1)
        if resol>1:
            rref1= rref1.reshape([nmon,nlat,resol,nlon,resol]).mean(axis=(2,4))
        rref1_gm= np.ma.average(rref1.reshape([nmon,nlat*nlon]),weights=ltw.reshape(-1),axis=1)
        rad_ref_all.append(rref1_gm)
        print(rref1_gm[:120].mean())

    ### For Figure
    rg_name_tit= '{a}\u00B0S\u2013{a}\u00B0N'.format(a=max_lat)
    suptit= '{} vs. Built by RFO in {}'.format(mdnm0, rg_name_tit)
    outdir= './Pics/'       
    outfn= outdir+'Fig02.EBAF_vs_Built.GM_tseries.{}_comp_half_{}.png'.format(tgt_date_names,mdnm0)

    pic_data= dict(rad_ref= rad_ref_all, rad_built= rad_built_all, xt=xt,
                   pn_tit= rad_names_tit, prd1= dr0, prd2= dr1, mon_days=mon_days,
                   suptit=suptit, outfn=outfn,)
    plot_main(pic_data)
                                                         
    return

import matplotlib as mpl
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
from matplotlib.dates import DateFormatter, YearLocator

def plot_main(pdata):
    rad_ref= pdata['rad_ref']
    rad_built= pdata['rad_built']
    xt= pdata['xt']
    pn_tit= pdata['pn_tit']
    prd1= pdata['prd1']
    prd2= pdata['prd2']
    mon_days= pdata['mon_days']
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

    unit0= r'$Wm^{-2}$'
    unit1= unit0+r'$yr^{-1}$'
    ai=0
    ###--- panel1
    for i,tit1 in enumerate(pn_tit):
        ax1= fig.add_axes([ix,iy-ly,lx,ly])
        rad0= rad_built[i][0] #-rad_ref[i]
        rad1= rad_built[i][1] #-rad_ref[i]

        pic1a= ax1.plot(xt,rad0,c='C0',alpha=0.6,lw=0.8)
        pic2a= ax1.plot(xt,rad1,c='C1',alpha=0.6,lw=0.8)
        pic0a= ax1.plot(xt,rad_ref[i],c='0.4',lw=0.8)
        
        pic1b= ax1.plot(xt[6:-6],running_mean_1d(rad0,12)[6:-6],c='C0',alpha=0.7,lw=3,label='Prd1_Model')        
        pic2b= ax1.plot(xt[6:-6],running_mean_1d(rad1,12)[6:-6],c='C1',alpha=0.7,lw=3,label='Prd2_Model')        
        pic0b= ax1.plot(xt[6:-6],running_mean_1d(rad_ref[i],12)[6:-6],c='0.1',alpha=0.75,lw=2.,label='Obs.')
        
        subtit= '({}) {} '.format(abc[ai],tit1); ai+=1
        ax1.set_title(subtit,fontsize=12,x=0,ha='left')

        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))    
        #ax1.xaxis.set_major_formatter(DateFormatter("%Y\n%b"))
        ax1.set_xticklabels('')
        ax1.yaxis.set_major_locator(MultipleLocator(4))    
        ax1.set_ylabel('Rad. ({})'.format(unit0),fontsize=10)
        ax1.grid(axis='both',ls=':',c='0.7',lw=1)
        ax1.tick_params(axis='y',which='major',labelsize=9)
        ax1.yaxis.set_ticks_position('both')

        for dd1,dd2 in zip(prd1,prd2):
            ax1.axvline(x=dd1,ls='--',c='0.6',lw=1.5)
            ax1.axvline(x=dd2,ls='--',c='0.6',lw=1.5)        
        ax1.legend(loc='upper left',fontsize=10,framealpha=0.9, borderaxespad=0.,bbox_to_anchor=(1.02,1.))
        if i==0:
            yr= ax1.get_ylim()
            yc= (yr[0]+yr[1])/2
            yr1= (yr[1]-yc)*0.88
            ax1.set_ylim( [yc-yr1,yc+yr1])
        else:
            yr= ax1.get_ylim()
            yc= (yr[0]+yr[1])/2
            ax1.set_ylim([yc-yr1,yc+yr1])
            
        iy-= ly+gapy/10
        ly2= ly*0.8
        ax2= fig.add_axes([ix,iy-ly2,lx,ly2])
        p2a= ax2.plot(xt,rad0-rad_ref[i],c='C0',alpha=0.6,lw=0.8)
        rm1= running_mean_1d(rad0-rad_ref[i],12)
        p2b= ax2.plot(xt[6:-6],rm1[6:-6],c='C0',alpha=0.7,lw=2.5,label='Prd1_M-Obs.')
        p3a= ax2.plot(xt,rad1-rad_ref[i],c='C1',alpha=0.6,lw=0.8)
        rm2= running_mean_1d(rad1-rad_ref[i],12)
        p3b= ax2.plot(xt[6:-6],rm2[6:-6],c='C1',alpha=0.7,lw=2.5,label='Prd2_M-Obs.')
        
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))    
        ax2.xaxis.set_major_formatter(DateFormatter("%Y\n%b"))        
        ax2.yaxis.set_major_locator(MultipleLocator(0.5))    
        ax2.set_ylabel('Rad. Diff. ({})'.format(unit0),fontsize=10)
        ax2.grid(axis='both',ls=':',c='0.7',lw=1)
        ax2.tick_params(axis='y',which='major',labelsize=9)
        ax2.yaxis.set_ticks_position('both')

        for dd1,dd2 in zip(prd1,prd2):
            ax2.axvline(x=dd1,ls='--',c='0.6',lw=1.5)
            ax2.axvline(x=dd2,ls='--',c='0.6',lw=1.5)    
        ax2.axhline(y=0.,ls='--',c='0.1',lw=1)
        ax2.legend(loc='upper left',fontsize=10,framealpha=0.9, borderaxespad=0.,bbox_to_anchor=(1.02,1.))

        yr= ax2.get_ylim()
        if i==0:            
            yc= (yr[0]+yr[1])/2
            yr2= (yr[1]-yc) #*0.9
            #ax1.set_ylim( [yc-yr1,yc+yr1])
        else:            
            #yc= (yr[0]+yr[1])/2
            yc= (min(rm1.min(),rm2.min())+max(rm1.max(),rm2.max()))/2
            ax2.set_ylim([yc-yr2,yc+yr2])

        nmon= cf.get_tot_months(*prd1)
        mm1= np.average(rad_ref[i][:nmon],weights=mon_days[:nmon])
        mm2= np.average(rad_ref[i][-nmon:],weights=mon_days[-nmon:])
        print('Mean Diff=',mm2-mm1)
        txt1= 'Period 1\n(Mean= {:.1f} {})'.format(mm1,unit0)
        txt2= 'Period 2\n(Mean= {:.1f} {})'.format(mm2,unit0)
        ax2.text(date(2007,9,1),yc+yr2,txt1,fontsize=12,va='center',ha='center',weight='semibold')
        ax2.text(date(2019,9,1),yc+yr2,txt2,fontsize=12,va='center',ha='center',weight='semibold')
        
        ix+= lx+gapx
        if ix+lx>rf:
            ix=lf
            iy-= ly2+gapy
    
        
    ###---
    print(pdata['outfn'])
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150) #
    #plt.show()
    
    return

def running_mean_1d(x, N):
    """
    Calculate running mean with "Cumulative Sum" function, asuming no missings.
    Ref: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    Input x: 1-d numpy array of time series
    Input N: Running Mean period
    Return: Same dimension with x; end points are averaged for less than N values
    """
    cumsum= np.cumsum(np.insert(x, 0, 0))
    new_x= (cumsum[N:] - cumsum[:-N]) / float(N)  ## Now it's running mean of [dim(x)-N] size
    pd0= N//2; pd1= N-1-pd0  ## Padding before and after. If N=5: pd0=2, pd1=2
    head=[]; tail=[]
    for i in range(pd0):
        head.append(x[:i+N-pd0].mean())
        tail.append(x[-i-1-pd1:].mean())
    new_x= np.concatenate((np.asarray(head),new_x,np.asarray(tail)[::-1][:pd1]))
    return new_x

if __name__=="__main__":
    main()
