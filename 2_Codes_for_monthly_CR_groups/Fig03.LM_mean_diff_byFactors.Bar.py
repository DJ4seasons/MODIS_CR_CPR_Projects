"""
Analyze which factors are responsible for the mean difference, 2003-2012 vs. 2015-2024

1. By model difference
2. By mean RFO difference
3. Combined

2025.04.17
Daeho Jin
---

Figure 2 for the paper
2025.05.13
---

For Rv1.
H1_tk and H2_tk, and geodetic weight+mon_day weight
2025.11.11 
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
    pd1_date_name= 'PD1: {}-{}'.format(*[dd.strftime('%Y.%m') for dd in dr0])
    imon2,nmon2= cf.get_tot_months(tgt_dates[0],dr1[0])-1,cf.get_tot_months(*dr1)
    iyr2,nyr2= imon2//nmon_yr, nmon2//nmon_yr
    pd2_date_name= 'PD2: {}-{}'.format(*[dd.strftime('%Y.%m') for dd in dr1])
        
    ##-- RFO clim 
    #rfos_clim_all= rfos.mean(axis=1)
    rfos_clim_pd1= rfos[:,iyr1:iyr1+nyr1,:,:,:].mean(axis=1)
    rfos_clim_pd2= rfos[:,iyr2:iyr2+nyr2,:,:,:].mean(axis=1)
    rfos= None
    
    ### Read RFO kernel
    indir= './Data/'
    LM_fn= indir+f'Linear_Model_Coef.4Outgoing_Radiation_w{ncr}CRgroups.vEBAF_regression.nc'
    fid= Dataset(LM_fn,'r')
    rad_clr_name= ['Insol','OLR_clr']
    #kc_all, rad_clr_all= [],[]
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
    rad_built_set, sn_set=[],[]
    ltw= ltw.reshape(-1)
    for kc1 in half_kc:
        by_clim=[]        
        for j,rc in enumerate([rfos_clim_pd1,rfos_clim_pd2]):
            rad_compo=[]
            rad_compo_sn=[]
            for icr in range(ncr):
                rad_tmp2= kc1[:,:,:,icr]*rc[icr,:]

                ## De-normalize
                rad_tmp2*= half_clr[j]

                rad_tmp2_mean0= (rad_tmp2*mon_days_clim_norm[:,None,None]).sum(axis=0).reshape(-1)
                rad_compo.append(np.average(rad_tmp2_mean0,weights= ltw))

                rad_tmp2_sn_mean0= (rad_tmp2*mon_days_clim_norm[:,None,None]).reshape([4,3,nlat*nlon])
                rad_tmp2_sn_mean0= rad_tmp2_sn_mean0.sum(axis=1)/mon_days_clim_norm.reshape([4,3]).sum(axis=1)[:,None]
                rad_compo_sn.append(np.average(rad_tmp2_sn_mean0,weights=ltw,axis=1))
                
            by_clim.append(rad_compo)
            sn_set.append(rad_compo_sn)
        rad_built_set.append(by_clim)
        
    rad_built_set= np.asarray(rad_built_set)
    sn_set= np.asarray(sn_set)
    print(rad_built_set.shape) #[n_model, n_rfo, ncr]
    for i in range(2):
        for j in range(2):
            print(rad_built_set[i,j,:])

    ### Semi-global mean difference of CERES reference
    var_names= ['toa_sw_all_mon','toa_lw_all_mon',]
    in_dir= './Data/'

    rad_var= var_names[rad_idx]
    rref1= cf.get_NRB_TOA_monthly(rad_var,tgt_dates,tgt_latlon1,in_dir=in_dir)
    if resol>1:
        rref1= rref1.reshape([nmon,nlat,resol,nlon,resol]).mean(axis=(2,4))

    rad_pd1= np.ma.average(rref1[imon1:imon1+nmon1,:],weights=mon_days[imon1:imon1+nmon1],axis=0) 
    rad_pd2= np.ma.average(rref1[imon2:imon2+nmon2,:],weights=mon_days[imon1:imon1+nmon1],axis=0) 
    rad_pd1_gm= np.ma.average(rad_pd1.reshape(-1),weights=ltw)
    rad_pd2_gm= np.ma.average(rad_pd2.reshape(-1),weights=ltw)
    rad_gm_diff= rad_pd2_gm-rad_pd1_gm
    print(rad_pd1_gm, rad_pd2_gm, rad_gm_diff)
    
        
    ### For Figure
    rg_name_tit= '{a}\u00B0S\u2013{a}\u00B0N'.format(a=max_lat)
    sn_names= ['SON','DJF','MAM','JJA']
    suptit= 'Contribution to {} mean difference in {}'.format(rad_name_tit, rg_name_tit)
    outdir= './Pics/'       
    outfn= outdir+'Fig03.Contribution2GM_mean_diff.{}_vs_{}.{}.{}.png'.format(*[val.split()[1] for val in [pd1_date_name,pd2_date_name]],rad_name,mdnm0)

    pn_tit= ['By Model Diff. (Period2_Model - Period1_Model)',
             'By RFO_clim Diff. (using Period2_RFO - using Period1_RFO)',
             'Combined (Period2_Model[Period2_RFO] - Period1_Model[Period1_RFO])',
             'Combined, by season',
    ]
    pn_labs= [('Period1\nRFO','Period2\nRFO'),('Period1\nModel','Period2\nModel'),('',)]
    pic_data= dict(rad_ref= rad_gm_diff, rad_built_set= rad_built_set,
                   ind_set1= [[(0,0),(1,0)],[(0,1),(1,1)]],
                   ind_set2= [[(0,0),(0,1)],[(1,0),(1,1)]],
                   ind_set3= [[(0,0),(1,1)],], sn_diff= sn_set[-1]-sn_set[0],
                   pn_tit= pn_tit, pn_labs= pn_labs,sn_names= sn_names,
                   cr_group_names=cr_names,
                   suptit=suptit, outfn=outfn,)
    plot_main(pic_data)
                                                         
    return

import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
from matplotlib.dates import DateFormatter, YearLocator

def plot_main(pdata):
    rad_ref= pdata['rad_ref']
    rad_built_set= pdata['rad_built_set']
    ind_set1= pdata['ind_set1']
    ind_set2= pdata['ind_set2']
    ind_set3= pdata['ind_set3']
    sn_diff= pdata['sn_diff']
    sn_names= pdata['sn_names']
    pn_tit= pdata['pn_tit']
    pn_labs= pdata['pn_labs']
    cr_group_names= pdata['cr_group_names']
    
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'

    ###---
    fig=plt.figure()
    fig.set_size_inches(7.6,8.)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=16,y=0.975,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    ncol,nrow=1,3.5
    lf,rf,bf,tf=0.05,0.95,0.1,0.925
    gapx, npnx=0.06,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.075,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    
    ix=lf; iy=tf    

    unit0= r'$Wm^{-2}$'
    ai=ci=0
    cc= ['C0','C1','C8','C6','C3','0.5']

    xtlabs= cr_group_names+['Sum']
    nv= len(xtlabs)
    wd= 0.75
    
    ###--- panel1
    axes=[]
    for j,ind0 in enumerate([ind_set1, ind_set2, ind_set3]):
        ax1= fig.add_axes([ix,iy-ly,lx,ly])
        diff_set=[]
        for ind1 in ind0:
            (i0,i1),(i2,i3)= ind1
            diff1= rad_built_set[i2,i3,:]-rad_built_set[i0,i1,:]
            diff1_sum= diff1.sum()
            data= list(diff1)+[diff1_sum]
            diff_set.append(data)
        ng= len(diff_set)
        wd1= wd/ng if ng>1 else 0.67
        xlocs= bar_x_locator(wd1,data_dim=[ng,nv])

        for xl,data,data_nm in zip(xlocs,diff_set,pn_labs[j]):
            bar1= ax1.bar(xl,data,width=wd1,color=cc[ci],alpha=0.8,label=data_nm); ci+=1
            for ix,val in enumerate(data):
                ax1.text(xl[ix],0.06,'{:.2f}'.format(val),ha='center',va='bottom',rotation=90,fontsize=9,color='k')
        subtit= '({}) {} '.format(abc[ai],pn_tit[ai],); ai+=1
        ax1.set_title(subtit,fontsize=12,x=0,ha='left')
        ax1.set_xticks(range(nv))
        ax1.set_xticklabels(xtlabs)
    
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    
        ax1.set_ylabel('Rad. Diff. ({})'.format(unit0),fontsize=10)    
        ax1.grid(axis='y',ls=':',c='0.7',lw=1)
        ax1.tick_params(axis='both',which='major',labelsize=9.5)

        ax1.axhline(y=0,ls='--',lw=1,c='0.3')
        ax1.axvline(x=nv-1.5,ls='--',c='0.6',lw=1.5)
        if j==2:
            ax1.axhline(y=rad_ref,ls=':',lw=2,c='0.1',label='Obs. Mean\nDifference')
        ax1.legend(loc='upper left',fontsize=10,framealpha=0.9, borderaxespad=0.,bbox_to_anchor=(1.02,1.))
        ax1.set_xlim([-0.5,nv-0.5])
        axes.append(ax1)
        
        ix+= lx+gapx
        if ix+lx>rf:
            ix=lf
            iy-= ly+gapy
    

    ## Modify y-axs range
    ax1,ax2,ax3= axes
    vmin1,vmax1= ax1.get_ylim()
    vmin2,vmax2= ax2.get_ylim()
    vmin,vmax= min(vmin1,vmin2), max(vmax1,vmax2)
    ax1.set_ylim([vmin,vmax])
    ax2.set_ylim([vmin,vmax])
    if vmax-vmin<2.:
        lnt=0.3
    else:
        lnt=1

    ax1.yaxis.set_major_locator(MultipleLocator(lnt))
    ax2.yaxis.set_major_locator(MultipleLocator(lnt))
    ax3.yaxis.set_major_locator(MultipleLocator(lnt))
    

    ### Seasonal view
    if True:
        ax1= fig.add_axes([ix,iy-ly,lx,ly])
        n_sn= len(sn_names)
        diff_set=[]
        for isn in range(n_sn):
            diff1= sn_diff[:,isn]/4
            diff1_sum= diff1.sum()
            data= list(diff1)+[diff1_sum]
            diff_set.append(data)
        
        ng= 1 #len(diff_set)
        wd1= wd/ng if ng>1 else 0.67
        xlocs= bar_x_locator(wd1,data_dim=[ng,nv])

        ax1.axhline(y=rad_ref,ls=':',lw=2,c='0.1',label='Obs. Mean\nDifference')

        t_crt= 0.195 if abs(rad_ref)>0.9 else 0.495
        ci=0
        for jj in range(nv):
            bm1=bp1=0
            for ii in range(len(diff_set)):
                diff1= diff_set[ii][jj]
                if diff1>=0:
                    if jj==0:
                        bar1= ax1.bar([xlocs[0][jj],],[diff1,],bottom=bp1,color=cc[ii],alpha=0.8,width=wd1,label=sn_names[ii])
                    else:
                        bar1= ax1.bar([xlocs[0][jj],],[diff1,],bottom=bp1,color=cc[ii],alpha=0.8,width=wd1,)
                    if diff1>t_crt:
                        ax1.text(xlocs[0][jj],bp1+diff1/2,'{:.2f}'.format(diff1),ha='center',va='center',fontsize=9,color='k')
                    bp1+= diff1
                else:
                    if jj==0:
                        bar1= ax1.bar([xlocs[0][jj],],[diff1,],bottom=bm1,color=cc[ii],alpha=0.8,width=wd1,label=sn_names[ii])
                    else:
                        bar1= ax1.bar([xlocs[0][jj],],[diff1,],bottom=bm1,color=cc[ii],alpha=0.8,width=wd1,)
                    if diff1<-t_crt:
                        ax1.text(xlocs[0][jj],bm1+diff1/2,'{:.2f}'.format(diff1),ha='center',va='center',fontsize=9,color='k')
                    bm1+= diff1
                    
        subtit= '({}) {} '.format(abc[ai],pn_tit[ai],); ai+=1
        ax1.set_title(subtit,fontsize=12,x=0,ha='left')
        ax1.set_xticks(range(nv))
        ax1.set_xticklabels(xtlabs)
    
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    
        ax1.set_ylabel('Rad. Diff. ({})'.format(unit0),fontsize=10)    
        ax1.grid(axis='y',ls=':',c='0.7',lw=1)
        ax1.tick_params(axis='both',which='major',labelsize=9.5)

        ax1.axhline(y=0,ls='--',lw=1,c='0.3')
        ax1.axvline(x=nv-1.5,ls='--',c='0.6',lw=1.5)        
            
        ax1.legend(loc='upper left',fontsize=10,framealpha=0.9, borderaxespad=0.,bbox_to_anchor=(1.02,1.))

        ax1.yaxis.set_major_locator(MultipleLocator(lnt))
        ax1.set_xlim([-0.5,nv-0.5])
        
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
