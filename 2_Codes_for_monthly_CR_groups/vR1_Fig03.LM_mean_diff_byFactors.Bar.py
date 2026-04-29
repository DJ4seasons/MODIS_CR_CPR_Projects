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
import common_functions as cf

def main():
    ###--- Parameters
    mdnm0= 'EBAF_4.2.1' #'FBCT_4.1'
    rad_names= ['SW','LW']
    rad_idx= 1
    rad_name= rad_names[rad_idx]+'_outgoing'
    #rad_names= [val+'_outgoing' for val in rad_names]
    rad_name_tit= 'O{}R'.format(rad_names[rad_idx][0])

    sat_nm= 'TAmean'
    #tgt_crg= [0,1,2,3,4]
    #rfo_crt=1.
    tgt_cr_groups= [
        ('H1_tk',(1,3,5)), ('H2_tk',(2,6)), 
        #('H1_tn',(7,9)),  ('H2_tn',(8,)),
        #('H_tk',(1,2,3,5,6,)),
        ('H_tn',(7,8,9)),
        ('Mid',(4,151,152)),
        ('L1_tk',(11,13)),
        ('L2_tk',(10,12)),                    
        ('L_tn',(14,)),
        ('S-Clr',(153,)),
        #('Clr',(0,)),        
    ]
    tgt_crg= range(len(tgt_cr_groups))
    #cr_group_name,tgt_cr= tgt_cr_groups[tgt_crg]
    cr_group_names= [tgt_cr_groups[k][0] for k in tgt_crg]
    tgt_crs= [tgt_cr_groups[k][1] for k in tgt_crg]
    ncr= len(tgt_crg)

    max_lat= 60 #
    tshs=   ['monthly',4] #['monthly',5] #

    ## Temporal info
    tgt_dates= (date(2002,9,1),date(2024,8,31))
    tgt_date_names= [d.strftime('%Y.%m') for d in tgt_dates]
    nmon= cf.get_tot_months(*tgt_dates)
    nmon_yr=12
    nyr= nmon//nmon_yr

    xt= cf.yield_monthly_date_range(*tgt_dates,mdelta=1)
    mon_days= cf.get_month_days(tgt_dates)    
    #print(np.asarray(mon_days).reshape(-1,12))
    #print(xt); sys.exit()
    mon_days_clim_norm= np.asarray(mon_days).reshape(-1,12)[:4,:].mean(axis=0)
    mon_days_clim_norm= mon_days_clim_norm/mon_days_clim_norm.sum()
    
    pd1_dates= (date(2002,9,1),date(2012,8,31))
    imon1,nmon1= cf.get_tot_months(tgt_dates[0],pd1_dates[0])-1,cf.get_tot_months(*pd1_dates)
    iyr1,nyr1= imon1//nmon_yr, nmon1//nmon_yr
    pd1_date_name= 'PD1: {}-{}'.format(*[dd.strftime('%Y.%m') for dd in pd1_dates])
    pd2_dates= (date(2014,9,1),date(2024,8,31))
    imon2,nmon2= cf.get_tot_months(tgt_dates[0],pd2_dates[0])-1,cf.get_tot_months(*pd2_dates)
    iyr2,nyr2= imon2//nmon_yr, nmon2//nmon_yr
    pd2_date_name= 'PD2: {}-{}'.format(*[dd.strftime('%Y.%m') for dd in pd2_dates])

    print(imon1,iyr1,nmon1,nyr1)
    print(imon2,iyr2,nmon2,nyr2) #; sys.exit()
                       
    ## Horizontal info
    hs1= tshs[1]
    tgt_latlon1, tgt_rg_name1= [-max_lat,max_lat,-180,180], '{a}S-{a}N'.format(a=max_lat)
    nlat,nlon= max_lat*2//hs1,360//hs1
    latinfo, loninfo = (-max_lat+hs1/2,hs1,nlat), (-180+hs1/2,hs1,nlon)
    latlon_info= dict(latinfo=latinfo, loninfo=loninfo)
    lats= np.arange(nlat)*hs1+latinfo[0]
    lons= np.arange(nlon)*hs1+loninfo[0]
    xy= np.meshgrid(lons,lats)
    #lat_weight= cf.apply_lat_weight(np.ones([nlat,nlon]),nlat,nlon,lats).squeeze()
    #print(lat_weight[:,0])
    #lats_2d= np.ones([nlat,nlon],dtype=float)*lats[:,None]
    nlat0,nlon0= max_lat*2,360
    lats0= np.arange(nlat0)-nlat0/2+0.5
    lat_weight= cf.apply_lat_weight(np.ones([nlat0,nlon0,]),nlat0,nlon0,lats0,geodetic=True).squeeze()
    lat_weight= lat_weight.reshape([nlat,hs1,nlon,hs1]).mean(axis=(1,3))

    if rad_idx==0:
        ### Read Solar Insolation data
        rad0= cf.get_NRB_TOA_monthly('solar_mon',tgt_dates,tgt_latlon1) #*-1
    elif rad_idx==1:
        rad0= cf.get_NRB_TOA_monthly('toa_lw_clr_c_mon',tgt_dates,tgt_latlon1)
        
    ## Degrading resolution
    if hs1>1:
        rad0= rad0.reshape([nmon,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nmon,nlat,nlon,hs1*hs1]).mean(axis=-1)
    rad0= rad0.reshape([nyr,nmon_yr,nlat,nlon])
    rad0_pd1= rad0[iyr1:iyr1+nyr1,:,:,:].mean(axis=0)
    rad0_pd2= rad0[iyr2:iyr2+nyr2,:,:,:].mean(axis=0)
    r0_clim= [rad0_pd1, rad0_pd2]
    rad0=0
    
    ### Read RFO kernel
    ncr2= ncr+1
    nm0= 'v{}RFOs'.format(ncr2)
    indim= [nmon_yr,nlat,nlon,ncr2]
    dim_txt= 'x'.join([str(val) for val in indim])
    kc_all=[]
    rad_ref_all=[]
    md_ver= 'v2b_m2r3'
    #for rad_idx,rad_name in enumerate(rad_names):
    if True:        
        half_kc=[]
        for half_idx in range(2):
            nm1= '_half{}_{}'.format(half_idx,md_ver)
            kc_name= './Data2/{}_kernel4{}_in{}.{}.{}.f32dat'.format('CR_groups',rad_name,tgt_rg_name1,nm0+nm1,dim_txt)
            kc= cf.bin_file_read2mtx(kc_name).reshape(indim) #[:,:,:,tgt_crg]
            kc= np.ma.masked_less(kc,-999.).filled(0.)
            print(kc.shape, kc.min(), kc.max(),kc.mean(axis=(0,1,2)))
            half_kc.append(kc)
        #kc_all.append(half_kc)

    
        ### For TOA CRE    
        var_names= ['toa_sw_all_mon','toa_sw_clr_t_mon','toa_lw_all_mon','toa_lw_clr_t_mon',]
        vn= var_names[rad_idx*2]
        #rad0= cf.get_NRB_TOA_monthly('solar_mon',tgt_dates,tgt_latlon1) #*-1
        rad1= cf.get_NRB_TOA_monthly(vn,tgt_dates,tgt_latlon1) #*-1
        #rad1= rad0-rad1
        print( rad1.shape, rad1.min(), rad1.max(),) #rad0.min(), rad0.max(),

        ## Degrading resolution
        if hs1>1:
            rad1= rad1.reshape([nmon,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nmon,nlat,nlon,hs1*hs1]).mean(axis=-1)
        #rad_ref_all.append(rad1)
        
    rad_pd1= np.average(rad1[imon1:imon1+nmon1,:],weights=mon_days[imon1:imon1+nmon1],axis=0) #.mean(axis=0)
    rad_pd2= np.average(rad1[imon2:imon2+nmon2,:],weights=mon_days[imon1:imon1+nmon1],axis=0) #.mean(axis=0)
    rad_pd1_gm= np.ma.average(rad_pd1.reshape(-1),weights=lat_weight.reshape(-1))
    rad_pd2_gm= np.ma.average(rad_pd2.reshape(-1),weights=lat_weight.reshape(-1))
    rad_gm_diff= rad_pd2_gm-rad_pd1_gm
    print(rad_pd1_gm, rad_pd2_gm, rad_gm_diff)

    ### Read regime RFO
    rg_set= dict(rg=50,nelemp=0,prwt=0,km=15)
    crmap= cf.read_cpr_map(rg_set,sat_nm,tgt_dates,tgt_latlon1)    
    print(crmap.shape)
    nt1,nlat1,nlon1= crmap.shape
    
    ## Degrading resolution
    crmap= crmap.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
    print(crmap.shape)
    if True: #np.any(np.asarray(tgt_cr)>rg_set['km']):
        crmap_sub= cf.read_cpr_map(rg_set,sat_nm,tgt_dates,tgt_latlon1,sub=True,tgt_cr=rg_set['km'],subk=3)
        crmap_sub= crmap_sub.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
        print(crmap_sub.shape)
    else:
        crmap_sub=[]

    ## Read Total CF
    cscf_crt= 5.
    cfmap= cf.get_Total_CF_daily(tgt_dates,tgt_latlon1,sat_nm=sat_nm)
    print(cfmap.shape,cfmap.min(), cfmap.max())
    cfmap= cfmap.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
    non_cs= cfmap>= cscf_crt/100. #
    cs_idx= np.logical_and(cfmap>-0.00001,cfmap< cscf_crt/100.)
    cfmap=0
    print(non_cs.sum(), cs_idx.sum())
    
    #rad0= rad0.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
    #rad1= rad1.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
    
    ## Transform to monthly
    y0,m0= tgt_dates[0].year, tgt_dates[0].month
    y1,m1= tgt_dates[1].year, tgt_dates[1].month
    by_month,by_month_zero=[],[]
    month_indicator=[]
    for yy in range(y0,y1+1,1):
        im=m0 if yy==y0 else 1
        em=m1 if yy==y1 else 12
        #print(yy,im,em)
        for mm in range(im,em+1,1):
            it= (date(yy,mm,1)-tgt_dates[0]).days
            yy1,mm1=yy,mm+1
            if mm1>12: yy1+=1; mm1-=12
            et= (date(yy1,mm1,1)-tgt_dates[0]).days
            tmp= crmap[it:et,:]
            if len(crmap_sub)>0: tmp_sub= crmap_sub[it:et,:]
            miss= (tmp==-1).sum(axis=(0,-1))
            tsz= (et-it)*hs1*hs1-miss
            tsz[tsz==0]=1

            #tmp_rad0= rad0[it:et,:].mean(axis=(0,-1))

            by_tcr=[]
            for tgt_cr in tgt_crs:
                idx_all=False
                for tcr in tgt_cr:
                    if tcr<=rg_set['km']:
                        idx= tmp==tcr                    
                    else:
                        tcr1= tcr-rg_set['km']*10
                        idx= tmp_sub==tcr1
                    idx_all= np.logical_or(idx_all,idx)
                ## Exclude new CS
                idx_all= np.logical_and(idx_all,non_cs[it:et,:])                
                by_tcr.append(idx_all)
                
            tmp_rfos= np.asarray(by_tcr).sum(axis=(1,-1))/tsz  #[ncr,nlat,nlon]
            by_month.append(tmp_rfos)
            rfo_zero= cs_idx[it:et,:].sum(axis=(0,-1))/tsz
            by_month_zero.append(rfo_zero)
            
    ### RFO                
    rfos= np.asarray(by_month).swapaxes(0,1) #*100 ## Now in %
    print(rfos.shape, rfos.min(), rfos.max(),rfos[:,4::12,:].mean(axis=(1,2,3))) #; sys.exit() #[ncr,nmon,nlat,nlon]
    rfos= rfos.reshape([ncr,nyr,nmon_yr,nlat,nlon])

    rfo0= np.asarray(by_month_zero); print(rfo0.shape,rfo0.min(), rfo0.max(),rfo0[4::12,:].mean()) #; sys.exit()
    rfo0= rfo0.reshape([1,nyr,nmon_yr,nlat,nlon])

    rfos= np.concatenate((rfos,rfo0),axis=0); print(rfos.shape)
    
    ### RFO clim and ano
    #rfos_clim_all= rfos.mean(axis=1)
    rfos_clim_pd1= rfos[:,iyr1:iyr1+nyr1,:,:,:].mean(axis=1)
    rfos_clim_pd2= rfos[:,iyr2:iyr2+nyr2,:,:,:].mean(axis=1)

    #rfos_pd1_ano= rfos[:,iyr1:iyr1+nyr1,:,:,:]-rfos_clim_pd1[:,None,:,:,:]
    #rfos_pd2_ano= rfos[:,iyr2:iyr2+nyr2,:,:,:]-rfos_clim_pd2[:,None,:,:,:]
    
    ### Build R from RFO
    ### kc: [nmon_yr,nlat,nlon,ncr+1]
    ## 1. Due to model diff
    rad_built_set=[]
    lw= lat_weight.reshape(-1)
    sn_set=[]
    for kc1 in half_kc:
        by_clim=[]        
        for j,rc in enumerate([rfos_clim_pd1,rfos_clim_pd2]):
            rad_compo=[]
            rad_compo_sn=[]
            for icr in range(ncr2):
                rad_tmp2= kc1[:,:,:,icr]*rc[icr,:]
                #if rad_idx==0: rad_tmp2*=r0_clim[j]
                if True: rad_tmp2*=r0_clim[j]
                #rad_tmp2_mean0= rad_tmp2.mean(axis=0)
                rad_tmp2_mean0= (rad_tmp2*mon_days_clim_norm[:,None,None]).sum(axis=0)
                rad_compo.append(np.average(rad_tmp2_mean0,weights= lat_weight))
                #rad_tmp2_sn_mean0= rad_tmp2.reshape([4,3,nlat*nlon]).mean(axis=1)
                rad_tmp2_sn_mean0= (rad_tmp2*mon_days_clim_norm[:,None,None]).reshape([4,3,nlat*nlon])
                rad_tmp2_sn_mean0= rad_tmp2_sn_mean0.sum(axis=1)/mon_days_clim_norm.reshape([4,3]).sum(axis=1)[:,None]
                rad_compo_sn.append(np.average(rad_tmp2_sn_mean0,weights=lw,axis=1))
            #rad_intp= kc1[:,:,:,-1].copy()  ## Intercept
            #if rad_idx==0: rad_intp*=r0_clim[j]
            #rad_compo.append(np.average(rad_intp.mean(axis=0),weights= lat_weight))
            #rad_compo_sn.append(np.average(rad_intp.reshape([4,3,nlat*nlon]).mean(axis=1),weights=lw,axis=1))
            by_clim.append(rad_compo)
            sn_set.append(rad_compo_sn)
        rad_built_set.append(by_clim)

        '''
        by_ano=[]
        for r1 in [rfos_pd1_ano,rfos_pd2_ano]:
            rad_compo=[]
            for icr in range(ncr):
                rad_tmp3= kc1[None,:,:,:,icr]*r1[icr,:]
                rad_compo.append(np.average(rad_tmp3.mean(axis=(0,1)),weights= lat_weight))
            rad_intp= kc1[:,:,:,-1]  ## Intercept
            rad_compo.append(np.average(rad_intp.mean(axis=0),weights= lat_weight))
            by_ano.append(rad_compo)
        rad_built_set3.append(by_ano)
        
        
        rad_built=0.
        for icr in range(ncr):
            rad_built+= kc1[None,:,:,:,icr]*rfos[icr,:]
        rad_built+= kc1[None,:,:,:,-1]  ## Intercept

        ## semi-global mean
        rbuilt1_gm= np.ma.average(rad_built.reshape([nmon,nlat*nlon]),weights=lat_weight.reshape(-1),axis=1)
        '''
        
    #rad_built_set3= [rad_built_set2[0][0],rad_built_set2[1][1]]
    rad_built_set= np.asarray(rad_built_set)
    sn_set= np.asarray(sn_set)
    print(rad_built_set.shape) #; sys.exit() #[n_model, n_rfo, ncr+1]
    for i in range(2):
        for j in range(2):
            print(rad_built_set[i,j,:])
    #sys.exit()
    ### For Figure
    rg_name_tit= '{a}\u00B0S\u2013{a}\u00B0N'.format(a=max_lat)
    sn_names= ['SON','DJF','MAM','JJA']
    suptit= 'Contribution to {} mean difference in {}'.format(rad_name_tit, rg_name_tit)
    outdir= '../../Writing_TOA_Rad_Trend+CR/Pics/'       
    outfn= outdir+'vR1_Fig03.Contribution2GM_mean_diff.{}_vs_{}.{}.{}.png'.format(*[val.split()[1] for val in [pd1_date_name,pd2_date_name]],rad_name,md_ver)

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
                   xt=xt, cr_group_names=cr_group_names+['Clear'],
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
    rad_built_set= pdata['rad_built_set']
    ind_set1= pdata['ind_set1']
    ind_set2= pdata['ind_set2']
    ind_set3= pdata['ind_set3']
    sn_diff= pdata['sn_diff']
    sn_names= pdata['sn_names']
    xt= pdata['xt']
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
    #unit1= unit0+r'$yr^{-1}$'
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
        #ax1.yaxis.set_ticks_position('both')

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
        #ax1.yaxis.set_ticks_position('both')

        ax1.axhline(y=0,ls='--',lw=1,c='0.3')
        ax1.axvline(x=nv-1.5,ls='--',c='0.6',lw=1.5)        
            
        ax1.legend(loc='upper left',fontsize=10,framealpha=0.9, borderaxespad=0.,bbox_to_anchor=(1.02,1.))
        #axes.append(ax1)

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
    main()
