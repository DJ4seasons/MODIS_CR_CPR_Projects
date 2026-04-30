"""
Compare albedo distribution between various methods

2025.07.02
---

Revision1
H_tk -> H1_tk and H2_tk
Geodetic weight + mon_days weight
2025.11.18
"""

import numpy as np
import sys
import os.path
from datetime import timedelta, date
import common_functions as cf

def main():
    ###--- Parameters
    mdnm0= 'EBAF_4.2.1' #
    mdnm1= 'FBCT_4.1' #'EBAF_4.2.1' #
    rad_names= ['SW','LW']
    rad_idx= 1
    rad_name= rad_names[rad_idx]+'_outgoing'
    #rad_names= [val+'_outgoing' for val in rad_names]
    #rad_names_tit= ['Outgoing Shortwave Radiation','Outgoing Longwave Radiation']
    rad_name_tit= 'O{}R'.format(rad_names[rad_idx][0])

    sat_nm= 'TAmean'
    #tgt_crg= [0,1,2,3,4]
    rfo_crt=5.
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
    #print(xt); sys.exit()
    mon_days= cf.get_month_days(tgt_dates)
    mon_days_clim_norm= np.asarray(mon_days).reshape(-1,12)[:4,:].mean(axis=0)
    mon_days_clim_norm= mon_days_clim_norm/mon_days_clim_norm.sum()
    
    dr0,dr1= (date(2002,9,1),date(2012,8,31)),(date(2014,9,1),date(2024,8,31))
    imon1,nmon1= cf.get_tot_months(tgt_dates[0],dr0[0])-1,cf.get_tot_months(*dr0)
    iyr1,nyr1= imon1//nmon_yr, nmon1//nmon_yr
    imon2,nmon2= cf.get_tot_months(tgt_dates[0],dr1[0])-1,cf.get_tot_months(*dr1)
    iyr2,nyr2= imon2//nmon_yr, nmon2//nmon_yr
    nyr2_F= nyr2-2

    dr0_name= '-'.join([d.strftime('%Y.%m') for d in dr0])
    dr1_name= '-'.join([d.strftime('%Y.%m') for d in dr1])
    ## Horizontal info
    hs1= tshs[1]
    tgt_latlon1, tgt_rg_name1= [-max_lat,max_lat,-180,180], '{a}S-{a}N'.format(a=max_lat)
    nlat,nlon= max_lat*2//hs1,360//hs1
    latinfo, loninfo = (-max_lat+hs1/2,hs1,nlat), (-180+hs1/2,hs1,nlon)
    latlon_info= dict(latinfo=latinfo, loninfo=loninfo)
    lats= np.arange(nlat)*hs1+latinfo[0]
    lons= np.arange(nlon)*hs1+loninfo[0]
    #xy= np.meshgrid(lons,lats)
    #lat_weight= cf.apply_lat_weight(np.ones([nlat,nlon]),nlat,nlon,lats).squeeze()
    #print(lat_weight[:,0])
    #lats_2d= np.ones([nlat,nlon],dtype=float)*lats[:,None]
    nlat0,nlon0= max_lat*2,360
    lats0= np.arange(nlat0)-nlat0/2+0.5
    lat_weight= cf.apply_lat_weight(np.ones([nlat0,nlon0,]),nlat0,nlon0,lats0,geodetic=True).squeeze()
    lat_weight= lat_weight.reshape([nlat,hs1,nlon,hs1]).mean(axis=(1,3))

    '''
    if rad_idx==0:
        ### Read Solar Insolation data
        #rad0_E= cf.get_NRB_TOA_monthly('solar_mon',tgt_dates,tgt_latlon1) #*-1
        rad0= cf.get_NRB_FBCT_monthly('SWTDN',tgt_dates,tgt_latlon1)
    elif rad_idx==1:
        #rad0_E= cf.get_NRB_TOA_monthly('toa_lw_clr_c_mon',tgt_dates,tgt_latlon1)
        rad0= cf.get_NRB_FBCT_monthly('LWTUPCLR',tgt_dates,tgt_latlon1)*-1

    ## Degrading resolution
    if hs1>1:
        rad0= rad0.reshape([nmon,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nmon,nlat,nlon,hs1*hs1]).mean(axis=-1)
        #rad0_E= rad0_E.reshape([nmon,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nmon,nlat,nlon,hs1*hs1]).mean(axis=-1)
        
    rad0= rad0.reshape([nyr,nmon_yr,nlat,nlon])
    rad0_pd1= rad0[iyr1:iyr1+nyr1,:,:,:].mean(axis=0)
    rad0_pd2= rad0[iyr2:iyr2+nyr2,:,:,:].mean(axis=0)
    r0_clim= [rad0_pd1, rad0_pd2]
    #rad0=0
    '''
    
    ### Read RFO kernel
    ncr2= ncr+1
    nm0= 'v{}RFOs'.format(ncr2)
    indim= [nmon_yr,nlat,nlon,ncr2]
    dim_txt= 'x'.join([str(val) for val in indim])
    kc_all=[]
    rad_ref_all=[]
    md_vers= ['v2b_m2r3','vFBCT_4.1','v2b_m2r3FBCT']
    mv_names= ['EBAF_regr','FBCT_comp','FBCT_regr']
    for ridx,md_ver in enumerate(md_vers):
        half_kc=[]
        for half_idx in range(2):
            nm1= '_half{}_{}'.format(half_idx,md_ver)
            kc_name= './Data2/{}_kernel4{}_in{}.{}.{}.f32dat'.format('CR_groups',rad_name,tgt_rg_name1,nm0+nm1,dim_txt)
            kc= cf.bin_file_read2mtx(kc_name).reshape(indim) #[:,:,:,tgt_crg]
            kc= np.ma.masked_less(kc,-999.)
            print(kc.shape, kc.min(), kc.max(),kc.mean(axis=(0,1,2)))
            half_kc.append(kc)

        kc_all.append(half_kc)
        '''
        ### For TOA CRE    
        var_names_E= ['toa_sw_all_mon','toa_sw_clr_c_mon','toa_lw_all_mon','toa_lw_clr_c_mon',]
        var_names= ['SWTNT', 'SWTNTCLR', 'LWTUP','LWTUPCLR',]
        vn= var_names[rad_idx*2]
        vn_E= var_names_E[rad_idx*2]
        
        rad1_E= cf.get_NRB_TOA_monthly(vn_E,tgt_dates,tgt_latlon1) #*-1
        rad1= cf.get_NRB_FBCT_monthly(vn,tgt_dates,tgt_latlon1) #*-1
        #rad1= rad0-rad1
        print( rad1_E.shape, rad1_E.min(), rad1_E.max(),) #rad0.min(), rad0.max(),
        print( rad1.shape, rad1.min(), rad1.max(),) #rad0.min(), rad0.max(),

        ## Degrading resolution
        if hs1>1:
            rad1= rad1.reshape([nmon,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nmon,nlat,nlon,hs1*hs1]).mean(axis=-1)
            rad1_E= rad1_E.reshape([nmon,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nmon,nlat,nlon,hs1*hs1]).mean(axis=-1)
            if rad_idx==0:
                rad1= rad0.reshape([nmon,nlat,nlon])-rad1
            elif rad_idx==1:
                rad1*=-1

        rad_ref_all= [rad1, rad1_E] #.append(rad1)
        '''
    ### Read regime RFO
    '''
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
    '''
    indir= '/Users/djin1/Documents/CLD_Work/Data_Obs/MODIS_c61/Composite/'
    infn= indir+'Monthly_Composite_Histogram+RFO_map.by{}CRgroups.nc'.format(ncr2)
    from netCDF4 import Dataset
    fid= Dataset(infn,'r')
    mrfo= fid.variables['mRFO'][:]
    print(mrfo.shape) #; sys.exit()
    rfos= mrfo.reshape([ncr2,nyr,nmon_yr,nlat,nlon])
    
    ### Screend out by clim_RFO
    cr_idx=[]
    for icr in range(ncr2):
        clim_rfo= rfos[icr,:].mean(axis=0)
        #rfos[icr,:,clim_rfo<rfo_crt/100]=0.
        cr_idx.append(clim_rfo>rfo_crt/100)
        print(icr,cr_idx[-1].mean())
        
    ### Collecting slope info
    ### kc: [nmon_yr,nlat,nlon,ncr+1]
    slope_all=[]
    for ik in range(2):
        slope_half=[]
        for ridx,half_kc in enumerate(kc_all):
            kc1= half_kc[ik]
            slope_byCR=[]
            for icr in range(ncr2):
                slope_byCR.append(kc1[:,:,:,icr][cr_idx[icr]])
            slope_half.append(slope_byCR)
        slope_all.append(slope_half)

    for r in range(3):
        print(r+1)
        for icr in range(ncr2):
            d0= slope_all[0][r][icr]
            d1= slope_all[1][r][icr]
            print(icr,np.median(d1)-np.median(d0), d1.mean()-d0.mean())
            
    sys.exit()

    ### For Figure
    rg_name_tit= '{a}\u00B0S\u2013{a}\u00B0N'.format(a=max_lat)
    suptit= 'Slope Comparison by CR groups [{} in {}, {}>{:.0f}%]'.format(rad_name_tit,rg_name_tit,r'$RFO_{clim}$',rfo_crt)
    outdir= '../../Writing_TOA_Rad_Trend+CR/Pics/'       
    outfn= outdir+'vR1_Fig09.regr_coeff_comparison_byCRgroup.{}-{}_comp_vs_regr.{}.png'.format(*tgt_date_names,rad_names[rad_idx])

    pic_data= dict(albedo_all= slope_all, mv_names= mv_names,
                   pn_tit= ['Period1 ({})'.format(dr0_name),'Period2 ({})'.format(dr1_name)],
                   cr_group_names=cr_group_names+['Clear'],
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

    #unit0= r'$Wm^{-2}$'
    #unit1= unit0+r'$yr^{-1}$'
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
