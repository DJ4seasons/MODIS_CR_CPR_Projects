'''
Perform linear regression between TOA_Radiation vs. RFOs 

Subtract F0*CS_albedo from total albedo
Monthly and 4-deg resolution
Train only for half of the period (10 years).
Apply non-negative Ridge regression
Apply Repeated K-fold CV to decide the best alpha

by Daeho Jin
2025.11.10
'''

import numpy as np
import sys
import os.path
from datetime import timedelta, date
from netCDF4 import Dataset, num2date
import common_functions as cf

def main(rad_idx,half_idx):
 
    ###--- Parameters
    mdnm0= 'EBAF_4.2.1' #'FBCT_4.1'
    rad_names= ['SW','LW']
    rad_name= rad_names[rad_idx]+'_outgoing'

    ## Regime Info
    sat_nm= 'TAmean' 
    rg,nelemp,prwt,km= 50,0,0,15 
    rg_set= dict(rg=rg,nelemp=nelemp,prwt=prwt,km=km)
    tgt_cr,subk= km,3
    nelemc,nelem= 42,42+nelemp
    p_letter= 'P' if prwt>0 else ''
    prset_nm = f'Cld{nelemc}+Pr{nelemp}x{prwt}' if prwt>0 else f'Cld{nelemc}'
    rg_nm= f'{rg}S-{rg}N'
    mdnm = 'MODIS_t+a_C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)    

    cscf_crt= 5.
    tgt_cr_groups= [
        ('H1_tk',(1,3,5)), ('H2_tk',(2,6)), 
        ('H_tn',(7,8,9)),
        ('Mid',(4,151,152)),
        ('L1_tk',(11,13)),
        ('L2_tk',(10,12)),                    
        ('L_tn',(14,)),
        ('S-Clr',(153,)),
        #('Clr',(0,)),        # Screend out since relaxed Clr criterion (5%) is applied
    ]
    ncr= len(tgt_cr_groups)
    tgt_crg= range(ncr)
    cr_group_names= [tgt_cr_groups[k][0] for k in tgt_crg]
    tgt_crs= [tgt_cr_groups[k][1] for k in tgt_crg]

    tshs= ['monthly',4] #
    hs1= tshs[1]
    nmon_yr = 12

    max_lat, lat_delta= 60,1    
    tgt_latlon1, tgt_rg_name1= [-max_lat-lat_delta,max_lat+lat_delta,-180,180], '{a}S-{a}N'.format(a=max_lat)
    
    ## Two periods
    dr0,dr1= (date(2002,9,1),date(2012,8,31)),(date(2014,9,1),date(2024,8,31))    
    train_dates= dr0 if half_idx==0 else dr1
    train_date_names= [d.strftime('%Y.%m') for d in train_dates]
    nmon= cf.get_tot_months(*train_dates)
    nyr= nmon//nmon_yr
    print(train_date_names,nmon,nyr)


    ### Open netCDF file for daily RFO data
    indir= './Data/'
    infn= indir+mdnm+'.nc'

    fid= Dataset(infn,'r')

    ##-- Read dimension info
    times= fid.variables['time']
    time_units = times.units
    times= num2date(times[:], units = times.units, calendar=times.calendar,
                      only_use_cftime_datetimes=True,)
    data_t_range= [date(t1.year, t1.month, t1.day) for t1 in [times[0],times[-1]]]    
    idy,ndy= (train_dates[0]-data_t_range[0]).days, (train_dates[1]-train_dates[0]).days+1    
    print(data_t_range, train_dates, idy, ndy)
    
    lats= fid.variables['lat'][:]
    lons= fid.variables['lon'][:]    

    nlat,nlon= len(lats), len(lons)
    resol=np.rint(lons[1]-lons[0]).astype(int)
    latinfo, loninfo= (lats[0],resol,nlat), (lons[0],resol,nlon)
    latlon_info= dict(latinfo=latinfo, loninfo=loninfo)
    lat_idx,lon_ids= cf.get_tgt_latlon_idx(latlon_info, tgt_latlon1[:2],tgt_latlon1[2:])
    
    
    ### Read regime RFO
    crmap= fid.variables['CRnum_on_map_'+sat_nm][idy:idy+ndy,lat_idx[0]:lat_idx[1],:]
    print(crmap.shape,np.unique(crmap[:10]))
    _,nlat,nlon= crmap.shape  ## Update dims
    nlat1,nlon1= nlat//hs1, nlon//hs1
    
    ### Read total CF
    infn_cf= indir+'MODIS_Cld_Fraction_at1deg_65S-65N.nc'
    fid_cf= Dataset(infn_cf,'r')
    cfmap= fid_cf.variables['CF_map_'+sat_nm][idy:idy+ndy,lat_idx[0]:lat_idx[1],:]/10000.
    print(type(cfmap),cfmap.shape,cfmap.min(), cfmap.max())
    cfmap= cfmap.filled(-999.9)
    non_cs= cfmap>= cscf_crt/100. #
    cs_idx= np.logical_and(cfmap>-0.00001,cfmap< cscf_crt/100.)
    cfmap=0
    print(non_cs.sum(), cs_idx.sum())
    

    ## Transform to monthly
    y0,m0= train_dates[0].year, train_dates[0].month
    y1,m1= train_dates[1].year, train_dates[1].month
    print(y0,m0,y1,m1)
    by_month,by_month_rad0,by_month_rad1=[],[],[]
    by_month_zero=[]
    month_indicator=[]
    for yy in range(y0,y1+1,1):
        im=m0 if yy==y0 else 1
        em=m1 if yy==y1 else 12
        print(yy,im,em)
        for mm in range(im,em+1,1):
            it= (date(yy,mm,1)-train_dates[0]).days
            yy1,mm1=yy,mm+1
            if mm1>12: yy1+=1; mm1-=12
            et= (date(yy1,mm1,1)-train_dates[0]).days
            tmp_crmap= crmap[it:et,:]
            miss_idx= (tmp_crmap==-1)
            tsz= (et-it)*hs1*hs1-degraded_sampling(miss_idx,et-it,nlat1,nlon1,hs1,lat_delta,lon_extend=True,is_rad=False)
            #print(tsz.shape, tsz.min(), tsz.max()) #[nlat,nlon,5]
            tsz[tsz==0]=1
                        
            by_tcr=[]
            for tgt_cr in tgt_crs:
                idx_all=False
                for tcr in tgt_cr:
                    idx= tmp_crmap==tcr                    
                    idx_all= np.logical_or(idx_all,idx)
                ## Exclude new CS
                idx_all= np.logical_and(idx_all,non_cs[it:et,:])
                idx_sum= degraded_sampling(idx_all,et-it,nlat1,nlon1,hs1,lat_delta,lon_extend=True,is_rad=False)
                by_tcr.append(idx_sum/tsz)

            tmp_rfos= np.array(by_tcr)  #[ncr,nlat,nlon,5]

            cs_idx_1mo= degraded_sampling(cs_idx[it:et,:],et-it,nlat1,nlon1,hs1,lat_delta,lon_extend=True,is_rad=False)
            rfo_zero= cs_idx_1mo/tsz
            #print(tmp_rfos.min(), tmp_rfos.max(), rfo_zero.min(), rfo_zero.max())
            #print(tmp_rfos.shape, rfo_zero.shape); sys.exit()

            by_month.append(tmp_rfos)
            by_month_zero.append(rfo_zero)
                
    rfos= np.asarray(by_month).swapaxes(0,1) 
    print(rfos.shape, rfos.min(), rfos.max(),rfos[:,4::12,:].mean(axis=(1,2,3))) #; sys.exit() #[ncr,nmon,nlat,nlon]
    rfo0= np.asarray(by_month_zero); print(rfo0.shape,rfo0.min(), rfo0.max(),rfo0[4::12,:].mean()) #; sys.exit()
    nd2i= rfo0.shape[-1]
    

    ### For TOA radiation    
    var_names= ['toa_sw_all_mon','toa_sw_clr_c_mon','toa_lw_all_mon','toa_lw_clr_c_mon',]
    vn= var_names[rad_idx*2]
    in_dir= './Data/'
    if rad_idx==0:
        rad0= cf.get_NRB_TOA_monthly('solar_mon',train_dates,tgt_latlon1,in_dir=in_dir) 
        print( rad0.shape, rad0.min(), rad0.max(),) #rad0.min(), rad0.max(),

        ## Degrading resolution        
        rad0= degraded_sampling(rad0,nmon,nlat1,nlon1,hs1,lat_delta,lon_extend=True,is_rad=True)
        
    rad1= cf.get_NRB_TOA_monthly(vn,train_dates,tgt_latlon1,in_dir=in_dir) 
    print( rad1.shape, rad1.min(), rad1.max(),) 
    
    vn2= var_names[rad_idx*2+1]
    rad2= cf.get_NRB_TOA_monthly(vn2,train_dates,tgt_latlon1,in_dir=in_dir) 
    print( rad2.shape, rad2.min(), rad2.max()) 
    
    if rad1.mask.sum()+rad2.mask.sum()>0:
        sys.exit('Miss in Rad')
       
    ## Degrading resolution
    rad1= degraded_sampling(rad1,nmon,nlat1,nlon1,hs1,lat_delta,lon_extend=True,is_rad=True)
    rad2= degraded_sampling(rad2,nmon,nlat1,nlon1,hs1,lat_delta,lon_extend=True,is_rad=True)
    
    if rad_idx==0:
        rad_ref= rad1/rad0    
        alb_CS= rad2/rad0
        print('Alb_CS',alb_CS.shape, alb_CS.min(), np.percentile(alb_CS.compressed(),[1,2,98,99]),alb_CS.max())
        rad1=rad2=0    
    elif rad_idx==1:
        rad_ref= rad1/olr_clr        
        olr_clr= rad2        
        print('OLR_Clr',olr_clr.shape, olr_clr.min(), np.percentile(olr_clr.compressed(),[1,2,98,99]),olr_clr.max())
    print('Rad_Ref.',rad_ref.min(), np.percentile(rad_ref.compressed(),[1,2,98,99]),rad_ref.max())


    ### Prepare train data
    rfos_train= rfos.reshape([ncr,nyr,nmon_yr,nlat1,nlon1,nd2i])
    rfo0_train= rfo0.reshape([nyr,nmon_yr,nlat1,nlon1,nd2i])
    R_train= rad_ref.reshape([nyr,nmon_yr,nlat1,nlon1,nd2i])
    if rad_idx==0:
        rad0_train= rad0.reshape([nyr,nmon_yr,nlat1,nlon1,nd2i])
        alb_CS_train= alb_CS.reshape([nyr,nmon_yr,nlat1,nlon1,nd2i])
        R_train= R_train-rfo0_train*alb_CS_train
        alb_CS_train_clim= alb_CS_train.mean(axis=0)
    elif rad_idx==1:
        R_train= R_train-rfo0_train
        olr_clr_train_clim= olr_clr.reshape([nyr,nmon_yr,nlat1,nlon1,nd2i]).mean(axis=0)
        
    print('R-train',R_train.min(), np.percentile(R_train.compressed(),[1,2,98,99]),R_train.max())

    
    ### Perform RKCV
    from sklearn.linear_model import Ridge #, RidgeCV
    from sklearn.model_selection import cross_val_score, RepeatedKFold
    #alphas = np.logspace(-6, -3, 25)  # Range from 0.000001 to 0.001
    alphas = np.logspace(-6, -4, 33)  # Range from 0.000001 to 0.0001
    alphas= alphas[:-4]
    
    print('Alphas2test', alphas)
    n_folds = 10
    n_repeats = 2 
    
    slope_by_month=[]
    rsq_by_month=[]
    #alb_crt_rate=0.05
    count=[]
    for imon in range(nmon_yr):
        #slope_map=[]
        rsq_map=[]
        for iy in range(nlat1):
            for ix in range(nlon1):
                yy1t= R_train[:,imon,iy,ix,:].copy().reshape(-1)
                xx1t= rfos_train[:,:,imon,iy,ix,:].copy().reshape([ncr,-1])
                    
                msidx1= yy1t.mask   
                if isinstance(msidx1, np.bool_) or np.logical_not(msidx1).sum()>=9:
                    if msidx1.sum()>0:
                        xx1t= xx1t[:,~msidx1]
                        yy1t= yy1t[~msidx1]

                    xx1t= xx1t.T
                    yy1t= yy1t.filled(np.nan)                    

                    print(xx1t)
                    print(yy1t); sys.exit()
                    cv_scores=[]
                    rkf_cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
                    for alp in alphas:
                        ridge = Ridge(alpha=alp,positive=True,fit_intercept=False,copy_X=True)
                        scores = cross_val_score(ridge, xx1t, yy1t, n_jobs=2, cv=rkf_cv, #LeaveOneOut(), 
                                                scoring='neg_mean_squared_error')                        
                        cv_scores.append(np.sqrt(-scores.mean()))
    
                    rsq_map.append(cv_scores)
                else:
                    #sl= [np.nan,]*ncr
                    #intercept= np.nan
                    rsq,rmse= np.nan,np.nan


        rsq_map= np.asarray(rsq_map)  ## [n_gridcell, n_alphas]
        mm_scores= rsq_map.mean(axis=0)
        mm_idx= np.argmin(mm_scores)
        print(imon, mm_idx, alphas[mm_idx], mm_scores)

        rsq_by_month.append(rsq_map)
    mm_scores= np.asarray(rsq_by_month).mean(axis=(0,1))
    mm_idx= np.argmin(mm_scores)
    print('Total', mm_idx, alphas[mm_idx], mm_scores)

    if True:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogx(alphas, mm_scores, 'b-', linewidth=2)
        plt.xlabel('Alpha (regularization strength)')
        plt.ylabel('Grid Mean Cross-validated RMSE')
        plt.title('Ridge Regression: CV Error vs Regularization Strength')
        plt.axvline(alphas[mm_idx], color='r', linestyle='--', 
           label=f'Optimal α = {alphas[mm_idx]:.3e}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        outdir= './Pics/'
        outfn= outdir+'Fig_RKCV.CV_Error_vs_Alphas.{}_pr{}.png'.format(rad_names[rad_idx],half_idx)
        plt.savefig(outfn,bbox_inches='tight',dpi=150) #
        #plt.show()
        #sys.exit()
        print(outfn)

    return

def degraded_sampling(arr,nt,nlat,nlon,hs1,delta1,lon_extend=True,is_rad=True):
    '''arr in shape of [nt, nlat*hs1+lat_delta+2, nlon*hs1]
    '''
    _,nlat0,nlon0= arr.shape
    if lon_extend:
        arr= np.pad(arr,((0,0),(0,0),(1,1)),mode='wrap')
        #print(arr.shape)
    arr_temp=[]
    for (ypad,xpad) in [(delta1,delta1),(delta1,0),(0,delta1),(delta1,delta1*2),(delta1*2,delta1)]:
        arr1= arr[:,ypad:ypad+nlat*hs1,xpad:xpad+nlon*hs1] #; print(ypad,xpad,arr.shape)
        arr1= arr1.reshape([nt,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt,nlat,nlon,hs1*hs1,1])
        if is_rad:
            arr_temp.append(arr1.mean(axis=-2))
        else:
            arr_temp.append(arr1.sum(axis=(0,3)))
        
    arr= np.ma.concatenate(arr_temp,axis=-1) #; print(arr.shape) #; sys.exit()
    return arr


if __name__=="__main__":
    
    #rad_idx,half_idx
    main(0,0)
    main(0,1)
    main(1,0)
    main(1,1)



