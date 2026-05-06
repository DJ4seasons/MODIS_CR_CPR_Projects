'''
Test CR-radiation kernel
Perform linear regression between R_all vs. RFOs 

: monthly and 4-deg resolution
: After normalizing R, regressed against all month RFO

Daeho Jin
2025.04.14
---

Train only for half of the period.
SW: Set albedo_surface (R_clr/Insol) as intercept after transformed to albedo
LW: Set OLR_clr as intercept

v2b: Apply non-negative least square function

2025.04.29
---

Subtract F0*CS_albedo from total albedo
2025.06.04
---

Apply Ridge regression
Train period is still set as 10 years, and training samples are obtained from 1-deg shifted 4-deg box
#Best_alpha= 6.31e-06 for SW, 2.0e-06 for LW

2025.06.17
---

Try with H1_tk and H2_tk + geodetic weight
2025.11.10
'''

import numpy as np
import sys
import os.path
from datetime import timedelta, date
import common_functions as cf

def rad_degrading_sampling(arr,nmon,nlat,nlon,hs1,delta1,lon_extend=True):
    '''arr in shape of [nmon, nlat*hs1+lat_delta+2, nlon*hs1]
    '''
    _,nlat0,nlon0= arr.shape
    if lon_extend:
        arr= np.pad(arr,((0,0),(0,0),(1,1)),mode='wrap')
        print(arr.shape)
    arr_temp=[]
    for (ypad,xpad) in [(delta1,delta1),(delta1,0),(0,delta1),(delta1,delta1*2),(delta1*2,delta1)]:
        arr1= arr[:,ypad:ypad+nlat*hs1,xpad:xpad+nlon*hs1] #; print(ypad,xpad,arr.shape)
        arr1= arr1.reshape([nmon,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nmon,nlat,nlon,hs1*hs1,1])
        arr_temp.append(arr1.mean(axis=-2))
        
    arr= np.ma.concatenate(arr_temp,axis=-1); print(arr.shape) #; sys.exit()
    return arr

def cr_degrading_sampling(arr,nt,nlat,nlon,hs1,delta1,lon_extend=True):
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
        arr_temp.append(arr1.sum(axis=(0,3)))
        
    arr= np.concatenate(arr_temp,axis=-1) #; print(arr.shape) #; sys.exit()
    return arr

def tcr_sampling(arr,deg2ind):
    '''arr in shape of [ncr,nmon, nlat*hs1+lat_delta+2, nlon*hs1]
    '''
    arr= arr.reshape(list(arr.shape)+[1,])
    arr_temp= [arr.sum(axis=(1,4)),]
    for d2i in deg2ind:
        arr_temp.append(arr[:,:,:,:,d2i,:].sum(axis=(1,4)))
    arr= np.ma.concatenate(arr_temp,axis=-1) #; print(arr.shape) #; sys.exit()
    return arr


def main():
 
    ###--- Parameters
    half_idx=1
    
    sat_nm= 'TAmean' #'CERES_FBCT-Day' #

    mdnm0= 'EBAF_4.2.1' #'FBCT_4.1'
    rad_names= ['SW','LW']
    rad_idx= 1
    rad_name= rad_names[rad_idx]+'_outgoing'

    # Pre-determined from RKCV
    if half_idx==0:
        alphas= [1.155e-06,1.0e-06]
    elif half_idx==1:
        alphas= [5.623e-06,1.155e-06] #
        
    best_alp= alphas[rad_idx]

    #tgt_crg= [0,1,2,3,4]
    cscf_crt= 5.
    rfo_crt= 1.
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
    
    #tgt_boxes= get_tgt_boxes()[:5]
    max_lat= 60 #
    lat_delta=1
    
    tshs=   ['monthly',4] #['monthly',5] #
    sn_names= ['All','SON','DJF','MAM','JJA']
    sn_idx= 0
    sn_name= sn_names[sn_idx]

    '''
    #vni= 1
    var_names= ['LTS (K)','EIS (K)','ECTEI (K)','ECF (%)', 'M (K)', ]
    var_names2= ['SST (K)','w700 (Pa/s)','w850 (Pa/s)','RH_free (%)','RH_2m (%)','Wspd_10m (m/s)','dT_sfc (K)','dq (g/kg)']
    vns= var_names+var_names2    
    vns= vns[:6]
    vni2= 5  ## SST
    vn_tit= '+'.join([vns[vi].split()[0] for vi in [vni,vni2]]) #vns[vni].split()[0]
    '''
    
    tgt_dates= (date(2002,9,1),date(2024,8,31))
    tgt_date_names= [d.strftime('%Y.%m') for d in tgt_dates]
    nmon= cf.get_tot_months(*tgt_dates)
    nmon_yr=12
    nyr= nmon//nmon_yr

    dr0,dr1= (date(2002,9,1),date(2012,8,31)),(date(2014,9,1),date(2024,8,31))
    nmon2= cf.get_tot_months(*dr0)
    nyr2= nmon2//nmon_yr
    if half_idx==0:
        train_dates, test_dates= dr0, dr1
        imon_tr,imon_te= cf.get_tot_months(tgt_dates[0],dr0[0])-1,cf.get_tot_months(tgt_dates[0],dr1[0])-1
    elif half_idx==1:
        train_dates, test_dates= dr1, dr0
        imon_tr,imon_te= cf.get_tot_months(tgt_dates[0],dr1[0])-1,cf.get_tot_months(tgt_dates[0],dr0[0])-1
    train_date_names= [d.strftime('%Y.%m') for d in train_dates]
    test_date_name= '-'.join([d.strftime('%Y') for d in test_dates])
    print(imon_tr, train_date_names)
    print(imon_te, test_date_name)

    #t_scale, hs1= tshs
    hs1= tshs[1]
    tgt_latlon1, tgt_rg_name1= [-max_lat-lat_delta,max_lat+lat_delta,-180,180], '{a}S-{a}N'.format(a=max_lat)
    nlat0,nlon0= (max_lat+lat_delta)*2,360
    nlat,nlon= max_lat*2//hs1,90
    
    latinfo, loninfo = (-max_lat+hs1/2,hs1,nlat), (-180+hs1/2,hs1,nlon)
    latlon_info= dict(latinfo=latinfo, loninfo=loninfo)
    lats= np.arange(nlat)*hs1+latinfo[0]
    lons= np.arange(nlon)*hs1+loninfo[0]
    xy= np.meshgrid(lons,lats)
    #lat_weight= cf.apply_lat_weight(np.ones([nlat,nlon]),nlat,nlon,lats).squeeze()
    #print(lat_weight[:,0])
    #lats_2d= np.ones([nlat,nlon],dtype=float)*lats[:,None]
    lats0= np.arange(nlat0)-nlat0/2+0.5
    lat_weight= cf.apply_lat_weight(np.ones([nlat0,nlon0,]),nlat0,nlon0,lats0,geodetic=True).squeeze()
    lat_weight= lat_weight[lat_delta:-lat_delta,:].reshape([nlat,hs1,nlon,hs1]).mean(axis=(1,3))

    
    ### For TOA CRE    
    var_names= ['toa_sw_all_mon','toa_sw_clr_c_mon','toa_lw_all_mon','toa_lw_clr_c_mon',]
    #in_data=[]
    #for vn in var_names[cre_idx*2:cre_idx*2+2]:
    #    rad0= cf.get_NRB_monthly(vn,tgt_dates,tgt_latlon1)
    #    in_data.append(rad0)
    vn= var_names[rad_idx*2]
    if rad_idx==0:
        rad0= cf.get_NRB_TOA_monthly('solar_mon',tgt_dates,tgt_latlon1) #*-1
        print( rad0.shape, rad0.min(), rad0.max(),) #rad0.min(), rad0.max(),
        '''
        rad0=rad0.reshape([nmon,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nmon,nlat,nlon,hs1*hs1]).mean(axis=-1)
        rad0_1= rad0[:120].mean(axis=0)
        rad0_2= rad0[-120:].mean(axis=0)
        rad0_diff= rad0_2-rad0_1
        print(rad0_diff.min(), rad0_diff.max(), np.percentile(rad0_diff.compressed(), [1,2,98,99])); sys.exit()
        rad0_gm= np.ma.average(rad0.reshape([nmon,nlat*nlon]),weights=lat_weight.reshape(-1),axis=1)
        rad0_gm1= rad0_gm[:120].mean()
        rad0_gm2= rad0_gm[-120:].mean()
        print(rad0_gm1, rad0_gm2, rad0_gm2-rad0_gm1); sys.exit()
        '''
        ## Degrading resolution
        rad0= rad_degrading_sampling(rad0,nmon,nlat,nlon,hs1,lat_delta,lon_extend=True)
        #rad0= rad0.reshape([nmon,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nmon,nlat,nlon,hs1*hs1]).mean(axis=-1)
        
    rad1= cf.get_NRB_TOA_monthly(vn,tgt_dates,tgt_latlon1) #*-1
    #rad1= rad0-rad1
    print( rad1.shape, rad1.min(), rad1.max(),) #rad0.min(), rad0.max(),
    vn2= var_names[rad_idx*2+1]
    rad2= cf.get_NRB_TOA_monthly(vn2,tgt_dates,tgt_latlon1) #*-1
    print( rad2.shape, rad2.min(), rad2.max()) #rad0.min(), rad0.max(),
    
    if rad1.mask.sum()+rad2.mask.sum()>0:
        sys.exit('Miss in Rad')
       
    ## Degrading resolution
    #rad1= rad1.reshape([nmon,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nmon,nlat,nlon,hs1*hs1]).mean(axis=-1)
    #rad2= rad2.reshape([nmon,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nmon,nlat,nlon,hs1*hs1]).mean(axis=-1)
    rad1= rad_degrading_sampling(rad1,nmon,nlat,nlon,hs1,lat_delta,lon_extend=True)
    rad2= rad_degrading_sampling(rad2,nmon,nlat,nlon,hs1,lat_delta,lon_extend=True)

    if rad_idx==0:
        rad1n= rad1/rad0    
        print('Rad_norm', rad1n.shape, rad1n.min(), rad1n.max(),)
        alb_CS= rad2/rad0
        print('Alb_CS',alb_CS.shape, alb_CS.min(), np.percentile(alb_CS.compressed(),[1,2,98,99]),alb_CS.max())
        diff= rad1n #-alb_CS
        rad1=rad2=0    
    elif rad_idx==1:        
        olr_clr= rad2
        diff= rad1/olr_clr        
        #rad1=0
    print('Diff.',diff.min(), np.percentile(diff.compressed(),[1,2,98,99]),diff.max())
    #sys.exit()
    
    ### Read CR set
    rg_set= dict(rg=50,nelemp=0,prwt=0,km=15)
    crmap= cf.read_cpr_map(rg_set,sat_nm,tgt_dates,tgt_latlon1)    
    print(crmap.shape)
    nt1,nlat1,nlon1= crmap.shape
    
    ## Degrading resolution
    #crmap= crmap.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
    #print(crmap.shape)
    if True: #np.any(np.asarray(tgt_cr)>rg_set['km']):
        crmap_sub= cf.read_cpr_map(rg_set,sat_nm,tgt_dates,tgt_latlon1,sub=True,tgt_cr=rg_set['km'],subk=3)
        #crmap_sub= crmap_sub.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
        print(crmap_sub.shape)
    else:
        crmap_sub=[]

    ## Read Total CF
    cfmap= cf.get_Total_CF_daily(tgt_dates,tgt_latlon1,sat_nm=sat_nm)
    print(cfmap.shape,cfmap.min(), cfmap.max())
    #cfmap= cfmap.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
    non_cs= cfmap>= cscf_crt/100. #
    cs_idx= np.logical_and(cfmap>-0.00001,cfmap< cscf_crt/100.)
    cfmap=0
    print(non_cs.sum(), cs_idx.sum())
    
    #rad0= rad0.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
    #rad1= rad1.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
    #rad_insol= rad_insol.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])

    
    ## Transform to monthly
    y0,m0= tgt_dates[0].year, tgt_dates[0].month
    y1,m1= tgt_dates[1].year, tgt_dates[1].month
    print(y0,m0,y1,m1)
    by_month,by_month_rad0,by_month_rad1=[],[],[]
    by_month_zero=[]
    month_indicator=[]
    for yy in range(y0,y1+1,1):
        im=m0 if yy==y0 else 1
        em=m1 if yy==y1 else 12
        print(yy,im,em)
        for mm in range(im,em+1,1):
            it= (date(yy,mm,1)-tgt_dates[0]).days
            yy1,mm1=yy,mm+1
            if mm1>12: yy1+=1; mm1-=12
            et= (date(yy1,mm1,1)-tgt_dates[0]).days
            tmp= crmap[it:et,:]
            if len(crmap_sub)>0: tmp_sub= crmap_sub[it:et,:]
            miss_idx= (tmp==-1)
            tsz= (et-it)*hs1*hs1-cr_degrading_sampling(miss_idx,et-it,nlat,nlon,hs1,lat_delta,lon_extend=True)
            #print(tsz.shape, tsz.min(), tsz.max()) #[nlat,nlon,5]
            tsz[tsz==0]=1
                        
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
                idx_sum= cr_degrading_sampling(idx_all,et-it,nlat,nlon,hs1,lat_delta,lon_extend=True)
                by_tcr.append(idx_sum/tsz)

            tmp_rfos= np.array(by_tcr)  #[ncr,nlat,nlon,5]
            
            cs_idx_1mo= cr_degrading_sampling(cs_idx[it:et,:],et-it,nlat,nlon,hs1,lat_delta,lon_extend=True)
            rfo_zero= cs_idx_1mo/tsz
            #print(tmp_rfos.min(), tmp_rfos.max(), rfo_zero.min(), rfo_zero.max())
            #print(tmp_rfos.shape, rfo_zero.shape); sys.exit()
            '''
            tmp_rad1 =[]
            for y1p in range(nlat):
                for x1p in range(nlon):
                    tmp_idx= idx_all[:,y1p,x1p,:]
                    if tmp_idx.sum()>5:                        
                        tmp_rad1.append(np.nanmean(rad1[it:et,y1p,x1p,:][tmp_idx].filled(np.nan)))
                    else:                        
                        tmp_rad1.append(np.nan)
            tmp_rad1= np.asarray(tmp_rad1).reshape([nlat,nlon])
            '''
            by_month.append(tmp_rfos)
            #by_month_rad1.append(tmp_rad1)
            #by_month_rad1.append(np.ma.masked_invalid(tmp_rad1))
            #by_month_insol.append(tmp_insol)
            by_month_zero.append(rfo_zero)
                
    rfos= np.asarray(by_month).swapaxes(0,1) #*100  ## Now in percent
    print(rfos.shape, rfos.min(), rfos.max(),rfos[:,4::12,:].mean(axis=(1,2,3))) #; sys.exit() #[ncr,nmon,nlat,nlon,3]
    rfo0= np.asarray(by_month_zero); print(rfo0.shape,rfo0.min(), rfo0.max(),rfo0[4::12,:].mean()) #; sys.exit()
    nd2i= rfo0.shape[-1]
    
    '''
    clim_rfos= rfos.reshape([ncr,nyr,nmon_yr,nlat,nlon])[:,:-test_years,:,:,:].mean(axis=1)    
    #rfos_mask= clim_rfos<rfo_crt #np.logical_or(clim_rfos<rfo_crt,w500_mask[None,:])
    #clim_rfos= np.ma.masked_array(clim_rfos,mask=rfos_mask)    
    clim_rfos= np.ma.masked_less(clim_rfos,rfo_crt)
    rfo_mask= clim_rfos.mask.sum(axis=0)==ncr
    
    R_all= rad1 #np.ma.asarray(by_month_rad1)
    
    clim_R_all= R_all.reshape([nyr,nmon_yr,nlat,nlon])[:-test_years,:,:,:].mean(axis=0)
    clim_R_all.mask= np.logical_or(clim_R_all.mask, rfo_mask)
    
    ### Deseasonalizing
    #rfo_ano= cf.monthly_deseasonalizing(rfo,nmon_yr)
    rfos_ano= rfos.reshape([ncr,nyr,nmon_yr,nlat,nlon])-clim_rfos[:,None,:]
    print(rfos_ano.shape, rfos_ano.min(), rfos_ano.max(),rfos_ano[:,:,4,:].mean(axis=(1,2,3)))
    #R_all_ano= cf.monthly_deseasonalizing(cre,nmon_yr)
    R_all_ano= R_all.reshape([nyr,nmon_yr,nlat,nlon])-clim_R_all[None,:]
    print(R_all_ano.shape, R_all_ano.min(), R_all_ano.max())
    #ts_ano= cf.monthly_deseasonalizing(ts.reshape([nmon,1]),nmon_yr).squeeze()

    #rfo_std= np.sqrt((rfos_ano**2).mean(axis=1))
    #rfos_mask= rfo_std<rfo_crt
    #rfo_mask= rfos_mask.sum(axis=0)==ncr
    #rfos_ano.mask= np.logical_or(rfos_ano.mask,rfos_mask[:,None,:])
    #R_all_ano.mask= np.logical_or(R_all_ano.mask, rfo_mask)
    '''

    rfos_train, rfos_test= rfos[:,imon_tr:imon_tr+nmon2,:].reshape([ncr,nyr2,nmon_yr,nlat,nlon,nd2i]), rfos[:,imon_te:imon_te+nmon2,:].reshape([ncr,nyr2,nmon_yr,nlat,nlon,nd2i])
    rfo0_train, rfo0_test= rfo0[imon_tr:imon_tr+nmon2,:].reshape([nyr2,nmon_yr,nlat,nlon,nd2i]), rfo0[imon_te:imon_te+nmon2,:].reshape([nyr2,nmon_yr,nlat,nlon,nd2i])
    R_train, R_test= diff[imon_tr:imon_tr+nmon2,:].reshape([nyr2,nmon_yr,nlat,nlon,nd2i]), diff[imon_te:imon_te+nmon2,:].reshape([nyr2,nmon_yr,nlat,nlon,nd2i])
    if rad_idx==0:
        rad0_train,rad0_test= rad0[imon_tr:imon_tr+nmon2,:].reshape([nyr2,nmon_yr,nlat,nlon,nd2i]), rad0[imon_te:imon_te+nmon2,:].reshape([nyr2,nmon_yr,nlat,nlon,nd2i])
        alb_CS_train= alb_CS[imon_tr:imon_tr+nmon2,:].reshape([nyr2,nmon_yr,nlat,nlon,nd2i])
        alb_CS_test= alb_CS[imon_te:imon_te+nmon2,:].reshape([nyr2,nmon_yr,nlat,nlon,nd2i])
        R_train, R_test= R_train-rfo0_train*alb_CS_train, R_test-rfo0_test*alb_CS_test
        alb_CS_train_clim= alb_CS_train.mean(axis=0)
    elif rad_idx==1:
        R_train, R_test= R_train-rfo0_train, R_test-rfo0_test
        olr_clr_train_clim= olr_clr[imon_tr:imon_tr+nmon2,:].reshape([nyr2,nmon_yr,nlat,nlon,nd2i]).mean(axis=0)
        print('R-train',R_train.min(), np.percentile(R_train.compressed(),[1,2,98,99]),R_train.max())
        rad0_train,rad0_test= olr_clr[imon_tr:imon_tr+nmon2,:].reshape([nyr2,nmon_yr,nlat,nlon,nd2i]), olr_clr[imon_te:imon_te+nmon2,:].reshape([nyr2,nmon_yr,nlat,nlon,nd2i])
        alb_CS_train,alb_CS_test= np.ones_like(R_train), np.ones_like(R_test)
        alb_CS_train_clim= alb_CS_train.mean(axis=0)
        
    ## Screend out by clim_RFO
    #for icr in range(ncr):
    #    clim_rfo= rfos_train[icr,:].mean(axis=0)
    #    rfos_train[icr,:,clim_rfo<rfo_crt/100]=0.    

    crfo_valid= rfos_train.mean(axis=1)>rfo_crt/100  ##[ncr,nmon_yr,nlat,nlon]
    
    ## Calc slope
    #from sklearn.linear_model import LinearRegression
    from scipy.optimize import nnls
    from sklearn.linear_model import Ridge
    
    slope_by_month=[]
    rsq_by_month=[]
    #alb_crt_rate=0.05
    count=[]
    '''
    if rad_idx==0:
        scr_clim= alb_CS_train_clim
        alb_crt_min= 0.01
    elif rad_idx==1:
        scr_clim= olr_clr_train_clim
        alb_crt_min= 10
    ''' 
    for imon in range(nmon_yr):
        #astc1= scr_clim[imon,:] 
        slope_map=[]
        rsq_map=[]
        for iy in range(nlat):
            for ix in range(nlon):
                yy1= R_train[:,imon,iy,ix,:]
                xx1= rfos_train[:,:,imon,iy,ix,:]
                #alb1= astc1[iy,ix]
                #alb_crt= max(alb1*alb_crt_rate,alb_crt_min)
                cv= crfo_valid[:,imon,iy,ix]
                
                yy_add, xx_add= [],[]
                '''
                ## Check albedo of neighbor
                for iy0 in [iy-1,iy+1]:
                    if iy0>=0 and iy0<nlat:
                        if abs(astc1[iy0,ix]-alb1)<alb_crt:
                            yy_add.append(R_train[:,imon,iy0,ix])
                            xx_add.append(rfos_train[:,:,imon,iy0,ix])
                for ix0 in [ix-1,ix+1]:
                    if ix0<0: ix0+=nlon
                    elif ix0>=nlon: ix0-=nlon
                    if abs(astc1[iy,ix0]-alb1)<alb_crt:
                        yy_add.append(R_train[:,imon,iy,ix0])
                        xx_add.append(rfos_train[:,:,imon,iy,ix0])
                ## Check adjacent month
                if len(yy_add)<2:
                    for im0 in [imon-1, imon+1]:
                        if im0<0: im0+= nmon_yr
                        elif im0>=nmon_yr: im0-= nmon_yr
                        if abs(scr_clim[im0,iy,ix]-alb1)<alb_crt:
                            yy_add.append(R_train[:,im0,iy,ix])
                            xx_add.append(rfos_train[:,:,im0,iy,ix])
                
                count.append(len(yy_add))

                if len(yy_add)>0:
                    yy1t= np.ma.concatenate([yy1,]*len(yy_add)*3+yy_add)
                    xx1t= np.concatenate([xx1,]*len(yy_add)*3+xx_add,axis=1)
                    #print(yy.shape, xx.shape); sys.exit()
                else:
                '''
                if True:
                    yy1t= yy1[:,:nd2i].copy().reshape(-1)
                    xx1t= xx1[:,:,:nd2i].copy().reshape([ncr,-1])
                if False:
                    #yy1t= yy1[:,0:1].copy().reshape(-1) #yy1[:,1:nd2i].copy().reshape(-1)
                    #xx1t= xx1[:,:,0:1].copy() #.reshape([ncr,-1]) #xx1[:,:,1:nd2i].copy().reshape([ncr,-1])

                    if np.any(cv==False):
                        xx1t[~cv[:,0],:]=0.
                    xx1t= xx1t.reshape([ncr,-1])
                    
                msidx1= yy1t.mask   
                if isinstance(msidx1, np.bool_) or np.logical_not(msidx1).sum()>=9:
                    if msidx1.sum()>0:
                        #xx1a,xx1b= xx1a[:,~msidx1], xx1b[~msidx1]
                        xx1t= xx1t[:,~msidx1]
                        yy1t= yy1t[~msidx1]

                    xx1t= xx1t.T
                    yy1t= yy1t.filled(np.nan)                    
                    #yy1t[yy1t<0]= 0.
                    
                    #lr= LinearRegression(fit_intercept=False)  
                    #lr.fit(xx1t,yy1t)
                    #sl,intercept= lr.coef_,alb1 #lr.intercept_
                    #rsq= lr.score(xx1t,yy1t)
                    
                    #sl,rnorm= nnls(xx1t,yy1t)
                    #intercept= 0 #alb1

                    ridge = Ridge(alpha=best_alp,positive=True,fit_intercept=False,copy_X=True)
                    ridge.fit(xx1t,yy1t)
                    sl,intercept= ridge.coef_, 0

                    #rsq= ridge.score(xx1t,yy1t)
                    rsq= 1-np.sum((yy1[:,0]-(xx1[:,:,0].T @ sl))**2)/np.sum((yy1[:,0]-yy1[:,0].mean())**2)
                    #print(rsq); sys.exit()
                    
                else:
                    sl= [np.nan,]*ncr
                    intercept= np.nan
                    rsq,rmse= np.nan,np.nan

                slope_map.append([*sl,])  #intercept
                rsq_map.append(rsq)
                    
        '''
        for v,c in zip(*np.unique(count,return_counts=True)):
            print(imon,v+1,c)
        ind= np.where(np.asarray(count)==0)[0]
        print(ind,np.floor(ind/nlon))
        count=[]
        '''

        ### Slopes
        #slopes= np.full([nlat,nlon,ncr+1],np.nan)
        #slopes[valid,:]= slope_map
        slopes= np.asarray(slope_map).reshape([nlat,nlon,ncr]) #+1
        print(np.nanmin(slopes), np.nanmax(slopes)) #; sys.exit()
        for k in range(ncr): #+1
            sl1= slopes[:,:,k]
            msidx= np.isnan(sl1)
            print(k,msidx.sum(),np.nanmin(sl1),np.nanmax(sl1),np.percentile(sl1[~msidx],[1,2,98,99]))
        #sys.exit()
        slopes= np.ma.masked_invalid(slopes)
        slope_map= 0
        slope_by_month.append(slopes)

        ### R^2 and RMSE measure
        rsqs= np.full([nlat,nlon,4],np.nan)
        rsqs[:,:,0]= np.asarray(rsq_map).reshape([nlat,nlon])

        ## Mean and RMSE from train data
        yy_train= R_train[:,imon,:,:,0].copy()
        xx_train= rfos_train[:,:,imon,:,:,0]
        
        yy_train+= rfo0_train[:,imon,:,:,0]*alb_CS_train[:,imon,:,:,0]
        
        y_pred=0
        for isl,x1 in enumerate(xx_train): #,xx1b
            y_pred+= slopes[None,:,:,isl].filled(0.)*x1 #.filled(0.)
        #y_pred+= slopes[None,:,:,-1]        
        y_pred+= rfo0_train[:,imon,:,:,0]*alb_CS_train_clim[None,imon,:,:,0]
        #print(y_pred.mean(axis=(1,2)))
        if True: #rad_idx==0:
            yy_train*=rad0_train[:,imon,:,:,0] #.mean(axis=0)[None,:]
            y_pred*=rad0_train[:,imon,:,:,0].mean(axis=0)[None,:] ## de-normalized
        #print(y_pred.mean(axis=(1,2))); sys.exit()
        
        rmse= np.sqrt(np.ma.mean((yy_train-y_pred)**2,axis=0)).filled(np.nan)
        mm_diff= (y_pred-yy_train).mean(axis=0)
        rsqs[:,:,1]= mm_diff
        rsqs[:,:,2]= rmse
        print('mm_diff',imon, np.average(mm_diff,weights=lat_weight)) #, np.average(yy_train.mean(axis=0),weights=lat_weight), np.average(y_pred.mean(axis=0),weights=lat_weight))
        print('r2',imon,rsqs[:,:,0].min(),rsqs[:,:,0].max(),rsqs[:,:,0].mean(),)
        #sys.exit()
        ## RMSE from test data
        yy_test= R_test[:,imon,:,:,0].copy() #*rad0_test[:,imon,:] #[None,:,:]  ## de-normalized
        xx_test= rfos_test[:,:,imon,:,:,0]
        
        yy_test+= rfo0_test[:,imon,:,:,0]*alb_CS_test[:,imon,:,:,0]
        
        y_pred=0
        for isl,x1 in enumerate(xx_test): #,xx1b
            y_pred+= slopes[None,:,:,isl].filled(0.)*x1 #.filled(0.)
        #y_pred+= slopes[None,:,:,-1]
        y_pred+= rfo0_test[:,imon,:,:,0]*alb_CS_train_clim[None,imon,:,:,0]

        if True: #rad_idx==0:
            yy_test*=rad0_test[:,imon,:,:,0] #.mean(axis=0)[None,:]
            y_pred*=rad0_test[:,imon,:,:,0].mean(axis=0)[None,:] ## de-normalized
        
        rmse= np.sqrt(np.ma.mean((yy_test-y_pred)**2,axis=0)).filled(np.nan)
        #print(np.nanmin(rmse), np.nanmax(rmse),rmse.max(),np.isnan(rmse).sum()) #; sys.exit()
        rsqs[:,:,3]= rmse
            
                    
        rsqs= np.ma.masked_invalid(rsqs)
        rsq_by_month.append(rsqs)
        
    slope_by_month= np.ma.asarray(slope_by_month)
    rsq_by_month= np.ma.asarray(rsq_by_month)
    print(slope_by_month.min(axis=(0,1,2)))
    print(slope_by_month.max(axis=(0,1,2)))

    slope_by_month= np.ma.concatenate((slope_by_month,alb_CS_train_clim[:,:,:,0:1]),axis=-1)
    print(slope_by_month.shape)
    ncr= ncr+1
    
    ### For Figure    
    #tshs_nm= '{} {}\u00B0 vs. {}\u00B0'.format(tsnm,hs1,hs2)
    #tshs_fn= '+'.join([nm1.split()[0][0]+nm1.split()[1][0] for nm1 in [comp_nm,]])
    tshs_fn= '{}_{}deg'.format(*tshs)
    rg_nm= tgt_rg_name1

    ## Save array
    if False: #True: #
        outdir= './Data2/'
        nm0= 'v{}RFOs_half{}_v2b_m2r3'.format(ncr,half_idx)
        dim_txt= 'x'.join([str(val) for val in slope_by_month.shape])
        
        outfn= outdir+'{}_kernel4{}_in{}.{}.{}.f32dat'.format('CR_groups',rad_name,rg_nm,nm0,dim_txt)
        with open(outfn,'bw') as fout:
            slope_by_month.filled(-999.9).astype(np.float32).tofile(fout)
        print(outfn)
        
    unit_nm= ['',]+[r'($Wm^{-2}$)',]*3
    unit_nm+= ['',]*ncr
    #if rad_idx==0: 
    #elif rad_idx==1:
    #    um2= ['(\u00D7100 '+r'$Wm^{-2}$)',]*ncr
    #    unit_nm= unit_nm+um2
    #print(len(unit_nm),type(unit_nm),unit_nm); sys.exit()
        
    #ts_nm, ts_fn= r'$T_s (K)$', 'Ts'
    outdir= '../../Writing_TOA_Rad_Trend+CR/Pics/'
    pn_tit0= [r'$R^2$','Mean Diff.','RMSE','RMSE in {}'.format(test_date_name)]+['Slope for {}'.format(crn) for crn in cr_group_names]+['Mean CS Albedo']
    mon_names= ['Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug']
    suptit_tail= '[{}, {}-{}, Ridge_\u03B1={:.2e}]'.format(mdnm0,*train_date_names,best_alp)
    for mm in range(1,nmon_yr,3): #[10,]: #
        suptit= '{} Regression against RFO in {}\n{}'.format(rad_name,mon_names[mm],suptit_tail) #, \u03C9{}>{} ,r'$_{500}$',w500_crt) ', {}>{:.0f}%'.format(r'$RFO_{clim}$',rfo_crt)
    
        outfn= outdir+'vR1_Fig.Pred_R_by{}RFO_test_byEBAF_v2b_m2r3.sl+rsq.{}_in{}.{}.{}-{}_{}.z.png'.format(
            ncr,rad_name,rg_nm,tshs_fn,*train_date_names,mon_names[mm])
    
    
        pic_data= dict(data_A=[slope_by_month[mm,:,:,k] for k in range(ncr)],
                       data_B=[rsq_by_month[mm,:,:,k] for k in range(4)],
                       #rfo_std= [rfo_std[mm,:] for mm in range(1,nmon_yr,3)],
                       pn_tit0=pn_tit0 ,#pn_tit1=pn_tit1,
                       xy=xy, unit_nm= unit_nm,
                       lw=lat_weight,
                       suptit=suptit, outfn=outfn, )
        plot_main(pic_data)
    #sys.exit()
        
    return


#import plot_common as pcf
import matplotlib as mpl
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
import cartopy.crs as ccrs
from cartopy.feature import LAND
def plot_main(pdata):
    xy= pdata['xy']
    data_A,data_B = pdata['data_A'], pdata['data_B']
    #rfo_std,lw= pdata['rfo_std'],pdata['lw']
    pn_tit0= pdata['pn_tit0'] #, pdata['pn_tit1']
    unit_nm= pdata['unit_nm']
        
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'    
    
    ###---
    fig=plt.figure()
    fig.set_size_inches(7.6,9.) #(10.6,9.)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=16,y=0.975,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    ncol,nrow=2,4
    lf,rf,bf,tf=0.05,0.95,0.1,0.925
    gapx, npnx=0.06,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.07,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    
    ix=lf; iy=tf


    lon_ext= [20,360+20]
    cm = (lon_ext[0]+lon_ext[1])/2 #180
    #map_proj = ccrs.Robinson(central_longitude=cm)
    map_proj= ccrs.PlateCarree(central_longitude=cm)
    data_crs= ccrs.PlateCarree()

    
    ## Find data range
    dr1= np.ma.asarray(data_A).compressed()
    #if pdata['suptit'][0]=='L': dr1/=100.
    #dr2= np.ma.asarray(data_B).compressed()
    dr1= np.percentile(dr1[dr1!=0.],[2,98])
    #dr2= np.percentile(dr2,[5,95])
    print(np.round(dr1,3),) #np.round(dr2,3),)
    
    ## Col1
    #vm= max(abs(dr1[0]),abs(dr1[1]),1.21) if pdata['suptit'][0]=='L' else max(abs(dr1[0]),abs(dr1[1]),1.01)
    if pdata['suptit'][0]=='S':
        vmin,vmax= 0,0.802 #dr1[0],dr1[1] #np.floor((dr1[0])/5)*5,np.ceil((dr1[1])/5)*5
        cmap0= mpl.colormaps['YlOrRd'] #'RdBu_r'] #'RdYlGn_r'] #'PuOr_r'] #'RdYlBu_r']
    elif pdata['suptit'][0]=='L':
        vmin,vmax= 0.3,1.305
        cmap0= mpl.colormaps['YlGnBu'] #'RdBu_r'] #'RdYlGn_r'] #'PuOr_r'] #'RdYlBu_r']    
    props_mesh0= dict(cmap=cmap0,alpha=0.75,vmin=vmin,vmax=vmax,transform=data_crs)


    ## Col2
    vmin,vmax= 0.1,0.9 #dr2[1] #np.floor((dr3[0])/5)*5,np.ceil((dr3[1])/5)*5
    #if vmax-vmin<0.0031:
    #    vc= (vmin+vmax)/2
    #    vmin, vmax= vc-0.00155, vc+0.00155
    cmap1= mpl.colormaps['plasma'] #'RdYlGn_r'] #'PuOr_r'] #'RdYlBu_r']
    props_mesh1= dict(cmap=cmap1,alpha=0.86,vmin=vmin,vmax=vmax,transform=data_crs)

    ## Col3
    vmin,vmax= -3,3 #dr2[1] #np.floor((dr3[0])/5)*5,np.ceil((dr3[1])/5)*5
    #if vmax-vmin<0.0031:
    #    vc= (vmin+vmax)/2
    #    vmin, vmax= vc-0.00155, vc+0.00155
    cmap2= mpl.colormaps['RdYlBu_r'] #'RdYlGn_r'] #'PuOr_r'] #'RdYlBu_r']
    #cmap2.set_bad('0.9')
    props_mesh2= dict(cmap=cmap2,alpha=0.75,vmin=vmin,vmax=vmax,transform=data_crs)

    ## Col4
    vmin,vmax= 0.9,11.1 #dr2[1] #np.floor((dr3[0])/5)*5,np.ceil((dr3[1])/5)*5
    #if vmax-vmin<0.0031:
    #    vc= (vmin+vmax)/2
    #    vmin, vmax= vc-0.00155, vc+0.00155
    cmap3= mpl.colormaps['viridis_r'] #'RdYlGn_r'] #'PuOr_r'] #'RdYlBu_r']
    #cmap2.set_bad('0.9')
    props_mesh3= dict(cmap=cmap3,alpha=0.75,vmin=vmin,vmax=vmax,transform=data_crs)

    '''
    ccb= np.arange(5,26,5)
    props_contour= dict(alpha=0.7,colors='0.1',linewidths=1,transform=data_crs)
    
    meanprops= dict(marker='x', markeredgecolor='k',
                        markerfacecolor='k', markersize=6,
                        markeredgewidth=1.2 )
    props= dict(showfliers=False,widths=0.7,showmeans=True,meanprops=meanprops,medianprops=dict(color='k'))
    '''
    
    ai=0
    for k,(data2,props) in enumerate(zip(data_B,[props_mesh1,props_mesh2,props_mesh3,props_mesh3])):
        ax2=fig.add_axes([ix,iy-ly,lx,ly], projection=map_proj)
        ax2.set_extent(lon_ext+[-61,61],data_crs)
        pic2= ax2.pcolormesh(*xy,data2,shading='nearest',**props)
        #rstd1= rfo_std[k]
        #pic2= ax1.contour(*xy,rstd1,ccb,**props_contour)
        #ax1.clabel(pic2,levels=ccb[1::2],fontsize=8)

        data2.mask= np.logical_or(data2.mask, data2==0.)
        q1,q3= np.percentile( data2.compressed() ,[25,75])
        mtxt= '1Q,3Q={:.3f}, {:.3f}'.format(q1,q3) if q3<1 else '1Q,3Q={:.2f}, {:.2f}'.format(q1,q3)
        #subtit= "({}) {} in {} ".format(abc[ai],pn_tit0[0],pn_tit1[k]); ai+=1
        subtit= "({}) {} ({})".format(abc[ai],pn_tit0[ai],mtxt); ai+=1
        ax2.set_title(subtit,fontsize=12,x=0,ha='left')

        right_label= True if ix+lx>=rf-0.01 else False
        map_common(ax2,data_crs,right_label=right_label,lon_ext=lon_ext)
        #ax2.add_feature(LAND,facecolor='0.8') #'#bfbfbf')

        #wt= (lw*rfo_std[k]).filled(0)
        #wMean= np.average(data2.compressed(),weights=wt[~data2.mask])
        #mtxt= mm_txt(data2.filled(np.nan))
        #ax2.text(4+lon_ext[0],58,mtxt,
        #         ha='left',va='top',c='k',fontsize=9,stretch='condensed',
        #         transform=data_crs,weight='semibold')

        ## Color bar for column 1&2
        vmin,vmax= props['vmin'],props['vmax']
        if vmax<1: lns= 0.1
        elif vmax<5: lns=1
        else: lns=2 #int((vmax-vmin)/60)*10    if lns==0: lns=5
    
        hh=0.016
        loc0= [ix,iy-ly-gapy*0.8,lx,hh]# [ix,iy-ly-gapy*0.65,lx,hh] #
        #tt= np.arange(vmin*10,vmax*10+0.01,lns*10,dtype=int)/10. if lns<1 else np.arange(vmin,vmax+0.01,lns,dtype=int)
        tt= np.arange(np.floor(vmin/lns),np.ceil(vmax/lns)+0.01,1,dtype=int)*lns
        tt2= tt.astype(np.float16) #['{}'.format(val) if i%2==0 else '' for i,val in enumerate(tt)]
        cb0 =draw_colorbar(fig,pic2,loc0,ft=9,extend='both',tt=tt,tt2=tt2)
        #vnm_unit= ''.join(xx_name.split()[1:])[1:-1] #r'$W/m^2$' #
        xlab= unit_nm[ai-1]
        cb0.ax.set_xlabel(xlab,fontsize=10,labelpad=2) #,x=1,ha='right') #,rotation=-90,va='bottom')
        cb0.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
        ix+= lx+gapx
        if ix+lx>rf:
            ix=lf
            iy-= ly+gapy*2

    
    for k,data1 in enumerate(data_A):
        #if pdata['suptit'][0]=='L': data1/=100.
        data1.mask= np.logical_or(data1.mask, data1==0.)
        q1,q3= np.percentile(data1.compressed(),[5,95])
        print(k,data1.min(), data1.max())
        #mtxt= r'$5^{th}pct=$'+'{:.2f}\n'.format(q1)
        #mtxt+= r'$95^{th}pct=$'+'{:.2f}'.format(q3)
        mtxt= r'$5^{th}, 95^{th}=$'+'{:.2f}, {:.2f}'.format(q1,q3)
        
        ax1=fig.add_axes([ix,iy-ly,lx,ly], projection=map_proj)
        ax1.set_extent(lon_ext+[-61,61],data_crs)
        pic1= ax1.pcolormesh(*xy,data1,shading='nearest',**props_mesh0)
        #rstd1= rfo_std[k]
        #pic2= ax1.contour(*xy,rstd1,ccb,**props_contour)
        #ax1.clabel(pic2,levels=ccb[1::2],fontsize=8)
        
        #subtit= "({}) {} in {} ".format(abc[ai],pn_tit0[0],pn_tit1[k]); ai+=1
        subtit= "({}) {} ({})".format(abc[ai],pn_tit0[ai],mtxt); ai+=1
        ax1.set_title(subtit,fontsize=12,x=0,ha='left')

        right_label= True if ai%ncol==0 else False
        map_common(ax1,data_crs,right_label=right_label,lon_ext=lon_ext)
        #ax1.add_feature(LAND,facecolor='0.8') #'#bfbfbf')

        #ax1.text(4+lon_ext[0],58,mtxt,
        #         ha='left',va='top',c='k',fontsize=9,stretch='condensed',
        #         transform=data_crs,weight='semibold')
            
        ix+= lx+gapx
        if ix+lx>rf:
            ix=lf
            iy-= ly+gapy
        if pdata['suptit'][0]=='L' and k==len(data_A)-2:
            break
    
    ## Color bar for column 0
    vmin,vmax= props_mesh0['vmin'],props_mesh0['vmax']
    if vmax<1.001:
        lns= 0.1 #int((vmax-vmin)/60)*10    if lns==0: lns=5
    else:
        lns= 0.2
    ext='max'
    
    hh=0.016
    if ix==lf:
        loc0= [0.15,iy+gapy*0.2,0.7,hh]# [ix,iy-ly-gapy*0.65,lx,hh] #
        iy-= gapy
    else:
        loc0= [ix,iy-ly+hh,lx,hh] #
        ix+= lx+gapx
        if ix+lx>rf:
            ix=lf
            iy-= ly+gapy
        
    #tt= np.arange(vmin*10,vmax*10+0.01,lns*10,dtype=int)/10. if lns<1 else np.arange(vmin,vmax+0.01,lns,dtype=int)
    tt= np.arange(np.floor(vmin/lns),np.ceil(vmax/lns)+0.01,1,dtype=int)*lns
    tt2= tt.astype(np.float16) #['{}'.format(val) if i%2==0 else '' for i,val in enumerate(tt)]
    cb0 =draw_colorbar(fig,pic1,loc0,ft=9,extend=ext,tt=tt,tt2=tt2)
    #vnm_unit= ''.join(xx_name.split()[1:])[1:-1] #r'$W/m^2$' #
    xlab= unit_nm[ai-1]
    cb0.ax.set_xlabel(xlab,fontsize=10,labelpad=2) #,x=1,ha='right') #,rotation=-90,va='bottom')
    cb0.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    
    

        
    ###---
    print(pdata['outfn'])
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150) #
    #plt.show()
    
    return

def mm_txt(amap):
    miss_idx= np.isnan(amap)
    tmp= amap[~miss_idx]    
    q1,q3= np.percentile(tmp,[25,75])
    mm1= tmp[tmp<q1].mean()
    mm3= tmp[tmp>q3].mean()
    txt1= '1Q_mm={:.3f}'.format(mm1) if abs(mm1)<1 else '1Q_mm={:.2f}'.format(mm1)
    txt3= '3Q_mm={:.3f}'.format(mm3) if abs(mm3)<1 else '3Q_mm={:.2f}'.format(mm3)
    txt= '{}\n{}'.format(txt1,txt3)    
    return txt

def map_common(ax,data_crs,right_label=False,lon_ext=[0,360]):
    #ax.set_extent([0,359.9,-24.1,24.1],data_crs)

    ax.coastlines(color='0.1',linewidth=1.) #'silver'
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
    label_idx=[False,False,False,True] #[True,True,False,True]
    gl.top_labels = label_idx[2]
    gl.left_labels = label_idx[0]
    gl.right_labels = label_idx[1]
    gl.bottom_labels = label_idx[3]
    gl.ylocator = MultipleLocator(20)
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 9, 'color': 'k'}
    gl.ylabel_style = {'size': 9, 'color': 'k'}

    ax.set_aspect('auto')
    for lt in range(-60,61,30):
        #ax.yaxis.set_major_locator(MultipleLocator(20))
        #ax.yaxis.set_major_formatter(FuncFormatter(cf.lat_formatter))
        #ax.tick_params(axis='y',which='major',labelsize=10)
        #ax.tick_params(left=True,right=True)
        ax.text(lon_ext[0]+0.01,lt,cf.lat_formatter(lt,0)+' ',ha='right',va='center',fontsize=9,c='k',transform=data_crs)
        if right_label:
            ax.text(lon_ext[1]-0.01,lt,' '+cf.lat_formatter(lt,0),ha='left',va='center',fontsize=9,c='k',transform=data_crs)

    tgt_boxes= get_tgt_boxes()
    #draw_box(ax,tgt_boxes,data_crs,ls='-',c='c')
    return
def draw_box(ax1,tgt_boxes,data_crs,ls='-',c='0.5'):
    alp=1
    for abox in tgt_boxes:
        y1,y2,x1,x2= abox[1]

        ax1.plot([x1,x1],[y1,y2],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
        ax1.plot([x2,x2],[y1,y2],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
        ax1.plot([x1,x2],[y1,y1],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
        ax1.plot([x1,x2],[y2,y2],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
    return

def get_tgt_boxes():
    tgt_boxes_tk= [ ('Peruvian',(-20,-12,-88,-80)), #-92,-84)), #-90,-80)),
                    ('Namibian',(-20,-12,0,8)), #-4,4)),
                    ('Californian',(24,32,-132,-124)), #-136,-128)),
                    ('Australian',(-36,-28,100,108)), #96,104)),

                    #('Azores',(40,50,-25,-15)),
                    #('SW.IndOce',(-52,-44,28,36)),

                    ('S.Pacifc',(-56,-48,-176,-168)),
                    #('SE.Atlantic',(-56,-48,4,12)),
                    #('N.Pacific',(48,56,172,180)),

                    #('Canarian',(15,25,-35,-25)),
                    #('China',(20,105)),
                    #('N.Pacific',(45,55,170,180)),  ## Modified
                    #('N.Atlantic',(50,60,-45,-35)),
                    #
    ]
    return tgt_boxes_tk


def draw_colorbar(fig,pic1,loc,ft=10,extend='both',tt=[],tt2=[]): #max_vals=[0,1],val_lin=0.02,unit=''):
    #tt=[0.1,0.3,1,3,10,30]
    #tt= np.arange(np.ceil(max_vals[0]*10)/10, max_vals[1]+val_lin/2, val_lin)
    #tt2=['{:.01f}{}'.format(x,unit) for x in tt] if val_lin<1 else ['{:.0f}{}'.format(x,unit) for x in tt]

        ###- Get position from previous subplot
#        pos1=ax1.get_position().bounds  ##<= (left,bottom,width,height)
#        cb_ax = fig.add_axes([0.1,pos1[1]-0.05,0.8,0.015])
        #cb = m.colorbar(cs,"bottom", size="5%", pad="10%")

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
    '''
    tgt_cr_groups= [('L_tk1',(11,13)),
                    ('L_tk2',(10,12)),                    
                    ('L_tn',(14,)),
                    ('S-Clr',(153,)),
                    ('L_tk_1+2',(10,11,12,13)),
    ]

    '''
    main();  sys.exit()
    #for tcr in range(0,4,1):
    #    main(tcr)


