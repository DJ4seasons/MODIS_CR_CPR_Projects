"""
Simple regression: LcRFO vs. Cloud-controlling factors

Calculate simple linear regression per each region and RFO

RFO: L1_tk, L2_tk, L_tn, and S-Clr

Variable for low cloud amount indices:
't700','t2m','sp','q700','q2m','t800','skt'
--> 'LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', Ts (K)

Target resolution: Monthly, and 4-deg
Sampling stratege: Quarter-sliding    
For a target region of 12x12-deg, it is expected to get
81 in horizontal, 10 in temporal, and 22 years= 17820


Apply test years (2018,2019).
NNet-like scaling is applied to all input variables

Daeho Jin
2026.04.14 
---

Updated to anomaly-based model
2026.07.31
---

Perform boot-strap method with seasonal block
2026.08.02
"""

import numpy as np
import sys
import os #.path
from datetime import timedelta, date
import math
import common_functions as cf

def main(rg_nm,K=1000):
    print(rg_nm)    
    
    ## Parameters
    mdnm1= 'ERA5'
    nyr,npt= 22,810 
    indir= './Input4ML_LcRFO/'
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']
    ncr= len(tgt_crs)
    
    input4LCAIs= ['t700','t2m','sp','q700','q2m','t800','skt',]
    basic_vars= ['LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', 'SST (K)',]
    nv= len(basic_vars)
    
    test_yr_idx= [yr-2003 for yr in [2018,2019]]
    train_yr_idx= [val for val in range(nyr) if val not in test_yr_idx]
    nyr_train= len(train_yr_idx)
    
    ## Read CR_rfo
    rfos= cf.collect_data2calc_LCidx_fromSamples(
        mdnm1,rg_nm,var_names=tgt_crs,indir=indir,in_dim=[nyr,npt])
    rfos= np.asarray(rfos).reshape([ncr,nyr*npt]).T
    ## Check RFO data
    for k,crn in enumerate(tgt_crs):
        rfo1= rfos[:,k]
        print(crn, rfo1.min(), np.percentile(rfo1,[5,50,95]), rfo1.max())
        
    ## Prepare LCAIs
    indata= cf.collect_data2calc_LCidx_fromSamples(
        mdnm1,rg_nm,indir=indir,var_names=input4LCAIs,in_dim=[nyr,npt])
    lcai1, sst1= cf.calc_LCidx(indata), indata[-1]
    print(indata[0].shape, lcai1.shape, sst1.shape) #; sys.exit() # [nyr,npt,nvar]
    
    lcai1[:,:,3]*=100  ## Now ECF in %    
    lcai1= np.concatenate((lcai1,sst1.reshape([nyr,npt,1])),axis=2).reshape([nyr*npt,nv])
    for k in range(6):
        a= lcai1[:,k]
        print(basic_vars[k],a.min(), np.percentile(a,[5,50,95]),a.max())
    #sys.exit()

    ## Train-Test split
    rfos= rfos.reshape([nyr,npt,ncr])
    lcai1= lcai1.reshape([nyr,npt,nv])
    
    ## Standardization
    rfos= cf.get_anomaly(rfos,train_yr_idx=train_yr_idx,flatten=False,standardization=True)
    lcai1= cf.get_anomaly(lcai1,train_yr_idx=train_yr_idx,flatten=False,standardization=True)
    
    #X_train, X_test= lcai1[train_yr_idx,:].reshape([-1,nv]),lcai1[test_yr_idx,:].reshape([-1,nv])
    #y_train, y_test= rfos[train_yr_idx,:].reshape([-1,ncr]),rfos[test_yr_idx,:].reshape([-1,ncr]) 
    X_train= lcai1[train_yr_idx,:] # n_year,npt,nv 
    y_train= rfos[train_yr_idx,:] # n_year,npt,ncr    
    print(X_train.shape, y_train.shape)
    
    ## Simple LR
    from scipy.stats import linregress as lr

    ## Shuffle by year for Bootstrap
    rng= np.random.default_rng(1234)
    out_coef=[]
    for k in range(K):
        ind= rng.choice(nyr_train,nyr_train,replace=True)
        #print(ind)
        X_tmp1= X_train[ind,:].reshape([-1,nv])
        y_tmp1= y_train[ind,:].reshape([-1,ncr])
        coef_by_cr=[]
        for cr1 in range(ncr):
            coef_by_x=[]
            yy= y_tmp1[:,cr1]
            for v in range(nv):
                xx= X_tmp1[:,v]
                slope1,intercept,rv,pv,_= lr(xx,yy)
                coef_by_x.append([slope1,intercept])
            coef_by_cr.append(coef_by_x)
        coef_by_cr= np.asarray(coef_by_cr) # [ncr,nv, (coef,intp)]
        out_coef.append(coef_by_cr)

    out_coef= np.asarray(out_coef).swapaxes(0,1).swapaxes(1,2)  
    print(out_coef.shape) # now [ncr,nv,K,n_coef+intp]
    print(out_coef[0,:,:,0].mean(axis=1))
    return out_coef

if __name__=="__main__":

    K=1000
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian']
    
    out_dir= './Bootstrap_result/'
    out_fn_h= out_dir+'Coef_set_BootStrap.SimpleLR_basic_ano.'.format()
    for i,rg_nm in enumerate(rg_names):
        output1= main(rg_nm,K)
        dim_txt= 'x'.join([str(v) for v in output1.shape])
        out_fn= out_fn_h+'{}_12deg.{}.f32dat'.format(rg_nm,dim_txt)
        with open(out_fn,'wb') as f:
            output1.astype(np.float32).tofile(f)


