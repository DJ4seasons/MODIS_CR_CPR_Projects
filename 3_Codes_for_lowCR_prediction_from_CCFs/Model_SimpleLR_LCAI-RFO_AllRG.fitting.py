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

Updated to anomaly-based All-RG model
2026.08.13
"""

import numpy as np
import sys
import os #.path
from datetime import timedelta, date
import math
import common_functions as cf

def main(rg_names):
    #print(rg_nm)
    nrg= len(rg_names)
    
    ## Parameters
    mdnm1= 'ERA5'
    nyr,npt= 22,810 
    indir= './Input4ML_LcRFO/'
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']
    ncr= len(tgt_crs)
    
    input4LCAIs= ['t700','t2m','sp','q700','q2m','t800','skt']
    basic_vars= ['LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', 'SST (K)']
    nv= len(basic_vars)
    
    test_yr_idx= [yr-2003 for yr in [2018,2019]]
    train_yr_idx= [val for val in range(nyr) if val not in test_yr_idx]
    
    ## Read CR_rfo
    rfo_all=[]
    for rg_nm in rg_names:
        rfos= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,var_names=tgt_crs,indir=indir,in_dim=[nyr,npt])
        rfos= np.asarray(rfos).reshape([ncr,nyr*npt]).T
        rfo_all.append(rfos)
    rfos= np.asarray(rfo_all).reshape([nrg,nyr,npt,ncr]).swapaxes(0,1).reshape([nyr*nrg*npt,ncr]) 
    ## Check RFO data
    for k,crn in enumerate(tgt_crs):
        rfo1= rfos[:,k]
        print(crn, rfo1.min(), np.percentile(rfo1,[5,50,95]), rfo1.max())

    ## Prepare LCAIs
    all_lcai=[]
    for rg_nm in rg_names:
        indata= cf.collect_data2calc_LCidx_fromSamples(mdnm1,rg_nm,indir=indir,in_dim=[nyr,npt])
        lcai1, sst1= cf.calc_LCidx(indata), indata[-1]
        lcai1[:,:,3]*=100  ## Now ECF in %    
        lcai1= np.concatenate((lcai1,sst1.reshape([nyr,npt,1])),axis=2) #.reshape([nyr*npt,nv])
        all_lcai.append(lcai1)
    lcai1= np.asarray(all_lcai).swapaxes(0,1).reshape([nyr*nrg*npt,nv])
    print(indata[0].shape, lcai1.shape, sst1.shape) #; sys.exit() 
    for k in range(6):
        a= lcai1[:,k]
        print(basic_vars[k],a.min(), np.percentile(a,[5,50,95]),a.max())
    #sys.exit()

    ## Train-Test split
    rfos= rfos.reshape([nyr,nrg*npt,ncr])
    lcai1= lcai1.reshape([nyr,nrg*npt,nv])
    
    ## Standardization
    rfos= cf.get_anomaly(rfos,train_yr_idx=train_yr_idx,flatten=False,standardization=True)
    lcai1= cf.get_anomaly(lcai1,train_yr_idx=train_yr_idx,flatten=False,standardization=True)
    
    X_train, X_test= lcai1[train_yr_idx,:].reshape([-1,nv]),lcai1[test_yr_idx,:].reshape([-1,nv])
    y_train, y_test= rfos[train_yr_idx,:].reshape([-1,ncr]),rfos[test_yr_idx,:].reshape([-1,ncr])
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    ## LR
    from scipy.stats import linregress as lr
    txt0= 'Var_names, Coefficient, Intercept'
    output=[txt0,]
    for k in range(ncr):
        yy= y_train[:,k]
        for j in range(nv):
            xx= X_train[:,j]
            txt_nm= basic_vars[j]+'-'+tgt_crs[k]
            slope,intercept,rv,pv,_= lr(xx,yy)
            txt1= f'{txt_nm},{slope:.8f},{intercept:.8f}'
            output.append(txt1)
        
            y_pred= X_test[:,j]*slope+intercept
            rmse= np.sqrt(((y_pred-y_test[:,k])**2).mean())
            print(txt_nm,rmse,rv**2)
            
    return output


if __name__=="__main__":

    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian']
    
    out_dir= './LR_Coef_data/'
    out_fn_h= out_dir+'Coef.SimpleLR_basic_ano.'
    #for i,rg_nm in enumerate(rg_names):
    if True:
        output1= main(rg_names)
        out_fn= out_fn_h+'{}_12deg.txt'.format(f'AllRG{len(rg_names)}')
        with open(out_fn,'w') as f:
            for txt1 in output1:
                print(txt1,file=f)


