"""
Ridge regression: LcRFO vs. Cloud-controlling factors

Calculate ridge regression per each region and RFO

RFO: L1_tk, L2_tk, L_tn, and S-Clr

Variable for low cloud amount indices:
't700','t2m','sp','q700','q2m','t800','skt'
--> 'LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', Ts (K)

Supplementary CCFs:
'skTadv','wspd10m','w700','r700'
--> 'T_adv (K/day)','WS10m (m/s)','w700 (hPa/day)','RH700 (%)'


Target resolution: Monthly, and 4-deg
Sampling stratege: Quarter-sliding    
For a target region of 12x12-deg, it is expected to get
81 in horizontal, 10 in temporal, and 22 years= 17820


Apply test years (2018,2019).
NNet-like scaling is applied to all input variables
Select LR6 or LR10
---

This code is for final fitting based on a select alpha

Daeho Jin
2026.04.14 
---

Updated to anomaly-based model, including LR6-variants
2026.07.31
"""

import numpy as np
import sys
import os 
from datetime import timedelta, date
import math
import common_functions as cf

def main(model,best_alpha,rg_nm,tgt_crs):
    print(rg_nm)    
    ncr= len(tgt_crs)
    
    ## Parameters
    mdnm1= 'ERA5'
    nyr,npt= 22,810 
    indir= './Input4ML_LcRFO/'
    
    input4LCAIs= ['t700','t2m','sp','q700','q2m','t800','skt',]
    input4add_CCFs= ['skTadv','wspd10m','w700','r700']
    all_vars= ['LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']
    if 'v' in model: #model[-3:]=='6v2':
        candidates= ['LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', 'SST (K)','RH700 (%)']
        vid2exclude= ['','',6,0,1,2,5]
        ver= int(model.strip().split('v')[1])
        if ver>=2 and ver<len(vid2exclude):
            basic_vars= []
            for i,vnm in enumerate(candidates):
                if i != vid2exclude[ver]:
                    basic_vars.append(vnm)
            print(model, basic_vars)
        else:
            print('Set proper model name, v2 to v6')
            sys.exit()
    elif model=='LR6':
        basic_vars= ['EIS (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']
    elif model=='LR10':
        basic_vars= all_vars    
    else:
        sys.exit(f'model name is incompatible: {model}')
        
    nv= len(basic_vars)
    nv4lcai= len(input4LCAIs)
    basic_var_ind= [all_vars.index(vn) for vn in basic_vars]
    
    test_yr_idx= [yr-2003 for yr in [2018,2019]]
    train_yr_idx= [val for val in range(nyr) if val not in test_yr_idx]    
    
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
        mdnm1,rg_nm,indir=indir,var_names=input4LCAIs+input4add_CCFs,in_dim=[nyr,npt])
    lcai1, ext1= cf.calc_LCidx(indata[:nv4lcai]), np.asarray(indata[nv4lcai-1:])
    print(indata[0].shape, lcai1.shape, ext1.shape) #; sys.exit() # [nyr,npt,nvar]
    
    lcai1= lcai1.reshape([nyr*npt,-1])
    ext1= ext1.reshape([-1,nyr*npt]).T
    lcai1[:,3]*=100  ## Now ECF in %    
    
    lci_var_ind, ext_var_ind= [],[]
    for iv in basic_var_ind:
        if iv<5:
            lci_var_ind.append(iv)
        else:
            ext_var_ind.append(iv-5)
        
    lcai1= lcai1[:,lci_var_ind]
    ext1= ext1[:,ext_var_ind]
    lcai1= np.concatenate((lcai1,ext1),axis=1)
    print(lcai1.shape)
    ## Check LCAI data 
    for k in range(nv):
        a= lcai1[:,k]
        print(basic_vars[k],a.min(), np.percentile(a,[5,50,95]),a.max())

    ## Train-Test split
    rfos= rfos.reshape([nyr,npt,ncr])
    lcai1= lcai1.reshape([nyr,npt,nv])
    
    ## Standardization
    rfos= cf.get_anomaly(rfos,train_yr_idx=train_yr_idx,flatten=False,standardization=True)
    lcai1= cf.get_anomaly(lcai1,train_yr_idx=train_yr_idx,flatten=False,standardization=True)
    
    X_train, X_test= lcai1[train_yr_idx,:].reshape([-1,nv]),lcai1[test_yr_idx,:].reshape([-1,nv])
    y_train, y_test= rfos[train_yr_idx,:].reshape([-1,ncr]),rfos[test_yr_idx,:].reshape([-1,ncr])
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    
    ## Ridge LR
    from sklearn.linear_model import Ridge
    txt0= ','.join(['Cld_name',]+basic_vars+['Intercept',])
    output=[txt0,]
    for k in range(ncr):
        ridge= Ridge(alpha=best_alpha[k],fit_intercept=True,copy_X=True).fit(X_train,y_train[:,k])
        coef= ridge.coef_
        intp= ridge.intercept_
        txt= ','.join(str(v) for v in np.round(coef,8))
        txt+= f',{intp:.8f}'
        txt= f'{tgt_crs[k]},'+txt
        output.append(txt)
        
        y_pred= (X_test*coef[None,:]).sum(axis=1)+intp
        rmse= np.sqrt(((y_pred-y_test[:,k])**2).mean())
        mae= np.mean(np.absolute(y_pred-y_test[:,k]))
        print(tgt_crs[k],rmse,mae,ridge.score(X_test,y_test[:,k]))
            
    return output


if __name__=="__main__":

    model, best_alpha= 'LR10' ,[6.31e02, 1.58e01, 1.58e02, 1.00e03]
    #model, best_alpha= 'LR6', [1.58e03,2.51e03,1.00e03,2.51e03]
    
    #model, best_alpha= 'LR6v2', [1.58e02,6.31e00,1.00e02,3.98e02]
    #model, best_alpha= 'LR6v3', [6.31e02,1.0e01,1.0e01,6.31]
    #model, best_alpha= 'LR6v4', [6.31e02,2.51e01,3.98e01,2.51e01]
    #model, best_alpha= 'LR6v5', [3.98e02,1.0e01,1.58e01,3.98e01]
    #model, best_alpha= 'LR6v6', [6.31e02,1.58e02,1.0e02,2.51e02]    
    
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian']
    
    out_dir= './LR_Coef_data/'
    out_fn_h= out_dir+'Coef.Ridge{}_basic_ano.'.format(model)
    for i,rg_nm in enumerate(rg_names):
        output1= main(model,best_alpha,rg_nm,tgt_crs)
        out_fn= out_fn_h+'{}_12deg.txt'.format(rg_nm)
        with open(out_fn,'w') as f:
            for txt1 in output1:
                print(txt1,file=f)


