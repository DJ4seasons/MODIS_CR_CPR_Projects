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

Daeho Jin
2026.04.14 
---

Perform boot-strap method with seasonal block
For anomaly-based ridge regressions
2026.08.02
---

Add MAE and R^2 drop to estiamte feature importance
2026.08.12
"""

import numpy as np
import sys
import os 
from datetime import timedelta, date
import math
import common_functions as cf

def main(model,best_alpha,rg_nm,tgt_crs,K=1000,drop_K=25):
    print(rg_nm)    
    
    ## Parameters
    mdnm1= 'ERA5'
    nyr,npt= 22,810 
    indir= './Input4ML_LcRFO/'
    ncr= len(tgt_crs)
    
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
    rfo_ref= dict(std=rfos[train_yr_idx,:].std(axis=0,ddof=1),mean=rfos[train_yr_idx,:].mean(axis=0))
    
    ## Standardization
    rfos= cf.get_anomaly(rfos,train_yr_idx=train_yr_idx,flatten=False,standardization=True)
    lcai1= cf.get_anomaly(lcai1,train_yr_idx=train_yr_idx,flatten=False,standardization=True)

    X_train, X_test= lcai1[train_yr_idx,:],lcai1[test_yr_idx,:]
    y_train, y_test= rfos[train_yr_idx,:],rfos[test_yr_idx,:]    
    print(X_train.shape, y_train.shape) # [n_year,npt,nv],[n_year,npt,ncr] 
    print(X_test.shape, y_test.shape)

    ## Prepare shuffled X_test for measuring performance drop
    rng0= np.random.default_rng(1234)
    test_ind= np.arange(X_test.shape[0]*X_test.shape[1],dtype=int)
    test_ind_shuffled=[]
    for k in range(drop_K):
        rng0.shuffle(test_ind)
        test_ind_shuffled.append(test_ind.copy())
        
    ## Ridge LR
    from sklearn.linear_model import Ridge

    ## Shuffle by year for Bootstrap
    rng= np.random.default_rng(1234)
    out_coef=[]
    out_drop=[]
    for k in range(K):
        ind= rng.choice(nyr_train,nyr_train,replace=True)

        X_tmp1= X_train[ind,:].reshape([-1,nv])
        y_tmp1= y_train[ind,:].reshape([-1,ncr])
        
        ridge_by_cr=[]
        drop_by_cr=[]
        for cr1 in range(ncr):
            ridge= Ridge(alpha=best_alpha[cr1],fit_intercept=True,copy_X=True).fit(X_tmp1,y_tmp1[:,cr1])
            coef= ridge.coef_
            intp= ridge.intercept_
            ridge_by_cr.append(np.append(coef,intp))

            drop_set= get_metric_drop(ridge_by_cr[-1],X_test,y_test[:,:,cr1:cr1+1],test_ind_shuffled,rfo_ref,cr1)
            drop_by_cr.append(drop_set)
            
        ridge_by_cr= np.asarray(ridge_by_cr)
        #print(ridge_by_cr.shape) # [ncr, n_coef+intp]
        drop_by_cr= np.asarray(drop_by_cr)
        #print(drop_by_cr.shape)
        
        out_coef.append(ridge_by_cr)
        out_drop.append(drop_by_cr)
        if (k+1)%100==0:
            print(k+1)
        
    out_coef= np.asarray(out_coef).swapaxes(0,1)  
    print(out_coef.shape) # now [ncr,K,n_coef+intp]
    print(out_coef[0,:].mean(axis=0))

    out_drop= np.asarray(out_drop).swapaxes(0,1)  # now [ncr,K,n_pair,2]
    return out_coef,out_drop
    
from itertools import combinations
def get_metric_drop(ridge_coef,X_test,yy,test_ind_shuffled,rfo_ref,cr_idx):
    rfo_std= rfo_ref['std'][:,cr_idx:cr_idx+1]
    rfo_mean= rfo_ref['mean'][:,cr_idx:cr_idx+1]
    
    ## Calc ref first
    yy0= cf.de_standardize(rfo_std,rfo_mean,yy.squeeze()[np.newaxis,:,:]).squeeze()
    y_pred= (X_test*ridge_coef[:-1][None,None,:]).sum(axis=2)+ridge_coef[-1]    
    y_pred= cf.de_standardize(rfo_std,rfo_mean,y_pred[np.newaxis,:,:]).squeeze() 
    mae_ref= np.abs(y_pred-yy0).mean()
    r2_ref= 1-((yy0-y_pred)**2).sum()/((yy0-yy0.mean())**2).sum()
    #print(mae_ref, r2_ref)
    
    nv= X_test.shape[2]
    drop_set=[]
    ## Individual predictor
    for k in range(nv):
        mae1,r2_1= [],[]
        for shuffle_ind in test_ind_shuffled:
            X_test1= X_test.copy().reshape([-1,nv])
            X_test1[:,k]= X_test1[shuffle_ind,k]
            y_pred1= (X_test1*ridge_coef[:-1][None,:]).sum(axis=1)+ridge_coef[-1]
            y_pred1= cf.de_standardize(rfo_std,rfo_mean,y_pred1.reshape(y_pred.shape)[np.newaxis,:,:]).squeeze() #

            mae= np.abs(y_pred1-yy0).mean()
            r2= 1-((yy0-y_pred1)**2).sum()/((yy0-yy0.mean())**2).sum()
            mae1.append(mae)
            r2_1.append(r2)
        drop1= [np.asarray(mae1).mean()-mae_ref, r2_ref-np.asarray(r2_1).mean()]
        drop_set.append(drop1)

    ## Pair predictor
    for ind in combinations(range(nv),2): ## All possible pairs
        mae1,r2_1= [],[]        
        for shuffle_ind in test_ind_shuffled:
            X_test1= X_test.copy().reshape([-1,nv])
            for ind1 in ind:
                X_test1[:,ind1]= X_test1[shuffle_ind,ind1]
            y_pred1= (X_test1*ridge_coef[:-1][None,:]).sum(axis=1)+ridge_coef[-1]
            y_pred1= cf.de_standardize(rfo_std,rfo_mean,y_pred1.reshape(y_pred.shape)[np.newaxis,:,:]).squeeze() #

            mae= np.abs(y_pred1-yy0).mean()
            r2= 1-((yy0-y_pred1)**2).sum()/((yy0-yy0.mean())**2).sum()
            mae1.append(mae)
            r2_1.append(r2)
        drop1= [np.asarray(mae1).mean()-mae_ref, r2_ref-np.asarray(r2_1).mean()]
        drop_set.append(drop1)
        
    return np.asarray(drop_set)

if __name__=="__main__":

    K= 1000  # Bootstrap
    drop_K= 25  # For performance drop by permutation, ensembles at each bootstrap step

    model, best_alpha= 'LR10' ,[6.31e02, 1.58e01, 1.58e02, 1.00e03]
    model, best_alpha= 'LR6', [1.58e03,2.51e03,1.00e03,2.51e03]    
    #model, best_alpha= 'LR6v2', [1.58e02,6.31e00,1.00e02,3.98e02]

    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian']
    
    out_dir= './Bootstrap_result/'
    out_fn_h= out_dir+'Coef_set_BootStrap.Ridge{}_basic_ano.'.format(model)
    out_fn_h2= out_dir+'Metric_drop_set_BootStrap.Ridge{}_basic_ano.'.format(model)
    for i,rg_nm in enumerate(rg_names):
        output1,output2= main(model,best_alpha,rg_nm,tgt_crs,K,drop_K)
        dim_txt= 'x'.join([str(v) for v in output1.shape])
        out_fn= out_fn_h+'{}_12deg.{}.f32dat'.format(rg_nm,dim_txt)
        with open(out_fn,'wb') as f:
            output1.astype(np.float32).tofile(f)

        dim_txt= 'x'.join([str(v) for v in output2.shape])
        out_fn= out_fn_h2+'{}_12deg.{}.f32dat'.format(rg_nm,dim_txt)
        with open(out_fn,'wb') as f:
            output2.astype(np.float32).tofile(f)


