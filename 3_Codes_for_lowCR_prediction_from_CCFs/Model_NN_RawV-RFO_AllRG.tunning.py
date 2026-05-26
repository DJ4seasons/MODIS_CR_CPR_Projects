"""
Non-linear regression: LcRFO vs. Cloud-controlling factors

Perform Neural Network non-linear regression per each region 

RFO: L1_tk, L2_tk, L_tn, and S-Clr

Use raw variables instead of LCAIs
T: Ts,T2M,T850,T700,T500, T_adv
q: q2m,q850,q700,q500, q800adv, q750adv
Misc: Ps, WS10m, WS850, WS700
Clim: Ps_clim, Ts_clim


Target resolution: Monthly, and 4-deg
Sampling stratege: Quarter-sliding    
For a target region of 12x12-deg, it is expected to get
81 in horizontal, 10 in temporal, and 22 years= 17820

Apply test years (2018,2019).

This code is to find best hyper-paramter set using cross-validation

By Daeho Jin
2026.03.06
---

Implement "other regime" to make all-regime sum=1.0
Change the activation to softmax (requiring no scaler)

Add additional target area if necessary
2026.04.10 
"""

import numpy as np
import sys
import os 
from datetime import timedelta, date
import math
import common_functions as cf
import NN_classes_y5 as NNc5

def main(tgt_crs,rg_names,param_grid,scoring='mse'):
    print(tgt_crs)
    ## Parameters
    ncr= len(tgt_crs)
    nrg= len(rg_names)    
    
    mdnm1= 'ERA5'
    nyr,npt= 22,810 
    indir= './Input4ML_LcRFO/'    

    var_names= [
        'SST (K)','T2M (K)','T850 (K)','T700 (K)',
        'Q2M (g/kg)','Q850 (g/kg)','Q700 (g/kg)',
        'PS (Pa)','T_adv (K/day)','WS10m (m/s)',
    ]
    vns= ['skt','t2m','t850','t700','t500',
          'q2m','q850','q700','q500',
          'wspd10m','wspd850', 'wspd700',
          'sp','skTadv', 'q850adv', 'q700adv',
    ]
    nv= len(vns)
    clim_vars= ['skt','sp']
    clim_vidx= [vns.index(name) for name in clim_vars]  # index for skt and sp
    nv2= len(clim_vars)
    
    test_yr_idx= [yr-2003 for yr in [2018,2019]]
    train_yr_idx= [val for val in range(nyr) if val not in test_yr_idx]

    ## Cross-validation parameters
    n_folds=5
    n_tr_years= len(train_yr_idx)
    #-- Grouping by year
    groups = create_year_groups(
        n_years=n_tr_years, n_folds=n_folds, n_points_per_year=nrg*npt,  
        random_state=1234
    )
    #print(groups.shape, np.unique(groups, return_counts=True))
    
    ## Read CR_rfo
    rfo_all=[]
    for rg_nm in rg_names:
        rfos= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,var_names=tgt_crs,indir=indir,in_dim=[nyr,npt]) 
        rfos= np.asarray(rfos).reshape([ncr,nyr*npt]).T
        rfo_all.append(rfos)
    rfo_all= np.asarray(rfo_all).reshape([nrg,nyr,npt,ncr]).swapaxes(0,1).reshape([nyr*nrg*npt,ncr])
    rfos=rfo_all
    
    #-- Check RFO data
    for k,crn in enumerate(tgt_crs):
        rfo1= rfos[:,k]
        print(crn, rfo1.min(), np.percentile(rfo1,[5,50,95]), rfo1.max())        
    #-- Check "other" regime
    rfo1= 1-rfos.sum(axis=1)
    print('Other',rfo1.min(), np.percentile(rfo1,[5,50,95]), rfo1.max(), rfo1.std())

    
    ## Prepare Phys vars
    all_indata=[]
    for rg_nm in rg_names:
        indata= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,var_names=vns,indir=indir,in_dim=[nyr,npt])
        indata= np.concatenate([np.expand_dims(arr,axis=-1) for arr in indata],axis=-1) 
        all_indata.append(indata)
    all_indata= np.asarray(all_indata) #[nrg,nyr,npt,nv]
    clim_indata= all_indata[:,:,:,clim_vidx].mean(axis=1) #[nrg,npt,nv2]
    all_indata= all_indata.swapaxes(0,1).reshape([nyr*nrg*npt,nv])
    clim_indata= np.tile(clim_indata,[nyr,1,1,1]).reshape([nyr*nrg*npt,nv2])

    ## Normalize x_inputs
    indata= cf.normalize_x_raw(all_indata,vns)

    ## Check x_input data after normalization
    for k in range(nv):
        a= indata[:,k]
        print(vns[k],a.min(), np.percentile(a,[5,50,95]),a.max())

    ## Normalize clim_inputs and check data range
    clim_indata= cf.normalize_x_raw(clim_indata,clim_vars)
    for k in range(nv2):
        a= clim_indata[:,k]
        print(clim_vars[k]+'_clim',a.min(), np.percentile(a,[5,50,95]),a.max())

    ## Combining with clim data
    indata= np.concatenate((indata,clim_indata),axis=1)
    nv+=nv2

    
    ## Train-Test split
    rfos= rfos.reshape([nyr,nrg*npt,ncr])
    indata= indata.reshape([nyr,nrg*npt,nv])
    X_train, X_test= indata[train_yr_idx,:].reshape([-1,nv]),indata[test_yr_idx,:].reshape([-1,nv])
    y_train, y_test= rfos[train_yr_idx,:].reshape([-1,ncr]),rfos[test_yr_idx,:].reshape([-1,ncr])
    #-- Implement "other regime"
    model= NNc5.NeuralNetworkRegressor()
    y_train= model.append_other_regime(y_train)
    y_test= model.append_other_regime(y_test)
    
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    ## Hyper-paramter search
    # Initialize tuner
    tuner = NNc5.CrossValidationTuner(NNc5.NeuralNetworkRegressor, cv_folds=n_folds)

    ## Add ReduceLROnPlateau
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    RLR= ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,                    # Reduce LR by half
        patience=7,                    # Wait 7 epochs before reducing
        min_lr=5e-5, 
        cooldown=0,min_delta= 1e-6,
        verbose=0
    )

    callback=[RLR,]
    
    results= tuner.tune_hyperparameters(
        X_train, y_train, param_grid=param_grid,
        groups=groups, scoring=scoring,
        callback2add=callback,
        max_epochs=300, verbose=0,
    )
            
    return results


def create_year_groups(n_years, n_folds, n_points_per_year, random_state=None):
    """
    Create group array for year-based GroupKFold cross-validation.
    
    Parameters:
    -----------
    n_years : int (e.g., 20)
    n_folds : int (e.g., 5)  
    n_points_per_year : int (your spatial points per year)
    random_state : int, optional
        
    Returns:
    --------
    groups : ndarray 
        Group numbers for each sample
    """
    if n_years % n_folds != 0:
        raise ValueError(f"n_years ({n_years}) must be divisible by n_folds ({n_folds})")
    
    # Shuffle years randomly
    years = np.arange(n_years)
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        rng.shuffle(years)
    else:
        np.random.shuffle(years)
    
    # Assign group numbers to years  
    years_per_fold = n_years // n_folds
    year_to_group = np.zeros(n_years, dtype=int)
    
    for fold in range(n_folds):
        start_idx = fold * years_per_fold
        end_idx = start_idx + years_per_fold
        fold_years = years[start_idx:end_idx]
        year_to_group[fold_years] = fold
    
    # Create group array for all samples
    groups = np.repeat(year_to_group, n_points_per_year)
    return groups #.reshape([n_years, n_points_per_year])


            
if __name__=="__main__":

    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian',
               #'SON_TIO','SON_StropEPO','DJF_NWPO',
               #'MAM_Canarian','MAM_SPCZ','JJA_SAO',
    ]

    # Define hyperparameter grid (including two hidden layers)

    param_grid = {
        'hidden_units': [
            #(40,), (60,), (80,), (100,), #(160,), (250,), (400,),
            (80,80), (100,100), (120,120), (160,160), # (172,172), #
            #(128,64),(160,80), (192,96), (224,112), (240,120), 
            #(90,60), (120,80),(150,100),(180,120), (210,140), 
            #(96,48), (84,56),  (80,40), (72,48), #(64,32),(60,40),             
        ],  
        'learning_rate': [0.005,0.01], 
        'patience': [20,],
        'batch_size': [512,], 
        'min_delta': [1e-6,],
        'l2_reg_str': [0.0001,0.00001,], #0.000001
        'other_regime_weight': [1.,],
        'random_state': [23,], #[37,], #
    }
    '''
    param_grid = {        
        'hidden_units': [
            (172,172), #(160,160), #(80,80), (100,100), (120,120), 
            #(224,112), #(240,120), #(128,64),(160,80), (192,96), 
            (210,140), #(90,60), (120,80),(150,100),(180,120),
        ],      # 
        'learning_rate': [0.003,0.005,0.006,0.012,0.02,0.025], #[0.002,0.004,0.008,0.016,], #
        'patience': [20,],
        'batch_size': [1600,1024,256,], #256,512,128
        'min_delta': [1e-6,],
        'l2_reg_str': [0.00001,], #0.0001,
        'other_regime_weight': [1.,],
        'random_state': [23,], #[37,], 
    }


    param_grid = {
        'hidden_units': [ #(64,), #(128,), (160,),
                          (172,172), #(64,64), # #(40,40),(48,48), #
                          #(64,32), #(80,40), #(48,24), #
                          (224,112), #(72,48), #(80,40),
                          #(126,84), (140,70), #(70,70), 
        ],      # 
        'learning_rate': [0.002,0.005,], #[0.003,0.005,0.007,0.01], #
        'patience': [20,],
        'batch_size': [512,1024], #
        'min_delta': [1e-6,],
        'l2_reg_str': [0.00001,], #0.0001, 0.00001,
        'other_regime_weight': [1.,],
        'random_state': [101,108,113,128,149], #[43,53,61,71,79], #[23,37,], #23, 37, 31 ##
    }
    '''

    scoring= 'mse' 

    results_all=[]    
    #for i,tgt_crs in enumerate([['L2_tk','L_tn'],]): #['L1_tk','S-Clr'],
    if True:
        res= main(tgt_crs,rg_names,param_grid,scoring=scoring)
        results_all.append(res)
        #print(res); sys.exit()
        tcr_names= ['All_Regions',]
        

    ### Print CV results
    select_keys= ['hidden_units', 'learning_rate', 'batch_size', 'l2_reg_str', f'std_{scoring}', f'mean_{scoring}','mean_epochs','first_lr_reduction_epoch_mean','lr_at_best_epoch_mean']
    for i,res in enumerate(results_all):
        sorted_results = sorted(res, key=lambda x: x[f"mean_{scoring}"])
        top_results = sorted_results[:10]
        print('\n***--- ',tcr_names[i])
        print(', '.join(select_keys))
        for r in top_results:
            txt=[f'{res.index(r)}']
            for key in select_keys:
                if 'mse' in key or 'mae' in key:
                    v= f'{r[key]:.6f}'
                elif 'r2' in key:
                    if key[0]=='m':
                        v= f'{-r[key]:.5f}'
                    else:
                        v= f'{r[key]:.5f}'
                else:
                    v= f'{r[key]}'
                txt.append(v)

            print('; '.join(txt))

    sys.exit()
