"""
Save "feature importance" of SHAP and ALE method for NNet model

Softmax model with "Other regime" and 18 vars with clims

Daeho Jin
2026.04.11
---

Version for NN with shadow variables
2026.07.27
"""

import numpy as np
import sys
import os 
from datetime import timedelta, date
import math
import common_functions as cf
import NN_classes_y5 as NNc5
import shap
import joblib

def main(rg_names,tgt_crs,model_name):
    
    ## Parameters
    ncr= len(tgt_crs); ncr2= ncr+1 # Include "other regime"
    nrg= len(rg_names)
    
    mdnm1= 'ERA5'
    nyr,npt= 22,810 
    indir= './Input4ML_LcRFO/'

    test_yr_idx= [yr-2003 for yr in [2018,2019]]
    train_yr_idx= [val for val in range(nyr) if val not in test_yr_idx]

    shap_samples= 250
    ale_bins= 50
    
    ## Read model data
    indir1b= './NN_Model_data/'

    var_names= [
        'SST (K)','T2M (K)','T850 (K)','T700 (K)','T500 (K)',
        'Q2M (g/kg)','Q850 (g/kg)','Q700 (g/kg)','Q500 (g/kg)',
        'WS10m (m/s)','WS850 (m/s)','WS700 (m/s)',
        'PS (Pa)','T_adv (K/day)',
        'q850_adv (g/kg/day)','q700_adv (g/kg/day)'
    ]
    vns= ['skt','t2m','t850','t700','t500',
              'q2m','q850','q700','q500',
              'wspd10m','wspd850', 'wspd700',
              'sp','skTadv','q850adv', 'q700adv', ]
    nv= len(vns)
    clim_vars,clim_vnames= ['skt','sp'], ['SST_clim (K)', 'PS_clim (hPa)',]
    clim_vidx= [vns.index(name) for name in clim_vars]  # index for skt and sp
    nv2= len(clim_vars)
    
    out_dir= './NN_feature_importance_data/'
    
    ## Prepare X_test, y_test, and model by region
    ## Read CR_rfo
    rfo_all=[]
    for rg_nm in rg_names:
        rfos= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,var_names=tgt_crs,indir=indir,in_dim=[nyr,npt])
        rfos= np.asarray(rfos).reshape([ncr,nyr*npt]).T
        rfo_all.append(rfos)
    rfo_all= np.asarray(rfo_all).reshape([nrg,nyr,npt,ncr]).swapaxes(0,1).reshape([nyr*nrg*npt,ncr])
    rfos= rfo_all
    
    ## Prepare Phys vars
    all_indata=[]
    for rg_nm in rg_names:
        indata= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,var_names=vns,indir=indir,in_dim=[nyr,npt])
        indata= np.concatenate([np.expand_dims(arr,axis=-1) for arr in indata],axis=-1) #.reshape([nyr,npt,nv])
        all_indata.append(indata)

    all_indata= np.asarray(all_indata) #[nrg,nyr,npt,nv]
    clim_indata= all_indata[:,:,:,clim_vidx].mean(axis=1) #[nrg,npt,nv2]
    all_indata= all_indata.swapaxes(0,1).reshape([nyr*nrg*npt,nv])
    clim_indata= np.tile(clim_indata,[nyr,1,1,1]).reshape([nyr*nrg*npt,nv2])
    #print(indata.shape, ) #; sys.exit() # [nyr,npt,nv]

    ## Normalize
    indata= cf.normalize_x_raw(all_indata,vns)
    clim_indata= cf.normalize_x_raw(clim_indata,clim_vars)

    ## Combine
    indata= np.concatenate((indata,clim_indata),axis=1)
    nv+=nv2
    var_names+= clim_vnames
    
    ## Add shadow variables
    indata2= np.copy(indata)
    rng = np.random.default_rng(seed=1234)
    rng.shuffle(indata2,axis=0)
    indata= np.concatenate((indata,indata2),axis=1)
    var_names+= [f'Random{i+1:02d}' for i in range(nv)]
    nv+=nv
    print(indata.shape)
    
    ## Train-Test split
    rfos= rfos.reshape([nyr,nrg*npt,ncr])
    indata= indata.reshape([nyr,nrg*npt,nv])
    X_train, X_test= indata[train_yr_idx,:].reshape([-1,nv]),indata[test_yr_idx,:].reshape([-1,nv])
    y_train, y_test= rfos[train_yr_idx,:].reshape([-1,ncr]),rfos[test_yr_idx,:].reshape([-1,ncr])
    ## Implement "other regime"
    model= NNc5.NeuralNetworkRegressor()
    y_train= model.append_other_regime(y_train)
    y_test= model.append_other_regime(y_test)
    
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape) #; sys.exit()

    ## Prepare NNet model
    in_fn_h= indir1b+f'NN_rawVar{nv//2}_scaled_12d_wShadow.'
    loaded_model= NNc5.NeuralNetworkRegressor()
    loaded_model.load_model(in_fn_h+f'{model_name}.h5')
    
    #-- Calculate SHAP
    shap_values,explainer, test_data= shap_feature_importance(
        loaded_model, X_train, X_test, method= 'kernel', max_samples = shap_samples
    )
    print(shap_values.shape) #; sys.exit() # [#_samples,#_features,#_output]
    print(test_data.shape)
    shape_txt= 'x'.join([str(v) for v in shap_values.shape])
    outfn= out_dir+model_name+'.shap_dict_{}.wShadow.joblib'.format(shape_txt)
    out_data= dict(shap_values=shap_values, test_data=test_data)
    joblib.dump(out_data,outfn)

    ## Calculate ALE
    ale_byVar=[]
    bin_bounds_byVar=[]
    for vidx in range(nv):
        ale_centered, bin_bounds= calculate_ale_1d_multi_Y(
            loaded_model, X_test, feature_index=vidx, var_name= var_names[vidx],
            y_dim=ncr2, n_bins=ale_bins)
        ale_byVar.append(ale_centered)
        bin_bounds_byVar.append(bin_bounds)
    ale_results= dict(
        feature_names=var_names, ale_values=np.asarray(ale_byVar),
        bin_bounds= np.asarray(bin_bounds_byVar),
    )
    outfn= out_dir+model_name+'.ale_result_dict_{}x{}x{}.wShadow.joblib'.format(nv,ale_bins,ncr2)
    joblib.dump(ale_results,outfn)
    print(ale_results['ale_values'].shape, ale_results['bin_bounds'].shape)                
        
    return 


def calculate_ale_1d_multi_Y(model, X_train, feature_index, var_name, y_dim=1, n_bins=40):
    """
    Computes 1D Accumulated Local Effects (ALE) for a single feature + multi-Y.
    """
    # 1. Define the bins using quantiles of the feature
    feature_values = X_train[:, feature_index]
    bins = np.quantile(feature_values, q=np.linspace(0, 1, n_bins + 1))
    
    # Ensure bins are unique
    bins = np.unique(bins)
    n_bins = len(bins) - 1

    # Initialize array to store the average local effects for each bin
    local_effects = np.zeros([n_bins,y_dim])
    print(f'Calculating ALE for {var_name}...')
    for i in range(n_bins):
        # Find data points within the current bin
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        
        # Get indices of data points in the current bin
        indices_in_bin = np.logical_and(feature_values >= bin_lower,feature_values < bin_upper)
        
        # Skip if bin is empty
        if len(indices_in_bin) == 0:
            continue

        # Get the data for this bin
        X_bin = X_train[indices_in_bin].copy()

        # 2. Calculate local effects by replacing feature values
        # Predict with feature value at lower bound of the bin
        X_bin_lower = X_bin.copy()
        X_bin_lower[:, feature_index] = bin_lower
        pred_lower = model.predict(X_bin_lower,verbose=0) #[:,y_index]

        # Predict with feature value at upper bound of the bin
        X_bin_upper = X_bin.copy()
        X_bin_upper[:, feature_index] = bin_upper
        pred_upper = model.predict(X_bin_upper,verbose=0) #[:,y_index]

        # The local effect is the difference in predictions
        effect_in_bin = pred_upper - pred_lower
        local_effects[i] = np.mean(effect_in_bin,axis=0)

    # 3. Accumulate the effects
    # The final ALE value for a bin is the cumulative sum of local effects
    ale = np.cumsum(local_effects,axis=0)
    
    # 4. Center the ALE plot
    # The mean of ALE is weighted by the number of samples in each bin
    bin_counts = np.histogram(feature_values, bins=bins)[0]
    ale_centered = ale - (np.sum(ale * bin_counts[:,None],axis=0) / np.sum(bin_counts))[None,:]

    return ale_centered, bins

def shap_feature_importance(model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: str = 'kernel',
    max_samples: int = 100
):
    """
    Calculate feature importance using SHAP values.
    Requires: pip install shap
    
    Parameters:
    -----------
    model : object
        Trained model
    X_train : ndarray
        Training data (for background)
    X_test : ndarray
        Test data (to explain)
    method : str
        - 'kernel': Model-agnostic (slower but works for any model)
        - 'tree': Fast for tree-based models (RF, XGBoost, etc.)
    max_samples : int
        Maximum samples to use for background/explanation
    
    Returns:
    --------
    shap_values : ndarray
        SHAP values for each sample and feature
    explainer : shap.Explainer
        SHAP explainer object
    """
    
    # Subsample if needed
    if len(X_train) > max_samples:
        indices = np.random.choice(len(X_train), max_samples, replace=False)
        background = X_train[indices]
    else:
        background = X_train
    
    if len(X_test) > max_samples:
        indices = np.random.choice(len(X_test), max_samples, replace=False)
        test_data = X_test[indices]
    else:
        test_data = X_test
    
    print(f"Calculating SHAP values using {method} method...")
    
    if method == 'kernel':
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(test_data)
    elif method == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_data)
    else:
        raise ValueError(f"Unknown method: {method}")

    for j in range(test_data.shape[1]):
        print(j,np.round(np.absolute(shap_values[:,j,:]).mean(axis=0),4))
                    
    return shap_values, explainer, test_data


if __name__=="__main__":
    
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']
    
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian',]
    model_name= 'AllRG6_ow10_100-100_rs37'
    main(rg_names,tgt_crs,model_name)

    '''
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian',
               'SON_TIO','SON_StropEPO','DJF_NWPO',
               'MAM_Canarian','MAM_SPCZ','JJA_SAO',]
    model_name= 'AllRG12_ow10_172-172_rs23'
    main(rg_names,tgt_crs,model_name)
    '''

