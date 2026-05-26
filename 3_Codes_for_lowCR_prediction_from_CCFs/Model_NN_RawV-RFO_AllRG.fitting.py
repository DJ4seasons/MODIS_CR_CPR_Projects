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

This code is to do a final fitting

By Daeho Jin
2026.03.06
---

Implement "other regime" to make all-regime sum=1.0
Change the activation to softmax (requiring no scaler)

2026.03.06
---

with cosine_decay LR scheduler
2026.03.19
---

Implement "other regime"
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

def main(tgt_crs,rg_names,nn_param,out_fn_h):
    
    ## Parameters
    mdnm1= 'ERA5'
    nyr,npt= 22,810 
    indir= './Input4ML_LcRFO/'

    ncr= len(tgt_crs)
    nrg= len(rg_names)
    
    var_names= [
        'SST (K)','T2M (K)','T850 (K)','T700 (K)',
        'Q2M (g/kg)','Q850 (g/kg)','Q700 (g/kg)',
        'PS (hPa)','T_adv (K/day)','WS10m (m/s)',
    ]
    vns= ['skt','t2m','t850','t700','t500',
          'q2m','q850','q700','q500',
          'wspd10m','wspd850', 'wspd700',
          'sp','skTadv', 'q850adv', 'q700adv',
    ]
    nv= len(vns)
    clim_vars,clim_vnames= ['skt','sp'], ['SST_clim (K)', 'PS_clim (hPa)',]
    clim_vidx= [vns.index(name) for name in clim_vars]  # index for skt and sp
    nv2= len(clim_vars)
    
    test_yr_idx= [yr-2003 for yr in [2018,2019]]    
    train_yr_idx= [val for val in range(nyr) if val not in test_yr_idx]
    
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
    print('Other',rfo1.min(), np.percentile(rfo1,[5,25,50,75,95]), rfo1.max(), rfo1.std())

        
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
    var_names+= clim_vnames
    
    
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


    ## Cosine decay LR scheduler
    initial_lr = nn_param['learning_rate']
    final_lr= nn_param['final_LR']
    batch_size = nn_param['batch_size']
    sample_size= X_train.shape[0]
    total_epochs=nn_param['max_epoch']
    steps_per_epoch = sample_size // batch_size
    total_steps = total_epochs * steps_per_epoch
    
    cosine_lr = NNc5.create_cosine_schedule(
        initial_lr=initial_lr,
        final_lr=final_lr,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=0,  
    )
    
    ## Neural Network
    model = NNc5.NeuralNetworkRegressor(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_units=nn_param['hidden_units'],      
        learning_rate=None, #nn_param['learning_rate'],
        LR_scheduler=cosine_lr,
        l2_reg_str=nn_param['l2_reg_str'],
        target_names=tgt_crs+['Other',],
        random_state=nn_param['random_state'],
        other_regime_weight= nn_param['other_regime_weight']
    )
    print("Model Architecture:")
    model.get_model_summary()
    
    callback=[] 
    print("\nTraining the model...")
    model.fit(X_train, y_train, epochs=nn_param['max_epoch'],
              batch_size=nn_param['batch_size'],
              patience=nn_param['patience'], 
              validation_split=0.,
              verbose=0,callback2add=callback) # restore_best_weights=True,
        
    # Evaluate the model
    metrics = model.evaluate(X_train, y_train,verbose=0,scatter_plot=True)
    print(f"\nTraining Results:")
    print(f"MSE: {metrics['mse']:.5f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Mean R²: {metrics['mean_r2']:.4f}")
    print(f"R² per output: {[float(f'{r2:.4f}') for r2 in metrics['r2_scores']]}")
    print(f"Corr per output: {[float(f'{c:.3f}') for c in metrics['corrs']]}")
    print(f"MAE per output: {[float(f'{v:.4f}') for v in metrics['mae_scores']]}")
    #print(' ')
    
    metrics = model.evaluate(X_test, y_test,verbose=0,scatter_plot=True)
    print(f"\nTest Results:")
    print(f"MSE: {metrics['mse']:.5f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Mean R²: {metrics['mean_r2']:.4f}")
    print(f"R² per output: {[float(f'{r2:.4f}') for r2 in metrics['r2_scores']]}")
    print(f"Corr per output: {[float(f'{c:.3f}') for c in metrics['corrs']]}")
    print(f"MAE per output: {[float(f'{v:.4f}') for v in metrics['mae_scores']]}")

    out_fn_fig= out_fn_h+'{}.png'.format(nn_param['model_name'])
    model.plot_training_history('All 6 Regions',out_fn=out_fn_fig,show_val=False)

    out_fn= out_fn_h+'{}.h5'.format(nn_param['model_name'])
    model.save_model(out_fn)

    return #model
    

if __name__=="__main__":
    
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']

    '''    
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian',]
    param_list= [

        { 'model_name': 'AllRG6_ow10_100-100_rs37',
          'hidden_units': (100,100),  
          'learning_rate': 0.004,
          'batch_size': 256, 
          'l2_reg_str': 0.0001, 'other_regime_weight': 1.,
          'max_epoch': 61, 'final_LR': 0.00071,
          'random_state': 37, 'patience': 0,
        },
        
    ]
    '''
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian',
               'SON_TIO','SON_StropEPO','DJF_NWPO',
               'MAM_Canarian','MAM_SPCZ','JJA_SAO',
    ]
    param_list= [

        { 'model_name': 'AllRG12_ow10_172-172_rs23',
          'hidden_units': (172,172),  #(80,), #
          'learning_rate': 0.005,
          'batch_size': 1024, #16,
          'l2_reg_str': 0.00001, 'other_regime_weight': 1.,
          'max_epoch': 57, 'final_LR': 0.002026,
          'random_state': 23, 'patience': 0,
        },
        
    ]

    
    out_dir= './NN_Model_data/'
    out_fn_h= out_dir+'NN_rawVar18_scaled_12d.'
    for i,nn_param in enumerate(param_list):        
        main(tgt_crs,rg_names,nn_param,out_fn_h)


 
