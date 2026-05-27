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

This code is for cross-validation, the version of AllRG

By Daeho Jin
2026.04.14 
"""

import numpy as np
import sys
import os 
from datetime import timedelta, date
import math
import common_functions as cf

def main(model,rg_names,tgt_crs):
    print(model, tgt_crs)
    ## Parameters
    ncr= len(tgt_crs)
    nrg= len(rg_names)    
    
    mdnm1= 'ERA5'
    nyr,npt= 22,810 
    indir= './Input4ML_LcRFO/'    

    input4LCAIs= ['t700','t2m','sp','q700','q2m','t800']
    input4add_CCFs= ['skt','skTadv','wspd10m','w700','r700']
    if model[-1]=='6':
        basic_vars= ['EIS (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']
    elif model[-1]=='0':
        basic_vars= ['LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']
    else:
        sys.exit(f'model name is incompatible: {model}')
        
    nv= len(basic_vars)
    nv4lcai= len(input4LCAIs)
        
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
    
    ## Read CR_rfo
    rfo_all=[]
    for rg_nm in rg_names:
        rfos= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,var_names=tgt_crs,indir=indir,in_dim=[nyr,npt]) 
        rfos= np.asarray(rfos).reshape([ncr,nyr*npt]).T
        rfo_all.append(rfos)
    rfo_all= np.asarray(rfo_all).reshape([nrg,nyr,npt,ncr]).swapaxes(0,1).reshape([nyr*nrg*npt,ncr])
    rfos=rfo_all
    ## Check RFO data
    for k,crn in enumerate(tgt_crs):
        rfo1= rfos[:,k]
        print(crn, rfo1.min(), np.percentile(rfo1,[5,50,95]), rfo1.max())
    
    ## Prepare LCAIs
    all_lcai=[]
    for rg_nm in rg_names:
        indata= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,indir=indir,var_names=input4LCAIs+input4add_CCFs,in_dim=[nyr,npt])
        lcai1, ext1= cf.calc_LCidx(indata[:nv4lcai]), np.asarray(indata[nv4lcai:])
        #print(indata[0].shape, lcai1.shape, ext1.shape) #; sys.exit() # [nyr,npt,nvar]
        
        lcai1= lcai1.reshape([nyr*npt,-1])
        ext1= ext1.reshape([-1,nyr*npt]).T

        if model[-1]=='6':
            lcai1= np.concatenate((lcai1[:,1:2],ext1),axis=1)
        elif model[-1]=='0':
            lcai1[:,3]*=100  ## Now ECF in %    
            lcai1= np.concatenate((lcai1,ext1),axis=1)
        print(lcai1.shape)
        all_lcai.append(lcai1.reshape([nyr,npt,nv]))

    all_lcai= np.asarray(all_lcai).swapaxes(0,1).reshape([nyr*nrg*npt,nv])
    
    ## Normalize LCAIs
    lcai1= cf.normalize_x_lcai(all_lcai,basic_vars)

    ## Check LCAI data after normalization
    for k in range(nv):
        a= lcai1[:,k]
        print(basic_vars[k],a.min(), np.percentile(a,[5,50,95]),a.max())

    ## Train-Test split
    rfos= rfos.reshape([nyr,nrg*npt,ncr])
    lcai1= lcai1.reshape([nyr,nrg*npt,nv])
    X_train, X_test= lcai1[train_yr_idx,:].reshape([-1,nv]),lcai1[test_yr_idx,:].reshape([-1,nv])
    y_train, y_test= rfos[train_yr_idx,:].reshape([-1,ncr]),rfos[test_yr_idx,:].reshape([-1,ncr])
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    ## Test various alphas
    from sklearn.linear_model import Ridge 
    from sklearn.model_selection import cross_val_score, GroupKFold
    
    alphas = np.logspace(-4, 1, 26)  # Range from 0.0001 to 10

    kf= GroupKFold(n_splits=n_folds)
    cv_scores=[]
    for k in range(ncr):
        by_cr=[]        
        for alp in alphas:
            ridge= Ridge(alpha=alp,fit_intercept=False,copy_X=True)
            scores= cross_val_score(ridge,X_train,y_train[:,k],n_jobs=2,
                                    cv=kf,groups=groups,
                                    scoring='neg_root_mean_squared_error',
            )
            by_cr.append((alp,-scores.mean()))
        cv_scores.append(by_cr)
            
    return cv_scores


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

import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
def plot(model,cv_data,rg_names,tgt_crs,out_fn):
    print(model,cv_data.shape)
    ncr,n_rg,n_alp,_= cv_data.shape
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'
    
    suptit= f'{model}_{rg_names[0]}: RMSE by Ridge_alpha'
    
    ###---
    fig=plt.figure()
    fig.set_size_inches(8.5,6)    ## (lx,ly)
    plt.suptitle(suptit,fontsize=17,y=0.98,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')

    ncol,nrow=2,2
    lf,rf,bf,tf=0.02,0.98,0.08,0.9
    gapx, npnx=0.1,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.13,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    
    ix=lf; iy=tf
    ai=0
    cc= ['C0','C1',]
    ls= ['-','-.',':']
    for k,data0 in enumerate(cv_data):
        ax1=fig.add_axes([ix,iy-ly,lx,ly])
        
        for j,data1 in enumerate(data0):
            pic1= ax1.semilogx(data1[:,0],data1[:,1],
                           lw=1.5,ls=ls[j%len(ls)],c=cc[j%len(cc)],
                           label=rg_names[j])
            y_min_idx= np.argmin(data1[:,1])
            sct1= ax1.scatter([data1[y_min_idx,0],],[data1[y_min_idx,1],],s=20,c=cc[j%len(cc)],marker='v')
            #y_max_idx= np.argmax(data1[:,1])
            #sct2= ax1.scatter([data1[y_max_idx,0],],[data1[y_max_idx,1],],s=12,c=cc[j%len(cc)],marker='^')
            
        subtit= '({}) {} RFO'.format(abc[ai],tgt_crs[k]); ai+=1
        ax1.set_title(subtit,fontsize=13,x=0,ha='left')
        ax1.set_xlabel('Alpha (regularization strength)',fontsize=11)
        ax1.set_ylabel('RMSE',fontsize=11)
        ax1.grid(alpha=0.5,ls=':')
        ax1.tick_params(labelsize=10,right=True)
        
        ax1.legend(loc='best',fontsize=10,framealpha=0.9,)
        y0,y1= ax1.get_ylim()
        if y1-y0<0.06:
            yc= (y0+y1)/2
            ax1.set_ylim(yc-0.03,yc+0.03)
            
        ix+= lx+gapx
        if ix+gapx>rf:            
            ix=lf
            iy-= ly+gapy

    ###---
    plt.savefig(out_fn,bbox_inches='tight',dpi=120) #
    #plt.show()
    print(out_fn)
    return
            
if __name__=="__main__":

    model= 'LR10' 
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian']
    rg_nm= f'All_RG{len(rg_names)}'
    
    cv_score_all=[]
    if True:
        cvs= main(model,rg_names,tgt_crs)
        cv_score_all.append(cvs)

    cv_score_all= np.asarray(cv_score_all).swapaxes(0,1)  #[ncr,n_rg,n_alp,2]
    out_fig= f'./Pics/FigX01.Ridge_alpha_CV.{model}_{rg_nm}.png'
    plot(model,cv_score_all,[rg_nm,],tgt_crs,out_fig)
