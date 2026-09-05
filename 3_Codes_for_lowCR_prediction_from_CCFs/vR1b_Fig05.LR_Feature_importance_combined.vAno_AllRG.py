"""
Simple regression: LcRFO vs. Cloud-controlling factors

RFO: L1_tk, L2_tk, L_tn, and S-Clr

Variable for low cloud indices:
'T700','T2M','PS','QV700','QV2M','T800','TS'
't700','t2m','sp','q700','q2m','t800','skt'
--> 'LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', Ts (K)

Extended:
'T_adv (K/day)','WS10m (m/s)','w850 (hPa/day)','RH850 (%)','w700 (hPa/day)','RH700 (%)'
'skTadv','si10','w850','r850','w700','r700'

---

Target resolution: Monthly, and 4-deg
Sampling stratege: Quarter-sliding    
In horizontal: 4x4 box slide by 1-deg
In temporal: for 91-day (=1 season), 28-day window moves by 7-day

For a target region of 10x10-deg, it is expected to get
49 in horizontal, 10 in temporal, and 22 years= 10780

---

Plot "feature importance"  for ridge regression: bar chart of metric drop
Import bootstrap results for showing range.

By Daeho Jin
2026.08.13
---

For LCAI_compo model (LR10c)
2026.08.13
---

Combine regression slope range and performance drop for L1_tk&L2_tk / L_tn&S-Clr
2026.08.14
"""

import numpy as np
import sys
import os #.path
from datetime import timedelta, date
import math
import common_functions as cf
#import NN_classes as NNc
#from sklearn.model_selection import train_test_split
#import shap
#import joblib
from itertools import combinations

def main(model):
    
    ## Parameters
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']; ncr= len(tgt_crs)
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian']
    #cr1= tgt_crs.index(tcr_nm)
    nrg= len(rg_names)
    model_nm_tail= 'AllRG{}'.format(nrg)
    model_nm_title= model+' (All_rg)'
    
    mdnm1= 'ERA5'
    nyr,npt= 22,810 #490

    test_yr_idx= [yr-2003 for yr in [2018,2019]]
    train_yr_idx= [val for val in range(nyr) if val not in test_yr_idx]
    
    #indir= '/Users/djin1/Documents/CLD_Work/Data_Obs/ERA5/Input4ML_LcRFO/'
    indir= '../Input4ML_LcRFO/'

    ## Read coefficients or model
    indir1a= '../Coef_data_12deg/'
    indir1b= '../Model_data_12deg/'

    ## Read CR_rfo
    rfo_all=[]
    for rg_nm in rg_names:
        rfos= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,var_names=tgt_crs,indir=indir,in_dim=[nyr,npt]) #[cr_id:cr_id+1]
        rfos= np.asarray(rfos).reshape([ncr,nyr*npt]).T
        rfo_all.append(rfos)
    rfo_all= np.asarray(rfo_all).reshape([nrg,nyr,npt,ncr]).swapaxes(0,1).reshape([nyr*nrg*npt,ncr])
    
    rfos=rfo_all    
    import scipy.stats as st
    for k,crn in enumerate(tgt_crs):
        rfo1= rfos[:,k]
        #print(crn, rfo1.min(), np.percentile(rfo1,[5,50,95]), rfo1.max())
        print(crn, rfo1.min(), np.percentile(rfo1,[5,25,50,75,95]), rfo1.max(), rfo1.var(), (np.abs(rfo1-rfo1.mean()).mean())) #,st.skew(rfo1, bias=False),st.kurtosis(rfo1, fisher=True, bias=False))

    ## Prepare LC_idx
    input4LCAIs= ['t700','t2m','sp','q700','q2m','t800','skt','skTadv','wspd10m','w700','r700']
    if model=='LR10c':
        input4LCAIs.remove('skt')
        pt= '\u03B8'
        basic_vars= [pt+'_sfc (K)',pt+'_800 (K)',pt+'_700 (K)','MAdC (K)', 'CTEnt (K)', 'ELF (%)', 'T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']
    elif model=='LR10':
        basic_vars= ['LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']
    elif model[-3:]=='6v2':
        basic_vars= ['LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', 'SST (K)',]
    elif model[-1]=='6':
        basic_vars= ['EIS (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']
    else:
        print(model,"is not supported.")
        
    nv= len(basic_vars)
    nv4lcai= 6
    #basic_var_ind= [all_vars.index(vn) for vn in basic_vars]
    var_names= [v.split()[0] for v in basic_vars]
    
    all_lci=[]
    for rg_nm in rg_names:
        indata= cf.collect_data2calc_LCidx_fromSamples(
                mdnm1,rg_nm,indir=indir,var_names=input4LCAIs,in_dim=[nyr,npt])
        if model=='LR10c':            
            lci1, ext1= cf.calc_LCidx_component(indata[:nv4lcai]), np.asarray(indata[nv4lcai:])
        else:
            lci1, ext1= cf.calc_LCidx(indata[:nv4lcai+1]), np.asarray(indata[nv4lcai:])
            
        #print(indata[0].shape, lci1.shape, sst1.shape) #; sys.exit() # [nyr,npt,nvar]
        lci1= lci1.reshape([nyr*npt,-1])
        ext1= ext1.reshape([-1,nyr*npt]).T
        try:
            i= var_names.index('ELF')
            lci1[:,i]*=100  ## Now ELF in %
        except: pass

        if model[-3:]=='6v2':            
            lci1= np.concatenate((lci1,ext1[:,:1]),axis=1)
        elif model[-1]=='6':
            lci1= np.concatenate((lci1[:,1:2],ext1),axis=1)
        else:
            lci1= np.concatenate((lci1,ext1),axis=1)
        #print(lci1.shape)
        all_lci.append(lci1.reshape([nyr,npt,nv]))

    lci1= np.asarray(all_lci).swapaxes(0,1).reshape([nyr*nrg*npt,nv])
    #lci1= cf.normalize_x_lcai(lci1,basic_vars)
    for k in range(nv):
        a= lci1[:,k]
        #print(basic_vars[k],a.min(), np.percentile(a,[5,50,95]),a.max())
        #print(basic_vars[k].split()[0],a.min(), np.percentile(a,[5,25,50,75,95]),a.max(),a.var(), (np.abs(a-a.mean()).mean()),st.skew(a, bias=False),st.kurtosis(a, fisher=True, bias=False))

    ## Train-Test split
    rfos= rfos.reshape([nyr,nrg*npt,ncr])
    lci1= lci1.reshape([nyr,nrg*npt,nv])
    rfo_ref= dict(std=rfos[train_yr_idx,:].std(axis=0,ddof=1),mean=rfos[train_yr_idx,:].mean(axis=0))
    
    ## Make them anomalous
    #rfos= get_anomaly(rfos,train_yr_idx=train_yr_idx,flatten=False,standardization=True)
    lci1= get_anomaly(lci1,train_yr_idx=train_yr_idx,flatten=False,standardization=True)
    for k in range(nv):
        a= lci1[:,k]
        #print(basic_vars[k],a.min(), np.percentile(a,[5,50,95]),a.max())
        print(basic_vars[k].split()[0],a.min(), np.percentile(a,[5,25,50,75,95]),a.max(),a.var(), (np.abs(a-a.mean()).mean()),)
    
    #X_train, X_test= lci1[train_yr_idx,:].reshape([-1,nv]),lci1[test_yr_idx,:].reshape([-1,nv])
    #y_train, y_test= rfos[train_yr_idx,:].reshape([-1,ncr]),rfos[test_yr_idx,:].reshape([-1,ncr])
    X_train= lci1[train_yr_idx,:] # n_year,npt,nv 
    y_train= rfos[train_yr_idx,:] # n_year,npt,ncr
    X_test= lci1[test_yr_idx,:] #.reshape([-1,nv]) # n_year*npt,nv 
    y_test= rfos[test_yr_idx,:] #.reshape([-1,ncr]) # n_year*npt,ncr
    
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    ## Prepare shuffled X_test
    rng0= np.random.default_rng(1234)
    test_ind= np.arange(X_test.shape[0]*X_test.shape[1],dtype=int)
    test_ind_shuffled=[]
    for k in range(25):
        rng0.shuffle(test_ind)
        test_ind_shuffled.append(test_ind.copy())
        
    ## Prepare X_test, y_test, and model by region
    #model_by_region=[]
    #for i,rg_nm in enumerate(rg_names):
    if True:
        rg_nm= model_nm_tail
        print(rg_nm)
        ## Prepare model
        if True:
            ## Ridge LR
            infn= indir1a+'Coef.Ridge{}_basic_ano.{}_12deg.txt'.format(model,rg_nm)
            ridge_LR= np.zeros([ncr,nv+1])            
            with open(infn,'r') as f:
                for k,line in enumerate(f):
                    if k>0:  ## skip header
                        ww= line.strip().split(',')
                        vn0= ww[0]                        
                        vals= [float(v) for v in ww[1:]]
                        ridge_LR[k-1,:]= vals

            drop_by_cr=[]
            for k in range(ncr):
                drop_set= get_metric_drop(ridge_LR[k],X_test,y_test[:,:,k],test_ind_shuffled,rfo_ref,k)
                drop_by_cr.append(drop_set[:,0])  # MAE only
                
            ## Read bootstrap result
            K=1000; n_combi= nv*(nv-1)//2
            nks= [0,nv,nv+n_combi]
            in_dim= [ncr,K,nks[-1],2]
            dim_txt= 'x'.join([str(v) for v in in_dim])
            infn= './Boot_strap_result/Metric_drop_set_BootStrap.Ridge{}_basic_ano.{}_12deg.{}.f32dat'.format(model,rg_nm,dim_txt)
            m_drop= cf.bin_file_read2mtx(infn).reshape(in_dim)[:,:,:,0]  # MAE only

            ## Collect var_names
            all_vns= var_names
            for ind in combinations(range(nv),2): ## All possible pairs
                vn1= '\n+'.join([var_names[i] for i in ind])
                all_vns.append(vn1)
            #for ind in combinations(range(6),3): ## Mix within 5 LCAIs and SST
            #    #vn1= '+'.join([var_names[i] for i in ind[:2]])+'\n+'+var_names[ind[2]]
            #    vn1= '\n+'.join([var_names[i] for i in ind])
            #    all_vns.append(vn1)
            #print(len(all_vns), all_vns); sys.exit()

            result_by_cr=[]
            for cr1,cr_nm in enumerate(tgt_crs):
                result1=[]
                ct=0
                n_sel= [6,4,0]
                for nk0,nk1 in zip(nks[:-1],nks[1:]):
                    #mean_drop= m_drop[cr1,:,nk0:nk1].mean(axis=0)
                    mean_drop= drop_by_cr[cr1][nk0:nk1]
                    if n_sel[ct]>0:
                        tgt_vind= np.argsort(mean_drop)[-n_sel[ct]:][::-1]
                    else:
                        tgt_vind=[]
                    for tv in tgt_vind:
                        m_drop1= m_drop[cr1,:,tv+nk0]
                        cls= np.percentile(m_drop1,[2.5,97.5])
                        result1.append([all_vns[tv+nk0],mean_drop[tv],*cls])
                    ct+=1
                result_by_cr.append(result1)
            
            #model_by_region.append(dict(rg_nm=model_nm_title,bs_range=result_by_cr))
            metric_drop= dict(rg_nm=model_nm_title,bs_range=result_by_cr)

            ## Read bootstrap result
            in_dim= [ncr,K,nv+1]
            dim_txt= 'x'.join([str(v) for v in in_dim])
            infn= './Boot_strap_result/Coef_set_BootStrap.Ridge{}_basic_ano.{}_12deg.{}.f32dat'.format(model,rg_nm,dim_txt)
            bs_coef= cf.bin_file_read2mtx(infn).reshape(in_dim) #[:,:,:-1] # Exclude intercept
            #bs_coef= bs_coef #*X_train_std[None,None,:]
            #model_by_region.append(dict(rg_nm=model_nm_title,model=fim,bs_coef=bs_coef))
            result_by_cr=[]
            for cr1 in range(ncr):
                result1=[]
                for iv in range(nv):
                    bs1= bs_coef[cr1,:,iv]
                    cls= np.percentile(bs1,[2.5,97.5])
                    result1.append([var_names[iv],ridge_LR[cr1,iv],*cls])
                result_by_cr.append(result1)
                    
            regr_coef= dict(rg_nm=model_nm_title,bs_range=result_by_cr)
            
        else:
            ## NN_basic_ano
            in_fn_h= indir1b+'NN_basic_ano1.'

    ### For Figure
    outdir= '../../../Writing_LCC_LCAI/Pics_Rv1/'
    md_nm= f'{model}_{rg_nm}'    
    suptit= f'Feature importance in {model_nm_title} model' #\n(Slope_coef. \u00D7 1STD of feature variable)'

    for i,cr_idx in enumerate([[0,1,2,3],]):
        outfn= outdir+f'vRv1b_Fig05.LR_Feature_importance_combined.{md_nm}.vAno.{i}.png'
        pic_data= dict(metric_drop=metric_drop, regr_coef=regr_coef, 
                       tgt_crs=tgt_crs, cr_idx=cr_idx,
                       outfn=outfn,suptit=suptit,
        )
        plot_main(pic_data)
    return 

def get_anomaly(arr,train_yr_idx=[],flatten=True,standardization=False):
    if len(train_yr_idx)>0:
        mm= arr[train_yr_idx,:].mean(axis=0)
        if standardization:
            std= arr[train_yr_idx,:].std(axis=0,ddof=1)
    else:
        mm= arr.mean(axis=0)
        if standardization:
            std= arr.std(axis=0,ddof=1)
    ano= arr-mm[None,:]
    if standardization:
        std_non_zero= std>0.
        ano[:,std_non_zero]/=std[std_non_zero]

    if flatten:
        if ano.ndim==2:
            ano= ano.reshape(-1)
        elif ano.ndim==3:
            nyr,npt,nv= ano.shape
            ano= ano.reshape([nyr*npt,nv])
    return ano

def de_standardize(ref_std,ref_mm,target,flatten=True):
        '''
        ref.shape= [npt,ncr]
        target.shape= [ncr,nyr2,npt]
        '''
        #print('de',ref.shape, target.shape)
        target= target*ref_std.T[:,None,:]+ref_mm.T[:,None,:]
        #ncr,nyr,npt= target.shape
        return target #.reshape([ncr,-1])

def get_metric_drop(ridge_coef,X_test,yy,test_ind_shuffled,rfo_ref,cr_idx):
    rfo_std= rfo_ref['std'][:,cr_idx:cr_idx+1]
    rfo_mean= rfo_ref['mean'][:,cr_idx:cr_idx+1]

    ## Calc ref first    
    y_pred= (X_test*ridge_coef[:-1][None,None,:]).sum(axis=2)+ridge_coef[-1]
    y_pred= de_standardize(rfo_std,rfo_mean,y_pred[np.newaxis,:,:]).squeeze()
    mae_ref= np.abs(y_pred-yy).mean()
    r2_ref= 1-((yy-y_pred)**2).sum()/((yy-yy.mean())**2).sum()

    nv= X_test.shape[2]
    drop_set=[]
    ## Individual predictor
    for k in range(nv):
        mae1,r2_1= [],[]
        for shuffle_ind in test_ind_shuffled:
            X_test1= X_test.copy().reshape([-1,nv])
            X_test1[:,k]= X_test1[shuffle_ind,k]
            y_pred1= (X_test1*ridge_coef[:-1][None,:]).sum(axis=1)+ridge_coef[-1]
            y_pred1= de_standardize(rfo_std,rfo_mean,y_pred1.reshape(y_pred.shape)[np.newaxis,:,:]).squeeze() #

            mae= np.abs(y_pred1-yy).mean()
            r2= 1-((yy-y_pred1)**2).sum()/((yy-yy.mean())**2).sum()
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
            y_pred1= de_standardize(rfo_std,rfo_mean,y_pred1.reshape(y_pred.shape)[np.newaxis,:,:]).squeeze() #

            mae= np.abs(y_pred1-yy).mean()
            r2= 1-((yy-y_pred1)**2).sum()/((yy-yy.mean())**2).sum()
            mae1.append(mae)
            r2_1.append(r2)
        drop1= [np.asarray(mae1).mean()-mae_ref, r2_ref-np.asarray(r2_1).mean()]
        drop_set.append(drop1)
    '''
    ## Triplet predictor
    for ind in combinations(range(6),3): ## Mix within 5 LCAIs and SST
        mae1,r2_1= [],[]        
        for shuffle_ind in test_ind_shuffled:
            X_test1= X_test.copy().reshape([-1,nv])
            for ind1 in ind:
                X_test1[:,ind1]= X_test1[shuffle_ind,ind1]
            y_pred1= (X_test1*ridge_coef[:-1][None,:]).sum(axis=1)+ridge_coef[-1]
            y_pred1= de_standardize(rfo_std,rfo_mean,y_pred1.reshape(y_pred.shape)[np.newaxis,:,:]).squeeze() #

            mae= np.abs(y_pred1-yy).mean()
            r2= 1-((yy-y_pred1)**2).sum()/((yy-yy.mean())**2).sum()
            mae1.append(mae)
            r2_1.append(r2)
        drop1= [np.asarray(mae1).mean()-mae_ref, r2_ref-np.asarray(r2_1).mean()]
        drop_set.append(drop1)
    '''
    return np.asarray(drop_set)

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
def plot_main(pdata):
    metric_drop= pdata['metric_drop']
    regr_coef= pdata['regr_coef'] 
    cr_idx= pdata['cr_idx']
    tgt_crs= pdata['tgt_crs']
    #rg_names= pdata['rg_names']
    #var_names= pdata['var_names']

    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'
    ncr, nrg= len(cr_idx), 1

    md_nm= regr_coef['rg_nm'].split()[0]
    nv= 10 if md_nm[2]=='1' else int(md_nm[2])
    nv2= 10
    
    ###---
    fig=plt.figure()
    fig.set_size_inches(9,8.)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=16,y=0.97,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    ncol,nrow=1,4
    lf,rf,bf,tf=0.12,0.88,0.2,0.925
    gapx, npnx=0.08,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.075,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)

    ix=lf; iy=tf
    ai=0

    mk= ['o','s','^','x','v','P','h']
    cc= [f'C{v}' for v in range(10)]; n_cc= len(cc) #[::-1]
    #xtlabs= var_names
    wd=0.76
    xlocs= bar_x_locator(wd,data_dim=[1,nv2])
    wd1= wd/ncr
    xlocs1= bar_x_locator(wd1,data_dim=[ncr,nv])
    
    ### Top: Regr. Coeff. for select CRs
    ly2= ly*1.3
    ax1= fig.add_axes([ix,iy-ly2,lx,ly2])
    for j,cr1 in enumerate(cr_idx):
        rc= regr_coef['bs_range'][cr1]
        vns= [vals[0] for vals in rc]
        md= np.array([vals[1] for vals in rc])
        lower= np.array([vals[2] for vals in rc])
        upper= np.array([vals[3] for vals in rc])
        yerr = np.vstack([ md-lower, upper-md ])
        
        pic1= ax1.errorbar(xlocs1[j],md,yerr=yerr,fmt=mk[j],capsize=2,color=cc[j],label=tgt_crs[cr1])

        for i,xl1 in enumerate(xlocs1[j]):
            if upper[i]*lower[i]<=0.:
                ax1.axvspan(xl1-wd1*0.4,xl1+wd1*0.4,color='0.7',alpha=0.6)
                    
    subtit= '({}) Regression coefficients'.format(abc[ai]); ai+=1
    ax1.set_title(subtit,fontsize=12,x=0,ha='left')
    ax1.set_ylabel('Regr. Coef.',fontsize=10)
    
    ax1.legend(loc='upper left',fontsize=9,#ncol=2,
               framealpha=0.9, borderaxespad=0.,bbox_to_anchor=(1.01,1.))
    ax1.axhline(y=0.,ls=':',c='k',lw=0.8) #,zorder=0)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.set_xticks(range(len(xlocs1[0])))
    ax1.set_xticklabels(vns)
    ax1.set_xlim([-0.5,len(vns)-0.5])
    
    for x1 in range(nv-1):
        ax1.axvline(x=x1+0.5,ls='-',c='k',lw=0.8)
    yr= ax1.get_ylim()
    yr_max= np.abs(yr).max()
    if yr_max<0.5:
        ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    if yr_max<0.75:
        ax1.yaxis.set_major_locator(MultipleLocator(0.3))
    elif yr_max<1.2:
        ax1.yaxis.set_major_locator(MultipleLocator(0.4))
    ax1.grid(axis='y',ls=':',c='0.7',lw=1)
    ax1.tick_params(axis='both',which='major',labelsize=9)
    
    #ix+= lx+gapx
    iy-= ly2+gapy*1.15

    ### Bottom: Performance drop by permutation
    axes=[]
    axes_ylim=[]
    xl= xlocs[0]
    for j,cr1 in enumerate(cr_idx):
        ax2= fig.add_axes([ix,iy-ly,lx,ly])

        rc= metric_drop['bs_range'][cr1]
        vns= [vals[0] for vals in rc]
        md= np.array([vals[1] for vals in rc])*100.
        lower= np.array([vals[2] for vals in rc])*100.
        upper= np.array([vals[3] for vals in rc])*100.
        yerr = np.vstack([ md-lower, upper-md ])
        
        pic2= ax2.errorbar(xl,md,yerr=yerr,fmt=mk[j],capsize=2,color=cc[j])

        subtit= '({}) {}: MAE increase by feature permutation '.format(abc[ai],tgt_crs[cr1],); ai+=1
        ax2.set_title(subtit,fontsize=12,x=0,ha='left')

        #ax1.axvline(0.,ls='--',c='k',lw=0.8)
        ax2.set_xticks(xl)
        ax2.set_xticklabels(vns)
        ax2.set_ylabel(r"$\Delta$MAE (%)")
        #ax1.invert_yaxis()
        ax2.axvline(x=5.49,ls='-',c='k',lw=0.6)
        ax2.axvline(x=5.51,ls='-',c='k',lw=0.6)
        ax2.axhline(y=0.,ls=':',c='k',lw=0.8) #,zorder=0)
        
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.grid(axis='y',ls=':',c='0.7',lw=1)
        ax2.tick_params(axis='both',which='major',labelsize=9)
            
        if False: #True: # j==0:
            ax1.legend(loc='upper left',fontsize=9.5,framealpha=0.9, borderaxespad=0.,bbox_to_anchor=(1.02,1.))
                            
        ix+= lx+gapx
        if ix+lx>rf:
            ix=lf
            iy-= ly+gapy

    '''
    axes_ymax= np.asarray(axes_ylim)
    y_common= axes_ymax[axes_ymax<2].max()
    for i,ax1 in enumerate(axes):
        if axes_ymax[i]<2:
            ax1.set_ylim([-y_common,y_common])
        else:
            ax1.set_ylim([-axes_ymax[i],axes_ymax[i]])
    '''

    ###---
    print(pdata['outfn'])
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150) #
    #plt.show()

    return

def bar_x_locator(width,data_dim=[1,10]):
    """
    Depending on width and number of bars,
    return bar location on x axis
    Input width: (0,1) range
    Input data_dim: [# of vars, # of bins]
    Output locs: list of 1-D array(s)
    """
    xx=np.arange(data_dim[1])
    shifter= -width/2*(data_dim[0]-1)
    locs=[]
    for x1 in range(data_dim[0]):
        locs.append(xx+(shifter+width*x1))
    return locs

if __name__=="__main__":
    '''
    tgt_boxes_JJA= [
        ('Peruvian',(-20,-10,-100,-90)), #-92,-84)), #-90,-80)),
        ('Namibian',(-20,-10,-10,0)), #-4,4)),
        ('Californian',(20,30,-145,-135)), #-136,-128)),
        #('Australian',(-40,-30,75,85)), #96,104)),
    ]

    tgt_boxes_DJF= [
        ('Peruvian',(-30,-20,-90,-80)), #-92,-84)), #-90,-80)),
        ('Namibian',(-25,-15,-5,5)), #-4,4)),
        #('Californian',(20,30,-145,-135)), #-136,-128)),
        ('Australian',(-35,-25,95,105)), #96,104)),

    tgt_cr_groups= [
        ('L1_tk',(11,13)),
        ('L2_tk',(10,12)),
        ('L_tn',(14,)),
        ('S-Clr',(153,)),
    ]
    '''

    
    #tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']
    #basic_vars= ['LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']
    #basic_vars= ['EIS (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']

    model= 'LR10' #  'LR10c' #'LR6v2' #
    main(model)

