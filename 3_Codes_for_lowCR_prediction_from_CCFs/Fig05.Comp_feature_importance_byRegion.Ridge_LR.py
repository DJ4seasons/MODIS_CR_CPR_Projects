"""
Plot "feature importance"  for ridge regression:
Bar chart of standardized coefficients

By Daeho Jin
2026.03.12
"""

import numpy as np
import sys
import os 
from datetime import timedelta, date
import math
import common_functions as cf

def main(model):
    
    ## Parameters
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']; ncr= len(tgt_crs)
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian']
    
    mdnm1= 'ERA5'
    nyr,npt= 22,810 

    test_yr_idx= [yr-2003 for yr in [2018,2019]]
    train_yr_idx= [val for val in range(nyr) if val not in test_yr_idx]

    ## Read regional samples
    indir= './Input4ML_LcRFO/'

    ## Read coefficients or model
    indir1a= './LR_Coef_data/'

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
    
    ## Prepare X_test, y_test, and model by region
    model_by_region=[]
    for i,rg_nm in enumerate(rg_names):
        print(rg_nm)
        ## Read CR_rfo
        rfos= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,var_names=tgt_crs,indir=indir,in_dim=[nyr,npt])
        rfos= np.asarray(rfos).reshape([ncr,nyr*npt]).T
    
        ## Prepare LC_idx
        indata= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,indir=indir,var_names=input4LCAIs+input4add_CCFs,in_dim=[nyr,npt])
        lcai1, ext1= cf.calc_LCidx(indata[:nv4lcai]), np.asarray(indata[nv4lcai:])
        #print(indata[0].shape, lcai1.shape, ext1.shape) #; sys.exit() # [nyr,npt,nvar]
        
        lcai1= lcai1.reshape([nyr*npt,-1])
        ext1= ext1.reshape([-1,nyr*npt]).T

        if nv==6:
            lcai1= np.concatenate((lcai1[:,1:2],ext1),axis=1)
        else:
            lcai1[:,3]*=100  ## Now ECF in %    
            lcai1= np.concatenate((lcai1,ext1),axis=1)
        #print(lcai1.shape)

        ## Normalize LCAIs
        lcai1= cf.normalize_x_lcai(lcai1,basic_vars)
        for k in range(nv):
            a= lcai1[:,k]
            print(basic_vars[k],a.min(), np.percentile(a,[5,50,95]),a.max())
    

        ## Train-Test split
        rfos= rfos.reshape([nyr,npt,ncr])
        lcai1= lcai1.reshape([nyr,npt,nv])
        X_train, X_test= lcai1[train_yr_idx,:].reshape([-1,nv]),lcai1[test_yr_idx,:].reshape([-1,nv])
        y_train, y_test= rfos[train_yr_idx,:].reshape([-1,ncr]),rfos[test_yr_idx,:].reshape([-1,ncr])
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        ## STD of X_train for standardization
        X_train_std= np.std(X_train,axis=0)
        
        ## Prepare Ridge LR model
        infn= indir1a+'Coef.Ridge{}_basic_scaledX.{}_12deg.txt'.format(model,rg_nm)
        ridge_LR= np.zeros([ncr,nv+1])            
        with open(infn,'r') as f:
            for k,line in enumerate(f):
                if k>0:  ## skip header
                    ww= line.strip().split(',')
                    vn0= ww[0]                        
                    vals= [float(v) for v in ww[1:]]
                    ridge_LR[k-1,:]= vals

        feature_im= ridge_LR[:,:-1]*X_train_std[None,:]
        #-- Make EIS value positive
        eis_idx= basic_vars.index('EIS (K)')        
        for k in range(ncr):
            if feature_im[k,eis_idx]<0:
                feature_im[k,:]*=-1.
        model_by_region.append(dict(rg_nm=rg_nm,model=feature_im))
    
    ### For Figure
    outdir= './Pics/'
    outfn= outdir+f'Fig05.Feature_importance_byRegion.LCAI_ridgeLR{nv}.png'
    suptit= f'Standardized Coefficients in LR{nv} model\n(Slope_coef. \u00D7 1STD of feature variable)'
    pic_data= dict(data= model_by_region, var_names=[vn.split()[0] for vn in basic_vars],
                   tgt_crs=tgt_crs, rg_names=rg_names,
                   outfn=outfn,suptit=suptit,
    )
    plot_main(pic_data)
    return 

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
def plot_main(pdata):
    data= pdata['data']
    tgt_crs= pdata['tgt_crs']
    rg_names= pdata['rg_names']
    var_names= pdata['var_names']

    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'
    ncr, nrg= len(tgt_crs), len(rg_names)
    nv= len(var_names)
    
    ###---
    fig=plt.figure()
    fig.set_size_inches(7.6,8.)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=16,y=0.97,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    ncol,nrow=1,3.5
    lf,rf,bf,tf=0.08,0.92,0.24,0.925
    gapx, npnx=0.06,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.075,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)

    ix=lf; iy=tf
    ai=0
    
    cc= [f'C{v}' for v in range(10)]; n_cc= len(cc) #[::-1]
    xtlabs= var_names
    wd=0.8
    wd1= wd/nrg
    xlocs= bar_x_locator(wd1,data_dim=[nrg,nv])

    axes=[]
    axes_ylim=[]
    for j in range(ncr):
        ax1= fig.add_axes([ix,iy-ly,lx,ly])
        
        ci=0
        for xl,data1 in zip(xlocs,data):
            rg_nm, coeff= data1['rg_nm'], data1['model'][j][:nv]
            rnm= rg_nm[:4]+'\n'+rg_nm[4:]
            bar1= ax1.bar(xl,coeff,width=wd1,color=cc[ci],alpha=0.8,label=rnm); ci+=1

        subtit= '({}) For {} '.format(abc[ai],tgt_crs[ai],); ai+=1
        ax1.set_title(subtit,fontsize=12,x=0,ha='left')
        ax1.set_xticks(range(nv))
        ax1.set_xticklabels(xtlabs)

        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

        #ax1.set_ylabel('Feature Importance',fontsize=10)
        ax1.grid(axis='y',ls=':',c='0.7',lw=1)
        ax1.tick_params(axis='both',which='major',labelsize=9)
        ax1.axhline(y=0,ls='--',lw=1,c='0.15')
        for xl in np.arange(0.5,nv-1,1):
            ax1.axvline(x=xl,ls='--',lw=1,c='0.4')
        ax1.set_xlim([-0.5,nv-0.5])
            
        if j==0:
            ax1.legend(loc='upper left',fontsize=9.5,framealpha=0.9, borderaxespad=0.,bbox_to_anchor=(1.02,1.))
        axes.append(ax1)
        axes_ylim.append(ax1.get_ylim())
        ix+= lx+gapx
        if ix+lx>rf:
            ix=lf
            iy-= ly+gapy

    ## Make consistent y-scale
    axes_ylim= np.asarray(axes_ylim)
    yr_max= np.max(axes_ylim,axis=0)
    yr_min= np.min(axes_ylim,axis=0)
    for i,ax1 in enumerate(axes):
        ax1.set_ylim([yr_min[0],yr_max[1]])
    
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

    model= 'LR10' #'LR6' #
    main(model)

