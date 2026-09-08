"""
Draw heat maps to check if one region's LR model can be used in other regions

Main metric: R^2 

By Daeho Jin
2026.03.12
---

Add LR10_AllRG model results, test vs. train
2026.08.05
"""

import numpy as np
import sys
import os 
from datetime import timedelta, date
import math
import common_functions as cf

def get_score(model,tcr_nm):
    print('**-- Start:',tcr_nm)
    
    ## Parameters
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian']
    cr1= tgt_crs.index(tcr_nm)
    
    mdnm1= 'ERA5'
    nyr,npt= 22,810

    test_yr_idx= [yr-2003 for yr in [2018,2019]]
    train_yr_idx= [val for val in range(nyr) if val not in test_yr_idx]
    nyr2= len(test_yr_idx)
    
    indir= './Input4ML_LcRFO/'

    ## Read coefficients or model
    indir1a= './LR_Coef_data/'

    input4LCAIs= ['t700','t2m','sp','q700','q2m','t800','skt',]
    input4add_CCFs= ['skTadv','wspd10m','w700','r700']
    if model[-1]=='6':
        basic_vars= ['EIS (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']
    elif model[-1]=='0':
        basic_vars= ['LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']
    else:
        sys.exit(f'model name is incompatible: {model}')
        
    nv= len(basic_vars)
    nv4lcai= len(input4LCAIs)    
    
    ## Prepare X_test, y_test, and model by region
    input_by_region=[]
    model_by_region=[]
    for i,rg_nm in enumerate(rg_names):
        print(rg_nm)
        ## Read CR_rfo
        rfo1= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,var_names=[tcr_nm,],indir=indir,in_dim=[nyr,npt])[0]
        rfo1= rfo1.reshape(-1)
        print(rfo1.shape) #[nyr*npt,ncr]
        print(tcr_nm, rfo1.min(), np.percentile(rfo1,[5,50,95]), rfo1.max())    
        ncr=1
        
        ## Prepare LC_idx
        indata= cf.collect_data2calc_LCidx_fromSamples(
            mdnm1,rg_nm,indir=indir,var_names=input4LCAIs+input4add_CCFs,in_dim=[nyr,npt])
        lcai1, ext1= cf.calc_LCidx(indata[:nv4lcai]), np.asarray(indata[nv4lcai-1:])
        #print(indata[0].shape, lcai1.shape, sst1.shape) #; sys.exit() # [nyr,npt,nvar]
        
        lcai1= lcai1.reshape([nyr*npt,-1])
        ext1= ext1.reshape([-1,nyr*npt]).T

        if nv==6:
            lcai1= np.concatenate((lcai1[:,1:2],ext1),axis=1)
        else:
            lcai1[:,3]*=100  ## Now ECF in %    
            lcai1= np.concatenate((lcai1,ext1),axis=1)
        print(lcai1.shape)

        for k in range(nv):
            a= lcai1[:,k]
            print(basic_vars[k],a.min(), np.percentile(a,[5,50,95]),a.max())

        ## Train-Test split
        rfo1= rfo1.reshape([nyr,npt])
        lcai1= lcai1.reshape([nyr,npt,nv])
        
        ## Standardization
        #rfos1= cf.get_anomaly(rfos,train_yr_idx=train_yr_idx,flatten=False,standardization=True)
        lcai1= cf.get_anomaly(lcai1,train_yr_idx=train_yr_idx,flatten=False,standardization=True)
        for k in range(nv):
            a= lcai1[:,:,k]
            print(basic_vars[k],a.min(), np.percentile(a,[5,50,95]),a.max())
            
        X_train, X_test= lcai1[train_yr_idx,:].reshape([-1,nv]),lcai1[test_yr_idx,:].reshape([-1,nv])
        y_train, y_test= rfo1[train_yr_idx,:],rfo1[test_yr_idx,:].reshape(-1)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        
        # Need mean and std for de-normalization
        rfos_mm,rfos_std= y_train.mean(axis=0), y_train.std(axis=0,ddof=1)
        
        # Save test data
        input_by_region.append(dict(rg_nm=rg_nm,X=X_test,y=y_test,y_stat=(rfos_mm,rfos_std)))

        ## Prepare model
        infn= indir1a+'Coef.Ridge{}_basic_ano.{}_12deg.txt'.format(model,rg_nm)
        with open(infn,'r') as f:
            for k,line in enumerate(f):
                if k>0:  ## skip header
                    ww= line.strip().split(',')
                    vn0= ww[0]
                    if vn0==tcr_nm:
                        vals= [float(v) for v in ww[1:]]
                        ridge_LR= np.asarray(vals)
                        break
        if not isinstance(ridge_LR, np.ndarray):
            print('Ridge_LR read error',infn,tcr_nm)
        model_by_region.append(dict(rg_nm=rg_nm,model=ridge_LR))

    ## Calculate R^2 for all combinations
    results_all=[]
    for j,model1 in enumerate(model_by_region):
        md= model1['model']
        results_by_model=[]
        for i,data1 in enumerate(input_by_region):
            X= data1['X']
            y_true= data1['y']
            y_stat= data1['y_stat']
            
            ## Ridge LR
            y_pred= np.dot(X,md[:-1])+md[-1]
            y_pred= y_pred.reshape([nyr2,npt])*y_stat[1]+y_stat[0]
            y_pred= y_pred.reshape(-1)
            
            r2= 1-((y_true-y_pred)**2).sum()/((y_true-y_true.mean())**2).sum()
            #mae= np.abs(y_pred-y_true).mean()
            results_by_model.append(r2)
        results_all.append(results_by_model)
        
    ## AllRG model
    rg_nm= 'AllRG6'
    infn= indir1a+'Coef.Ridge{}_basic_ano.{}_12deg.txt'.format(model,rg_nm)
    with open(infn,'r') as f:
        for k,line in enumerate(f):
            if k>0:  ## skip header
                ww= line.strip().split(',')
                vn0= ww[0]
                if vn0==tcr_nm:
                    vals= [float(v) for v in ww[1:]]
                    break
    allrg_model= np.asarray(vals)
    
    results_allrg=[]
    if True:
        md= allrg_model
        for i,data1 in enumerate(input_by_region):
            X= data1['X']
            y_true= data1['y']
            y_stat= data1['y_stat']
            
            ## Ridge LR
            y_pred= np.dot(X,md[:-1])+md[-1]
            y_pred= y_pred.reshape([nyr2,npt])*y_stat[1]+y_stat[0]
            y_pred= y_pred.reshape(-1)
            
            r2= 1-((y_true-y_pred)**2).sum()/((y_true-y_true.mean())**2).sum()
            #mae= np.abs(y_pred-y_true).mean()
            results_allrg.append(r2)
    
    return rg_names,np.asarray(results_all),np.asarray(results_allrg)

    
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
def plot_main(pdata):
    results= pdata['results']
    tcr_names= pdata['tcr_names']
    
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'

    ###---
    fig=plt.figure()
    fig.set_size_inches(7.5,9)    ## (lx,ly)
    
    plt.suptitle(pdata['suptit'],fontsize=17,y=0.965,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    
    ncol,nrow=2,2
    lf,rf,bf,tf=0.02,0.98,0.15,0.92
    gapx, npnx=0.175,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.13,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)

    ix=lf; iy=tf
    gapx0= 0.01
    lx0= lx/6.4 
    lx1= lx0*6 

    cm = plt.get_cmap('inferno_r')
    props= dict(cmap=cm,vmin=0.1,vmax=0.9,origin='upper',alpha=0.7)

    ai=0
    for ii,(rg_nm,output1,output2) in enumerate(results):
        ax1=fig.add_axes([ix,iy-ly,lx1,ly])
        pic1= ax1.imshow(output1.T,**props)
        # Confusion matrix
        nv= len(rg_nm)
        ax1.set_xticks(range(nv))
        ax1.set_xticklabels(rg_nm,rotation=35,ha='right',va='top')
        ax1.set_yticks(range(nv))
        ax1.set_yticklabels(rg_nm,rotation=35,ha='right',va='top')
        ax1.tick_params(labelsize=9)
        ax1.set_xlabel('Model Trained',fontsize=11,weight='bold')
        ax1.set_ylabel('Tested',fontsize=11,weight='bold')

        subtit= '({}) {}'.format(abc[ai],tcr_names[ii]); ai+=1
        ax1.set_title(subtit,fontsize=14,x=0,ha='left')
        ax1.plot([-0.5,nv-0.5],[-0.5,nv-0.5],ls='--',lw=3,c='silver',alpha=0.75)
        write_val(ax1,output1.T,threshold=0.695,fmt='{:.2f}')
        
        # AllRG column
        ix= ix+lx1+gapx0
        ax0= fig.add_axes([ix,iy-ly,lx0,ly])
        output2= output2.reshape([-1,1])
        pic0= ax0.imshow(output2,**props)
        ax0.set_yticklabels('')
        ax0.set_xticks(range(1))
        ax0.set_xticklabels(['All_rg',],rotation=35,ha='right',va='top')
        ax0.tick_params(axis='y', right=True)
        write_val(ax0,output2,threshold=0.695,fmt='{:.2f}')
        
        ix+= lx0+gapx
        cnt= ix-gapx
        if ix>rf:        
            ix=lf
            iy-= ly+gapy
                 
    ## Colorbar
    hh= 0.02
    loc0= [0.15,iy-hh,0.7,hh]
    tt= np.round(np.arange(1,10)/10,1)
    cb0 =draw_colorbar(fig,pic1,loc0,ft=9,extend='both',tt=tt,tt2=tt)
    cb0.ax.set_xlabel(r'$R^2$',fontsize=10)
                    
    ###---
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=125) #
    #plt.show()
    print(pdata['outfn'])
    return

def write_val(ax1,arr1,threshold=0.7,fmt='{:.1f}'):
    ny,nx= arr1.shape
    props= dict(ha='center',va='center',stretch='semi-condensed',fontsize=11,weight='bold')
    for j in range(ny):
        for i in range(nx):
            if arr1[j,i]>threshold: #abs(arr1[j,i])>threshold:                
                ax1.text(i,j,fmt.format(arr1[j,i]),color='c',**props)
    return

def draw_colorbar(fig,pic1,loc,ft=10,extend='both',tt=[],tt2=[]): 

    cb_ax = fig.add_axes(loc)  ##<= (left,bottom,width,height)
    if loc[2]<loc[3]:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='vertical',ticks=tt,extend=extend)
        cb.ax.set_yticklabels(tt2,size=ft,stretch='condensed')
    else:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='horizontal',ticks=tt,extend=extend)
        cb.ax.set_xticklabels(tt2,size=ft,stretch='condensed')
    cb_ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    return cb

if __name__=="__main__":

    model= 'LR10' #'LR6' #
    
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']
    result_by_tcr=[]
    for i,tcr_nm in enumerate(tgt_crs):
        output1= get_score(model,tcr_nm)
        result_by_tcr.append(output1)

    ## Plot the results
    outdir= './Pics/'
    mdnm= model 

    if True:
        outfn= outdir+'Fig09.Model_Interchangable_byRegion.R2_{}.png'.format(mdnm)
        suptit= r'$R^2$ Metric for {} Model Transferability'.format(mdnm)
        pic_data= dict(results= result_by_tcr,
                       tcr_names=tgt_crs,
                       outfn=outfn,suptit=suptit,
        )
        plot_main(pic_data)

