"""
Compare the performances of regression models

R^2 vs. MAE
Collect y_pred from all regions, and calculate a score 

By Daeho Jin
2026.04.15
"""

import numpy as np
import sys
import os 
from datetime import timedelta, date
import math
import common_functions as cf
import NN_classes_y5 as NNc5

def get_y_pred(rg_nm,tgt_crs): 
    print(rg_nm)
    ncr= len(tgt_crs)
    
    ## Parameters
    mdnm1= 'ERA5'
    nyr,npt= 22,810 
    indir= './Input4ML_LcRFO/'

    test_yr_idx= [yr-2003 for yr in [2018,2019]]
    train_yr_idx= [val for val in range(nyr) if val not in test_yr_idx]

    ## Read CR_rfo
    rfos= cf.collect_data2calc_LCidx_fromSamples(
        mdnm1,rg_nm,var_names=tgt_crs,indir=indir,in_dim=[nyr,npt])
    rfos= np.asarray(rfos).reshape([ncr,nyr*npt]).T

    
    ## Parameters for simple and ridge regression
    input4LCAIs= ['t700','t2m','sp','q700','q2m','t800']
    input4add_CCFs= ['skt','skTadv','wspd10m','w700','r700']
    basic_vars= ['LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']

    nv= len(basic_vars)
    nv4lcai= len(input4LCAIs)        
    
    ## Prepare LC_idx
    indata= cf.collect_data2calc_LCidx_fromSamples(
        mdnm1,rg_nm,indir=indir,var_names=input4LCAIs+input4add_CCFs,in_dim=[nyr,npt])
    lcai1, ext1= cf.calc_LCidx(indata[:nv4lcai]), np.asarray(indata[nv4lcai:])
    print(indata[0].shape, lcai1.shape, ext1.shape) #; sys.exit() # [nyr,npt,nvar]
    
    lcai1= lcai1.reshape([nyr*npt,-1])
    ext1= ext1.reshape([-1,nyr*npt]).T
    
    lcai1[:,3]*=100  ## Now ECF in %    
    lcai1= np.concatenate((lcai1,ext1),axis=1)
    print(lcai1.shape)

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

    output=dict(ref= [tgt_crs,y_test.T])
    
    ## Read coefficients or model
    indir1a= './LR_Coef_data/'
    
    ##-- Simple LR
    infn= indir1a+'Coef.SimpleLR_basic_scaledX.{}_12deg.txt'.format(rg_nm)
    
    simple_LR= np.empty([ncr,nv4lcai,2])
    with open(infn,'r') as f:
        for k,line in enumerate(f):
            if k>0:  ## skip header
                ww= line.strip().split(','); print(ww)
                vns= ww[0].split('-')
                if len(vns)==2:
                    vn0,vn1= vns
                elif len(vns)==3:
                    vn0= vns[0]; vn1= '-'.join(vns[1:])
                else:
                    print(vns); sys.exit()
                slope,intp= [float(v) for v in ww[1:3]]
                i,j= basic_vars.index(vn0), tgt_crs.index(vn1)
                simple_LR[j,i,:]= [slope,intp]

    simple_LR_output= [] 
    for j in range(ncr):
        yy= y_test[:,j]
        by_tcr=[]
        for i in range(nv4lcai):
            xx= X_test[:,i]        
            y_pred= xx*simple_LR[j,i,0]+simple_LR[j,i,1]
            by_tcr.append(y_pred)
        simple_LR_output.append(by_tcr)
    simple_LR_output= np.asarray(simple_LR_output)
    #print(simple_LR_output.shape); sys.exit() # [ncr,nv4lcai,nyr*npt]
    output['simple_LR']= [tgt_crs,basic_vars[:nv4lcai],simple_LR_output]
            
    ##-- Ridge LR10    
    infn= indir1a+'Coef.RidgeLR10_basic_scaledX.{}_12deg.txt'.format(rg_nm)    
    
    ridge_LR= np.zeros([ncr,nv+1])
    with open(infn,'r') as f:
        for k,line in enumerate(f):
            if k>0:  ## skip header
                ww= line.strip().split(',')
                vn0= ww[0]
                vals= [float(v) for v in ww[1:]]
                ridge_LR[k-1,:]= vals

    ridge_LR_output= [] 
    for j in range(ncr):
        yy= y_test[:,j]
        y_pred= (X_test*ridge_LR[j,:-1][None,:]).sum(axis=1)+ridge_LR[j,-1]
        ridge_LR_output.append(y_pred)
    ridge_LR_output= np.asarray(ridge_LR_output)
    #print(ridge_LR_output.shape); sys.exit() # [ncr,nyr*npt]
    output['ridge_LR']= [tgt_crs,ridge_LR_output]

    ##-- LR10 for All_RG6
    infn2= indir1a+'Coef.RidgeLR10_basic_scaledX.{}_12deg.txt'.format('AllRG6')
    ridge_LR2= np.zeros([ncr,nv+1])
    with open(infn2,'r') as f:
        for k,line in enumerate(f):
            if k>0:  ## skip header
                ww= line.strip().split(',')
                vn0= ww[0]
                vals= [float(v) for v in ww[1:]]
                ridge_LR2[k-1,:]= vals

    ridge_LR2_output= [] 
    for j in range(ncr):
        yy= y_test[:,j]
        y_pred= (X_test*ridge_LR2[j,:-1][None,:]).sum(axis=1)+ridge_LR2[j,-1]
        ridge_LR2_output.append(y_pred)
    ridge_LR2_output= np.asarray(ridge_LR2_output)
    output['ridge_LR2']= [tgt_crs,ridge_LR2_output]

    ##-- Ridge LR6
    basic_vars2= ['EIS (K)', 'SST (K)','T_adv (K/day)','WS10m (m/s)','w700 (Pa/s)','RH700 (%)']
    v_idx= [basic_vars.index(vn) for vn in basic_vars2]
    X_test2= X_test[:,v_idx]
    nv2= len(basic_vars2)
    
    infn= indir1a+'Coef.RidgeLR{}_basic_scaledX.{}_12deg.txt'.format(nv2, rg_nm)    
    
    ridge_LR= np.zeros([ncr,nv2+1])
    with open(infn,'r') as f:
        for k,line in enumerate(f):
            if k>0:  ## skip header
                ww= line.strip().split(',')
                vn0= ww[0]
                vals= [float(v) for v in ww[1:]]
                ridge_LR[k-1,:]= vals

    ridge_LR_output= [] 
    for j in range(ncr):
        yy= y_test[:,j]
        y_pred= (X_test2*ridge_LR[j,:-1][None,:]).sum(axis=1)+ridge_LR[j,-1]
        ridge_LR_output.append(y_pred)
    ridge_LR_output= np.asarray(ridge_LR_output)
    #print(ridge_LR_output.shape); sys.exit() # [ncr,nyr*npt]
    output['ridge_LR0']= [tgt_crs,ridge_LR_output]

    
    ##-- Neural Net_RawV
    ## Prepare Phys vars
    raw_var_names= [
        'SST (K)','T2M (K)','T850 (K)','T700 (K)','T500 (K)',
        'Q2M (g/kg)','Q850 (g/kg)','Q700 (g/kg)','Q500 (g/kg)',
        'WS10m (m/s)','WS850 (m/s)','WS700 (m/s)',
        'PS (Pa)','T_adv (K/day)',
        'q850_adv (g/kg/day)','q700_adv (g/kg/day)'
    ]
    raw_vns= ['skt','t2m','t850','t700','t500',
              'q2m','q850','q700','q500',
              'wspd10m','wspd850', 'wspd700',
              'sp','skTadv','q850adv', 'q700adv',]
    nv= len(raw_vns)
    clim_vars,clim_vnames= ['skt','sp'], ['SST_clim (K)', 'PS_clim (hPa)',]
    clim_vidx= [raw_vns.index(name) for name in clim_vars]  # index for skt and sp
    nv2= len(clim_vars)
    
    indata= cf.collect_data2calc_LCidx_fromSamples(
        mdnm1,rg_nm,var_names=raw_vns,indir=indir,in_dim=[nyr,npt])
    indata= np.concatenate([np.expand_dims(arr,axis=-1) for arr in indata],axis=-1)
    print(indata.shape, ) #; sys.exit() # [nyr,npt,nvar]

    ## Clim data
    clim_indata= indata[:,:,clim_vidx].mean(axis=0) #[npt,nv2]
    clim_indata= np.tile(clim_indata,[nyr,1,1]).reshape([nyr*npt,nv2])
    
    ## Normalize
    indata= cf.normalize_x_raw(indata.reshape([nyr*npt,nv]),raw_vns)
    for k in range(nv):
        a= indata[:,k]
        print(raw_vns[k],a.min(), np.percentile(a,[5,50,95]),a.max())

    clim_indata= cf.normalize_x_raw(clim_indata,clim_vars)
    for k in range(nv2):
        a= clim_indata[:,k]
        print(clim_vars[k]+'_clim',a.min(), np.percentile(a,[5,50,95]),a.max()) 
    indata= np.concatenate((indata,clim_indata),axis=1)
    nv+=nv2
    raw_var_names+= clim_vnames
    
    ## Train-Test split
    indata= indata.reshape([nyr,npt,nv])
    X_train, X_test= indata[train_yr_idx,:].reshape([-1,nv]),indata[test_yr_idx,:].reshape([-1,nv])
    y_train, y_test= rfos[train_yr_idx,:].reshape([-1,ncr]),rfos[test_yr_idx,:].reshape([-1,ncr])
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    indir1b= './NN_Model_data/'
    out_fn_h= indir1b+f'NN_rawVar{nv}_scaled_12d.'
    loaded_model= NNc5.NeuralNetworkRegressor()
    allrg_nm= 'AllRG6_ow10_100-100_rs37' 
    loaded_model.load_model(out_fn_h+f'{allrg_nm}.h5')
    loaded_model.get_model_summary()
    predictions= loaded_model.predict(X_test)
    
    NN_output= predictions.T 
    #print(NN_output.shape); sys.exit() # [ncr,nyr*npt]
    output['NN_rawVar1']= [tgt_crs,NN_output]
    
    
    return output


import matplotlib as mpl
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
def plot_main(pdata):
    output1= pdata['results']
    tgt_crs= pdata['tgt_crs']
    
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'

    ###---
    fig=plt.figure()
    fig.set_size_inches(6,6)    ## (lx,ly)
    
    plt.suptitle(pdata['suptit'],fontsize=16,y=0.98,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    
    ncol,nrow=2,2
    lf,rf,bf,tf=0.04,0.96,0.16,0.92
    gapx, npnx=0.085,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.095,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)

    ix=lf; iy=tf

    cc= [f'C{v}' for v in range(10)][::-1]
    sct_props_n= dict(s=75,alpha=0.9)
    sct_props_a= dict(s=50,alpha=0.9)
    sct_props_l= dict(s=40,alpha=0.8)

    axes,yr,xr=[],[],[]
    tcr_nms,y_true= output1['ref']
    for ii,tcr in enumerate(tgt_crs):
        ax1=fig.add_axes([ix,iy-ly,lx,ly])
        y_true1= y_true[ii]

        ## NNet_rawVar
        tcr_nms,result1= output1['NN_rawVar1']
        res1= result1[ii]
        mae= np.abs(res1-y_true1).mean()
        r2_score= 1-((y_true1-res1)**2).sum()/((y_true1-y_true1.mean())**2).sum()
        sct4= ax1.scatter(mae*100,np.clip(r2_score,-0.5,1.0),c='k',marker='*',label='NNet (All_rg)',**sct_props_n)
        
        
        ## Ridge LR
        output1c= output1['ridge_LR2']  #['ridge_LR']= (tgt_crs,ridge_LR_output)
        tcr_nms,result1= output1c
        res1= result1[ii]
        mae= np.abs(res1-y_true1).mean()
        r2_score= 1-((y_true1-res1)**2).sum()/((y_true1-y_true1.mean())**2).sum()
        sct2= ax1.scatter(mae*100,np.clip(r2_score,-0.5,1.0),c='r',marker='d',label='LR10 (All_rg)',**sct_props_a)

        tcr_nms,result1= output1['ridge_LR'] 
        res1= result1[ii]
        mae= np.abs(res1-y_true1).mean()
        r2_score= 1-((y_true1-res1)**2).sum()/((y_true1-y_true1.mean())**2).sum()
        sct2= ax1.scatter(mae*100,np.clip(r2_score,-0.5,1.0),c='b',marker='d',label='LR10',**sct_props_a)

        tcr_nms,result1= output1['ridge_LR0'] 
        res1= result1[ii]
        mae= np.abs(res1-y_true1).mean()
        r2_score= 1-((y_true1-res1)**2).sum()/((y_true1-y_true1.mean())**2).sum()
        sct2= ax1.scatter(mae*100,np.clip(r2_score,-0.5,1.0),c='g',marker='^',label='LR6',**sct_props_a)

        
        ## Simple LR
        #output1a= output1['simple_LR'] #simple_LR= (tgt_crs,basic_vars,simple_LR_output)
        tcr_nms,v_nms,result1= output1['simple_LR']        
        for jj,vn in enumerate(v_nms):
            res1= result1[ii][jj]
            mae= np.abs(res1-y_true1).mean()
            r2_score= 1-((y_true1-res1)**2).sum()/((y_true1-y_true1.mean())**2).sum()
            sct1= ax1.scatter(mae*100,np.clip(r2_score,-0.5,1.0),c=cc[jj],marker='o',label=vn.split()[0],**sct_props_l)
            
        
        ##--
        subtit= '({}) {}'.format(abc[ii],tcr)
        ax1.set_title(subtit,fontsize=12,x=0,ha='left')
        ax1.grid(ls=':')
        ax1.tick_params(labelsize=9)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        if ii%ncol==ncol-1:
            ax1.legend(loc='upper left',bbox_to_anchor=[1.04,1.],fontsize=9,borderaxespad=0)
        if ii%ncol==0:
            ax1.set_ylabel(r'$R^2$',fontsize=10)
        if ii//ncol==nrow-1:
            ax1.set_xlabel('Mean Abs. Error (%)')

        axes.append(ax1)
        yr.append(ax1.get_ylim())
        xr.append(ax1.get_xlim())
        
        ix+= lx+gapx
        if ix+gapx>rf:
            ix=lf
            iy-= ly+gapy

    yr,xr= np.asarray(yr), np.asarray(xr)
    yr1= [min(yr[:,0].min(),-0.01), max(yr[:,1].max(),1.01)]
    xr1= [min(xr[:,0].min(),2.75), xr[:,1].max()]
    for ax1 in axes:
        ax1.set_xlim(xr1)
        ax1.set_ylim(yr1)
        
                                 
    ###---
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150) #
    #plt.show()
    print(pdata['outfn'])
    return


if __name__=="__main__":
    
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']
    rg_names= ['DJF_Peruvian','DJF_Namibian','DJF_Australian',
               'JJA_Peruvian','JJA_Namibian','JJA_Californian']

    ## Collect y_pred by region and concatenate them
    result_by_region=[]
    for i,rg_nm in enumerate(rg_names):
        output1= get_y_pred(rg_nm,tgt_crs) 

        if i==0:
            keys= output1.keys()
            res= output1.copy()
        else:
            for k in keys:
                res[k][-1]= np.concatenate((res[k][-1],output1[k][-1]),axis=-1)     
    

    ## Plot the results
    outdir= './Pics/'
    if True:
        outfn= outdir+'Fig04.Regr_performance_All_RG6_togehter.png'
        suptit= "Regression Performance\n(All regions' predictions are aggregated)"
        pic_data= dict(results= res,
                       tgt_crs= tgt_crs,
                       outfn=outfn,suptit=suptit,
        )
        plot_main(pic_data)

