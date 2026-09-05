"""
Compare the feature importances of NNet model

SHAP vs. ALE 

By Daeho Jin
2026.04.15 
"""

import numpy as np
import sys
import os
from datetime import timedelta, date
import math
import common_functions as cf
import joblib

def get_score(model_name,tgt_crs):
    print(model_name)
    
    ## Parameters
    mdnm1= 'ERA5'
    nyr,npt= 22,810
    indir= './Input4ML_LcRFO/'

    ncr= len(tgt_crs); ncr2= ncr+1
    
    var_names= [
        'SST (K)','T2m (K)','T850 (K)','T700 (K)','T500 (K)',
        'q2m (g/kg)','q850 (g/kg)','q700 (g/kg)','q500 (g/kg)',
        'WS10m (m/s)','WS850 (m/s)','WS700 (m/s)',
        'Ps (Pa)','T_adv (K/day)',
        'q_adv850 (g/kg/day)','q_adv700 (g/kg/day)'   ]
    nv= len(var_names)
    clim_vars,clim_vnames= ['skt','sp'], ['SST_clim (K)', 'Ps_clim (hPa)',]
    nv2= len(clim_vars)

    nv+= nv2
    var_names+= clim_vnames
    
    ## Read feature importance data
    indir1= './NN_feature_importance_data/'
    shap_samples= 250
    ale_bins= 50

    ## Read SHAP data
    data_shape= [shap_samples,nv,ncr2]
    shape_txt= 'x'.join([str(v) for v in data_shape])
    infn1= indir1+model_name+'.shap_dict_{}.joblib'.format(shape_txt)
    shap_values= joblib.load(infn1)['shap_values']
    print(shap_values.shape)
    
    ## Read ALE_dict data
    data_shape= [nv,ale_bins,ncr2]
    ale_txt= 'x'.join([str(v) for v in data_shape])
    infn2= indir1+model_name+'.ale_result_dict_{}.joblib'.format(ale_txt)
    ale_results= joblib.load(infn2)
    print(ale_results['ale_values'].shape)
            
    output=[]
    for j in range(nv):
        mean_abs_shap= np.absolute(shap_values[:,j,:]).mean(axis=0)
        mean_abs_ale= np.absolute(ale_results['ale_values'][j,:,:]).mean(axis=0)
        feature_name= ale_results['feature_names'][j]
        print(feature_name,mean_abs_shap,mean_abs_ale) #; sys.exit()
        out1= dict(feature_name=feature_name,
                   mean_abs_shap= mean_abs_shap,
                   mean_abs_ale= mean_abs_ale,
        )
        output.append(out1)

    return output
    

import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
def plot_avg_main(pdata):
    results= pdata['results']
    tcr_names= pdata['tcr_names']
    
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'
    nrg= len(results)
    
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

    cc= [f'C{v}' for v in range(10)][::-1]; n_cc= len(cc)
    mk= ['d','o','^','v','s','P']; n_mk= len(mk)
    sct_props= dict(s=50)

    xy_vals=[]
    if True:
        md_nm,output1= results

        tx1,ty1= [],[]
        labels=[]
        for jj,out_dict in enumerate(output1):            
            xx= out_dict['mean_abs_ale'] 
            yy= out_dict['mean_abs_shap'] 
            label= out_dict['feature_name'].split()[0]
            if label[0]=='Q': label= 'q'+label[1:]
            elif label[0]=='q' and label[-1]=='v': label= 'q_adv'+label[1:4]
            elif label[:2]=='PS': label='Ps'+label[2:]
            if label[-1]=='M': label= label[:-1]+'m'
            
            tx1.append(xx)
            ty1.append(yy)
            labels.append(label)
        xy_vals.append([tx1,ty1])

    xy_vals= np.asarray(xy_vals).squeeze()    # [x/y,nv,ncr]
    #print(xy_vals.shape) ; sys.exit() 
    _,nv,ncr= xy_vals.shape

    ncr0= min(ncr,len(tcr_names))
    tot_panels= ncr0
    axes,yr,xr=[],[],[]
    for k in range(ncr0):    
        ax1=fig.add_axes([ix,iy-ly,lx,ly])
        xx,yy= xy_vals[0,:,k], xy_vals[1,:,k]
        for j in range(nv):        
            sct1= ax1.scatter(xx[j:j+1],yy[j:j+1],c=cc[j%n_cc],marker=mk[j%n_mk],label=labels[j],alpha=1-j*0.015,**sct_props)
        
        ##--
        subtit= '({}) For {} RFO'.format(abc[k],tcr_names[k])
        ax1.set_title(subtit,fontsize=12,x=0,ha='left')
        ax1.grid(ls=':')
        ax1.tick_params(labelsize=9)
        
        if k==ncol-1:
            ax1.legend(loc='upper left',bbox_to_anchor=[1.04,1.],fontsize=10,borderaxespad=0)
        if k%ncol==0:
            ax1.set_ylabel('Mean '+r'$|SHAP|$',fontsize=10)
        if k>=tot_panels-ncol:
            ax1.set_xlabel('Mean '+r'$|ALE|$',fontsize=10)

        axes.append(ax1)
        yr.append(ax1.get_ylim())
        xr.append(ax1.get_xlim())
        
        ix+= lx+gapx
        if ix+gapx>rf:
            ix=lf
            iy-= ly+gapy


    ## Make consistent scale of axis 
    yr,xr= np.asarray(yr), np.asarray(xr)
    yr1= [-0.002, yr[:,1].max()]
    xr1= [-0.002, xr[:,1].max()]
    for ax1 in axes:
        ax1.set_xlim(xr1)
        ax1.yaxis.set_major_locator(MultipleLocator(0.05))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        
        ax1.set_ylim(yr1)
        ax1.yaxis.set_major_locator(MultipleLocator(0.05))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
                                 
    ###---
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150) #
    #plt.show()
    print(pdata['outfn'])

    return

if __name__=="__main__":

    outdir= './Pics/'
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']

    md_name= 'AllRG6_ow10_100-100_rs37' #'AllRG12_ow10_172-172_rs23' #
    
    mdnm_head= md_name.split('_')[0]
    outfn= outdir+f'Fig06.Feature_importance_NNet_RawVar_{mdnm_head}.png'
    if mdnm_head[-1]=='6':
        suptit= 'Feature Importance in NNet model' #.format(tcr_nm)
    else:
        suptit= f'Feature Importance in NNet_{mdnm_head} model' #.format(tcr_nm)
    
    
    ## Read metrics
    output1= get_score(md_name,tgt_crs)
        
    ## Plot the results    
    pic_data= dict(results= [md_name,output1],
                   tcr_names= tgt_crs,
                   outfn=outfn,suptit=suptit,
    )
    plot_avg_main(pic_data)
    
