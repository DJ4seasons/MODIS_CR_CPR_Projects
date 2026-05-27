"""
Draw local sensitivity of each feature based on SHAP and ALE values
x-axis sets as standardized range of select features

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

def main(tgt_crs,md_nm,model_name):
    print(md_nm,model_name)
    
    ## Parameters
    ncr= len(tgt_crs); ncr2= ncr+1 #Consider "other" regime
    
    mdnm1= 'ERA5'
    nyr,npt= 22,810

    indir= './Input4ML_LcRFO/'

    var_names= [
        'SST (K)','T2m (K)','T850 (K)','T700 (K)','T500 (K)',
        'q2m (g/kg)','q850 (g/kg)','q700 (g/kg)','q500 (g/kg)',
        'WS10m (m/s)','WS850 (m/s)','WS700 (m/s)',
        'Ps (Pa)','T_adv (K/day)',
        'q_adv850 (g/kg/day)','q_adv700 (g/kg/day)'
    ]
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
    shap_data_shape= [shap_samples,nv,ncr2]
    shap_shape_txt= 'x'.join([str(v) for v in shap_data_shape])
    ale_data_shape= [nv,ale_bins,ncr2]
    ale_shape_txt= 'x'.join([str(v) for v in ale_data_shape])    
    
    ## Prepare X_test, y_test, and model by region
    ale_by_region,shap_by_region= [],[]
    if True:
        ## Read SHAP data
        infn1= indir1+model_name+'.shap_dict_{}.joblib'.format(shap_shape_txt)
        shap_dict= joblib.load(infn1)
        print(shap_dict['shap_values'].shape)  

        ## Read ALE_dict data
        infn2= indir1+model_name+'.ale_result_dict_{}.joblib'.format(ale_shape_txt)
        ale_results= joblib.load(infn2)
        print(ale_results.keys(),ale_results['ale_values'].shape) # [nv,n_bins,ncr]
        print(ale_results['bin_bounds'].shape)

        ## Collect local sensitivity data
        ale_byVar=[]
        shap_mean_byVar=[]
        for vidx in range(nv):
            ale_centered= ale_results['ale_values'][vidx,:,:ncr]
            bin_bounds= ale_results['bin_bounds'][vidx,:]
            bin_centers= (bin_bounds[:-1]+bin_bounds[1:])/2                
            

            td1= shap_dict['test_data'][:,vidx]
            xm,xstd= td1.mean(), td1.std()
            shap_values= shap_dict['shap_values'][:,vidx,:ncr]
            shap_xx, shap_yy=[],[]
            for b0,b1 in zip(bin_bounds[:-1],bin_bounds[1:]):
                test_ind= np.logical_and(td1>=b0,td1<b1)
                if test_ind.sum()>1:
                    shap_xx.append(td1[test_ind].mean())
                    shap_yy.append(shap_values[test_ind,:].mean(axis=0))
                else:
                    shap_xx.append(np.nan)
                    shap_yy.append(np.asarray([np.nan,]*ncr))
            shap_xx= np.asarray(shap_xx)
            shap_yy= np.asarray(shap_yy)
            #-- Standardization
            shap_mean_byVar.append([(shap_xx-xm)/xstd, shap_yy])
            #print(shap_yy.shape, ale_centered.shape); sys.exit()
            ale_byVar.append([(bin_centers-xm)/xstd,ale_centered])
        ale_by_region.append(ale_byVar)
        shap_by_region.append(shap_mean_byVar)
         
    ## Plot the results
    outdir= './Pics/'
    
    if True:
        outfn= outdir+f'Fig07.Local_feature_sensitivity.{md_nm}_RawVar.png' #.format(tcr_nm)
        suptit= f'Local Feature Sensitivity of {md_nm} model' #.format(tcr_nm)
        pic_data= dict(ale= ale_by_region[0], shap= shap_by_region[0],
                       var_names=var_names,
                       tgt_crs=tgt_crs,
                       outfn=outfn,suptit=suptit,
        )
        plot_main(pic_data)
        
    return 
    
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
def plot_main(pdata):
    '''
    ale= ale_by_region, shap= shap_by_region,
    var_names=basic_vars, y_ind=k,
    '''
    ale= pdata['ale']
    shap= pdata['shap']
    var_names= pdata['var_names']
    tgt_crs= pdata['tgt_crs']
    
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'
    max_var2display=6
    
    ## Convert by target
    ncr= len(tgt_crs) 
    ale_new, shap_new=[],[]
    for i in range(ncr):
        ## ale1 and shap1 are "by Var"
        ale1, shap1=[],[]
        for j,(ale2,shap2) in enumerate(zip(ale,shap)):
            ale1.append([ale2[0],ale2[1][:,i]])
            shap1.append([shap2[0],shap2[1][:,i]])
        ale_new.append(ale1)
        shap_new.append(shap1)
    ale_new= np.asarray(ale_new)
    shap_new= np.asarray(shap_new)
    #print(ale_new.shape, shap_new.shape); sys.exit()
    
    ###---
    fig=plt.figure()
    fig.set_size_inches(6.5,6)    ## (lx,ly)
    
    plt.suptitle(pdata['suptit'],fontsize=16,y=0.98,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    
    ncol,nrow=2,2
    lf,rf,bf,tf=0.06,0.94,0.24,0.92
    gapx, npnx=0.115,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.12,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)

    ix=lf; iy=tf

    cc= [f'C{v}' for v in range(10)]; n_cc= len(cc)
    
    props= dict(lw=2,ls='-',alpha=0.8)
    hv_props= dict(ls=':',lw=1,c='0.3',alpha=0.6,zorder=0)
    props_mav= dict(ha='right',va='bottom',color='k',fontsize=12)
    axes_l,yr_l=[],[]
    axes_r,yr_r=[],[]
    ai=0
    for i in range(ncr):
        ale1,shap1= ale_new[i], shap_new[i]
        mav_a= np.nanmean(ale1[:,1,:]**2,axis=1)
        mav_s= np.nanmean(shap1[:,1,:]**2,axis=1)
        v_idx= np.argsort(mav_a+mav_s)[-max_var2display:][::-1]
        print(i,[var_names[iv] for iv in v_idx]) 
        
        ax1=fig.add_axes([ix,iy-ly,lx,ly])
        for j,iv in enumerate(v_idx):
            xx,yy= shap1[iv,0,:], shap1[iv,1,:]
            nan_idx= np.isnan(yy)
            if nan_idx.sum()>0:
                xp1,yy= xx[~nan_idx],yy[~nan_idx]
            else:
                xp1=xx
            pic1= ax1.plot(xp1,yy,c=cc[j%n_cc],label=var_names[iv].split()[0],**props)
        
        ##--
        ax1.tick_params(labelsize=9)
        ax1.set_xlabel('Features are standardized',fontsize=10)
        ax1.set_ylabel('SHAP values',fontsize=10)
        ax1.axhline(y=0,**hv_props)
        ax1.axvline(x=0,**hv_props)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        
        subtit= '({}) For {}'.format(abc[ai],tgt_crs[i]); ai+=1
        ax1.set_title(subtit,fontsize=13,x=0,ha='left')
        axes_l.append(ax1); yr_l.append(ax1.get_ylim())
        
        ix+= lx+gapx
        ax2= fig.add_axes([ix,iy-ly,lx,ly])
        for j,iv in enumerate(v_idx):
            xx,yy= ale1[iv,0,:], ale1[iv,1,:]
            pic2= ax2.plot(xx,yy,c=cc[j%n_cc],label=var_names[iv].split()[0],**props)
                    
        ##--
        ax2.tick_params(labelsize=9)
        ax2.set_xlabel('Features are standardized',fontsize=10)
        ax2.set_ylabel('ALE values',fontsize=10)
        ax2.legend(loc='upper left',bbox_to_anchor=[1.04,1.],fontsize=10,borderaxespad=0)
        ax2.axhline(y=0,**hv_props)
        ax2.axvline(x=0,**hv_props)
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        
        xr1,xr2= ax1.get_xlim(),ax2.get_xlim()
        xr0= [min(xr1[0],xr2[0]),max(xr1[1],xr2[1])]
        ax1.set_xlim(xr0); ax2.set_xlim(xr0)
        
        axes_r.append(ax2); yr_r.append(ax2.get_ylim())

        ix=lf
        iy-= ly+gapy

    ## Make the same scale for y-axis
    yr_l,yr_r= np.asarray(yr_l), np.asarray(yr_r)
    yr1= [yr_l[:,0].min(), yr_l[:,1].max()]
    yr2= [yr_r[:,0].min(), yr_r[:,1].max()]
    for ax1,ax2 in zip(axes_l,axes_r):
        ax1.set_ylim(yr1)
        ax2.set_ylim(yr2)
        
                                 
    ###---
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150) #
    #plt.show()
    print(pdata['outfn'])
    return


if __name__=="__main__":
    
    tgt_crs= ['L1_tk','L2_tk','L_tn','S-Clr']

    md_nm, model_name= 'NNet', 'AllRG6_ow10_100-100_rs37'
    #md_nm, model_name= 'NNet_AllRG12', 'AllRG12_ow10_172-172_rs23'
    
    main(tgt_crs,md_nm, model_name)


    
    
   

