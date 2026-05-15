"""
Illustrate sample strategy + sample distribution example
Linear regression info in samples are added.

With help from Claude
Daeho Jin, 2026.03.12
"""

import numpy as np
import sys
import os #.path
from datetime import timedelta, date
import math
import common_functions as cf

def main():
    rg_nm='DJF_Peruvian'
    
    ## Parameters
    mdnm1= 'ERA5'
    nyr,npt= 22,810 
    indir= './Input4ML_LcRFO/'
    tgt_crs= ['L1_tk','L2_tk'] #,'L_tn','S-Clr']
    
    input4LCAIs= ['t700','t2m','sp','q700','q2m','t800','skt']
    basic_vars= ['LTS (K)','EIS (K)','ECTEI (K)','ELF (%)', 'M (K)', 'SST (K)']
    
    
    ## Read CR_rfo from samples
    rfos= cf.collect_data2calc_LCidx_fromSamples(mdnm1,rg_nm,var_names=tgt_crs,indir=indir,in_dim=[nyr,npt])

    ## Check RFO data
    for k,crn in enumerate(tgt_crs):
        rfo1= rfos[k]
        print(crn, rfo1.min(), np.percentile(rfo1,[5,50,95]), rfo1.max())    
    
    ## Prepare LC_idx
    indata= cf.collect_data2calc_LCidx_fromSamples(mdnm1,rg_nm,indir=indir,in_dim=[nyr,npt])
    lci1, sst1= cf.calc_LCidx(indata[:-1]), indata[-1]
    print(indata[0].shape, lci1.shape, sst1.shape) # [nyr,npt,nvar]
    
    ## Select year 2003 and EIS only
    rfos= [rfos[0][0,:]*100,rfos[1][0,:]*100] #[:490,:], now in %
    lci1= lci1[0,:,1] #[:490,1]
    
    #---
    suptit= 'Sampling Method and Examples'
    outdir= './Pics/'       
    outfn= outdir+'Fig03.Sampling_Illust+EIS_examples.png'
    pic_data= dict(data_set= [(lci1,rfos[0]),(lci1,rfos[1])],
                   var_names= [(basic_vars[1],tgt_crs[0]),(basic_vars[1],tgt_crs[1])],
                   subtit= '2003 '+rg_nm,
                   suptit=suptit, outfn=outfn,
    )
    plot_main(pic_data)
        
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator

def plot_main(pdata):
    data_set= pdata['data_set']
    var_names= pdata['var_names']
    subtit0= pdata['subtit']
    bbox_sz= 12
    sbox_sz= 4
    
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'
    ai=0
    
    ###---    
    fig = plt.figure(figsize=(8.5, 9))  ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=17,y=0.965,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    ncol,nrow=2,2
    lf,rf,bf,tf=0.02,0.98,0.1,0.92
    gapx, npnx=0.09,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.095,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)

    lx0, lx1= lx*2*0.4, lx*2*0.6
    ix=lf; iy=tf
    
    # First subplot: Show the sampling concept with a few example boxes
    ax1 = fig.add_axes([ix,iy-ly,lx,ly])
    ax1.set_xlim(-0.5, bbox_sz+0.5)
    ax1.set_ylim(-0.5, bbox_sz+0.5)
    #ax1.set_aspect('equal')

    # Draw domain boundary
    boundary = patches.Rectangle((0, 0), bbox_sz, bbox_sz, linewidth=3,
                            edgecolor='c', facecolor='none', zorder=2)
    ax1.add_patch(boundary)

    # Draw grid lines for reference
    for i in range(bbox_sz+1):
        ax1.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.5, zorder=0)
        ax1.axvline(x=i, color='lightgray', linewidth=0.5, alpha=0.5, zorder=0)

    # Draw example sampling boxes at different positions
    nx=ny=bbox_sz-sbox_sz+1
    for box_num,fc in zip([1,31,81],['red','blue','green']):
        x0,y0= (box_num-1)%nx, (box_num-1)//ny
        box1 = patches.Rectangle((x0, y0), sbox_sz, sbox_sz, linewidth=1.,
                         edgecolor='k', facecolor=fc, alpha=0.2, zorder=1)
        ax1.add_patch(box1)
        ax1.text(x0+2, y0+2, f'Box {box_num}\n({x0+sbox_sz//2},{y0+sbox_sz//2})',
                 ha='center', va='center', fontsize=10, fontweight='bold')
        
    # Add arrows to show 1-degree shift
    arrow1 = patches.FancyArrowPatch((sbox_sz, 1), (sbox_sz+1.1, 1),
                                 connectionstyle="arc3",
                                 arrowstyle='->', mutation_scale=12,
                                 color='darkred', linewidth=2)
    ax1.add_patch(arrow1)
    ax1.text(4.15, 0.6, '1° shift', ha='left', va='center',fontsize=9, color='darkred', stretch='semi-condensed')

    arrow2 = patches.FancyArrowPatch((1, sbox_sz), (1, sbox_sz+1.1),
                                 connectionstyle="arc3",
                                 arrowstyle='->', mutation_scale=12,
                                 color='darkred', linewidth=2)
    ax1.add_patch(arrow2)
    ax1.text(0.6, 4.15, '1° shift', ha='center', va='bottom',fontsize=9, color='darkred', rotation=90, stretch='semi-condensed')

    ax1.set_xlabel('Longitude (degrees)', fontsize=11)
    ax1.set_ylabel('Latitude (degrees)', fontsize=11)
    ax1.set_title(f'({abc[ai]}) Spatial Sampling', fontsize=13, ha='left',x=0.) #fontweight='bold')
    ai+=1
    ax1.set_xticks(range(bbox_sz+1))
    ax1.set_yticks(range(bbox_sz+1))
    ax1.tick_params(labelsize=10)

    ix+= lx+gapx

    #---
    # Second subplot: Timeline view showing the sampling concept
    ax2 = fig.add_axes([ix,iy-ly,lx,ly])
    ax2.set_xlim(-2, 93)
    ax2.set_ylim(-0.2, 11.2)

    # Draw the total 91-day duration
    total_duration = patches.Rectangle((0, 10.), 91, 0.9, linewidth=2,
                                  edgecolor='black', facecolor='lightgray',
                                  alpha=0.5, zorder=1)
    ax2.add_patch(total_duration)
    ax2.text(45.5, 10.45, '91-Day Total Duration', ha='center', va='center',
             fontsize=11, fontweight='bold')

    # Draw sample windows at different y-positions
    colors = plt.cm.tab10(np.linspace(0, 0.9, 10))
    window_labels = []

    for i in range(10):
        start_day = i * 7
        y_pos = i #9 - i * 1.0
   
        # Draw the 28-day window
        window = patches.Rectangle((start_day, y_pos), 28, 0.88, linewidth=1.,
                              edgecolor='k', facecolor=colors[i],
                              alpha=0.6, zorder=2)
        ax2.add_patch(window)
   
        # Add window label
        ax2.text(start_day + 14, y_pos + 0.44, f'Window{i+1}', ha='center', va='center',
             fontsize=9, fontweight='bold', color='white',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='darkblue', alpha=0.7))
   
   
    # Labels and formatting
    ax2.set_xlabel('Days', fontsize=11,) # fontweight='bold')
    ax2.set_title(f'({abc[ai]}) Temporal Sampling',
                  fontsize=13, ha='left',x=0.) #fontweight='bold')
    ai+=1
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(axis='x', which='major',alpha=0.5, linestyle='--')
    ax2.grid(axis='x', which='minor',alpha=0.4, linestyle=':')
    ax2.set_xlim(-2, 93)
    ax2.set_xticks(np.arange(0,92,14))
    ax2.set_yticks([])
    ax2.tick_params(labelsize=10)
    
    ix=lf
    iy-= ly+gapy
    
    ###--- Draw sample distribution
    sct_props= dict(s=10)

    from scipy.stats import linregress
    for ii,((xdata,ydata),(xlab,ylab)) in enumerate(zip(data_set,var_names)):
        ax1= fig.add_axes([ix,iy-ly,lx,ly])
        sct1= ax1.scatter(xdata,ydata,c=f'C{ii}',marker='o',**sct_props)

        subtit= '({}) {}, {}'.format(abc[ai],subtit0,ylab); ai+=1
        ax1.set_title(subtit,fontsize=13,x=0,ha='left')
        ax1.grid(ls=':')
        ax1.tick_params(labelsize=10)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax1.set_xlabel(xlab,fontsize=11)
        ax1.set_ylabel('RFO (%)',fontsize=11)

        ## Add linear regression line and information
        sl,intercept,rvalue,pvalue,stderr= linregress(xdata,ydata)
        print(sl,intercept,rvalue,pvalue)
        xlim= ax1.get_xlim()
        ylim= ax1.get_ylim()
        new_x= np.linspace(xlim[0],xlim[1],100)
        new_y= new_x*sl+intercept
        y_valid= np.logical_and(new_y>ylim[0],new_y<ylim[1])
        ax1.plot(new_x[y_valid],new_y[y_valid],c='k',lw=1,ls='--')

        res1= sl*xdata+intercept
        mae= np.abs(res1-ydata).mean()
        r2_score= 1-((ydata-res1)**2).sum()/((ydata-ydata.mean())**2).sum()
        reg_txt= 'Linear Regr.\n{}= {:.3f}\nMAE= {:.1f}'.format(r'$R^2$',r2_score,mae)
        ax1.text(0.03,0.97,reg_txt,fontsize=10,weight=600,transform= ax1.transAxes,ha='left',va='top')

        ix+= lx+gapx
        
    ###---
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150) #
    #plt.show()
    print(pdata['outfn'])

if __name__ == "__main__":
    main()
