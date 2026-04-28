"""
# 
# Read and display centroids
# for 42(cld) + 6(pr) histogram bins clustering results
#
# Daeho Jin, 2026.01.06
#
"""

import numpy as np
import sys
from netCDF4 import Dataset

def get_km(cr_params):
    '''
    Return total number of clusters (km) depending on cr_params
    '''
    prwt_set= [0,1,7]
    try:
        ind= prwt_set.index(cr_params['prwt'])
    except:
        print('prwt should be 0, 1, or 7, but now prwt=',cr_params['prwt'])
        sys.exit()
    if cr_params['rg']==15:
        km_set= [14,16,19]
        return km_set[ind]
    elif cr_params['rg']==50:
        km_set= [15,20,22]
        return km_set[ind]
    else:
        print('rg should be 15 or 50, but now rg=',cr_params['rg'])
        sys.exit()
        
def main(cr_params):
    ###-- Parameters and defalut values
    ###-------------------------------------
    rg, nelemp, prwt= cr_params['rg'], cr_params['nelemp'], cr_params['prwt']
    nelemc= 42
    nelem=nelemc+nelemp
    km= get_km(cr_params)   # total number of clusters  
    p_letter= 'P' if prwt>0 else ''
    prset_nm = f'Cld{nelemc}+Pr{nelemp}x{prwt}' if prwt>0 else f'Cld{nelemc}'
    rg_nm= f'{rg}S-{rg}N'
    
    indir= '../'
    mdnm= 'MODIS_t+a_C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)
    infn= indir+f'{mdnm}.nc'
    
    ## Read centroids
    fid= Dataset(infn,'r')    
    ctd_cld= fid.variables['Centroid_cloud_part'][:].filled(-999.)
    ctd_phist= fid.variables['Centroid_precipitation_part'][:].filled(-999.)
    
    cot_b= fid.variables['cloud_optical_thickness_bounds'][:].filled(-999.)
    ctp_b= fid.variables['cloud_top_pressure_bounds'][:].filled(-999.)
    phist_b= fid.variables['Prec_histogram_bin_bound'][:].filled(-999.)
    phist_c= fid.variables['Prec_histogram_bin_center'][:].filled(-999.)
    fid.close()
    
    ## Centroid info
    ctd_cf = np.sum(ctd_cld,axis=(1,2))*100.
    ctd_pf = np.sum(ctd_phist,axis=1)*100.
    #- Estimating precip_rate from pr_hist info
    print(phist_c)
    pm=np.concatenate((np.array([0.,]),phist_c))
    ctd_pr=[]
    for k in range(km):
        wt=np.concatenate(([100-ctd_pf[k],],ctd_phist[k,:]*100))
        ctd_pr.append(np.average(pm,weights=wt))

    np.set_printoptions(precision=3,suppress=True)
    print(ctd_cf)
    print(ctd_pf)
    print(ctd_pr)

    ###---- Plot
    suptit= "MODIS_C6.1 T+A C{}R in {}, {}, k={}".format(p_letter,rg_nm,prset_nm,km)
    outdir = './Pics/'
    outfn = outdir+ f"Fig.{mdnm}.png"
    pic_data= dict(
        ctd_cld= ctd_cld, ctd_phist= ctd_phist,
        ctd_cf= ctd_cf, ctd_pf= ctd_pf, ctd_pr= ctd_pr,
        labels= dict(cot= cot_b, ctp= ctp_b, phist= phist_b),
        p_letter= p_letter,
        suptit=suptit, outfn=outfn,
    )
    plot_main(pic_data)

    return

import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator, FuncFormatter
#import cartopy.crs as ccrs

def plot_main(pdata):
    ## Parameters and variables
    ctd_cld= pdata['ctd_cld'] 
    ctd_phist= pdata['ctd_phist']
    ctd_cf= pdata['ctd_cf'] 
    ctd_pf= pdata['ctd_pf'] 
    ctd_pr= pdata['ctd_pr']
    labels= pdata['labels']
    p_letter= pdata['p_letter']
    km= ctd_cld.shape[0]
    
    ###---
    fig=plt.figure()
    fig.set_size_inches(10,18)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=19,y=0.99)
    lf=0.04;rf=0.94
    bf=0.05;tf=0.95
    gapx=0.05; npnx=4
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy=0.043; npny=6
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    ly0=ly*7/9.; ly1=ly/9.

    cm = plt.get_cmap('jet',256)       
    cmnew = cm(np.arange(256))
    cmnew = cmnew[36:,:]
    newcm = cls.LinearSegmentedColormap.from_list("newJET",cmnew)
    newcm.set_under('white')
    props = dict(norm=cls.LogNorm(vmin=0.1,vmax=30),cmap=newcm,alpha=0.8)
    cm2 = plt.get_cmap('viridis').copy(); cm2.set_under('white')
    props2 = dict(norm=cls.LogNorm(vmin=0.1,vmax=30),cmap=cm2,alpha=0.8)

    ix=lf; iy=tf
    for ii in range(km):
        ax1= fig.add_axes([ix,iy-ly0,lx,ly0])
        vv= ctd_cld[ii,:]*100.
        vv= vv[::-1,:]
        pic1= cent_show(ax1,vv,props,labels,ytlabs='l')

        subtit= "C{}R{}. {:.0f}%, {:.0f}% {:.1f}mm/h".format(p_letter,ii+1,ctd_cf[ii],ctd_pf[ii],ctd_pr[ii])
        cent_show_common(ax1,subtit)

        ax2= fig.add_axes([ix,iy-ly0-ly1*2,lx,ly1])
        vv2= ctd_phist[ii,:]*100.
        pic2= cent_pr_show(ax2,vv2,props2,labels)

        ix+=(lx+gapx)
        if ix>rf:
            ix=lf
            iy=iy-ly-gapy
    
    ###-----------------------------------
    if ix==lf:
        iy+= ly+gapy
    loc1= [lf,iy-ly-gapy*0.94,lx*2+gapx,0.01]
    loc2= [lf+lx*2+gapx*2,iy-ly-gapy*0.94,lx*2+gapx,0.01]
    cb1=draw_colorbar(fig,pic1,loc1)
    cb2=draw_colorbar(fig,pic2,loc2)
    cb1.ax.set_xlabel('Cloud Fraction',fontsize=11)
    cb2.ax.set_xlabel('Precip. Fraction',fontsize=11)

    ### Show or Save
    #plt.show()
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150)
    print(pdata['outfn'])
    return

def cent_show(ax1,ctd,props,labels,ytlabs='l'):     

    nx = 6 #TAU (Optical Thickness)
    ny = 7 #CTP
    if len(ctd.reshape(-1)) != nx*ny:
        print("Error: centroid data size is bad:",ctd.shape)
        sys.exit()

    pic1=ax1.imshow(ctd,interpolation='nearest',aspect=0.8,**props)

    ### Axis Control
    xlabs= [f'{v:.0f}' if v>10 else f'{v:.1f}' for v in labels['cot']] #['0','1.3','3.6','9.4','23','60','150']
    ylabs= [f'{v:.0f}' for v in labels['ctp']] #[1100,800,680,560,440,310,180,0]

    ax1.set_xlim(-0.5,5.5)
    ax1.set_ylim(-0.5,6.5)
    ax1.set_xticks(np.arange(nx+1)-0.5)
    ax1.set_xticklabels(xlabs)

    ax1.set_yticks(np.arange(ny+1)-0.5)

    if ytlabs.lower()=='l':
        ax1.set_yticklabels(ylabs)
    elif ytlabs.lower()=='n':
        ax1.set_yticklabels([])
    elif ytlabs.lower()=='r':
        ax1.set_yticklabels(ylabs)
        ax1.yaxis.tick_right()
        
    for j in range(7):
        for i in range(6):
            if abs(ctd[j,i])>4.5:
                ax1.annotate("%.0f" %(ctd[j,i]),xy=(i,j),ha='center',va='center',stretch='semi-condensed',fontsize=10)
    return pic1

def cent_pr_show(ax2,pctd,props,labels):
    pctd=np.reshape(pctd,[1,-1])
    pic2=ax2.imshow(pctd,interpolation='nearest',aspect=0.8,**props)

    ### Axis Control
    xlabs= [str(v) for v in labels['phist']] #['0.03','0.1','0.33','1','3.33','10','(mm/h)']
    xlabs[-1]= '(mm/h)'
    
    ax2.set_xlim(-0.5,5.5)
    ax2.set_xticks(np.arange(-0.5,6,1.))
    ax2.set_xticklabels(xlabs,rotation=35,ha='right')

    ax2.set_ylim(-0.5,0.5)
    ax2.set_yticks([0.,])
    ax2.set_yticklabels(['Pr',])

    ### Ticks
    ax2.tick_params(axis='both',which='major',labelsize=10,pad=0)
    ax2.tick_params(left=False,right=False)

    for i in range(pctd.shape[1]):
        if abs(pctd[0,i])>4.5:
            ax2.annotate("%.0f" %(pctd[0,i]),xy=(i,0),ha='center',va='center',stretch='semi-condensed',fontsize=10)

    return pic2

def draw_colorbar(fig,pic1,loc):
    tt=[0.1,0.3,1,3,10,30]
    tt2=[str(x)+'%' for x in tt]
    cb_ax = fig.add_axes(loc)  ##<= (left,bottom,width,height)
    if loc[2]<loc[3]:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='vertical',ticks=tt,extend='both')
        cb.ax.set_yticklabels(tt2,size=10,stretch='condensed')
    else:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='horizontal',ticks=tt,extend='both')
        cb.ax.set_xticklabels(tt2,size=10,stretch='condensed')
    cb.ax.minorticks_off()
    return cb

def cent_show_common(ax1,subtit):

    ### add a title
    print(subtit)
    ax1.set_title(subtit,x=-0.1,ha='left',fontsize=12,stretch='condensed')

    ### Draw Guide Line
    ax1.axvline(x=1.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axvline(x=3.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axhline(y=1.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axhline(y=3.5,linewidth=0.7,color='k',linestyle=':')

    ### Ticks
    ax1.tick_params(axis='both',which='major',labelsize=10,pad=2)
    ax1.tick_params(left=True,right=True)

    return


###-------------------------------------

if __name__=="__main__":    
    cr_params= dict(
        rg= 50,      # domain max latitude, 15 or 50
        nelemp= 6,   # dimension of pr_hist, fixed to 6
        prwt= 0,     # weight of pr_hist, 0, 1, or 7
    )
    main(cr_params)
