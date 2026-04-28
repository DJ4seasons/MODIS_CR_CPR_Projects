"""
Draw Centroids and mean RFOs of sub-regimes
Sub-regimes are now available only in Cld-only regimes

Daeho Jin, 2026.01.07
---

"""

import numpy as np
import sys
from netCDF4 import Dataset, num2date
from datetime import timedelta, date

from math import ceil
lat_deg2y = lambda lat,lat0,dlat: ceil((lat-lat0)/dlat)

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

def main(cr_params, sat_nm='TAmean'):
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
    
    ## Parameters for CR_nums 
    tgt_lats= [-rg,rg]  ## It is limited to the original domain, although data were extended.
    tgt_dates= [date(2014,6,1),date(2019,5,31)]
    tgt_date_names= '-'.join([dd.strftime('%Y.%m') for dd in tgt_dates])
    ndy= (tgt_dates[1]-tgt_dates[0]).days+1
    
    #-- Open netCDF file
    fid= Dataset(infn,'r')    
    
    #-- Read dimension info
    times= fid.variables['time']
    time_units = times.units
    times= num2date(times[:], units = times.units, calendar=times.calendar,
                      only_use_cftime_datetimes=True,)
    date_range= [date(t1.year, t1.month, t1.day) for t1 in [times[0],times[-1]]]
    itidx= (tgt_dates[0]-date_range[0]).days
    
    lons= fid.variables['lon'][:]
    lats= fid.variables['lat']
    lat0, dlat= lats[0], (lats[-1]-lats[0])/(len(lats)-1)
    lat_idx= [lat_deg2y(lt,lat0,dlat) for lt in tgt_lats]
    print(lat0,dlat,lat_idx) #; sys.exit()
    lats= lats[lat_idx[0]:lat_idx[1]]
    
    cot_b= fid.variables['cloud_optical_thickness_bounds'][:].filled(-999.)
    ctp_b= fid.variables['cloud_top_pressure_bounds'][:].filled(-999.)
    phist_b= fid.variables['Prec_histogram_bin_bound'][:].filled(-999.)
    phist_c= fid.variables['Prec_histogram_bin_center'][:].filled(-999.)
    
    #-- Centroid info
    ctd_cld= fid.variables['SubRegime_Centroid_cloud_part'][:].filled(-999.)
    ctd_phist= fid.variables['SubRegime_Centroid_precipitation_part'][:].filled(-999.)
    ctd_cf = np.sum(ctd_cld,axis=(1,2))*100.    
    ctd_pf = np.sum(ctd_phist,axis=1)*100.
    subk= ctd_cld.shape[0]
    
    #- Estimating precip_rate from pr_hist info
    print(phist_c)
    pm=np.concatenate((np.array([0.,]),phist_c))
    ctd_pr=[]
    for k in range(subk):
        wt=np.concatenate(([100-ctd_pf[k],],ctd_phist[k,:]*100))
        ctd_pr.append(np.average(pm,weights=wt))

    np.set_printoptions(precision=3,suppress=True)
    print(ctd_cf)
    print(ctd_pf)
    print(ctd_pr)
    
    #-- Read CR-nums
    vn= 'CRnum_on_map_TAmean' if sat_nm=='TAmean' else 'CRnum_on_map_{}'.format(sat_nm.title())
    crnums= fid.variables[vn]
    crnums= crnums[itidx:itidx+ndy,lat_idx[0]:lat_idx[1],:]
    print(crnums.shape)
    fid.close()
    
    #-- Calculate RFO    
    tgt_crs= np.arange(1,subk+1,1,dtype=int)+km*10  ## Considering sub-regime numbering
    map0=[]
    for cr in tgt_crs:  
        idx1= crnums==cr  
        map0.append(idx1.mean(axis=0))  ## All time mean
    
    map0= np.asarray(map0)*100.  ## Now in percent

    #-- Global mean RFO; latitude weights are not considered here.
    grfo=map0.mean(axis=(1,2))
    print(grfo)


    ###---- Plot
    suptit= "Sub-regimes of C{}R{} in {}, {}, k={}".format(p_letter,km,rg_nm,prset_nm,km,)
    rfo_comment= "{}, {}".format(sat_nm,tgt_date_names)
    outdir = './Pics/'
    outfn = outdir+"Fig.Sub_regime_CTD+RFO."+mdnm+f".{sat_nm}_{tgt_date_names}.png"
    xy= np.meshgrid(lons,lats)
    abc= 'abcdefghijklmn'
    tgt_cr_names= ['C{}R{}{}'.format(p_letter,km,abc[i].upper()) for i in range(subk)]
    
    pic_data= dict(
        ctd_cld= ctd_cld, ctd_phist= ctd_phist,
        ctd_cf= ctd_cf, ctd_pf= ctd_pf, ctd_pr= ctd_pr,
        labels= dict(cot= cot_b, ctp= ctp_b, phist= phist_b),
        rfo_maps= map0, grfo= grfo, rfo_comment=rfo_comment,
        tgt_cr_names= tgt_cr_names, 
        xy=xy, #p_letter= p_letter,
        suptit=suptit, outfn=outfn,
    )
    plot_main(pic_data)

    return

import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
import cartopy.crs as ccrs

def plot_main(pdata):
    ## Parameters and variables
    ctd_cld= pdata['ctd_cld'] 
    ctd_phist= pdata['ctd_phist']
    ctd_cf= pdata['ctd_cf'] 
    ctd_pf= pdata['ctd_pf'] 
    ctd_pr= pdata['ctd_pr']
    labels= pdata['labels']
    #p_letter= pdata['p_letter']
    tgt_cr_names= pdata['tgt_cr_names']
    rfos= pdata['rfo_maps']
    grfo= pdata['grfo']
    xy= pdata['xy']
    rfo_comment= pdata['rfo_comment']
    
    subk= ctd_cld.shape[0]
    lat_max= rfos.shape[1]//2
    
    ###---
    fig=plt.figure()
    fig.set_size_inches(10,15)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=19,x=0.425,y=0.99)
    lf,rf,bf,tf=0.04,0.94,0.05,0.95
    gapx, npnx=0.07,3.5
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.05,5.2
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    ly0=ly*1.3*7/9.; ly1=ly*1.3/9.

    abc='abcdefghijklmn'
    
    cm = plt.get_cmap('jet',256)       
    cmnew = cm(np.arange(256))
    cmnew = cmnew[36:,:]
    newcm = cls.LinearSegmentedColormap.from_list("newJET",cmnew)
    newcm.set_under('white')
    props = dict(norm=cls.LogNorm(vmin=0.1,vmax=30),cmap=newcm,alpha=0.8)
    cm2 = plt.get_cmap('viridis').copy(); cm2.set_under('white')
    props2 = dict(norm=cls.LogNorm(vmin=0.1,vmax=30),cmap=cm2,alpha=0.8)

    ### Plot Centroid
    ix=lf; iy=tf
    ai=0
    for ii in range(subk):
        ax1= fig.add_axes([ix,iy-ly0,lx,ly0])
        vv= ctd_cld[ii,:]*100.
        vv= vv[::-1,:]
        pic1= cent_show(ax1,vv,props,labels,ytlabs='l')

        subtit= "({}) {} {:.0f}% {:.0f}% {:.1f}mm/h".format(
            abc[ai],tgt_cr_names[ii],ctd_cf[ii],ctd_pf[ii],ctd_pr[ii]); ai+=1
        cent_show_common(ax1,subtit)

        ax2= fig.add_axes([ix,iy-ly0-ly1*2.5,lx,ly1])
        vv2= ctd_phist[ii,:]*100.
        pic2= cent_pr_show(ax2,vv2,props2,labels)
        #    if ii==int((km-1)/3)*3:
        #ax1.set_xlabel('Optical Thickness',fontsize=10,va='top',labelpad=0)
        #ax1.set_ylabel('Pressure (hPa)',fontsize=10,labelpad=-2,va='bottom')

        ix+=(lx+gapx)
        
    hh=0.015
    lyt= ly0+ly1*2.8+gapy/3
    loc1= [ix-gapx/1.5,iy-lyt/2.05,hh,lyt/2.05]
    cb1=draw_colorbar(fig,pic1,loc1,ft=9.25)
    cb1.ax.set_ylabel('Cloud Fraction',fontsize=11,rotation=-90,va='bottom',labelpad=0)
    
    loc2= [ix-gapx/1.5,iy-lyt,hh,lyt/2.05]
    cb2=draw_colorbar(fig,pic2,loc2,ft=9.25)
    cb2.ax.set_ylabel('Precip. Fraction',fontsize=11,rotation=-90,va='bottom',labelpad=0)
            

    ix=lf
    iy=iy-ly*1.3-gapy*1.3
    ###---- Plot RFO Map
    cm = plt.get_cmap('magma_r',80) #'CMRmap_r' 'YlOrBr' 'Accent' 'afmhot_r'
    cmnew = cm(np.arange(80)) #; print(cmnew[0,:])
    cmnew = np.concatenate((np.array([1,1,1,1]).reshape([1,-1]),cmnew[:-1,:])) #print cmnew[0,:],cmnew[-1,:]
    newcm = cls.LinearSegmentedColormap.from_list("newCMR",cmnew)
    newcm.set_under("white")
    
    lon_ext= [20,360+20]
    lc= (lon_ext[0]+lon_ext[1])/2
    data_crs= ccrs.PlateCarree()
    map_proj= ccrs.PlateCarree(central_longitude=lc)
    cmax=50
    props_pc = dict(cmap=newcm,alpha=0.9,transform=data_crs,vmin=0.,vmax=cmax)
    cb_xt= range(0,cmax+1,10)
    
    ly2= ly/2 if lat_max<30 else ly
    lx2= (rf-lf)*0.85 #*2+gapx
    for ii in range(subk):
        ax=fig.add_axes([ix,iy-ly2,lx2,ly2],projection=map_proj)
        ax.set_extent(lon_ext+[-lat_max*1.03,lat_max*1.03],data_crs)
        
        # add a title.
        subtit='({}) {} [CF={:.1f}%]: RFO={:.1f}%'.format(
            abc[ai],tgt_cr_names[ii],ctd_cf[ii],grfo[ii]); ai+=1
        if ii==0:
            subtit+= f'  [{rfo_comment}]'
        print(subtit)
        ax.set_title(subtit,x=0.0,ha='left',fontsize=13,stretch='semi-condensed')

        cs=ax.pcolormesh(*xy,rfos[ii,:,:],**props_pc)

        label_idx=[True,False,False,True]
        map_common(ax,lat_max,label_idx)

        iy=iy-ly2-gapy*0.75


    loc2= [ix+lx2+gapx/3,iy+gapy*0.75,hh,ly2*3+gapy*1.5]
    cb= draw_colorbar(fig,cs,loc2,cb_xt,ft=9,extend='max')
    cb.ax.set_ylabel('RFO',fontsize=11,rotation=-90,va='bottom',labelpad=0)
    '''
    tt=np.arange(0,cmax+0.1,5)
    tt2=['{:.0f}%'.format(x) for x in tt]
    cb.set_ticks(tt)
    cb.ax.set_yticklabels(tt2,size=10)
    '''

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

def cent_show_common(ax1,subtit):

    ### add a title
    print(subtit)
    ax1.set_title(subtit,x=-0.1,ha='left',fontsize=13,stretch='condensed')

    ### Draw Guide Line
    ax1.axvline(x=1.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axvline(x=3.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axhline(y=1.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axhline(y=3.5,linewidth=0.7,color='k',linestyle=':')

    ### Ticks
    ax1.tick_params(axis='both',which='major',labelsize=10,pad=2)
    ax1.tick_params(left=True,right=True)

    return

def map_common(ax,lat_max,label_idx=[True,True,False,True]):

    if lat_max<=30:
        #ax.set_extent([0.,360,-16,16],ccrs.PlateCarree())
        ax.set_yticks([-15,0,15])
    else:
        #ax.set_extent([0.,360,-52,52],ccrs.PlateCarree())
        ax.set_yticks([-50,-25,0,25,50])

    ax.tick_params(axis='y',labelright=False,labelleft=False,labelsize=11)
    ax.tick_params(direction='in',left=True,right=True,top=True,bottom=True,)
    
    ax.coastlines(color='silver',linewidth=1.)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = label_idx[2]
    gl.left_labels = label_idx[0]
    gl.right_labels = label_idx[1]
    gl.bottom_labels = label_idx[3]

    gl.xlocator = MultipleLocator(60) 

    if lat_max>30:
        gl.ylocator = MultipleLocator(25)
    else:
        gl.ylocator = MultipleLocator(15)
        
    gl.xlabel_style = {'size': 11, 'color': 'k'}
    gl.ylabel_style = {'size': 11, 'color': 'k'}

    ax.set_aspect('auto')
    return

def draw_colorbar(fig,pic1,loc,tt=[0.1,0.3,1,3,10,30],ft=10,extend='both'):    
    tt2=[str(x)+'%' for x in tt]
    
    cb_ax = fig.add_axes(loc)  ##<= (left,bottom,width,height)
    if loc[2]<loc[3]:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='vertical',ticks=tt,extend=extend)
        cb.ax.set_yticklabels(tt2,size=ft,stretch='condensed')
    else:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='horizontal',ticks=tt,extend=extend)
        cb.ax.set_xticklabels(tt2,size=ft,stretch='condensed')
    cb.ax.minorticks_off()
    return cb


if __name__=='__main__':
    cr_params= dict(
        rg= 50,      # domain max latitude, 15 or 50
        nelemp= 6,   # dimension of pr_hist, fixed to 6
        prwt= 0,     # weight of pr_hist, should be 0 for sub-regime
    )
    sat_nm= 'TAmean' #'TAmean' # 'terra', 'aqua', or 'TAmean'
    main(cr_params,sat_nm)



    
