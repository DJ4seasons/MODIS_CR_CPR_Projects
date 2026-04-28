"""
It shows C-group centroid and RFO maps based on RFO-weighted average.
It is based on Cloud-only regime derived in 50S-50N.
tgt_cr_groups= [
        ('H1_tk',(1,3,5)), ('H2_tk',(2,6)),
        ('H_tn',(7,8,9)),
        ('Mid',(4,151,152)),
        ('L1_tk',(11,13)), ('L2_tk',(10,12)),
        ('L_tn',(14,)),
        ('S-Clr',(153,)),
        ('Clr',(0,)),
]
Lastly, Clr group is defined as 1-deg grid CF<5%, 
    which is a relaxed condition compared to the strict CF=0% condition in original CR.

Daeho Jin, 2026.01.08
"""

import numpy as np
import sys
from datetime import timedelta, date
from netCDF4 import Dataset, num2date

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

def main(cr_params, tgt_cr_groups):
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
    mdnm2= 'C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)
    
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
    ctd_cld= fid.variables['Centroid_cloud_part'][:].filled(-999.)
    ctd_phist= fid.variables['Centroid_precipitation_part'][:].filled(-999.)
    ctd_cld_sub= fid.variables['SubRegime_Centroid_cloud_part'][:].filled(-999.)
    ctd_phist_sub= fid.variables['SubRegime_Centroid_precipitation_part'][:].filled(-999.)
    subk= ctd_cld_sub.shape[0]
    ## Combining centroids with sub-regimes
    ctd_cld= np.concatenate((ctd_cld[:-1,:],ctd_cld_sub),axis=0)
    ctd_phist= np.concatenate((ctd_phist[:-1,:],ctd_phist_sub),axis=0)
    
    ctd_cf = np.sum(ctd_cld,axis=(1,2))*100.    
    ctd_pf = np.sum(ctd_phist,axis=1)*100.
    print(ctd_cf)
    km_all= km-1+subk
    
    #- Estimating precip_rate from pr_hist info
    print(phist_c)
    pm=np.concatenate((np.array([0.,]),phist_c))
    ctd_pr=[]
    for k in range(km_all):
        wt=np.concatenate(([100-ctd_pf[k],],ctd_phist[k,:]*100))
        ctd_pr.append(np.average(pm,weights=wt))

    np.set_printoptions(precision=3,suppress=True)
    print(ctd_cf)
    print(ctd_pf)
    print(ctd_pr)

    #-- Read total CF data
    infn2= indir+'MODIS_Cld_Fraction_at1deg_65S-65N.nc'
    fid2= Dataset(infn2,'r')        
    ## Assume that this data has the identical lat/lon dimension with CR-num data
    
    #-- Read CR-nums
    crnums=[]
    for sat_nm in ['terra','aqua']:
        ## Total CF first
        sat1= sat_nm.title()
        total_CF1= fid2.variables[f'CF_map_{sat1}']
        total_CF1= total_CF1[itidx:itidx+ndy,lat_idx[0]:lat_idx[1],:]
        
        ## CR-num data
        crnums1= fid.variables[f'CRnum_on_map_{sat1}']
        crnums1= crnums1[itidx:itidx+ndy,lat_idx[0]:lat_idx[1],:]
        print(total_CF1.shape, crnums1.shape)
        
        ## Modify Clr regime
        clr_idx= total_CF1<0.05
        crnums1[clr_idx]= 0
        
        crnums.append(crnums1)
    fid.close()
    fid2.close()
    crnums= np.concatenate(crnums,axis=0)
    print(crnums.shape)
    
    ###--- Prepare for grouping    
    #-- Composite by C-group
    group_ctd, group_rfo_map=[],[]
    group_rfo_mean, cg_names=[],[]
    for tcr_nm, tcr in tgt_cr_groups:
        cg_names.append(tcr_nm)
        rfo_sum=0.; rfo_map= np.zeros(crnums.shape[1:3],dtype=float)
        ctd_cld0, ctd_ph0=[],[]
        for i,cr1 in enumerate(tcr):
            rfo1= (crnums==cr1).mean(axis=0)*100.  ## Now in percent
            rfo_map+= rfo1
            mrfo1=rfo1.mean()  ## Latitude weights are not considered here.
            cr_idx= cr1-1 if cr1<km else cr1-km*10+km-2
            if cr_idx>=0:
                ctd_cld0.append(ctd_cld[cr_idx]*mrfo1)
                ctd_ph0.append(ctd_phist[cr_idx]*mrfo1)
            rfo_sum+= mrfo1
        group_rfo_map.append(rfo_map)        
        group_rfo_mean.append(rfo_sum)
        if tcr[0]!=0:
            ctd_cld0= np.asarray(ctd_cld0).sum(axis=0)/rfo_sum 
            ctd_ph0= np.asarray(ctd_ph0).sum(axis=0)/rfo_sum
            group_ctd.append([ctd_cld0, ctd_ph0])        
            
    print(group_rfo_mean)
    #-- Re-calculate CF, PF, and Pr
    ctd_cf = np.array([cld.sum()*100. for (cld,ph) in group_ctd]) 
    ctd_pf = np.array([ph.sum()*100. for (cld,ph) in group_ctd]) 
    print(ctd_cf)
    
    #- Estimating precip_rate from pr_hist info
    ctd_pr=[]
    for k,(cld,ph) in enumerate(group_ctd):
        wt=np.concatenate(([100-ctd_pf[k],],ph*100))
        ctd_pr.append(np.average(pm,weights=wt))
    print(ctd_cf)
    print(ctd_pf)
    print(ctd_pr)

    
    ###-------------------------------------
    ## For plotting a figure
    suptit= "CR Group by {}: Mean Histogram (left) and RFO map (right) [{}]".format(mdnm2,tgt_date_names)
    outdir = './Pics/'
    outfn = outdir+ "Fig.CR-group_CTD+RFO_wRelaxedClr.{}.{}.png".format(mdnm2,tgt_date_names)
    xy= np.meshgrid(lons,lats)
    
    pic_data= dict(
        group_ctd= group_ctd, group_rfo_map= group_rfo_map,
        ctd_cf= ctd_cf, ctd_pf= ctd_pf, ctd_pr= ctd_pr,
        labels= dict(cot= cot_b, ctp= ctp_b, phist= phist_b),
        grfo= group_rfo_mean, #rfo_comment=rfo_comment,
        tgt_cr_names= cg_names, 
        xy=xy, #p_letter= p_letter,
        suptit=suptit, outfn=outfn,
    )
    plot_main(pic_data)

    return

#import matplotlib as mpl
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator, FuncFormatter
import cartopy.crs as ccrs

###-- Plotting
def plot_main(pdata):
    ## Parameters and variables
    cent_all= pdata['group_ctd'] 
    ctd_cf= pdata['ctd_cf'] 
    ctd_pf= pdata['ctd_pf'] 
    ctd_pr= pdata['ctd_pr']
    labels= pdata['labels']
    cr_names= pdata['tgt_cr_names']
    rfo_maps= pdata['group_rfo_map']
    grfo= pdata['grfo']
    xy= pdata['xy']
     
    ncr= len(cr_names)
    lat_max= rfo_maps[0].shape[0]//2

    ###---
    fig= plt.figure()
    fig.set_size_inches(14.5,16)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=20,y=0.98,va='bottom')
    lf=0.05;rf=0.95
    bf=0.05;tf=0.95
    gapx0,lx0=0.04, 0.13 #;
    npnx=2
    #lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy0=0.043; npny=6.5
    ly0=(tf-bf-gapy0*(npny-1))/float(npny)

    gapx1=0.018
    lx1=(rf-lf-npnx*(gapx0+lx0)-gapx1*(npnx-1))/float(npnx)

    ## Centroids
    cm = plt.get_cmap('jet',256)       
    cmnew = cm(np.arange(256))
    cmnew = cmnew[36:,:]
    newcm = cls.LinearSegmentedColormap.from_list("newJET",cmnew)
    newcm.set_under('white')
    props = dict(norm=cls.LogNorm(vmin=0.1,vmax=30),cmap=newcm,alpha=0.8)
    cm2 = plt.get_cmap('viridis').copy(); cm2.set_under('white')
    props2 = dict(norm=cls.LogNorm(vmin=0.1,vmax=30),cmap=cm2,alpha=0.8)
    
    ix0,ix1, iy= lf,lf+2*(lx0+gapx0),tf
    ix=ix0
    for ii,(crnm,cent1) in enumerate(zip(cr_names,cent_all)):
        ax1= fig.add_axes([ix,iy-ly0,lx0,ly0])
        chist,phist= cent1
        vv1= chist[::-1,:]*100.
        pic1= cent_show(ax1,vv1,props,labels,ytlabs='l')
        
        subtit= "{} [CF={:.1f}%]".format(crnm,ctd_cf[ii])
        cent_show_common(ax1,subtit)
        '''
        subtit= "{} {:.0f}% {:.0f}% {:.1f}mm/h".format(crnm,ctd_cf[ii],ctd_pf[ii],ctd_pr[ii])
        ax2= fig.add_axes([ix,iy-ly0-ly1*2.5,lx,ly1])
        vv2= ctd_phist[ii,:]*100.
        pic2= cent_pr_show(ax2,vv2,props2,labels)
        '''

        if ix==ix0:
            ax1.set_ylabel('Pressure (hPa)',fontsize=13,labelpad=0)

        ix+=(lx0+gapx0)
        
        if ix>ix0+lx0*2: # or l==2:
            ix=ix0
            iy-=ly0+gapy0

        if ii>=len(cent_all)-npnx:
            ax1.set_xlabel('Optical Thickness',fontsize=13)

    ## Colorbar for centroids
    if True:
        tt=[0.1,0.3,1,3,10,30]
        tt2=[str(x)+'%' for x in tt]
        hh= ly0/10
        loc1= [ix,iy-hh/2,lx1,hh] if ix==ix0 else [ix,iy-ly0*0.75,lx0,hh]
        #loc1= [ix,iy-ly0,hh,ly0]
        cb1=draw_colorbar(fig,pic1,loc1,tt,tt2,ft=12)
        cb1.ax.set_xlabel('Cloud Fraction',fontsize=14) #,rotation=-90,va='bottom') #,labelpad=0)
        cb1.ax.minorticks_off()
        
    ###--- RFO
    cm = plt.colormaps['magma_r'].resampled(80) #'CMRmap_r' 'YlOrBr' 'Accent' 'afmhot_r'
    cmnew = cm(np.arange(80)) #; print(cmnew[0,:])
    cmnew = np.concatenate((np.array([1,1,1,1]).reshape([1,-1]),cmnew[:-1,:])) #print cmnew[0,:],cmnew[-1,:]
    newcm = cls.LinearSegmentedColormap.from_list("newCMR",cmnew)
    newcm.set_under("white")

    lon_ext= [20,360+20]
    lc= (lon_ext[0]+lon_ext[1])/2
    data_crs= ccrs.PlateCarree()
    map_proj= ccrs.PlateCarree(central_longitude=lc)
    cmax=60
    props_pc = dict(cmap=newcm,alpha=0.9,transform=data_crs,vmin=0.,vmax=cmax,shading='nearest')
    cb_xt= range(0,cmax+1,10)
    
    ix,iy= ix1,tf
    for ii,(crnm,amap,grfo) in enumerate(zip(cr_names,rfo_maps,grfo)):
        ax=fig.add_axes([ix,iy-ly0,lx1,ly0],projection=map_proj)
        ax.set_extent(lon_ext+[-lat_max*1.03,lat_max*1.03],data_crs)

        rfo_text= 'RFO= {:.1f}%'.format(grfo) if grfo>9.995 else 'RFO={:.2f}%'.format(grfo)
        subtit='{} [{}]'.format(crnm,rfo_text)
        print(subtit)
        ax.set_title(subtit,fontsize=14,stretch='condensed') #x=0.0,ha='left',
        cs=ax.pcolormesh(*xy,amap,**props_pc)

        if ix==ix1:
            ll,lr= True,False
        else:   
            ll,lr= False,True
        #ax.set_yticks([-60,-30,0,30,60])
        #ax.tick_params(axis='y',labelright=lr,labelleft=ll,labelsize=11)
        #ax.yaxis.set_major_formatter(FuncFormatter(cf.lat_formatter))
        #ax.tick_params(direction='in',left=True,right=True,top=True,bottom=True,)
    
        label_idx=[ll,lr,False,True]
        map_common(ax,lat_max,label_idx)

        ix+=(lx1+gapx1)
        
        if ii==ncr-1:
            loc1= [ix,iy-hh/2,lx1,hh] if ix==ix1 else [ix,iy-ly0*0.75,lx1,hh]
            tt= cb_xt
            tt2=[str(x)+'%' for x in tt]
            cb1=draw_colorbar(fig,cs,loc1,tt,tt2,ft=12,extend='max')
            cb1.ax.set_xlabel('RFO',fontsize=14) #,rotation=-90,va='bottom')
           
        if ix>ix1+lx1*2: # or l==2:
            ix=ix1
            iy-=ly0+gapy0
        
    ### Save
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
    ax1.set_title(subtit,x=0.,ha='left',fontsize=13,stretch='condensed')

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


def draw_colorbar(fig,pic1,loc,tt,tt2,ft=10,extend='both'):

    cb_ax = fig.add_axes(loc)  ##<= (left,bottom,width,height)
    if loc[2]<loc[3]:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='vertical',ticks=tt,extend=extend)
        cb.ax.set_yticklabels(tt2,size=ft,stretch='condensed')
    else:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='horizontal',ticks=tt,extend=extend)
        cb.ax.set_xticklabels(tt2,size=ft,stretch='condensed')
    return cb
    
if __name__=="__main__":
    cr_params= dict(rg=50, nelemp=6, prwt=0)  
    tgt_cr_groups= [
        ('H1_tk',(1,3,5)), ('H2_tk',(2,6)),
        ('H_tn',(7,8,9)),
        ('Mid',(4,151,152)),
        ('L1_tk',(11,13)), ('L2_tk',(10,12)),
        ('L_tn',(14,)),
        ('S-Clr',(153,)), ('Clr',(0,)),
    ]
    main(cr_params,tgt_cr_groups)