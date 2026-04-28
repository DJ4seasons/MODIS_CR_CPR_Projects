"""
Read CR-group NC file and draw a figure showing centroids (left) and RFO maps (right)

Clear-sky regime here is defined with relaxed criterion of 5%

Apply geodetic weight + month_day weight for mean RFO
month_day weight for centroid
2025.11.13
"""

import numpy as np
import sys
import os.path
from datetime import timedelta, date
from netCDF4 import Dataset

import common_functions as cf
#from subprocess import call
#import matplotlib as mpl
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator, FuncFormatter
import cartopy.crs as ccrs


def cent_show(ax1,ctd):     

    nx = 6 #TAU (Optical Thickness)
    ny = 7 #CTP
    if len(ctd.reshape(-1)) != nx*ny:
        print("Error: centroid data size is bad:",ctd.shape)
        sys.exit()

    cm = plt.get_cmap('jet',512)       
    cmnew = cm(np.arange(512))
    cmnew = cmnew[72:,:]
    newcm = cls.LinearSegmentedColormap.from_list("newJET",cmnew)
    newcm.set_under('white')

    props = dict(norm=cls.LogNorm(vmin=0.1,vmax=30),cmap=newcm,alpha=0.9)

    pic1=ax1.imshow(ctd,interpolation='nearest',aspect=0.84,**props)

    ### Axis Control
    xlabs=['0','1.3','3.6','9.4','23','60','150']
    ylabs=[1100,800,680,560,440,310,180,0]

    ax1.set_xlim(-0.5,5.5)
    ax1.set_ylim(-0.5,6.5)
    ax1.set_xticks(np.arange(nx+1)-0.5)
    ax1.set_xticklabels(xlabs)

    ax1.set_yticks(np.arange(ny+1)-0.5)
    ax1.set_yticklabels(ylabs)        
        
    for j in range(7):
        for i in range(6):
            if abs(ctd[j,i])>4.5:
                ax1.annotate("%.0f" %(ctd[j,i]),xy=(i,j),ha='center',va='center',stretch='semi-condensed',fontsize=11)
    return pic1

def cent_show_common(ax1,crnm,cf,km=0):

    ### add a title.
    subtit= "{} [CF={:.1f}%]".format(crnm,cf); print(subtit)
    ax1.set_title(subtit,fontsize=14,stretch='condensed') #x=0.0,ha='left',

    ### Draw Guide Line
    ax1.axvline(x=1.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axvline(x=3.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axhline(y=1.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axhline(y=3.5,linewidth=0.7,color='k',linestyle=':')

    ### Ticks
    ax1.tick_params(axis='both',which='major',labelsize=11)
    ax1.tick_params(left=True,right=True)
    return

def lat_formatter(x,pos):
    if x>0:
        return "{:.0f}\u00B0N".format(x)
    elif x<0:
        return "{:.0f}\u00B0S".format(-x)
    else:
        return "{:.0f}\u00B0".format(x)

def map_common(ax,label_idx=[True,True,False,True]):
    ax.set_extent([0.,360,-61,61],ccrs.PlateCarree())

    ax.coastlines(color='silver',linewidth=1.)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = label_idx[2]
    gl.left_labels = label_idx[0]
    gl.right_labels = label_idx[1]
    gl.bottom_labels = label_idx[3]

    gl.xlocator = MultipleLocator(60) 
    gl.ylocator = MultipleLocator(30)
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

def main():
    ###-- Parameters and defalut values
    ###-------------------------------------
    rg, km, sid= 50, 15, 0
    ncl=km; nelem=42

    mdnm = 't+a_cld_hist_{}S-{}N'.format(rg,rg)
    dir1 = '/Users/djin1/Documents/CLD_Work/Data_Obs/Cld+Pr_Regime/New_CTD/'
    ctdfnm= 'Centroid.MODIS_{}.k{}x{}'.format(mdnm,km,nelem)
    fnm=dir1+ctdfnm+'.f64dat'

    tgt_cr,subk= km,3
    ctdfnm_sub= ctdfnm+'.CR{}_subk{}'.format(tgt_cr,subk)
    fnm_sub=dir1+ctdfnm_sub+'.f64dat'


    ###
    ###---- Read Centroid
    obsctd = bin_file_read2mtx(fnm,dtp=np.float64)
    obsctd = obsctd.reshape([ncl,nelem])

    obsctd2 = bin_file_read2mtx(fnm_sub,dtp=np.float64)
    obsctd2 = obsctd2.reshape([subk,nelem])



    obscf = np.sum(obsctd,axis=1)*100.
    np.set_printoptions(precision=3,suppress=True)  
    print( obscf)
    print(np.sum(obsctd2,axis=1)*100.)


    #tgt_crs= [[1,3,5,],[7,9],[2,6],[8],[4,151,152],[10,11,12,13],[14,],[153,],] #[0,]]
    #cr_name=['H1_tk','H1_tn','H2_tk','H2_tn','Mid','L_tk','L_tn','S-Clr',] #'CS+Ms']
    #tgt_crs= [[11,13],[10,12],[11,13,10,12],[14,],[153,],] #[0,]]
    #cr_name=['L_tk1','L_tk2','L_tk_1+2', 'L_tn','S-Clr',] #'CS+Ms']
    cr_name= ['H1_tk','H2_tk','H_tn','Mid','L1_tk','L2_tk','L_tn','S-Clr','Clr (CF<5%)']
    ncr= len(cr_name)

    ###--- RFO
    tgt_dates= (date(2002,9,1),date(2024,8,31))
    tgt_date_names= [tgt_dates[0].year+1,tgt_dates[1].year]
    tgt_latlon, tgt_rg_name = [-60,60,-180,180], '60S-60N'
    nyr= tgt_dates[1].year - tgt_dates[0].year
    month_days= np.asarray(cf.get_month_days(tgt_dates))

    nlat,nlon=120,360
    lats= np.arange(nlat)-nlat/2+0.5
    lat_weight= cf.apply_lat_weight(np.ones([nlat,nlon,]),nlat,nlon,lats,geodetic=True).squeeze()
    #print(lat_weight.shape,lat_weight[nlat//2,0]); sys.exit()
    lons= np.arange(nlon)-nlon/2+0.5
    #lons2d,lats2d= np.meshgrid(lons,lats)


    satnm= 'TAmean'
    rg,nelemp,prwt,km= 50,0,0,15 #15,6,1,16 #7,22 #
    rg_set= dict(rg=rg,nelemp=nelemp,prwt=prwt,km=km)
    tgt_cr,subk= km,3
    #crnum= cf.read_cpr_map(rg_set,satnm,tgt_dates,tgt_latlon).reshape([-1,nlat,nlon])
    #crnum_sub= cf.read_cpr_map(rg_set,satnm,tgt_dates,tgt_latlon,sub=True,tgt_cr=tgt_cr,subk=subk).reshape([-1,nlat,nlon])


    
    indir= '/Users/djin1/Documents/CLD_Work/Data_Obs/MODIS_c61/Composite/'
    infn= indir+'Monthly_Composite_Histogram+RFO_map.by{}CRgroups.nc'.format(ncr)

    fid= Dataset(infn,'r')
    mrfo= fid.variables['mRFO'][:]
    jh= fid.variables['Monthly_CF_Joint_Hist']
    ncr,nmon,nlat,nlon,nctp,ntau= jh.shape
    #nlat,nlon=30,90  ## Now in 4-deg resolution
    resol=4
    if len(lats)//resol!=nlat or len(lons)//resol!=nlon:
        print("Dimension dismatch",len(lats)//resol,nlat, len(lons)//resol,nlon)
        sys.exit()
    else:
        lats,lons= lats.reshape([-1,resol]).mean(axis=1), lons.reshape([-1,resol]).mean(axis=1) 
        #ltw= cf.apply_lat_weight(np.ones([nlat,nlon,]),nlat,nlon,lats).squeeze()
        ltw= lat_weight.reshape([nlat,resol,nlon,resol]).sum(axis=1)[:,:,0]
        print(ltw.shape,ltw[0,0],ltw[nlat//2,0])
    lons2d,lats2d= np.meshgrid(lons,lats)

    cent_all=[]
    #rfo_all=mrfo.mean(axis=1)*100.
    rfo_all= np.average(mrfo,weights=month_days,axis=1)*100.
    grfo_all=[]
    for icr in range(ncr):
        wt1= (mrfo[icr]*month_days[:,None,None]).reshape(-1)  #*ltw[None,:,:]
        cent1= jh[icr,:].reshape([nmon*nlat*nlon,nctp*ntau])
        cent1= np.ma.average(cent1,weights=wt1,axis=0)*100.
        cent_all.append(cent1)
        grfo_all.append(np.ma.average(rfo_all[icr],weights=ltw))
    cent_all= np.asarray(cent_all)
    print(cent_all.sum(axis=1)) #; sys.exit()
    grfo_all= np.asarray(grfo_all)
    print(grfo_all)

    #tgt_crs= [[1,3,5],[7,9],[2,6],[8],[4,151,152],[10,11,12,13],[14,],[153,],[0,-1]]
    #cr_name=['H1_tk','H1_tn','H2_tk','H2_tn','Mid','L_tk','L_tn','S-Clr','CS+Miss']

    ### Read LO Mask
    indir= '/Users/djin1/Documents/CLD_Work/Data_Obs/'
    infn= indir+'PctWater.dat'
    lat_idx= [90+tgt_latlon[0],90+tgt_latlon[1]]
    lomask0= cf.bin_file_read2mtx(infn).reshape([180,360])[lat_idx[0]:lat_idx[1],:]
    print(lomask0.min(), lomask0.max())    
    lomask0= lomask0 >= 90  ## Ocean only
    

    ###---- Plot
    suptit= "MODIS_C6.1 T+A C{}R in {}, {}, k={}".format(p_letter,rg_nm,prset_nm,km)
    outdir = './Pics/'
    outfn = outdir+ f"Fig01.Monthly_C-group8_ctd+rfo.{ctdfnm}.png"
    pic_data= dict(
        ctd_cld= ctd_cld, ctd_phist= ctd_phist,
        ctd_cf= ctd_cf, ctd_pf= ctd_pf, ctd_pr= ctd_pr,
        labels= dict(cot= cot_b, ctp= ctp_b, phist= phist_b),
        p_letter= p_letter,
        suptit=suptit, outfn=outfn,
    )
    plot_main(pic_data)
    return
        
###-------------------------------------
def plot_main(pdata):

    ###-- Plotting basics
    #fig, axs = plt.subplots(5,3)  ## (ny,nx)
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

    ix0,ix1, iy= lf,lf+2*(lx0+gapx0),tf
    ix=ix0
    for l,(crnm,vv) in enumerate(zip(cr_name,cent_all)):
        ax1= fig.add_axes([ix,iy-ly0,lx0,ly0])
        vv1= vv.reshape([nctp,ntau])[::-1,:]
        pic1= cent_show(ax1,vv1)
        cent_show_common(ax1,crnm,vv1.sum())
    
        if ix==ix0:
            ax1.set_ylabel('Pressure (hPa)',fontsize=13,labelpad=0)
    
        ix+=(lx0+gapx0)
        
        if ix>ix0+lx0*2: # or l==2:
            ix=ix0
            iy-=ly0+gapy0

        if l>=len(cent_all)-npnx:
            ax1.set_xlabel('Optical Thickness',fontsize=13)

    #ax1= fig.add_axes([ix,iy-ly0,lx0,ly0])
    #ix+=(lx0+gapx0)
    #ax1= fig.add_axes([ix,iy-ly0,lx0,ly0])
    #ix=ix0
    ###--- Colorbar
    if True:
        tt=[0.1,0.3,1,3,10,30]
        tt2=[str(x)+'%' for x in tt]
        hh= ly0/10
        #loc1= [ix,iy-ly0*0.25,lx0*2+gapx1,hh]
        loc1= [ix,iy-ly0,hh,ly0]
        cb1=draw_colorbar(fig,pic1,loc1,tt,tt2,ft=12)
        cb1.ax.set_ylabel('Cloud Fraction',fontsize=14) #,rotation=-90,va='bottom') #,labelpad=0)
        cb1.ax.minorticks_off()
        

    ###--- RFO
    cm = mpl.colormaps['magma_r'].resampled(80) #'CMRmap_r' 'YlOrBr' 'Accent' 'afmhot_r'
    cmnew = cm(np.arange(80)) #; print(cmnew[0,:])
    cmnew = np.concatenate((np.array([1,1,1,1]).reshape([1,-1]),cmnew[:-1,:])) #print cmnew[0,:],cmnew[-1,:]

    newcm = cls.LinearSegmentedColormap.from_list("newCMR",cmnew)
    newcm.set_under("white")


    #props = dict(cmap=newcm,alpha=0.9,transform=ccrs.PlateCarree()) #,projection=ccrs.PlateCarree())
    lon_ext= [20,360+20]
    cm= (lon_ext[0]+lon_ext[1])/2   
    data_crs= ccrs.PlateCarree()
    props_pc = dict(cmap=newcm,alpha=0.9,transform=data_crs,vmin=0.,vmax=60,shading='nearest')
    ix,iy= ix1,tf
    for l,(crnm,amap,grfo) in enumerate(zip(cr_name,rfo_all,grfo_all)):
        ax=fig.add_axes([ix,iy-ly0,lx1,ly0],projection=ccrs.PlateCarree(central_longitude=cm))
        ax.set_extent(lon_ext+[-61,61],data_crs)
        #rfo= np.average(amap,weights=lat_weight)
        rfo_text= 'RFO= {:.1f}%'.format(grfo) #if rfo>9.995 else 'RFO={:.2f}%'.format(grfo)
        #rfo_text= 'RFO_All={:.1f}%, RFO_Oce={:.1f}%'.format(sum(grfo[l]),sum(orfo[l]))
        subtit='{} [{}]'.format(crnm,rfo_text)
        print(subtit)
        ax.set_title(subtit,fontsize=14,stretch='condensed') #x=0.0,ha='left',
        #cs=ax.contourf(lons_new,lats_new,map0[ii,:,:],np.linspace(0,75,101),**props)
        cs=ax.pcolormesh(lons2d,lats2d,amap,**props_pc)

        if ix==ix1:
            ll,lr= True,False
        else:
            ll,lr= False,True
        ax.set_yticks([-60,-30,0,30,60])
        ax.tick_params(axis='y',labelright=lr,labelleft=ll,labelsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))

        ax.tick_params(direction='in',left=True,right=True,top=True,bottom=True,)
    
        label_idx=[False,False,False,True]
        map_common(ax,label_idx)

        ix+=(lx1+gapx1)
        if l==ncr-1:
            loc1= [ix,iy-hh/2,lx1,hh] if ix==ix1 else [ix,iy-ly0*0.25,lx1,hh]
            tt=range(0,61,10)
            tt2=[str(x)+'%' for x in tt]
            cb1=draw_colorbar(fig,cs,loc1,tt,tt2,ft=12,extend='max')
            cb1.ax.set_xlabel('RFO',fontsize=14) #,rotation=-90,va='bottom')
        
        if ix>ix1+lx1*2: # or l==2:
            ix=ix1
            iy-=ly0+gapy0
        


    ###--- Save
    fnout = pdata['outfn']
    ### Show or Save
    plt.savefig(fnout,bbox_inches='tight',dpi=150)
    #plt.show()

    print(outdir+fnout)
    return
