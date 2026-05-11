'''
Climatology of Low-CR group RFOs and a map of all RFOs > rfo_crt
2002.09-2024.08 (22 years)

By Daeho Jin
2026.03.24
'''

import numpy as np
import sys
import os.path
from datetime import timedelta, date
from netCDF4 import Dataset, num2date
import common_functions as cf


def main():
    ###--- Parameters
    sat_nm= 'TAmean' #'CERES_FBCT-Day' #
    
    #tgt_boxes= get_tgt_boxes()[:5]

    max_lat= 64 #
    #trop_lat= 32 #
    #trop= True #False #
    
    tshs=   ['monthly',1] #['monthly',5] #

    tgt_dates= (date(2002,9,1),date(2024,8,31))
    tgt_date_names= [d.strftime('%Y.%m') for d in tgt_dates]
    nmon= cf.get_tot_months(*tgt_dates)
    nmon_yr= 12
    nyr= nmon//12

    sn_names= ['All','SON','DJF','MAM','JJA']
    #sn_idx= 4
    #sn_name= sn_names[sn_idx]
    #sn_mons= np.arange(3,dtype=int)+sn_idx*3+6 if sn_idx>0 else np.arange(12,dtype=int)+9
    #sn_mons[sn_mons>12]-=12
    #print(sn_name, sn_mons) #; sys.exit()
    
    sn_mon_idx=[]
    sn_nms=[]
    #all_months= cf.yield_monthly_date_range(*tgt_dates)
    all_months= np.asarray([dd.month for dd in cf.yield_date_range(*tgt_dates)])
    for sn_idx in [2,4]: # DJF and JJA
        sn_mons= np.arange(3,dtype=int)+sn_idx*3+6 if sn_idx>0 else np.arange(12,dtype=int)+9
        sn_mons[sn_mons>12]-=12
        sn_idx_bool=np.isin(all_months,sn_mons)
        sn_mon_idx.append(sn_idx_bool)
        print(sn_names[sn_idx],sn_mons,sn_idx_bool.sum())
        sn_nms.append(sn_names[sn_idx])
    #print(sn_mon_idx[-1].shape, sn_mon_idx[-1].dtype); sys.exit()
        
    ### Read data
    hs1= tshs[1]
    tgt_latlon1, tgt_rg_name1= [-max_lat,max_lat,-180,180], '{a}S-{a}N_Ocean'.format(a=max_lat)
    nlat,nlon= max_lat*2//hs1,360//hs1
    latinfo, loninfo = (-max_lat+hs1/2,hs1,nlat), (-180+hs1/2,hs1,nlon)
    latlon_info= dict(latinfo=latinfo, loninfo=loninfo)
    lats= np.arange(nlat)*hs1+latinfo[0]
    lons= np.arange(nlon)*hs1+loninfo[0]
    xy= np.meshgrid(lons,lats)
    lat_weight= cf.apply_lat_weight(np.ones([nlat,nlon]),nlat,nlon,lats,geodetic=True).squeeze()
    
    ### Read LO Mask
    wpct= cf.get_Water_Pct(tgt_latlon1,hs1)
    print(wpct.shape)
    print(wpct.min(), wpct.max())
    lomask0= wpct>= 90  ## Ocean only
    lomask1= wpct<90

    ### Read regime RFO
    rg_set= dict(rg=50,nelemp=0,prwt=0,km=15)
    tgt_cr,subk= rg_set['km'],3
    
    tgt_cr_groups= [
        ##('H1_tk',(1,3,5)), ('H2_tk',(2,6)), 
        ##('H1_tn',(7,9)),  ('H2_tn',(8,)),
        #('H_tk',(1,2,3,5,6,)),
        #('H_tn',(7,8,9)),
        #('Mid',(4,151,152)),
        ('L1_tk',(11,13)),
        ('L2_tk',(10,12)),                    
        ('L_tn',(14,)),
        ('S-Clr',(153,0)),
        #('Clr',(0,)),        
    ]
    tgt_crs= [item[1] for item in tgt_cr_groups]
    cr_name= [item[0] for item in tgt_cr_groups]
    ncr= len(tgt_cr_groups)
    #tgt_cr= [11,13,10,12,] #14,153]
    
    crmap= cf.read_cpr_map(rg_set,sat_nm,tgt_dates,tgt_latlon1)
    crmap_sub= cf.read_cpr_map(rg_set,sat_nm,tgt_dates,tgt_latlon1,sub=True,tgt_cr=tgt_cr,subk=subk)
    print(crmap.shape,crmap_sub.shape)
    nt1,nlat1,nlon1= crmap.shape
    ## Degrading resolution
    if hs1>1:
        crmap= crmap.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
        crmap_sub= crmap_sub.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
    elif hs1==1:
        crmap= crmap.reshape([nt1,nlat,nlon,1])
        crmap_sub= crmap_sub.reshape([nt1,nlat,nlon,1])

    '''
    ## Read Total CF
    cscf_crt= 5.
    cfmap= cf.get_Total_CF_daily(tgt_dates,tgt_latlon1,sat_nm=sat_nm)
    print(cfmap.shape,cfmap.min(), cfmap.max())
    if hs1>1:
        cfmap= cfmap.reshape([nt1,nlat,hs1,nlon,hs1]).swapaxes(2,3).reshape([nt1,nlat,nlon,hs1*hs1])
    elif hs1==1:
        cfmap= cfmap.reshape([nt1,nlat,nlon,1])
    non_cs= cfmap>= cscf_crt/100. #
    cs_idx= np.logical_and(cfmap>-0.00001,cfmap< cscf_crt/100.)
    cfmap=0
    print(cs_idx.sum(), cs_idx.sum()/non_cs.sum()*100.)
    '''
    ## Seasonal filtering
    mrfo_all=[]
    for sn_idx in sn_mon_idx:
        
        by_tcr=[]
        for tgt_cr in tgt_crs:
            idx_all=False
            for tcr in tgt_cr:
                if tcr<=rg_set['km']:
                    idx= crmap[sn_idx,:]==tcr                    
                else:
                    tcr1= tcr-rg_set['km']*10
                    idx= crmap_sub[sn_idx,:]==tcr1
                idx_all= np.logical_or(idx_all,idx)
            ### Exclude new CS
            #idx_all= np.logical_and(idx_all,non_cs[sn_idx,:])
            by_tcr.append(idx_all.mean(axis=(0,-1)))
            #print(idx_all.shape, by_tcr[-1].shape)

        tmp_rfos= np.asarray(by_tcr)*100 ## Now in %  #[ncr,nlat,nlon]
        mrfo_all.append(tmp_rfos)

    ### RFO                
    #rfos= np.asarray(by_month).swapaxes(0,1)
    for rfos in mrfo_all:
        print(rfos.shape, rfos.min(), rfos.max(),rfos[:,4::12,:].mean(axis=(1,2))) #; sys.exit() #[ncr,nmon,nlat,nlon]
    #sys.exit()
    #rfos= rfos.reshape([ncr,nyr,nmon_yr,nlat,nlon])

    #rfo0= np.asarray(by_month_zero); print(rfo0.shape,rfo0.min(), rfo0.max(),rfo0[4::12,:].mean()) #; sys.exit()
    #rfo0= rfo0.reshape([1,nyr,nmon_yr,nlat,nlon])
    #rfos= np.concatenate((rfos,rfo0),axis=0); print(rfos.shape)
    
    ### RFO clim and intersection
    for i,rfos in enumerate(mrfo_all):
        rfos_clim= np.ma.masked_array(rfos,mask=np.tile(lomask1,(ncr,1)))
        clim_crt= 10

        rfos_common= (rfos_clim>=clim_crt).sum(axis=0)==ncr
        print(rfos_common.sum()) #; sys.exit()
        mrfo_all[i]=[rfos_clim,rfos_common]

    ### For Figure    
    #tshs_nm= '{} {}\u00B0 vs. {}\u00B0'.format(tsnm,hs1,hs2)
    #tshs_fn= '+'.join([nm1.split()[0][0]+nm1.split()[1][0] for nm1 in [comp_nm,]])
    #tshs_fn= '{}_{}deg'.format(*tshs)
    
    
    #rg_nm= '{a}S-{a}N_Ocean'.format(a=trop_lat) if trop else 
    #rg_nm= 'Select_Boxes'
    rg_nm= tgt_rg_name1
    

    suptit= 'Mean RFO of Low Cloud Groups [{}-{}]'.format(*tgt_date_names)
    #if sn_idx>0:
    #    suptit= suptit[:-1]+', {}]'.format(sn_name)
    outdir= '../../Writing_LCC_LCAI/Pics/'       
    outfn= outdir+'v2_Fig01b.CR_RFO_clim_map.{}.{}-{}_{}.png'.format(
         rg_nm,*tgt_date_names,'+'.join(sn_nms))
    pic_data= dict(rfos_clim=[item[0] for item in mrfo_all],
                   rfos_common=[item[1] for item in mrfo_all],
                   cld_names= cr_name, sn_names=sn_nms,
                   xy=xy,lw=lat_weight,clim_crt=clim_crt,
                   suptit=suptit, outfn=outfn, )
    plot_main0(pic_data)

    '''
    suptit= 'Low Cloud Groups: Commonly RFO\u2265{}% [{}-{}]'.format(clim_crt,*tgt_date_names)
    #if sn_idx>0:
    #    suptit= suptit[:-1]+', {}]'.format(sn_name)
    outdir= '../../Writing_LCC_LCAI/Pics/'       
    outfn= outdir+'v1_Fig01b.CR_RFO_clim_map_common.{}.{}-{}_{}.png'.format(
         rg_nm,*tgt_date_names,'+'.join(sn_nms))
    pic_data= dict(rfos_common=[item[1] for item in mrfo_all],
                   cld_names= cr_name, sn_names=sn_nms,
                   xy=xy,lw=lat_weight,clim_crt=clim_crt,
                   suptit=suptit, outfn=outfn, )
    plot_main1(pic_data)
    '''
    return


#import plot_common as pcf
import matplotlib as mpl
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
import cartopy.crs as ccrs
from cartopy.feature import LAND
def plot_main0(pdata):
    rfos_clim = pdata['rfos_clim']
    rfos_common= pdata['rfos_common']
    cld_names= pdata['cld_names']
    xy,lw= pdata['xy'], pdata['lw']
    clim_crt= pdata['clim_crt']
    sn_names= pdata['sn_names']
    
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'
    
  
    
    ###---
    fig=plt.figure()
    fig.set_size_inches(8.5,6)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=17,y=0.98,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    ncol,nrow=2,2
    lf,rf,bf,tf=0.02,0.98,0.3,0.92
    gapx, npnx=0.054,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.095,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    
    ix=lf; iy=tf

    lon_ext= [20,360+20]
    cm = (lon_ext[0]+lon_ext[1])/2 #180
    #map_proj = ccrs.Robinson(central_longitude=cm)
    map_proj= ccrs.PlateCarree(central_longitude=cm)
    data_crs= ccrs.PlateCarree()

    vmin,vmax= clim_crt,90
    ccb= range(vmin,vmax+1,15)
    n_ccb= len(ccb)+1
    cm= mpl.colormaps['plasma_r'].resampled(n_ccb)(np.arange(n_ccb))
    #cm= plt.get_cmap('plasma_r').resampled(n_ccb)(np.arange(n_ccb))
    newcm= cls.ListedColormap(cm[1:-1,:]).with_extremes(over=cm[-1,:],under=[1.,1.,1.,1.]) #cm[0,:]) #[1,1,1,1])
    norm = cls.BoundaryNorm(ccb, newcm.N)

    props_mesh= dict(cmap=newcm,alpha=0.86,norm=norm,transform=data_crs)
    props_contour= dict(alpha=0.7,colors='0.1',linewidths=1,transform=data_crs)
    mpl.rcParams["hatch.color"]='0.4'
    
    fw= ['roman','demibold', 'demi', 'bold', 'semibold']
    ### Plot maps    
    ai=0    
    #for ii,(amap,tit) in enumerate(zip(rfos_clim,cld_names)):
    for ii,crnm in enumerate(cld_names):
        for jj,snm in enumerate(sn_names):
            tgt_boxes= get_tgt_boxes_12d(snm)  
            amap= rfos_clim[jj][ii,:]
            ax1=fig.add_axes([ix,iy-ly,lx,ly], projection=map_proj)
            #ax1.set_extent([0,360,-61,61],data_crs)
            ax1.set_extent(lon_ext+[-64,64],data_crs)
            
            pic1= ax1.pcolormesh(*xy,amap,shading='nearest',**props_mesh)
            pic2= ax1.contour(*xy,amap,ccb[0::2],**props_contour)
            ax1.clabel(pic2,ccb[0::2],inline=True,fontsize=8)
            
            subtit= "({}) {} in {}".format(abc[ai],crnm,snm); ai+=1
            ax1.set_title(subtit,fontsize=13,x=0,ha='left')
            #right_label= True if (ii+1)%ncol==0 else False
        
            ix+= lx+gapx
            if ix+gapx>rf:
                right_label=True
                ix=lf
                iy-= ly+gapy
            else:
                right_label=False
            map_common(ax1,data_crs,right_label=right_label,lon_ext=lon_ext)
            ax1.add_feature(LAND,facecolor='0.85') #'#bfbfbf')
            draw_box(ax1,tgt_boxes,data_crs,ls='-',c='c')

            mtxt= 'Mean= {:.1f}%'.format(np.average(amap.compressed(),weights=lw[~amap.mask]))

    
    if True:        
        hh=0.025 #; dx= 0.25
        #loc0= [ix,iy,lx,hh]
        loc0= [0.2,iy+gapy*0.15,0.6,hh]
        tt= ccb #np.arange(minv,maxv+0.01,0.2)
        #tt2= ['{}'.format(val) if i%2==0 else '' for i,val in enumerate(tt)]
        tt2= [f'{v}' for v in tt]
        cb0 =draw_colorbar(fig,pic1,loc0,ft=10,extend='both',tt=tt,tt2=tt2)
        cb0.ax.set_xlabel('RFO (%)',fontsize=11,labelpad=0) #,x=1,ha='right') #,rotation=-90,va='bottom')
        #txt= fig.text(lf,iy-ly-gapy,'Hatch: \u03C9{}<{} Pa/s'.format(r'$_{500}$',w500_crt),
        #              ha='left',va='bottom',color='k',fontsize=11, )

    iy-= gapy*1
    ## Intersection map
    if True:
        for jj,snm in enumerate(sn_names):
            tgt_boxes= get_tgt_boxes_12d(snm)  
            amap= rfos_common[jj]
                
            ax2=fig.add_axes([ix,iy-ly,lx,ly], projection=map_proj)    
            ax2.set_extent(lon_ext+[-64,64],data_crs)
            pmap= np.ma.masked_less(amap,0.5)
            pic2= ax2.pcolor(*xy,pmap,hatch='xxxxx',alpha=0.,transform=data_crs)
            map_common(ax2,data_crs,right_label=False,lon_ext=lon_ext)
            ax2.add_feature(LAND,facecolor='0.85') #'#bfbfbf')
            draw_box(ax2,tgt_boxes,data_crs,ls='-',c='c')
            subtit= "({}) All RFOs \u2265 {}% in {}".format(abc[ai],clim_crt,snm); ai+=1
            #subtit= '({}) {}'.format(abc[ai],snm); ai+=1
            ax2.set_title(subtit,fontsize=13,x=0,ha='left')
        
            ix+= lx+gapx    
    

    ###---
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150) #
    #plt.show()
    print(pdata['outfn'])
    return


def map_common(ax,data_crs,right_label=False,lon_ext=[0,360]):
    #ax.set_extent([0,359.9,-24.1,24.1],data_crs)

    ax.coastlines(color='silver',linewidth=1.)
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
    label_idx=[False,False,False,True] #[True,True,False,True]
    gl.top_labels = label_idx[2]
    gl.left_labels = label_idx[0]
    gl.right_labels = label_idx[1]
    gl.bottom_labels = label_idx[3]
    gl.ylocator = MultipleLocator(30)
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}

    ax.set_aspect('auto')

    for lt in range(-60,61,30):
        #ax.yaxis.set_major_locator(MultipleLocator(20))
        #ax.yaxis.set_major_formatter(FuncFormatter(cf.lat_formatter))
        #ax.tick_params(axis='y',which='major',labelsize=10)
        #ax.tick_params(left=True,right=True)
        ax.text(lon_ext[0]+0.01,lt,cf.lat_formatter(lt,0)+' ',ha='right',va='center',fontsize=10,c='k',transform=data_crs)
        if right_label:
            ax.text(lon_ext[1]-0.01,lt,' '+cf.lat_formatter(lt,0),ha='left',va='center',fontsize=10,c='k',transform=data_crs)

    
    return


def get_tgt_boxes_12d(sn_idx):
    tgt_boxes_JJA= [
        ('Peruvian',(-22,-10,-100,-88)), #-92,-84)), #-90,-80)),
        ('Namibian',(-22,-10,-10,2)), #-4,4)),
        ('Californian',(18,30,-146,-134)), #-136,-128)),
        #('Australian',(-40,-30,75,85)), #96,104)),
    ]

    tgt_boxes_DJF= [
        ('Peruvian',(-30,-18,-90,-78)), #-92,-84)), #-90,-80)),
        ('Namibian',(-26,-14,-6,6)), #-4,4)),
        #('Californian',(20,30,-145,-135)), #-136,-128)),
        ('Australian',(-36,-24,94,106)), #96,104)),
    ]
    if isinstance(sn_idx,int):
        if sn_idx==2:
            return tgt_boxes_DJF
        elif sn_idx==4:
            return tgt_boxes_JJA
        else:
            return tgt_boxes_tk
    elif isinstance(sn_idx,str):
        if sn_idx.lower()=='djf':
            return tgt_boxes_DJF
        elif sn_idx.lower()=='jja':
            return tgt_boxes_JJA
        else:
            return tgt_boxes_tk
    else:
        print('Check argument',sn_idx)
        sys.exit()
    
def get_tgt_boxes(sn_idx):
    tgt_boxes_JJA= [
        ('Peruvian',(-20,-10,-100,-90)), #-92,-84)), #-90,-80)),
        ('Namibian',(-20,-10,-10,0)), #-4,4)),
        ('Californian',(20,30,-145,-135)), #-136,-128)),
        #('Australian',(-40,-30,75,85)), #96,104)),
    ]

    tgt_boxes_DJF= [
        ('Peruvian',(-30,-20,-90,-80)), #-92,-84)), #-90,-80)),
        ('Namibian',(-25,-15,-5,5)), #-4,4)),
        #('Californian',(20,30,-145,-135)), #-136,-128)),
        ('Australian',(-35,-25,95,105)), #96,104)),
    ]
        
    tgt_boxes_tk= [ ('Peruvian',(-20,-12,-88,-80)), #-92,-84)), #-90,-80)),
                    ('Namibian',(-20,-12,0,8)), #-4,4)),
                    ('Californian',(24,32,-132,-124)), #-136,-128)),
                    ('Australian',(-36,-28,100,108)), #96,104)),

                    #('Azores',(40,50,-25,-15)),
                    #('SW.IndOce',(-52,-44,28,36)),

                    ('S.Pacifc',(-56,-48,-176,-168)),
                    #('SE.Atlantic',(-56,-48,4,12)),
                    #('N.Pacific',(48,56,172,180)),

                    #('Canarian',(15,25,-35,-25)),
                    #('China',(20,105)),
                    #('N.Pacific',(45,55,170,180)),  ## Modified
                    #('N.Atlantic',(50,60,-45,-35)),
                    #
    ]
    if isinstance(sn_idx,int):
        if sn_idx==2:
            return tgt_boxes_DJF
        elif sn_idx==4:
            return tgt_boxes_JJA
        else:
            return tgt_boxes_tk
    elif isinstance(sn_idx,str):
        if sn_idx.lower()=='djf':
            return tgt_boxes_DJF
        elif sn_idx.lower()=='jja':
            return tgt_boxes_JJA
        else:
            return tgt_boxes_tk
    else:
        print('Check argument',sn_idx)
        sys.exit()

def draw_box(ax1,tgt_boxes,data_crs,ls='-',c='0.5'):
    alp=1
    for abox in tgt_boxes:
        y1,y2,x1,x2= abox[1]

        ax1.plot([x1,x1],[y1,y2],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
        ax1.plot([x2,x2],[y1,y2],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
        ax1.plot([x1,x2],[y1,y1],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
        ax1.plot([x1,x2],[y2,y2],c=c,lw=1.5,alpha=alp,ls=ls,transform=data_crs)
    return

def draw_colorbar(fig,pic1,loc,ft=10,extend='both',tt=[],tt2=[]): #max_vals=[0,1],val_lin=0.02,unit=''):
    #tt=[0.1,0.3,1,3,10,30]
    #tt= np.arange(np.ceil(max_vals[0]*10)/10, max_vals[1]+val_lin/2, val_lin)
    #tt2=['{:.01f}{}'.format(x,unit) for x in tt] if val_lin<1 else ['{:.0f}{}'.format(x,unit) for x in tt]

        ###- Get position from previous subplot
#        pos1=ax1.get_position().bounds  ##<= (left,bottom,width,height)
#        cb_ax = fig.add_axes([0.1,pos1[1]-0.05,0.8,0.015])
        #cb = m.colorbar(cs,"bottom", size="5%", pad="10%")

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
    main()

