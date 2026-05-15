"""
Display centroids and mean RFO for select low-CR groups
: based on 50S-50N cloud only set

By Cld regime group
H1_tk: 1,3,5
H2_tk: 2,6
H_tn: 7,8,9
Mid: 4, 15A, 15B
L1_tk: 11,13
L2_tk: 10,12
L_tn: 14
S-Clr: 15C
Clear: 0
(Miss: -1)

This code shows L1_tk, L2_tk, L_tn, and S-Clr+Clear

Daeho Jin
2026.01.12
"""

import numpy as np
import sys
import os.path
from datetime import timedelta, date
from netCDF4 import Dataset, num2date
import common_functions as cf

def main():
    ###-- Parameters and defalut values
    ###-------------------------------------
    rg, nelemp, prwt, km= 50, 0, 0, 15
    nelemc= 42
    nelem=nelemc+nelemp

    p_letter= 'P' if prwt>0 else ''
    prset_nm = f'Cld{nelemc}+Pr{nelemp}x{prwt}' if prwt>0 else f'Cld{nelemc}'
    rg_nm= f'{rg}S-{rg}N'

    indir= '/Users/djin1/Documents/CLD_Work/Data_Ref2upload/CPR_update_wIMv07/' #'./Data/'
    mdnm= 'MODIS_t+a_C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)
    infn= indir+f'{mdnm}.nc'
    mdnm2= 'C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)

    ## Parameters for CR_nums
    sat_nm= 'TAmean'
    tgt_lats= [-60,60]  ## Data is available from 65S to 65N
    tgt_dates= [date(2002,9,1),date(2024,8,31)]
    tgt_date_names= '-'.join([dd.strftime('%Y.%m') for dd in tgt_dates])
    ndy= (tgt_dates[1]-tgt_dates[0]).days+1

    tgt_cr_groups= [
        ('L1_tk',(11,13)), ('L2_tk',(10,12)),                    
        ('L_tn',(14,)),    ('S-Clr',(153,0)),
    ]
    
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
    lat_idx= [cf.lat_deg2y(lt,lat0,dlat) for lt in tgt_lats]
    print(lat0,dlat,lat_idx) #; sys.exit()
    lats= lats[lat_idx[0]:lat_idx[1]]
    nlat,nlon= len(lats), len(lons)
    lat_weight= cf.apply_lat_weight(np.ones([nlat,nlon,]),nlat,nlon,lats,geodetic=True)
    xy= np.meshgrid(lons,lats)
    
    cot_b= fid.variables['cloud_optical_thickness_bounds'][:].filled(-999.)
    ctp_b= fid.variables['cloud_top_pressure_bounds'][:].filled(-999.)

    #-- Centroid info
    ctd_cld= fid.variables['Centroid_cloud_part'][:].filled(-999.)
    ctd_cld_sub= fid.variables['SubRegime_Centroid_cloud_part'][:].filled(-999.)
    subk= ctd_cld_sub.shape[0]
    ## Combining centroids with sub-regimes
    ctd_cld= np.concatenate((ctd_cld[:-1,:],ctd_cld_sub),axis=0)

    ctd_cf = np.sum(ctd_cld,axis=(1,2))*100.
    print(ctd_cf)
    km_all= km-1+subk

    #-- Read CR-nums
    ## CR-num data
    crnums= fid.variables[f'CRnum_on_map_{sat_nm}']
    crnums= crnums[itidx:itidx+ndy,lat_idx[0]:lat_idx[1],:]
    print(crnums.shape)
    
    ###--- Prepare for grouping
    #-- Composite by C-group
    group_ctd, group_rfo_map=[],[]
    group_rfo_mean, cg_names=[],[]
    for tcr_nm, tcr in tgt_cr_groups:
        cg_names.append(tcr_nm)
        rfo_sum=0.; rfo_map= np.zeros(crnums.shape[1:3],dtype=float)
        ctd_cld0=[]
        for i,cr1 in enumerate(tcr):
            rfo1= (crnums==cr1).mean(axis=0)*100.  ## Now in percent
            rfo_map+= rfo1
            mrfo1= np.average(rfo1,weights=lat_weight) ## Latitude weights are applied
            cr_idx= cr1-1 if cr1<km else cr1-km*10+km-2
            if cr_idx>=0:
                ctd_cld0.append(ctd_cld[cr_idx]*mrfo1)                
            rfo_sum+= mrfo1
        group_rfo_map.append(rfo_map)
        group_rfo_mean.append(rfo_sum)
        ctd_cld0= np.asarray(ctd_cld0).sum(axis=0)/rfo_sum
        group_ctd.append(ctd_cld0)

    print('Mean RFO:',np.round(group_rfo_mean,3))
    #-- Re-calculate CF
    ctd_cf = np.array([cld.sum()*100. for cld in group_ctd])
    print('Mean CF:',np.round(ctd_cf,3))

    ###-------------------------------------
    ## For plotting a figure
    suptit= "MODIS_C6.1 TA_mean Cloud Group: Mean Histogram (left) and RFO map (right)"
    outdir = './Pics/'
    outfn = outdir+ "Fig01.CR-group_CTD+mean_RFO_map.{}.{}.png".format(mdnm2,tgt_date_names)
    
    pic_data= dict(
        group_ctd= group_ctd, group_rfo_map= group_rfo_map,        
        labels= dict(cot= cot_b, ctp= ctp_b),
        grfo= group_rfo_mean,
        tgt_cr_names= cg_names, xy=xy,
        suptit=suptit, outfn=outfn,
    )
    plot_main(pic_data)
    
    return


    
import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator, FuncFormatter
import cartopy.crs as ccrs

###-- Plotting
def plot_main(pdata):
    ## Parameters and variables
    cent_all= pdata['group_ctd']
    labels= pdata['labels']
    cr_names= pdata['tgt_cr_names']
    rfo_maps= pdata['group_rfo_map']
    grfo= pdata['grfo']
    lons2d,lats2d= pdata['xy']

    ncr= len(cr_names)
    lat_max= rfo_maps[0].shape[0]//2

    ###---
    fig= plt.figure()
    fig.set_size_inches(14.5,16)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=20,y=0.98,va='bottom')
    
    lf=0.05;rf=0.95
    bf=0.05;tf=0.95
    ## Centroid panels
    gapx0,lx0=0.04, 0.13 
    npnx=2
    gapy0=0.043; npny=6.5
    ly0=(tf-bf-gapy0*(npny-1))/float(npny)
    ## RFO map panels
    gapx1=0.018
    lx1=(rf-lf-npnx*(gapx0+lx0)-gapx1*(npnx-1))/float(npnx)

    ###-- Draw Centroids
    cm = plt.get_cmap('jet',256)
    cmnew = cm(np.arange(256))
    cmnew = cmnew[36:,:]
    newcm = cls.LinearSegmentedColormap.from_list("newJET",cmnew)
    newcm.set_under('white')
    props = dict(norm=cls.LogNorm(vmin=0.1,vmax=30),cmap=newcm,alpha=0.8)
    
    ix0,ix1, iy= lf,lf+npnx*(lx0+gapx0),tf
    ix=ix0
    
    for ii,(crnm,cent1) in enumerate(zip(cr_names,cent_all)):
        ax1= fig.add_axes([ix,iy-ly0,lx0,ly0])
        vv1= cent1[::-1,:]*100.
        pic1= cent_show(ax1,vv1,props,labels,ytlabs='l')
        #pic1= cent_show(ax1,cent1)

        subtit= "{} [CF={:.1f}%]".format(crnm,vv1.sum())
        cent_show_common(ax1,subtit)
        #cent_show_common(ax1,crnm,cent1.sum())

        if ix==ix0:
            ax1.set_ylabel('Pressure (hPa)',fontsize=13,labelpad=0)

        ix+=(lx0+gapx0)
        
        if ix>ix0+lx0*2: 
            ix=ix0
            iy-=ly0+gapy0

        if ii>=ncr-npnx:
            ax1.set_xlabel('Optical Thickness',fontsize=13)

    ## Colorbar for centroids
    tt=[0.1,0.3,1,3,10,30]
    tt2=[str(x)+'%' for x in tt]
    hh= ly0/10
    loc1= [ix,iy-hh/2,lx0*2+gapx1,hh] if ix==ix0 else [ix,iy-ly0*0.25,lx0,hh]
    cb1=draw_colorbar(fig,pic1,loc1,tt,tt2,ft=12)
    cb1.ax.set_xlabel('Cloud Fraction',fontsize=14) #,rotation=-90,va='bottom') #,labelpad=0)
    cb1.ax.minorticks_off()


    ###--- Draw RFO maps
    cm = plt.get_cmap('magma_r').resampled(80) #'CMRmap_r' 'YlOrBr' 'Accent' 'afmhot_r'
    cmnew = cm(np.arange(80)) #; print(cmnew[0,:])
    cmnew = np.concatenate((np.array([1,1,1,1]).reshape([1,-1]),cmnew[:-1,:])) #print cmnew[0,:],cmnew[-1,:]

    newcm = cls.LinearSegmentedColormap.from_list("newCMR",cmnew)
    newcm.set_under("white")

    lon_ext= [20,360+20]
    cm= (lon_ext[0]+lon_ext[1])/2
    data_crs= ccrs.PlateCarree()
    props_pc = dict(cmap=newcm,alpha=0.9,transform=data_crs,vmin=0.,vmax=60,shading='nearest')
    
    ix,iy= ix1,tf
    for ii,(crnm,amap) in enumerate(zip(cr_names,rfo_maps)):
        ax=fig.add_axes([ix,iy-ly0,lx1,ly0],projection=ccrs.PlateCarree(central_longitude=cm))
        ax.set_extent(lon_ext+[-61,61],data_crs)

        rfo1= grfo[ii]
        rfo_text= 'RFO={:.1f}%'.format(rfo1) if rfo1>9.995 else 'RFO={:.2f}%'.format(rfo1)
        subtit='{} [{}]'.format(crnm,rfo_text)
        print(subtit)
        ax.set_title(subtit,fontsize=14,stretch='condensed') #x=0.0,ha='left',
        cs=ax.pcolormesh(lons2d,lats2d,amap,**props_pc)

        if ix==ix1:
            ll,lr= True,False
        else:
            ll,lr= False,True
        ax.set_yticks([-60,-30,0,30,60])
        ax.tick_params(axis='y',labelright=lr,labelleft=ll,labelsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(cf.lat_formatter))

        ax.tick_params(direction='in',left=True,right=True,top=True,bottom=True,)
    
        label_idx=[False,False,False,True]
        map_common(ax,label_idx)

        ix+=(lx1+gapx1)
        
        if ix>ix1+lx1*2: 
            ix=ix1
            iy-=ly0+gapy0
        
    ## Colorbar for RFO maps            
    loc1= [ix,iy-hh/2,lx1,hh] if ix==ix1 else [ix,iy-ly0*0.25,lx1,hh]
    tt=range(0,61,10)
    tt2=[str(x)+'%' for x in tt]
    cb1=draw_colorbar(fig,cs,loc1,tt,tt2,ft=12,extend='max')
    cb1.ax.set_xlabel('RFO',fontsize=14) #,rotation=-90,va='bottom')

    ###--------
    ### Save pic
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

def draw_colorbar(fig,pic1,loc,tt,tt2,ft=10,extend='both'):
    
    cb_ax = fig.add_axes(loc)  ##<= (left,bottom,width,height)
    if loc[2]<loc[3]:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='vertical',ticks=tt,extend=extend)
        cb.ax.set_yticklabels(tt2,size=ft,stretch='condensed')
    else:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='horizontal',ticks=tt,extend=extend)
        cb.ax.set_xticklabels(tt2,size=ft,stretch='condensed')
    return cb

def map_common(ax,label_idx=[True,True,False,True]):
    ax.set_extent([0.,360,-61,61],ccrs.PlateCarree())

    ax.coastlines(color='silver',linewidth=1.)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = label_idx[2]
    gl.left_labels = label_idx[0]
    gl.right_labels = label_idx[1]
    gl.bottom_labels = label_idx[3]

    gl.xlocator = MultipleLocator(60) #FixedLocator(range(-120,361,60)) #[0,60,180,240,360]) #np.arange(-180,181,60))
    gl.ylocator = MultipleLocator(30)
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 11, 'color': 'k'}
    gl.ylabel_style = {'size': 11, 'color': 'k'}

    ax.set_aspect('auto')

if __name__=="__main__":
    main()
    sys.exit()
    
    






###-- Parameters and defalut values
###-------------------------------------
#rg  = int(sys.argv[1])
#km    = int(sys.argv[2])       # Number of Clusters
#sid  = int(sys.argv[3])      # id

#rg, km, sid= 15, 14, 31
rg, km, sid= 50, 15, 0
ncl=km; nelem=42

mdnm = 't+a_cld_hist_{}S-{}N'.format(rg,rg)
#dir1 = '/Users/djin1/Documents/CLD_Work/Data_Obs/Cld+Pr_Regime/New_CTD/'
dir1= '/Users/djin1/Documents/CLD_Work/Data_Obs/Cld+Pr_Regime/CTD_wRelaxed_Clr/'
ctdfnm= 'Centroid.MODIS_{}.k{}x{}'.format(mdnm,km,nelem)
ctd_tail= '.wRelaxed_Clr+TAmean.f64dat'
fnm=dir1+ctdfnm+ctd_tail

tgt_cr,subk= km,3
ctdfnm_sub= ctdfnm+'.CR{}_subk{}'.format(tgt_cr,subk)
fnm_sub=dir1+ctdfnm_sub+ctd_tail #'.f64dat'


###
###---- Read Centroid
obsctd = bin_file_read2mtx(fnm,dtp=np.float64)
obsctd = obsctd.reshape([ncl+1,nelem])  # Includes new Clr

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
tgt_cr_groups= [
        #('H1_tk',(1,3,5)), ('H2_tk',(2,6)), 
        #('H1_tn',(7,9)),  ('H2_tn',(8,)),
        #('H_tk',(1,2,3,5,6,)),
        #('H_tn',(7,8,9)),
        #('Mid',(4,151,152)),
        ('L1_tk',(11,13)),
        ('L2_tk',(10,12)),                    
        ('L_tn',(14,)),
        ('S-Clr',(153,)),
        #('Clr',(0,)),        
]
tgt_crs= [item[1] for item in tgt_cr_groups]
cr_name= [item[0] for item in tgt_cr_groups]
ncr= len(tgt_cr_groups)
         
cent1=[]
for l,crs in enumerate(tgt_crs):
    crnm= cr_name[l]
    cent0= []
    for k,cr in enumerate(crs):
        if cr<ncl:
            vv=np.copy(obsctd[cr-1,:].reshape([7,6]))*100.
        else:
            vv=np.copy(obsctd2[cr-km*10-1,:].reshape([7,6]))*100.
        vv=vv[::-1,:]
        cent0.append(vv)
    cent1.append([crnm,np.array(cent0)])
        
###--- RFO
tgt_dates= (date(2002,9,1),date(2024,8,31))
tgt_date_names= [tgt_dates[0].year+1,tgt_dates[1].year]
tgt_latlon, tgt_rg_name = [-60,60,-180,180], '60S-60N'
nyr= tgt_dates[1].year - tgt_dates[0].year
nlat,nlon=120,360
lats= np.arange(nlat)-nlat/2+0.5
lat_weight= cf.apply_lat_weight(np.ones([nlat,nlon,]),nlat,nlon,lats,geodetic=True).squeeze()
lons= np.arange(nlon)-nlon/2+0.5
lons2d,lats2d= np.meshgrid(lons,lats)
        
satnm= 'TAmean'
rg,nelemp,prwt,km= 50,0,0,15 #15,6,1,16 #7,22 #
rg_set= dict(rg=rg,nelemp=nelemp,prwt=prwt,km=km)
tgt_cr,subk= km,3
crnum= cf.read_cpr_map(rg_set,satnm,tgt_dates,tgt_latlon).reshape([-1,nlat,nlon])
crnum_sub= cf.read_cpr_map(rg_set,satnm,tgt_dates,tgt_latlon,sub=True,tgt_cr=tgt_cr,subk=subk).reshape([-1,nlat,nlon])

## Read Total CF
if True:
    cscf_crt=5.
    cfmap= cf.get_Total_CF_daily(tgt_dates,tgt_latlon,sat_nm=satnm)
    print(cfmap.shape,cfmap.min(), cfmap.max())
    non_cs= cfmap>= cscf_crt/100. #
    cs_idx= np.logical_and(cfmap>-0.00001,cfmap< cscf_crt/100.)
    cfmap=0
    print(non_cs.sum(), cs_idx.sum())
    
#tgt_crs= [[1,3,5],[7,9],[2,6],[8],[4,151,152],[10,11,12,13],[14,],[153,],[0,-1]]
#cr_name=['H1_tk','H1_tn','H2_tk','H2_tn','Mid','L_tk','L_tn','S-Clr','CS+Miss']
### Read LO Mask
indir= '/Users/djin1/Documents/CLD_Work/Data_Obs/'
infn= indir+'PctWater.dat'
lat_idx= [90+tgt_latlon[0],90+tgt_latlon[1]]
lomask0= cf.bin_file_read2mtx(infn).reshape([180,360])[lat_idx[0]:lat_idx[1],:]
print(lomask0.min(), lomask0.max())    
lomask0= lomask0 >= 90  ## Ocean only
    
map1,grfo=[],[]
orfo=[]
for l,crs in enumerate(tgt_crs):
    crnm= cr_name[l]
    map0,grfo0,orfo0=[],[],[]
    for k,cr in enumerate(crs):
        if cr<km:
            idx1= crnum==cr            
        else:
            cr1= cr%(tgt_cr*10)
            idx1= crnum_sub==cr1
        ## Exclude new CS
        idx1= np.logical_and(idx1,non_cs)
        
        map0.append(idx1.mean(axis=0)*100)
        grfo0.append(np.average(map0[-1],weights=lat_weight))
        orfo0.append(np.average(map0[-1][lomask0],weights=lat_weight[lomask0]))
    map1.append([crnm,np.array(map0).sum(axis=0)])
    grfo.append(grfo0)
    orfo.append(orfo0)        


'''
## Clear-sky
map0=cs_idx.mean(axis=0)*100
map1.append(['Clr (CF<5%)',map0])
grfo.append([np.average(map0,weights=lat_weight),])
orfo.append([np.average(map0[lomask0],weights=lat_weight[lomask0]),])
'''

##-- Weighted sum of centroid
#obscf=[]
for i,(_,cent0) in enumerate(cent1):
    cent1[i][1]= np.average(cent0,weights=grfo[i],axis=0)
    #obscf.append(cent1[i].sum(axis=(1,2)))
#print(np.array(cent1).sum(axis=(1,2))); sys.exit()

        
###-------------------------------------

###-- Plotting basics
#fig, axs = plt.subplots(5,3)  ## (ny,nx)

###--- Save
#outdir = "/discover/nobackup/djin1/Clustering/TP_DCov/Pics_MODIS_TRCR_test/"
outdir = '../../Writing_LCC_LCAI/Pics/'
fnout = "v1_Fig00.4low_C-group_ctd+rfo.{}.wNewCS.png".format(ctdfnm)
### Show or Save

plt.savefig(outdir+fnout,bbox_inches='tight',dpi=150)
#plt.savefig(outdir+fnout,dpi=160)

print(outdir+fnout)

