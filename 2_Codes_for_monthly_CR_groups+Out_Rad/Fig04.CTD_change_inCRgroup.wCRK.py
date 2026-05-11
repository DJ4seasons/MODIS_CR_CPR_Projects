"""
Compare mean CTD between 2003-2012 vs. 2015-2024

3-lat_zone for select CR groups
Apply lat-mean CRK to final CTD difference

By Daeho Jin
2025.12.02
---

"""

import numpy as np
import sys
import os.path
from datetime import timedelta, date
from netCDF4 import Dataset, num2date
import common_functions as cf

def main(tgt_crg):
    ###--- Parameters
    '''
    mdnm0= 'EBAF_4.2.1' #'FBCT_4.1'
    rad_names= ['SW','LW']
    rad_idx= 0
    rad_name= rad_names[rad_idx]+'_outgoing'
    rad_name_tit= 'O{}R'.format(rad_names[rad_idx][0])
    '''

    sat_nm= 'TAmean'
    rg,nelemp,prwt,km= 50,0,0,15 
    rg_set= dict(rg=rg,nelemp=nelemp,prwt=prwt,km=km)
    tgt_cr,subk= km,3
    nelemc,nelem= 42,42+nelemp
    p_letter= 'P' if prwt>0 else ''
    prset_nm = f'Cld{nelemc}+Pr{nelemp}x{prwt}' if prwt>0 else f'Cld{nelemc}'
    rg_nm= f'{rg}S-{rg}N'
    mdnm = 'MODIS_t+a_C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)

    cr_names= ['H1_tk','H2_tk','H_tn','Mid','L1_tk','L2_tk','L_tn','S-Clr','Clear']
    ncr= len(cr_names)

    tgt_dates= (date(2002,9,1),date(2024,8,31)) 
    tgt_date_names= '-'.join([dd.strftime('%Y.%m') for dd in tgt_dates])
    
    ### Open netCDF file
    indir= './Data/'
    infn= indir+'Monthly_Composite_Histogram+RFO_map.by{}CRgroups.nc'.format(ncr)
    fid= Dataset(infn,'r')

    ##-- Read dimension info
    times= fid.variables['time']
    time_units = times.units
    times= num2date(times[:], units = times.units, calendar=times.calendar,
                      only_use_cftime_datetimes=True,)
    date_range= [date(t1.year, t1.month, t1.day) for t1 in [times[0],times[-1]]]

    imon,nmon= cf.get_tot_months(date_range[0],tgt_dates[0])-1, cf.get_tot_months(*tgt_dates)
    nmon_yr,nyr= 12, nmon//12
    mon_days= np.asarray(cf.get_month_days(tgt_dates))
    mon_days_clim_norm= np.asarray(mon_days).reshape(-1,12)[:4,:].mean(axis=0)
    mon_days_clim_norm= mon_days_clim_norm/mon_days_clim_norm.sum()
    #xt= cf.yield_monthly_date_range(*tgt_dates,mdelta=1)
    
    lats= fid.variables['lat'][:]
    lons= fid.variables['lon'][:]
    resol=np.round(lons[1]-lons[0],0).astype(int)  ## Now in 4-deg resolution
    max_lat= lats[-1]+resol/2
    tgt_latlon1, tgt_rg_name1= [-max_lat,max_lat,-180,180], '{a}S-{a}N'.format(a=max_lat)
    
    ##-- Geodetic weight is available at 1-deg resolution
    nlat,nlon= lats.shape[0], lons.shape[0]
    lats_1deg= np.arange(nlat*resol)+lats[0]-resol/2+0.5
    nlat1,nlon1= nlat*resol, nlon*resol
    lat_weight= cf.apply_lat_weight(np.ones([nlat1,nlon1,]),nlat1,nlon1,lats_1deg,geodetic=True)
    ltw= lat_weight.reshape([nlat,resol,nlon,resol]).sum(axis=1)[:,:,0]
    print(ltw.shape,ltw[0,0],ltw[nlat//2,0])
    xy= np.meshgrid(lons,lats)
    
    ##-- Two periods    
    dr0,dr1= (date(2002,9,1),date(2012,8,31)),(date(2014,9,1),date(2024,8,31))
    imon1,nmon1= cf.get_tot_months(tgt_dates[0],dr0[0])-1,cf.get_tot_months(*dr0)
    iyr1,nyr1= imon1//nmon_yr, nmon1//nmon_yr
    pd1_date_name= 'PD1: {}-{}'.format(*[dd.strftime('%Y.%m') for dd in dr0])
    imon2,nmon2= cf.get_tot_months(tgt_dates[0],dr1[0])-1,cf.get_tot_months(*dr1)
    iyr2,nyr2= imon2//nmon_yr, nmon2//nmon_yr
    pd2_date_name= 'PD2: {}-{}'.format(*[dd.strftime('%Y.%m') for dd in dr1])
     
    ##-- Read data 
    mrfo= fid.variables['mRFO'][:]
    jh= fid.variables['Monthly_CF_Joint_Hist']
    ncr,nmon,nlat,nlon,nctp,ntau= jh.shape

    ncr2= len(tgt_crg)
    mrfo= mrfo[tgt_crg,:]
    rfo1= mrfo[:,imon1:imon1+nmon1,:,:]
    rfo2= mrfo[:,imon2:imon2+nmon2,:,:]
    clim_rfo= mrfo.reshape(ncr2,nyr,nmon_yr,nlat,nlon).mean(axis=1)
    cr_group_names= [cr_names[v] for v in tgt_crg]
    mrfo=None
    
    ### Read Cloud-radiative kernel
    nbin= nctp*ntau #42
    indir= './Data/'
    nlat_crk, nlon_crk= 150,nlon1
    dlat= (nlat_crk-nlat1)//2
    indim= [nmon_yr,nlat_crk,nlon_crk,nbin]
    dim_txt= 'x'.join([str(v) for v in indim])
    fn_header= indir+'CERES_FBCT-MON_Terra-Aqua-MODIS_Ed4.1.'
    crk_all=[]
    rad_names= ['sw','lw',]
    for ridx,rn in enumerate(rad_names):
        infn= fn_header+rn+'_CRK.{}.f32dat'.format(dim_txt)
        crk= cf.bin_file_read2mtx(infn).reshape(indim)[:,dlat:-dlat,:,:]
        crk= np.ma.masked_less(crk,-999.)
        ## Degrading
        crk= crk.reshape([nmon_yr,nlat,resol,nlon,resol,nbin]).mean(axis=(2,4))
        print(crk.shape, crk.min(), crk.max()) # [nmon_yr,nlat,nlon,nbin]
        print(np.round(crk.mean(axis=(0,1,2)),3)) 
        crk_all.append(crk)

            
    '''
    ### Read EBAF radiaiton for comparison with CRK
    var_names= ['toa_sw_all_mon','toa_lw_all_mon',]
    in_dir= './Data/'
    rad_sw= cf.get_NRB_TOA_monthly(var_names[0],tgt_dates,tgt_latlon1,in_dir=in_dir)
    rad_lw= cf.get_NRB_TOA_monthly(var_names[1],tgt_dates,tgt_latlon1,in_dir=in_dir)
    ##-- Degrading
    if resol>1:
        rad_sw= rad_sw.reshape([nmon,nlat,resol,nlon,resol]).mean(axis=(2,4))
        rad_lw= rad_lw.reshape([nmon,nlat,resol,nlon,resol]).mean(axis=(2,4))
    rad_sw1= rad_sw[imon1:imon1+nmon1,:,:]
    rad_sw2= rad_sw[imon2:imon2+nmon2,:,:]
    rad_lw1= rad_lw[imon1:imon1+nmon1,:,:]
    rad_lw2= rad_lw[imon2:imon2+nmon2,:,:]
    '''
    
    ### 2D Joint-Hist Composite and applying CRK
    jh_data, rad_data=[],[]
    rfo_data=[]
    ebaf_data=[]
    for k,cr_id in enumerate(tgt_crg):
        mon_jh= jh[cr_id].reshape([nmon,nlat,nlon,nbin]) 
        mask1= mon_jh.mask[:,:,:,0]
        
        wt1= rfo1[k].reshape([nmon1,nlat,nlon,1])*ltw[None,:,:,None]*mon_days[imon1:imon1+nmon1,None,None,None]
        mjh1= mon_jh[imon1:imon1+nmon1]
        wt1[mask1[imon1:imon1+nmon1],:]=0.
        
        wt2= rfo2[k].reshape([nmon2,nlat,nlon,1])*ltw[None,:,:,None]*mon_days[imon2:imon2+nmon2,None,None,None]
        mjh2= mon_jh[imon2:imon2+nmon2]
        wt2[mask1[imon2:imon2+nmon2],:]=0.

        crk_wt= clim_rfo[k]*ltw[None,:,:]*mon_days_clim_norm[:,None,None]  #[nmon_yr,nlat,nlon]

        '''
        ## Test CRK results with EBAF (reference)
        emr1s= rad_sw1.reshape([nmon1,nlat,nlon,1])*wt1
        emr1l= rad_lw1.reshape([nmon1,nlat,nlon,1])*wt1
        emr2s= rad_sw2.reshape([nmon2,nlat,nlon,1])*wt2
        emr2l= rad_lw2.reshape([nmon2,nlat,nlon,1])*wt2
        '''
        
        by_lat1,by_lat2=[],[]
        by_lat1f,by_lat2f=[],[]
        by_lat_crk=[]
        lat_idx=[0,8,22,30]
        for lt0,lt1 in zip(lat_idx[:-1],lat_idx[1:]):
            mtmp1= (mjh1[:,lt0:lt1,:].filled(0.)*wt1[:,lt0:lt1,:]).sum(axis=(0,1,2))
            mwt1= np.sum(wt1[:,lt0:lt1,:])
            by_lat1.append(mtmp1/mwt1)
            by_lat1f.append(mwt1/ltw.sum()/mon_days[imon1:imon1+nmon1].sum()*100)

            mtmp2= np.sum(mjh2[:,lt0:lt1,:]*wt2[:,lt0:lt1,:],axis=(0,1,2))
            mwt2= np.sum(wt2[:,lt0:lt1,:])
            by_lat2.append(mtmp2/mwt2)
            by_lat2f.append(mwt2/ltw.sum()/mon_days[imon2:imon2+nmon2].sum()*100)
            
            crk_tmp=[]
            crk_wt1= crk_wt[:,lt0:lt1,:]
            for ridx,rn in enumerate(rad_names):
                crk1= crk_all[ridx][:,lt0:lt1,:,:]  # [nmon_yr,nlat,nlon,nbin]
                crk1_tmp= (crk1*crk_wt1[:,:,:,None]).sum(axis=(0,1,2))
                crk_tmp.append(crk1_tmp/crk_wt1.sum())
            by_lat_crk.append(crk_tmp)

            '''
            ## Print EBAF results for comparison
            e1s= emr1s[:,lt0:lt1,:].sum()/mwt1
            e2s= emr2s[:,lt0:lt1,:].sum()/mwt2
            e1l= emr1l[:,lt0:lt1,:].sum()/mwt1
            e2l= emr2l[:,lt0:lt1,:].sum()/mwt2
            print(cr_group_names[k],lt0,e1s,e2s,e2s-e1s)
            print(e1l,e2l,e2l-e1l)
            '''
        by_lat1= np.asarray(by_lat1)
        by_lat2= np.asarray(by_lat2)
        
        mm= (by_lat1+by_lat2)/2
        df= by_lat2-by_lat1
        print(mm.sum(axis=1), df.sum(axis=1))
        jh_data.append([mm,df])
        
        by_lat_crk= np.asarray(by_lat_crk) #[lat_zone,n_rad,nbin]
        rad_data.append((df[:,None,:]*by_lat_crk).sum(axis=-1)*100)  #[lat_zone,n_rad]
        
        by_lat1f= np.asarray(by_lat1f)
        by_lat2f= np.asarray(by_lat2f)
        mm= (by_lat1f+by_lat2f)/2
        df= by_lat2f-by_lat1f
        print(mm, df)
        rfo_data.append([mm,df])
        

    ### For Figure
    lat_names= ['60\u00B0S\u201028\u00B0S','28\u00B0S\u201028\u00B0N','28\u00B0N\u201060\u00B0N']
    suptit= 'Mean and Difference (Period2-Period1) of Cloud 2D Histogram' #.format(rad_name_tit, tgt_rg_name1)
    outdir= './Pics/'       
    outfn= outdir+'Fig04.mean_JH_diff_byLatZone.Prd1_vs_Prd2.{}.wCRK.png'.format('+'.join(cr_group_names))

    pic_data= dict(jh_data= jh_data, pn_tit= ['',''], #['Mean','Prd2-Prd1'],
                   rad_data= rad_data, rfo_data= rfo_data,
                   lat_names=lat_names, cr_group_names= cr_group_names,
                   suptit=suptit, outfn=outfn,)
    plot_main(pic_data)
                                                         
    return

import matplotlib.colors as cls
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator,FuncFormatter, MultipleLocator
from matplotlib.dates import DateFormatter, YearLocator

def plot_main(pdata):
    jh_data= pdata['jh_data']
    rad_data= pdata['rad_data']
    rfo_data= pdata['rfo_data']
    pn_tit= pdata['pn_tit']
    lat_names= pdata['lat_names']
    cr_group_names= pdata['cr_group_names']
    
    abc= 'abcdefghijklmnopqrstuvwxyzabcdefg'

    ###---
    fig=plt.figure()
    fig.set_size_inches(10.6,8.)    ## (lx,ly)
    plt.suptitle(pdata['suptit'],fontsize=16,y=0.975,va='bottom',stretch='semi-condensed') #,x=0.1,ha='left')
    ncol,nrow=3,3
    lf,rf,bf,tf=0.045,0.95,0.1,0.9
    gapx, npnx=0.05,ncol
    lx=(rf-lf-gapx*(npnx-1))/float(npnx)
    gapy, npny=0.13,nrow
    ly=(tf-bf-gapy*(npny-1))/float(npny)
    
    ix=lf; iy=bf    

    gapx2=0.014
    lx2= (lx-gapx2)/2
    for i,(mm0,df0) in enumerate(jh_data):
        fig.text(ix+lx/2,0.935,cr_group_names[i],color='k',fontsize=12,weight='bold',ha='center',va='bottom')
        rdf0= rad_data[i]
        fmm0,fdf0= rfo_data[i]
        for j,(mm1,df1) in enumerate(zip(mm0,df0)):
            rdf1= rdf0[j]
            fmm1,fdf1= fmm0[j], fdf0[j]
            
            ax1= fig.add_axes([ix,iy,lx2,ly])
            pic1= cent_show(ax1,mm1*100)

            ### add a title.
            subtit= "{} [CF={:.1f}%]".format(pn_tit[0],mm1.sum()*100.) 
            print(subtit)
            ax1.set_title(subtit,fontsize=12,stretch='condensed') #x=0.0,ha='left',

            ix2= ix+lx2+gapx2
            ax2= fig.add_axes([ix2,iy,lx2,ly])
            pic2= cent_show(ax2,df1*100,diff=True)

            ### add a title.
            subtit= "{} [\u0394CF={:.2f}%]".format(pn_tit[1],df1.sum()*100.) 
            print(subtit)
            ax2.set_title(subtit,fontsize=12,stretch='condensed') #x=0.0,ha='left',

            if ix2+lx>rf:
                ax2.tick_params(labelleft=False, labelright=True)
            else:
                ax2.tick_params(labelleft=False, labelright=False)

            if ix==lf: # and iy-gapy<bf:            
                ax1.set_ylabel('Pressure (hPa)',fontsize=10,labelpad=0)
                ax1.set_xlabel('Optical Thickness',fontsize=10)
            
            if i==0:
                fig.text(lf-0.06,iy+ly/2,lat_names[j],color='k',fontsize=12,weight='bold',rotation=90,ha='right',va='center')

            ### Add rad info
            gt= 0.17
            yt0= -0.24 
            rn= ['\u0394OSR=','\u0394OLR=']
            txt_props= dict(c='k',ha='center',va='top',fontsize=10,weight='bold')
            unit= r'$Wm^{-2}$'
            
            for k,rd in enumerate(rdf1):                
                ax2.text(0.5,yt0,rn[k]+' {:.2f} {}'.format(rd,unit),transform=ax2.transAxes,**txt_props)
                yt0-= gt
            ax1.text(0.5,yt0+gt,'RFO= {:.2f}%'.format(fmm1),transform=ax1.transAxes,**txt_props)
            
            iy+= ly+gapy
        
        ix+= lx+gapx
        iy=bf

    tt=[0.1,0.3,1,3,10,30]
    tt2=[str(x)+'%' for x in tt]
    hh= 0.015
    loc1= [0.18,bf-gapy*0.9,0.3,hh]
    cb1= cf.draw_colorbar(fig,pic1,loc1,tt,tt2,ft=9)
    cb1.ax.set_xlabel('Cloud Fraction',fontsize=10) #,rotation=-90,va='bottom') #,labelpad=0)
    cb1.ax.minorticks_off()

    tt= [-1,-0.5,0,0.5,1]
    tt2=[str(x)+'%' for x in tt]
    hh= 0.015
    loc1= [0.52,bf-gapy*0.9,0.3,hh]
    cb1= cf.draw_colorbar(fig,pic2,loc1,tt,tt2,ft=9)
    cb1.ax.set_xlabel('\u0394CF',fontsize=10) #,rotation=-90,va='bottom') #,labelpad=0)
    
    ###---
    print(pdata['outfn'])
    plt.savefig(pdata['outfn'],bbox_inches='tight',dpi=150) #
    #plt.show()
    
    return

def cent_show(ax1,ctd,diff=False):     

    nx = 6 #TAU (Optical Thickness)
    ny = 7 #CTP
    if len(ctd.reshape(-1)) != nx*ny:
        print("Error: centroid data size is bad:",ctd.shape)
        sys.exit()
    elif ctd.ndim!=2:
        ctd= ctd.reshape([ny,nx])[::-1,:]

    if diff:
        newcm= 'BrBG_r'
        props= dict(vmin=-1,vmax=1,cmap=newcm,alpha=0.9)
    else:
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
            if not diff:
                if abs(ctd[j,i])>4.95:
                    ax1.annotate("%.0f" %(ctd[j,i]),xy=(i,j),ha='center',va='center',stretch='semi-condensed',fontsize=9)
            else:
                if abs(ctd[j,i])>0.095:
                    ax1.annotate("%.1f" %(ctd[j,i]),xy=(i,j),ha='center',va='center',stretch='semi-condensed',fontsize=9)

    ax1.axvline(x=1.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axvline(x=3.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axhline(y=1.5,linewidth=0.7,color='k',linestyle=':')
    ax1.axhline(y=3.5,linewidth=0.7,color='k',linestyle=':')

    ### Ticks
    ax1.tick_params(axis='both',which='major',labelsize=9)
    ax1.tick_params(left=True,right=True)    
    
    return pic1


if __name__=="__main__":
    # cr_names= ['H1_tk','H2_tk','H_tn','Mid','L1_tk','L2_tk','L_tn','S-Clr','Clr (CF<5%)']
    tgt_crg= [4,5,6] #[0,1,3] #[2,7,8] #
    main(tgt_crg)
