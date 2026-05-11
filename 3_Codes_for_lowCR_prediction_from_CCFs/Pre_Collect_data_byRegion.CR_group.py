"""
Collect input data for ML project: LcRFO vs. Cloud-controlling factors

RFO: L1_tk, L2_tk, L_tn, and S-Clr+Clr

---

Target resolution: Monthly, and 4-deg
Sampling stratege: Quarter-sliding    
In horizontal: 4x4 box slide by 1-deg
In temporal: for 91-day (=1 season), 28-day window moves by 7-day

For a target region of 12x12-deg, it is expected to get
81 in horizontal, 10 in temporal, and 22 years= 17820

By Daeho Jin
2026.02.06
---

"""

import numpy as np
import sys
import os #.path
from datetime import timedelta, date
import math
from netCDF4 import Dataset, num2date
import common_functions as cf

def main(sn_idx,tgt_boxes):
    ## Parameters
    sn_names= ['SON','DJF','MAM','JJA']
    t_prd, t_wnd= 91,28
    h_wnd= 4

    ## Regime Info
    sat_nm= 'TAmean'
    rg,nelemp,prwt,km= 50,0,0,15 
    rg_set= dict(rg=rg,nelemp=nelemp,prwt=prwt,km=km)
    tgt_cr,subk= km,3
    nelemc,nelem= 42,42+nelemp
    p_letter= 'P' if prwt>0 else ''
    prset_nm = f'Cld{nelemc}+Pr{nelemp}x{prwt}' if prwt>0 else f'Cld{nelemc}'
    rg_nm= f'{rg}S-{rg}N'
    mdnm = 'MODIS_t+a_C{}R_set.{}_{}'.format(p_letter,rg_nm,prset_nm)    

    tgt_cr_groups= [
        ('L1_tk',(11,13)),
        ('L2_tk',(10,12)),
        ('L_tn',(14,)),
        ('S-Clr',(153,0)),  # Include Clear [regime number=0]
    ]
    tgt_crs= [item[1] for item in tgt_cr_groups]
    cr_name= [item[0] for item in tgt_cr_groups]
    ncr= len(tgt_cr_groups)

    tgt_dates= (date(2002,9,1),date(2024,9,30))
    sn_imon= sn_idx*3+9    
    if sn_imon>12: sn_imon-=12

    ### Open netCDF file for daily RFO data
    indir= './Data/'
    infn= indir+mdnm+'.nc'

    fid= Dataset(infn,'r')

    ##-- Read dimension info
    times= fid.variables['time']
    time_units = times.units
    times= num2date(times[:], units = times.units, calendar=times.calendar,
                      only_use_cftime_datetimes=True,)
    data_t_range= [date(t1.year, t1.month, t1.day) for t1 in [times[0],times[-1]]]    
    data_yy0,data_yy1= data_t_range[0].year, data_t_range[1].year
    data_mm0,data_mm1= data_t_range[0].month, data_t_range[1].month
    max_nyr= data_yy1-data_yy0
    idy,ndy= (tgt_dates[0]-data_t_range[0]).days, (tgt_dates[1]-tgt_dates[0]).days+1    
    print(data_t_range, tgt_dates, idy, ndy)
    
    lats= fid.variables['lat'][:]
    lons= fid.variables['lon'][:]    
    resol=np.rint(lons[1]-lons[0]).astype(int)  
    max_lat= (lats[-1]+resol/2).astype(int)
    tgt_latlon1, tgt_rg_name1= [-max_lat,max_lat,-180,180], '{a}S-{a}N'.format(a=max_lat)
    
    nlat,nlon= len(lats), len(lons) 
    latinfo, loninfo = (lats[0],resol,nlat), (lons[0],resol,nlon)
    latlon_info= dict(latinfo=latinfo, loninfo=loninfo)

    ### Read regime RFO
    crmap= fid.variables['CRnum_on_map_'+sat_nm][idy:idy+ndy,:]
    print(crmap.shape,np.unique(crmap[:10]))

    ### Collect and save data by region
    outdir0= './Input4ML_LcRFO/'
    ## Loop for each target area
    for (tb_name,tb_latlon) in tgt_boxes:
        mdnm= '{}_{}'.format(sn_names[sn_idx],tb_name)
        outdir= outdir0+f'{mdnm}/'
        if not os.path.isdir(outdir):  # Check if the directory already exists
            os.mkdir(outdir)  # Create a directory
            print(outdir,"is created.")
        
        ## Location info
        lat_idx, lon_ids= cf.get_tgt_latlon_idx(latlon_info, tb_latlon[:2], tb_latlon[2:])
        print(mdnm,tb_latlon) 
        print(lat_idx,lats[lat_idx])
        print(lon_ids,lons[lon_ids])
                

        if True:                        
            ## Read by year
            by_season=[]
            mon_indicator= False
            day_counter=0
            yr_counter=0
            for yy in range(data_yy0,data_yy1+1,1):
                im=data_mm0 if yy==data_yy0 else 1
                em=data_mm1 if yy==data_yy1 else 12
                for mm in range(im,em+1,1):
                    if mm==sn_imon and yr_counter<max_nyr:
                        it= (date(yy,mm,1)-tgt_dates[0]).days
                        et= it+t_prd
                        tmp_crmap= crmap[it:et,lat_idx[0]:lat_idx[1],:]
                        tmp_crmap= tmp_crmap[:,:,lon_ids]
                        miss= tmp_crmap==-1 

                        by_tcr=[]
                        for tgt_cr in tgt_crs:
                            idx_all=False
                            for tcr in tgt_cr:
                                idx= tmp_crmap==tcr
                                idx_all= np.logical_or(idx_all,idx)
                            by_tcr.append(np.ma.masked_array(idx_all,mask=miss,dtype=np.float32))
                    
                        by_season.append(np.ma.asarray(by_tcr))
                        yr_counter+=1

            by_season= np.ma.asarray(by_season).swapaxes(0,1);
            print(type(by_season),by_season.shape,by_season.mask.sum()) # [ncr,nyr,t_prd,nlat1,nlon1]

            all_samples=[]
            for k,data1 in enumerate(by_season):
                sampled_data= quarter_slide_sampling(data1,t_wnd,h_wnd,anomaly=False)
                all_samples.append(sampled_data)
                print(sampled_data.shape)
                print(cr_name[k],sampled_data.min(),np.percentile(sampled_data,[1,5,50,95,99]),sampled_data.max())

                ## Save data
                out_dim= sampled_data.shape
                dim_txt= 'x'.join(str(v) for v in out_dim)
                outfn1= outdir+'{}.{}_sampled.{}.f32dat'.format(cr_name[k],mdnm,dim_txt)
                with open(outfn1,'wb') as fout:
                    sampled_data.astype(np.float32).tofile(fout)
                print(outfn1)
            
    return
    

def quarter_slide_sampling(arr,t_wnd,h_wnd,anomaly=False):
    ## It is assumed that arr is of 4-D [nyr,t_prd,nlat1,nlon1]
    nyr,t_prd,nlat1,nlon1= arr.shape
    if arr.min()<-99.:
        print('Check data')
        sys.exit()
        
    out_data=[]
    for iy in range(nlat1-h_wnd+1):
        for ix in range(nlon1-h_wnd+1):            
            for it in range(0,t_prd-t_wnd+1,t_wnd//4):
                tmp_arr= arr[:,it:it+t_wnd,iy:iy+h_wnd,ix:ix+h_wnd].mean(axis=(1,2,3))
                
                if anomaly:
                    tmp_arr= tmp_arr-tmp_arr.mean()
                out_data.append(tmp_arr)
    
    out_data= np.asarray(out_data).T
    return out_data
    

def get_tgt_boxes_12d(sn_idx):
    tgt_boxes_JJA= [
        ('Peruvian',(-22,-10,-100,-88)), #-92,-84)), #-90,-80)),
        ('Namibian',(-22,-10,-10,2)), #-4,4)),
        ('Californian',(18,30,-146,-134)), #-136,-128)),
    ]

    tgt_boxes_DJF= [
        ('Peruvian',(-30,-18,-90,-78)), #-92,-84)), #-90,-80)),
        ('Namibian',(-26,-14,-6,6)), #-4,4)),        
        ('Australian',(-36,-24,94,106)), #96,104)),
    ]
    if sn_idx==1:
        return tgt_boxes_DJF
    elif sn_idx==3:
        return tgt_boxes_JJA
    else:
        return []

def get_tgt_boxes_12d_add(sn_idx):
    tgt_boxes_JJA= [
        ('SAO',(-56,-44,-40,-28)),
    ]

    tgt_boxes_DJF= [
        ('NWPO',(40,52,152,164)),
    ]
    tgt_boxes_MAM= [
        ('Canarian',(8,20,-36,-24)), #-136,-128)),
        ('SPCZ',(-16,-4,-160,-148)), 
    ]
    tgt_boxes_SON= [
        ('TIO',(-8,4,60,72)),
        ('STropEPO',(-20,-8,-112,-100)), 
    ]
    if sn_idx==1:
        return tgt_boxes_DJF
    elif sn_idx==3:
        return tgt_boxes_JJA
    elif sn_idx==0:
        return tgt_boxes_SON
    elif sn_idx==2:
        return tgt_boxes_MAM
    else:
        return [] 


if __name__=="__main__":
    #sn_names= ['SON','DJF','MAM','JJA']
    
    for sn_idx in [1,3]:
        tgt_boxes= get_tgt_boxes_12d(sn_idx)
        main(sn_idx,tgt_boxes)
    '''
    for sn_idx in [0,1,2,3]:
        tgt_boxes= get_tgt_boxes_12d_add(sn_idx)
        main(sn_idx,tgt_boxes)
    '''

