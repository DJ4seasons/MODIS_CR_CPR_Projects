import sys
import os.path
import numpy as np
from datetime import date,datetime, timedelta
from netCDF4 import Dataset

def yield_date_range(start_date, end_date, tdelta=1):
    ### Including end date
    for n in range(0,int((end_date - start_date).days)+1,tdelta):
        yield start_date + timedelta(n)

def yield_monthly_date_range(start_date,end_date,mdelta=1,ref_dy=15):
    nmon= get_tot_months(start_date,end_date)
    iyr,imo= start_date.year, start_date.month
    xt=[]
    for mm in range(0,nmon,mdelta):
        xt.append(date(iyr,imo,ref_dy))
        imo+= mdelta
        if imo>12:
            while(imo>12):
                iyr+=1
                imo-=12

    return xt

def get_month_days(tgt_dates):
    days=[]
    y0,m0= tgt_dates[0].year, tgt_dates[0].month
    y1,m1= tgt_dates[1].year, tgt_dates[1].month
    for yy in range(y0,y1+1,1):
        im=m0 if yy==y0 else 1
        em=m1 if yy==y1 else 12
        #print(yy,im,em)
        for mm in range(im,em+1,1):
            it= (date(yy,mm,1)-tgt_dates[0]).days
            yy1,mm1=yy,mm+1
            if mm1>12: yy1+=1; mm1-=12
            et= (date(yy1,mm1,1)-tgt_dates[0]).days
            ndy= et-it
            days.append(ndy)
    return days

def apply_lat_weight(arr,nlat,nlon,lats,geodetic=True):
    '''
    Build 2D array containing latitude weights
    Make sure that arr should be 2-Dimensional with [nlat,nlon]
    '''
    if geodetic:
        if lats[1]-lats[0] != 1.0:
            print('Geodetic weight is available for 1-deg resolution', lats[1]-lats[0])
            sys.exit()
            
        infn= './Data/geodetic_weight_1deg.txt'
        lat_ref, wt= [],[]
        with open(infn,'r') as f:
            for i,line in enumerate(f):
                if i>0:  ## Skip head
                    ww=line.strip().split()
                    lat_ref1,wt1= [float(val) for val in ww]
                    lat_ref.append(lat_ref1)
                    wt.append(wt1)
        ilat,elat=-999,-999
        for i,lat1 in enumerate(lat_ref):
            if lat1==lats[0]:
                ilat=i
            elif lat1==lats[-1]:
                elat=i
                break
        if ilat<0 or elat<0:
            print('Lat_ref is not matched to given lats',lats[0],lats[-1])
            sys.exit()
        else:
            lat_weight= np.asarray(wt)[ilat:elat+1]
            #print(lat_weight.dtype,lat_weight[0])
    else:
        lat_weight= np.cos(np.deg2rad(lats))
        #lat_weight= lat_weight/lat_weight.sum()
    return (arr*lat_weight[:,None]) 

from math import ceil
def lon_deg2x(lon,lon0,dlon):
    x=ceil((lon-lon0)/dlon)
    nx= int(360/dlon)
    if x<0:
        while(x<0):
            x+= nx
    if x>=nx: x=x%nx
    return x
lat_deg2y = lambda lat,lat0,dlat: ceil((lat-lat0)/dlat)

def get_tgt_latlon_idx(latlons, tgt_lats, tgt_lons):
    lon0,dlon,nlon= latlons['loninfo']
    lat0,dlat,nlat= latlons['latinfo']
    ##-- Regional index
    if isinstance(tgt_lons,(list,tuple,np.ndarray)):
        lon_idx= [lon_deg2x(ll,lon0,dlon) for ll in tgt_lons]
        if lon_idx[0]==lon_idx[1]:
            if tgt_lons[0]!=tgt_lons[1]:
                lon_ids= np.arange(nlon)+lon_idx[0]
                lon_ids[lon_ids>=nlon] -= nlon
            else:
                lon_ids= np.array([lon_idx,])
        elif lon_idx[1]<lon_idx[0]:
            lon_ids= np.arange(lon_idx[0]-nlon,lon_idx[1],1)
        else:
            if lon_idx[1]-lon_idx[1]<nlon and tgt_lons[1]-tgt_lons[0]>360:
                lon_idx[1]+=int(360/dlon)   
            lon_ids= np.arange(lon_idx[0], lon_idx[1], 1)
    else:
        lon_ids= np.arange(nlon,dtype=int)
    lat_idx= [lat_deg2y(ll,lat0,dlat) for ll in tgt_lats]
    return lat_idx, lon_ids

def lon_formatter(x,pos):
    if x<=-180: x+=360
    elif x>=360: x-=360

    if x>0 and x<180:
        return "{:.0f}\u00B0E".format(x)
    elif x>180 and x<360:
        return "{:.0f}\u00B0W".format(360-x)
    elif x>-180 and x<0:
        return "{:.0f}\u00B0W".format(-x)
    else:
        return "{:.0f}\u00B0".format(x)

def lat_formatter(x,pos):
    if x>0:
        return "{:.0f}\u00B0N".format(x)
    elif x<0:
        return "{:.0f}\u00B0S".format(-x)
    else:
        return "{:.0f}\u00B0".format(x)
    
def bin_file_read2mtx(fname,dtype=np.float32):
    """ Open a binary file, and read data 
        fname : file name
        dtp   : data type; np.float32 or np.float64, etc. """

    if not os.path.isfile(fname):
        print("File does not exist:"+fname)
        sys.exit()

    with open(fname,'rb') as fd:
        bin_mat = np.fromfile(file=fd,dtype=dtype)

    return bin_mat

def get_tot_months(date0,date1):
    iyr,imon= date0.year, date0.month
    eyr,emon= date1.year, date1.month
    tot_mon= (eyr-iyr-1)*12+ (13-imon) + emon
    return tot_mon

def get_NRB_TOA_monthly(vn,tgt_dates,tgt_latlon,in_dir='./Data/'):
    date_range=[date(2002,7,1),date(2024,12,31)]
    date_names=[d.strftime('%Y%m') for d in date_range]

    nmon= get_tot_months(*tgt_dates)
    imon= get_tot_months(date_range[0],tgt_dates[0])-1
    print(imon,nmon)

    fn= in_dir+'CERES_EBAF-TOA_Ed4.2.1_Subset_{}-{}.nc'.format(*date_names)
    fid=Dataset(fn,'r')

    lats= fid.variables['lat']
    lons= fid.variables['lon']
    latinfo, loninfo = (lats[0],lats[1]-lats[0],len(lats)), (lons[0],lons[1]-lons[0],len(lons))
    latlon_info= dict(latinfo=latinfo, loninfo=loninfo)
    nlat,nlon= latinfo[-1],loninfo[-1]
    lat_idx, lon_ids= get_tgt_latlon_idx(latlon_info, tgt_latlon[:2], tgt_latlon[2:])
    print(lat_idx, lon_ids[[0,180,-1]]) #; sys.exit()
    #domain_size= (lat_idx[1]-lat_idx[0])*len(lon_ids)

    vdata = fid.variables[vn][imon:imon+nmon,lat_idx[0]:lat_idx[1],lon_ids]
    print(vn,type(vdata),vdata.shape,vdata.min(),vdata.max(),vdata.mean(),vdata.mask.sum())
    return vdata 

def draw_colorbar(fig,pic1,loc,tt,tt2,ft=10,extend='both'):
    
    cb_ax = fig.add_axes(loc)  ##<= (left,bottom,width,height)
    if loc[2]<loc[3]:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='vertical',ticks=tt,extend=extend)
        cb.ax.set_yticklabels(tt2,size=ft,stretch='condensed')
    else:
        cb = fig.colorbar(pic1,cax=cb_ax,orientation='horizontal',ticks=tt,extend=extend)
        cb.ax.set_xticklabels(tt2,size=ft,stretch='condensed')
    return cb


