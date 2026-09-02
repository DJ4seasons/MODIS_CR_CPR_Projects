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

def get_Water_Pct(tgt_latlon=[],hs=1):
    latinfo, loninfo = (-89.5,1,180), (-179.5,1,360)
    latlon_info= dict(latinfo=latinfo, loninfo=loninfo)
    nlat,nlon= latinfo[-1],loninfo[-1]

    indir= './Data/'
    infn= indir+'PctWater.dat'
    wpct= bin_file_read2mtx(infn).reshape([nlat,nlon])
    if len(tgt_latlon)>0 and tgt_latlon!=[-90,90,-180,180]:
        lat_idx, lon_ids= get_tgt_latlon_idx(latlon_info, tgt_latlon[:2], tgt_latlon[2:])
        wpct= wpct[lat_idx[0]:lat_idx[1],lon_ids]

    nlat2,nlon2= wpct.shape
    if hs>1:
        ny,nx= nlat2//hs, nlon2//hs
        wpct= wpct.reshape([ny,hs,nx,hs]).mean(axis=(1,3))
    return wpct

def normalize_x_raw(arr,v_names):
    npt,nv= arr.shape
    arr2= np.copy(arr)
    for k in range(nv):
        vn= v_names[k] #.split()[0].lower()
        if vn=='sp':
            arr2[:,k]= (arr2[:,k]-1000)/25.
        elif vn=='skTadv':
            arr2[:,k]/=5.
        elif vn[:4]=='divg':
            arr2[:,k]/=1.
        elif vn[0]=='q' and vn[-3:]=='adv':
            arr2[:,k]*=(1000./4)
        elif vn[:4]=='wspd':
            arr2[:,k]/=10.
        elif vn[0]=='q':
            arr2[:,k]*= (1000/15)
        elif vn=='r2m':
            continue
        elif vn[0]=='r' and vn[-1]=='0':
            arr2[:,k]/= 100.
        elif 't' in vn:
            arr2[:,k]= (arr2[:,k]-273.15)/30.
        else:
            sys.exit(f'No matching variable: {vn}')            
    return arr2

def normalize_x_lcai(arr,v_names):
    npt,nv= arr.shape
    arr2= np.copy(arr)
    for k in range(nv):
        vn= v_names[k].split()[0].lower()
        if vn=='eis' or vn=='ectei' or vn=='m':
            arr2[:,k]/= 20.
        elif vn=='lts':
            arr2[:,k]= (arr2[:,k]-10)/20.
        elif vn=='elf':
            arr2[:,k]/= 100.
        elif vn=='t_adv':
            arr2[:,k]/=5.
        elif vn=='ws10m':
            arr2[:,k]/=10.
        elif vn=='w700':
            arr2[:,k]/=50.
        elif vn[:2]=='rh':
            arr2[:,k]/= 50.
        elif 't' in vn:
            arr2[:,k]= (arr2[:,k]-273.15)/30.
        else:
            sys.exit(f'No matching variable: {vn}')      
    return arr2

def de_standardize(ref_std,ref_mm,target,flatten=True):
        '''
        ref.shape= [npt,ncr]
        target.shape= [ncr,nyr2,npt]
        '''
        #print('de',ref.shape, target.shape)
        target= target*ref_std.T[:,None,:]+ref_mm.T[:,None,:]
        #ncr,nyr,npt= target.shape
        return target #.reshape([ncr,-1])
    
def get_anomaly(arr,train_yr_idx=[],flatten=False,standardization=False):
    if len(train_yr_idx)>0:
        mm= arr[train_yr_idx,:].mean(axis=0)
        if standardization:
            std= arr[train_yr_idx,:].std(axis=0,ddof=1)
    else:
        mm= arr.mean(axis=0)
        if standardization:
            std= arr.std(axis=0,ddof=1)
    ano= arr-mm[None,:]
    if standardization:
        std_non_zero= std>0.
        ano[:,std_non_zero]/=std[std_non_zero]

    if flatten:
        if ano.ndim==2:
            ano= ano.reshape(-1)
        elif ano.ndim==3:
            nyr,npt,nv= ano.shape
            ano= ano.reshape([nyr*npt,nv])
    return ano

def collect_data2calc_LCidx_fromSamples(mdnm,rg_name,var_names=[],in_dim=[22,490],
                indir= './Input4ML_LcRFO/'):
    indata=[]
    if mdnm.upper()=='MERRA2':
        return None
    elif mdnm.upper()=='ERA5':        
        dim_txt= 'x'.join([str(v) for v in in_dim])
        fn_t= '.{}_sampled.{}.f32dat'.format(rg_name,dim_txt)
                    
        if len(var_names)==0:
            var_names= ['t700','t2m','sp','q700','q2m','t800','skt']
            
        for vn in var_names:
            infn= indir+f'{rg_name}/{vn}'+fn_t
            tmp= bin_file_read2mtx(infn).reshape(in_dim)
                
            if vn=='sp':
                tmp/=100.  ## Change to hPa
            elif vn[-3:]=='adv' or vn[:4]=='divg':
                tmp*=86400  ## Change to K/day
            elif vn[0]=='w' and vn[1]!='s':
                tmp*=864  ## Change to hPa/day

            if tmp.min()<-99.:
                print('***----')
                print('Warning: too large negative values found:',vn,tmp.min())
                print('***----')
            indata.append(tmp)
    else:
        print("mdnm is not supported",mdnm )
        sys.exit()
    return indata

def calc_LCidx_component(indata,undef= -9999.9):
    '''
    indata= ['T700','T2M','PS','QV700','QV2M','T800']
        or  ['t700','t2m','sp','q700','q2m','t800']
    '''
    params= dict(
        gravity= 9.80665,
        R_dry= 287.05,
        R_vapor= 461.51,
        Cp_dry= 1004.,
        undef= undef
    )
    
    if not np.isnan(undef):
        for k,arr in enumerate(indata):
            arr[arr<undef+1]= np.nan
            indata[k]=arr

    ### Calc LCC
    noms_idx= indata[2]>900  ## Only for PS>900hPa

    ### LTS
    theta_sfc= get_potential_temp(indata[1][noms_idx],indata[2][noms_idx],params)
    theta_800= get_potential_temp(indata[5][noms_idx],800.,params)
    theta_700= get_potential_temp(indata[0][noms_idx],700.,params)

    if_calc_ELF=True
    EIS_supple, ELF_compo=calc_EIS_supplement(
            indata[0][noms_idx],
            indata[1][noms_idx],
            indata[2][noms_idx],
            None,
            indata[3][noms_idx],
            indata[4][noms_idx],
            params, ELF_return=if_calc_ELF
            )

    ECTEI_supple= calc_ECTEI_supplement(
            indata[0][noms_idx],
            indata[1][noms_idx],
            indata[3][noms_idx],
            indata[4][noms_idx],
            params
            )
    #ECTEI= EIS+ECTEI_supple

    if if_calc_ELF:
        LTS= theta_700-theta_sfc
        ELF= calc_ELF(
            indata[0][noms_idx],
            indata[1][noms_idx],
            indata[2][noms_idx],
            indata[3][noms_idx],
            indata[4][noms_idx],
            LTS, ELF_compo,
            params
            )
    ###---
    nvar= 5+int(if_calc_ELF)
    dim0= noms_idx.shape
    dim1= list(dim0)+[nvar,]
    out_arr= np.full(dim1,params['undef'])
    out_arr[noms_idx,0]= theta_sfc
    out_arr[noms_idx,1]= theta_800
    out_arr[noms_idx,2]= theta_700
    out_arr[noms_idx,3]= EIS_supple
    out_arr[noms_idx,4]= ECTEI_supple
    ai=5
    if if_calc_ELF:
        out_arr[noms_idx,ai]= ELF; ai+=1

    #nan_idx= np.isnan(out_arr)
    #if nan_idx.sum()>0:
    #    out_arr[nan_idx]= params['undef']

    return out_arr

def calc_LCidx(indata,undef= -9999.9):
    '''
    indata= ['T700','T2M','PS','QV700','QV2M','T800','Tsfc']
        or  ['t700','t2m','sp','q700','q2m','t800','skt']
    '''
    params= dict(
        gravity= 9.80665,
        R_dry= 287.05,
        R_vapor= 461.51,
        Cp_dry= 1004.,
        undef= undef
    )
    
    if not np.isnan(undef):
        for k,arr in enumerate(indata):
            arr[arr<undef+1]= np.nan
            indata[k]=arr

    ### Calc LCC
    noms_idx= indata[2]>900  ## Only for PS>900hPa

    ### LTS
    LTS= calc_LTS(
            indata[0][noms_idx],
            indata[1][noms_idx],
            indata[2][noms_idx],
            indata[3][noms_idx],
            indata[4][noms_idx],
            params
            )

    if_calc_M=True
    if if_calc_M:
        M= calc_LTS(
            indata[5][noms_idx],
            indata[6][noms_idx],
            indata[2][noms_idx],
            indata[3][noms_idx],
            indata[4][noms_idx],
            params, tgt_lev=800,
            )

    if_calc_ELF=True
    EIS_supple, ELF_compo=calc_EIS_supplement(
            indata[0][noms_idx],
            indata[1][noms_idx],
            indata[2][noms_idx],
            None,
            indata[3][noms_idx],
            indata[4][noms_idx],
            params, ELF_return=calc_ELF
            )
    EIS= LTS+EIS_supple

    ECTEI_supple= calc_ECTEI_supplement(
            indata[0][noms_idx],
            indata[1][noms_idx],
            indata[3][noms_idx],
            indata[4][noms_idx],
            params
            )
    ECTEI= EIS+ECTEI_supple

    if if_calc_ELF:
        ELF= calc_ELF(
            indata[0][noms_idx],
            indata[1][noms_idx],
            indata[2][noms_idx],
            indata[3][noms_idx],
            indata[4][noms_idx],
            LTS, ELF_compo,
            params
            )
    ###---
    nvar= 3+int(if_calc_ELF)+int(if_calc_M)
    dim0= noms_idx.shape
    dim1= list(dim0)+[nvar,]
    out_arr= np.full(dim1,params['undef'])
    out_arr[noms_idx,0]= LTS
    out_arr[noms_idx,1]= EIS
    out_arr[noms_idx,2]= ECTEI
    ai=3
    if if_calc_ELF:
        out_arr[noms_idx,ai]= ELF; ai+=1
    if if_calc_M:
        out_arr[noms_idx,ai]= M; ai+=1

    #nan_idx= np.isnan(out_arr)
    #if nan_idx.sum()>0:
    #    out_arr[nan_idx]= params['undef']

    LTS=EIS=ECTEI=ELF=M=0
    return out_arr

def get_moist_R(QV,params):
    return (1+0.61*QV)*params['R_dry']

def get_moist_Cp(QV,params):
    return (1+0.87*QV)*params['Cp_dry']

def get_potential_temp(T,P,params):
    '''
    Calculate dry potential temperature
    '''
    theta= T*(1000/P)**(params['R_dry']/params['Cp_dry'])
    return theta
    
def calc_LTS(T700,Tsfc,Psfc,QV700,QVsfc,params,tgt_lev=700):
    '''
    LTS= theta_700 - theta_sfc
    theta= T*(1000/P)**(R/Cp)
    '''
    theta_700= get_potential_temp(T700,tgt_lev,params)
    theta_sfc= get_potential_temp(Tsfc,Psfc,params)
    return theta_700-theta_sfc

def get_latent_heat_vapor(T):
    return (2.5015-0.0024*(T-273.15))*10**6

def get_moist_adiabatic_theta_gradient(T,QV,target_P,params):
    g,R_dry,R_vapor= params['gravity'],params['R_dry'],params['R_vapor']
    qs= get_saturation_mixing_ratio(T,target_P)
    Lv= get_latent_heat_vapor(T)
    Cp= get_moist_Cp(QV,params)
    Gamma_m= g/Cp*(1-(1+Lv*qs/R_dry/T)/(1+Lv*Lv*qs/Cp/R_vapor/T/T))
    return Gamma_m
    
def calc_EIS_supplement(T700,Tsfc,Psfc,Tsfc_dew,QV700,QVsfc,params,ELF_return=True):
    '''
    EIS= LTS + supplement
    supplement= -Gamma_850*(z_700-z_lcl)
    Gamma_850= g/Cp*(1-(1+Lv*qs/R_dry/Tm)/(1+Lv**2*qs/Cp/R_vapor/Tm**2))
    Tm= (T700+Tsfc)/2
    Lv= (2.5015-0.0024*(T-273.15))*10**6
    qs= 0.622*e_w/(850-e_w)
    e_w= 0.61078*exp(17.27*(T-273.15)/(T-273.15+237.3))*10
    z_700= R_dry*Tsfc/g*ln(Psfc/700)
    #z_lcl= 125*(Tsfc-Tsfc_dew)
    z_lcl is calculated by module "lcl"
    '''
    Tm= (T700+Tsfc)/2
    QVm= (QV700+QVsfc)/2
    Gamma_850= get_moist_adiabatic_theta_gradient(Tm,QVm,850,params)
    #=g/Cp*(1-(1+Lv*qs/R_dry/Tm)/(1+Lv*Lv*qs/Cp/R_vapor/Tm/Tm))
    
    g,R_dry= params['gravity'],params['R_dry']
    z_700= R_dry*Tsfc/g*np.log(Psfc/700)
    ### Get z_LCL
    #z_lcl= 125*(Tsfc-Tsfc_dew)
    if type(Tsfc_dew)!=np.ndarray:
        r= QVsfc/(1-QVsfc)
        e= Psfc*r/(0.622+r)
        rh= e/get_saturation_vapor_pressure(Tsfc)
    else:
        rh =100*(np.exp((17.625*(Tsfc_dew-273.15))/(243.04+Tsfc_dew-273.15))/np.exp((17.625*(Tsfc-273.15))/(243.04+Tsfc-273.15)))
    #print(rh.min(), rh.max()); sys.exit()
    z_lcl= np.maximum(0,lcl(Psfc*100,Tsfc,rh=rh))
    #print(z_lcl.shape, z_lcl.max(), np.argmax(z_lcl)); sys.exit()
    if ELF_return:
        compo= dict(z_lcl=z_lcl, z_700=z_700)
    else:
        compo= None
    return -Gamma_850*(z_700-z_lcl), compo
    #return Gamma_850,z_700,z_lcl,Gamma_850*(z_700-z_lcl)

def get_saturation_mixing_ratio(T,P):
    '''
    qs= 0.622*e_w/(P-e_w)
    '''

    e_w= get_saturation_vapor_pressure(T)
    qs= 0.622*e_w/(P-e_w)
    return qs

def get_saturation_vapor_pressure(T):
    '''
    Temp to pressure (hPa)
    e_w= 0.61078*np.exp(17.27*(T-273.15)/(T-273.15+237.3))*10
    '''
    e_w= 0.61078*np.exp(17.27*(T-273.15)/(T-273.15+237.3))*10
    return e_w

def get_saturation_specific_humidity(T,P,params):
    R_ratio= params['R_dry']/params['R_vapor']
    e_w= get_saturation_vapor_pressure(T)
    q_sat= R_ratio*e_w/(P-((1-R_ratio)*e_w))
    return q_sat

def calc_ECTEI_supplement(T700,Tsfc,QV700,QVsfc,params):
    '''
    ECTEI= EIS - beta*Lv/Cp*(QVsfc-QV700)
    beta= (1-k)*C_qgap= 0.23
    '''
    beta= 0.23
    Tm= (T700+Tsfc)/2
    QVm= (QV700+QVsfc)/2
    Lv= get_latent_heat_vapor(Tm)
    Cp= get_moist_Cp(QVm,params)
    return -beta*Lv/Cp*(QVsfc-QV700)
    
def calc_ELF(T700,Tsfc,Psfc,QV700,QVsfc,LTS,ELF_compo,params):
    """
    ELF= f*(1-root(z_inv*z_lcl)/delta_zs)
    f= max(0.15, min(1, QV_ML/0.003))
    z_inv= -(LTS/Gamma_m_700)+z_700+delta_zs*Gamma_m_DL/Gamma_m_700
    Gamma_m_DL is based on T at LCL, QV at sfc, and P at LCL
    """

    delta_zs= 2750
    f= np.maximum(0.15, np.minimum(1,QVsfc/0.003))
    g,R_dry= params['gravity'],params['R_dry']
    z_700= ELF_compo['z_700']
    z_lcl= ELF_compo['z_lcl'] #; print('LCL',np.min(z_lcl),np.percentile(z_lcl,[5,25,75,95]))
    Gamma_m_700= get_moist_adiabatic_theta_gradient(T700,QV700,700,params)
    #z_700= R_dry*Tsfc/g*np.log(Psfc/700)
    P_lcl= Psfc*np.exp(-g*z_lcl/R_dry/Tsfc)
    T_lcl= Tsfc*(P_lcl/Psfc)**0.286
    Gamma_m_lcl= get_moist_adiabatic_theta_gradient(T_lcl,QVsfc,P_lcl,params)
    z_inv= -(LTS/Gamma_m_700)+z_700+delta_zs*Gamma_m_lcl/Gamma_m_700 #; print('INV',np.percentile(z_inv,[5,25,75,95]))
    z_inv= np.clip(z_inv,z_lcl,z_lcl+delta_zs) #; print('INV',np.percentile(z_inv,[5,25,75,95]))
    ELF= f*(1-np.sqrt(z_inv*z_lcl)/delta_zs) #; print('ELF',np.percentile(ELF,[5,25,75,95])); 
    '''
    if np.any(ELF<-0.6):
        ind= ELF<-0.6
        for t7,ts,t0,t1,q1 in zip(ELF[ind],z_inv[ind], z_lcl[ind],Tsfc[ind],Psfc[ind]):
            print(t7,ts,t0,t1,q1)
        sys.exit()
    '''
    return ELF

# Version 1.0 released by David Romps on September 12, 2017.
# Version 1.1 vectorized lcl.R, released on May 24, 2021.
# 
# When using this code, please cite:
# 10.1175/JAS-D-17-0102.1; https://romps.berkeley.edu/papers/pubs-2016-lcl.html
# --> https://davidromps.com/papers/pubs-2016-lcl.html
#
# @article{16lcl,
#   Title   = {Exact expression for the lifting condensation level},
#   Author  = {David M. Romps},
#   Journal = {Journal of the Atmospheric Sciences},
#   Year    = {2017},
#   Month   = dec,
#   Number  = {12},
#   Pages   = {3891--3900},
#   Volume  = {74}
# }
#
# This lcl function returns the height of the lifting condensation level
# (LCL) in meters.  The inputs are:
# - p in Pascals
# - T in Kelvins
# - Exactly one of rh, rhl, and rhs (dimensionless, from 0 to 1):
#    * The value of rh is interpreted to be the relative humidity with
#      respect to liquid water if T >= 273.15 K and with respect to ice if
#      T < 273.15 K. 
#    * The value of rhl is interpreted to be the relative humidity with
#      respect to liquid water
#    * The value of rhs is interpreted to be the relative humidity with
#      respect to ice
# - return_ldl is an optional logical flag.  If true, the lifting deposition
#   level (LDL) is returned instead of the LCL. 
# - return_min_lcl_ldl is an optional logical flag.  If true, the minimum of the
#   LCL and LDL is returned.

def lcl(p,T,rh=None,rhl=None,rhs=None,return_ldl=False,return_min_lcl_ldl=False):
    

   import scipy.special

   # Parameters
   Ttrip = 273.16     # K
   ptrip = 611.65     # Pa
   E0v   = 2.3740e6   # J/kg
   E0s   = 0.3337e6   # J/kg
   ggr   = 9.81       # m/s^2
   rgasa = 287.04     # J/kg/K 
   rgasv = 461        # J/kg/K 
   cva   = 719        # J/kg/K
   cvv   = 1418       # J/kg/K 
   cvl   = 4119       # J/kg/K 
   cvs   = 1861       # J/kg/K 
   cpa   = cva + rgasa
   cpv   = cvv + rgasv

   # The saturation vapor pressure over liquid water
   def pvstarl(T):
      return ptrip * (T/Ttrip)**((cpv-cvl)/rgasv) * \
         np.exp( (E0v - (cvv-cvl)*Ttrip) / rgasv * (1/Ttrip - 1/T) )
   
   # The saturation vapor pressure over solid ice
   def pvstars(T):
      return ptrip * (T/Ttrip)**((cpv-cvs)/rgasv) * \
         np.exp( (E0v + E0s - (cvv-cvs)*Ttrip) / rgasv * (1/Ttrip - 1/T) )

   # Calculate pv from rh, rhl, or rhs
   rh_counter = 0
   if rh  is not None:
      rh_counter = rh_counter + 1
   if rhl is not None:
      rh_counter = rh_counter + 1
   if rhs is not None:
      rh_counter = rh_counter + 1
   if rh_counter != 1:
      print(rh_counter)
      exit('Error in lcl: Exactly one of rh, rhl, and rhs must be specified')
   if rh is not None:
       
      # The variable rh is assumed to be 
      # with respect to liquid if T > Ttrip and 
      # with respect to solid if T < Ttrip
       
      '''
      if T > Ttrip:
         pv = rh * pvstarl(T)
      else:
         pv = rh * pvstars(T)
      '''
      pv= np.empty_like(T)
      Tidx= T > Ttrip
      if isinstance(rh,np.ndarray):
         pv[Tidx] = rh[Tidx] * pvstarl(T[Tidx])
         pv[~Tidx] = rh[~Tidx] * pvstars(T[~Tidx])
      else:
         pv[Tidx] = rh * pvstarl(T[Tidx])
         pv[~Tidx] = rh * pvstars(T[~Tidx])
      
      rhl = pv / pvstarl(T)
      rhs = pv / pvstars(T)
   elif rhl is not None:
      pv = rhl * pvstarl(T)
      rhs = pv / pvstars(T)
      #if T > Ttrip: rh = rhl
      #else: rh = rhs
      rh= np.copy(rhs)
      rh[T>Ttrip]=rhl[T>Ttrip]
   elif rhs is not None:
      pv = rhs * pvstars(T)
      rhl = pv / pvstarl(T)
      #if T > Ttrip: rh = rhl
      #else: rh = rhs
      rh= np.copy(rhl)
      rh[T<=Ttrip]=rhs[T<=Ttrip]
   #if pv > p:
   #   return NA
   ms_idx= pv>p
   #pv= np.ma.masked_array(pv,mask=ms_idx)

   # Calculate lcl_liquid and lcl_solid
   qv = rgasa*pv / (rgasv*p + (rgasa-rgasv)*pv)
   rgasm = (1-qv)*rgasa + qv*rgasv
   cpm = (1-qv)*cpa + qv*cpv
   #if rh == 0:
   #   return cpm*T/ggr
   aL = -(cpv-cvl)/rgasv + cpm/rgasm
   bL = -(E0v-(cvv-cvl)*Ttrip)/(rgasv*T)
   cL = pv/pvstarl(T)*np.exp(-(E0v-(cvv-cvl)*Ttrip)/(rgasv*T))
   aS = -(cpv-cvs)/rgasv + cpm/rgasm
   bS = -(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T)
   cS = pv/pvstars(T)*np.exp(-(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T))
   lcl = cpm*T/ggr*( 1 - \
      bL/(aL*scipy.special.lambertw(bL/aL*cL**(1/aL),-1).real) )
   ldl = cpm*T/ggr*( 1 - \
      bS/(aS*scipy.special.lambertw(bS/aS*cS**(1/aS),-1).real) )
   if rh.any()==0:
      lcl[rh==0]= cpm*T[rh==0]/ggr
      ldl[rh==0]= cpm*T[rh==0]/ggr
   #if pv > p:
   if ms_idx.sum()>0:
      lcl[ms_idx]= np.nan
      
   # Return either lcl or ldl
   if return_ldl and return_min_lcl_ldl:
      exit('return_ldl and return_min_lcl_ldl cannot both be true')
   elif return_ldl:
      return ldl
   elif return_min_lcl_ldl:
      return np.minimum(lcl,ldl)
   else:
      return lcl

