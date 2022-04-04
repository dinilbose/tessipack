import numpy as np
import pandas as pd
import sys
# print(sys.path)
from ..functions import utils
from ..functions import aperture
from .. import config
from pathlib import Path
import os
import ast

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

#package_path='/home/dinilbose/PycharmProjects/light_cluster/'
#package_path='/home/dinilbose/mypackage/tessipack/tessipack/'
package_path=str(Path(PACKAGEDIR).parents[0])+'/'

#Data_path='/home/dinilbose/PycharmProjects/light_cluster/cluster/Collinder_69/Data/'
Data_path=config.data_folder
#catalog_version='01'
def pointer(catalog='mycatalog',pmemb=0,cluster='',id_mycatalog='',id_gaia='',integrate=True,real=False):
    '''Points to the catalog area'''

    data=''

    mycatalog=pd.read_csv(config.catalog_path)
    mycatalog=mycatalog.query('PMemb>=@pmemb')
    if not id_gaia=='':
        mycatalog=mycatalog.query('id_gaia==@id_gaia')
    if not cluster=='':
        mycatalog=mycatalog.query('cluster==@cluster')

    if real==True:
        whole=pd.read_csv(package_path+'/My_catalog/Source/Collinder_69_aperture_pixel.csv')
        Real_source=whole.drop_duplicates(subset=['pixel_x','pixel_y'],keep='first').id_gaia.unique().tolist()
        mycatalog=mycatalog.query('id_gaia==@Real_source')


    if not id_mycatalog=='':

        mycatalog=mycatalog.query('id_mycatalog==@id_mycatalog')
        id_gaia_n=mycatalog.id_gaia.values.tolist()
        if id_mycatalog=='custom_star':
            mycatalog.loc[0,'id_mycatalog']='custom_star'
            mycatalog.loc[0,'id_gaia']='custom_star'
            mycatalog.loc[0,'cluster']='Custom_star'
            mycatalog.loc[0,'Sector']='[0]'
            mycatalog=mycatalog.fillna(0)

    if catalog=='mycatalog':
        # data=pd.read_csv('My_catalog/my_catalog_v02.csv')
        # data=data.query('PMemb>=@pmemb')
        data=mycatalog

    if catalog=='gaia':
        data=utils.votable_to_pandas(full_path+'My_catalog/Source/Collinder69_gaia.vot')
        data=utils.votable_to_pandas(full_path+'My_catalog/Source/Gaia_open_cluster_members.vot')
        data['Source']='DR2_'+data['Source'].astype(str)
        data['id_gaia']=data['Source']
        data['Cluster'] = data['Cluster'].str.decode('utf-8')
        id_gaia_n=mycatalog.id_gaia.values.tolist()
        data=data.query('id_gaia==@id_gaia_n')

    if catalog=='apogee':
        # /home/dinilbose/PycharmProjects/light_cluster/My_catalog/Source
        # apogee=utils.votable_to_pandas(package_path+'My_Catalog/Source/Apogee.vot')
        apogee=utils.votable_to_pandas('/home/dinilbose/PycharmProjects/light_cluster/My_catalog/Source/Apogee.vot')

        apogee['SourceId']='DR2_'+apogee['SourceId'].astype('str')
        apogee['id_gaia']=apogee['SourceId']
        apogee['Cluster'] = apogee['Cluster'].str.decode('utf-8')
        apogee['APOGEE'] = apogee['APOGEE'].str.decode('utf-8')
        id_gaia_n=mycatalog.id_gaia.values.tolist()
        data=apogee.query('id_gaia==@id_gaia_n')

    if catalog=='oscillation':
        data=pd.read_csv(package_path+'My_catalog/Source/Gaia_all_astrobase_oscillation_source.csv')
        # data['Source']='DR2_'+data['Source'].astype(str)
        data['id_gaia']=data['Source']
        id_gaia_n=mycatalog.id_gaia.values.tolist()
        data=data.query('id_gaia==@id_gaia_n')


    if catalog=='tic':
        tic=utils.votable_to_pandas(package_path+'My_catalog/Source/gaia_mast_crossmatch.xml')
        tic['id_gaia'] = tic['id_gaia'].str.decode('utf-8')
        id_gaia_n=mycatalog.id_gaia.values.tolist()
        data=tic.query('id_gaia==@id_gaia_n')

    if integrate==True:
        data['id_mycatalog'] = data.id_gaia.map(mycatalog.set_index('id_gaia')['id_mycatalog'].to_dict())
    # data['Cluster']=data['Cluster'].str.decode('utf-8')
    return data

def filename(name='',id_gaia='',id_mycatalog='',cluster='Collinder_69',sector=''):
    '''Filename manager: Gives file names and path for all the files'''

    #Data_path='/home/dinilbose/PycharmProjects/light_cluster/cluster/Collinder_69/Data/'
    path_sys=Path()

    my_catalog=pointer(catalog='mycatalog')

    if sector=='':
        Sector=''
    else:
        Sector='_s'+str(sector)

    if not id_gaia=='':
        my_catalog=my_catalog.query('id_gaia==@id_gaia')
        cluster=my_catalog['cluster'].values[0]
    if not id_mycatalog=='':
        my_catalog=my_catalog.query('id_mycatalog==@id_mycatalog')
        if id_mycatalog=='custom_star':
            my_catalog=my_catalog.query('id_mycatalog=="7"')
            my_catalog.loc[0,'id_mycatalog']='custom_star'
            my_catalog.loc[0,'id_gaia']='custom_star'
            my_catalog.loc[0,'cluster']='Custom_star'
            my_catalog.loc[0,'Sector']='[0]'
            my_catalog=my_catalog.fillna(0)

        cluster=my_catalog['cluster'].values[0]

    if name=='eleanor_flux':
        id=my_catalog.id_gaia.item()
        filename=Data_path+name+'/id_gaia'+'/'+cluster+'_'+id+Sector+'_ffi.csv'
        full_path=Path(filename)

    if name=='eleanor_flux_current':
        id=my_catalog.id_gaia.item()
        filename=Data_path+name+'/id_gaia'+'/'+cluster+'_'+id+Sector+'_ffic.csv'
        full_path=Path(filename)

    if name=='eleanor_aperture':
        id=my_catalog.id_gaia.item()
        filename=Data_path+name+'/id_gaia'+'/'+cluster+'_'+id+Sector+'_ap.txt'
        full_path=Path(filename)

    if name=='eleanor_aperture_current':
        id=my_catalog.id_gaia.item()
        filename=Data_path+name+'/id_gaia'+'/'+cluster+'_'+id+Sector+'_apc.txt'
        full_path=Path(filename)


    if name=='eleanor_tpf':
        id=my_catalog.id_gaia.item()
        filename=Data_path+name+'/id_gaia'+'/'+cluster+'_'+id+Sector+'_tpf.npy'
        full_path=Path(filename)
    if name=='eleanor_header':
        id=my_catalog.id_gaia.item()
        filename=Data_path+name+'/id_gaia'+'/'+cluster+'_'+id+Sector+'_hd.txt'
        full_path=Path(filename)

    if name=='eleanor_time_flag':
        id=my_catalog.id_gaia.item()
        filename=Data_path+name+'/id_gaia'+'/'+cluster+'_'+id+Sector+'_tflag.txt'
        full_path=Path(filename)

    if name=='bokeh_periodogram_table':
        id=my_catalog.id_gaia.item()
        filename=Data_path+name+'/id_gaia'+'/'+cluster+'_'+id+Sector+'_bkprtb.csv'
        full_path=Path(filename)

    if name=='whole_gaia_data':
        #id=my_catalog.id_gaia.item()
        filename=Data_path+name+'/'+cluster+'.csv'
        full_path=Path(filename)

    if name=='extra_flag_file':
        #id=my_catalog.id_gaia.item()
        filename=Data_path+'extra_flag.flag'
        full_path=Path(filename)

    if name=='cluster_ffi':
        import glob
        filename=Data_path+name+'/'+cluster+'/'+'*.fits'
        filename=glob.glob(filename)
        full_path=filename


    if name=='temp':
        filename=Data_path+name
        full_path=Path(filename)

    return full_path

def update(id_mycatalog,**kwargs):
        '''Updates Catalog based on user inputs'''
        mycatalog=pd.read_csv(package_path+'My_catalog/my_catalog_v'+catalog_version+'.csv')
        mycatalog=mycatalog.set_index('id_mycatalog')
        #print('Catalog Version:',catalog_version,'Updated')
        for key in kwargs:
            print('Version:',catalog_version,'  ',id_mycatalog,"Updated: %s: %s" % (key, kwargs[key]))
            mycatalog.loc[id_mycatalog,key]=kwargs[key]
            # print(mycatalog.head)
        mycatalog.to_csv(package_path+'/My_catalog/my_catalog_v'+catalog_version+
'.csv')

def download_data(id_mycatalog=None,ra=None,dec=None):
    "Download data"
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import tessipack.eleanor as eleanor
    list_d=['bokeh_periodogram_table', 'eleanor_aperture', 'eleanor_aperture_current',
            'eleanor_flux', 'eleanor_flux_current', 'eleanor_header', 'eleanor_time_flag', 'eleanor_tpf', ]

    if not os.path.exists(os.path.expanduser(Data_path)):
        print('Does not exist')
        os.mkdir(os.path.expanduser(Data_path))
        os.mknod(os.path.expanduser(Data_path+'extra_flag.flag'))

        # 'extra_flag.flag'

    for dir_l in list_d:
        if not os.path.exists(os.path.expanduser(Data_path+dir_l)):
            print('Creating dir' + Data_path+dir_l)
            os.mkdir(os.path.expanduser(Data_path+dir_l))
            os.mkdir(os.path.expanduser(Data_path+dir_l+'/id_gaia'))


    # print(len(Full_catalog),'/###############################################',i,'#####################')
    center=0
    data_post=''
    #source=mycatalog.pointer(catalog='mycatalog').query('cluster=="NGC_2477" & flag_duplicate==1 & PMemb>=0.5').iloc[i]
    source=pointer(catalog='mycatalog').query('id_mycatalog==@id_mycatalog').iloc[0]
    # source=mycatalog.pointer(catalog='mycatalog').query('cluster=="Collinder_69"').iloc[i]
    print(source)
    dec=source.DE_ICRS
    ra=source.RA_ICRS
    id_mycatalog=source.id_mycatalog
    sec=source.Sector
    if not type(sec)==str:
        if np.isnan(sec):
            center = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            star_all = eleanor.multi_sectors(coords=center,sectors='all')
            all_sectors=[star.sector for star in star_all]
            update(id_mycatalog,Sector=str(all_sectors))
    else:
        all_sectors=ast.literal_eval(sec)

    if sec=='[0]':
        center = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        star_all = eleanor.multi_sectors(coords=center,sectors='all')
        all_sectors=[star.sector for star in star_all]
        update(id_mycatalog,Sector=str(all_sectors))
    else:
        all_sectors=ast.literal_eval(sec)



    for sector in all_sectors:
        print(sector)

        filename_ap=filename(id_mycatalog=id_mycatalog,name='eleanor_aperture',sector=int(sector))
        my_file = Path(filename_ap)
        print(filename_ap)
        if not os.path.isfile(filename_ap):
            center = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            star = eleanor.Source(coords=center,sector=int(sector))

            # for star in star_all:
                # print('%%%%%%%%sector',star.sector)
            data_post=eleanor.TargetData(star,do_psf=True,do_pca=True)
            mywcs=utils.extract_essential_wcs_postcard(data_post)
            radec=utils.pixe2radec(wcs=mywcs,aperture=data_post.aperture)
            for k in range(len(radec)):
                aperture.add_aperture(id_mycatalog=id_mycatalog,ra=radec[k][0],dec=radec[k][1],name='eleanor_aperture',sector=int(sector))
                aperture.add_aperture(id_mycatalog=id_mycatalog,ra=radec[k][0],dec=radec[k][1],name='eleanor_aperture_current',sector=int(sector))


        filename_flux=filename(id_mycatalog=id_mycatalog,name='eleanor_flux',sector=int(sector))
        filename_flux_current=filename(id_mycatalog=id_mycatalog,name='eleanor_flux_current',sector=int(sector))

        if not os.path.isfile(filename_flux):
            if type(data_post)==str:
                center = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
                star = eleanor.Source(coords=center,sector=int(sector))

                data_post=eleanor.TargetData(star,do_psf=True,do_pca=True)


            data_frame=pd.DataFrame()
            data_frame['time']=data_post.time
            data_frame['corr_flux']=data_post.corr_flux
            data_frame['pca_flux']=data_post.pca_flux
            data_frame['psf_flux']=data_post.psf_flux
            data_frame['flux_err']=data_post.flux_err
            data_frame['time_flag']=data_post.quality
            data_frame['sector']=star.sector
            data_frame.to_csv(filename_flux)
            data_frame.to_csv(filename_flux_current)

        filename_tpf=filename(id_mycatalog=id_mycatalog,name='eleanor_tpf',sector=int(sector))
        if os.path.isfile(filename_tpf):
            print('Tpf File exist',id_mycatalog,'sector',sector)
        else:
            if type(data_post)==str:
                print('Tpf Computing sector',id_mycatalog)
                center = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
                star = eleanor.Source(coords=center,sector=int(sector))
                data_post=eleanor.TargetData(star,do_psf=True,do_pca=True)

            np.save(filename_tpf,data_post.tpf)

        filename_header=filename(id_mycatalog=id_mycatalog,name='eleanor_header',sector=int(sector))
        if os.path.isfile(filename_header):
            print('File exist',id_mycatalog)
        else:
            if type(data_post)==str:
                print('header Computing',id_mycatalog)
                center = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
                star = eleanor.Source(coords=center,sector=int(sector))
                data_post=eleanor.TargetData(star,do_psf=True,do_pca=True)
                np.save(filename_tpf,data_post.tpf)

            data_post.header.totextfile(filename_header,overwrite=True)

        filename_time=filename(id_mycatalog=id_mycatalog,name='eleanor_time_flag',sector=int(sector))

        print('time_computing',id_mycatalog)
        if not os.path.isfile(filename_time):
            if type(data_post)==str:
                center = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
                star = eleanor.Source(coords=center,sector=int(sector))
                data_post=eleanor.TargetData(star,do_psf=True,do_pca=True)

            data_frame=pd.DataFrame()
            q=data_post.quality == 0
            data_frame['time_flag']=q
            data_frame.to_csv(filename_time)


        #setting up custom star

        id_mycatalog='custom_star'

        mywcs=utils.extract_essential_wcs_postcard(data_post)
        radec=utils.pixe2radec(wcs=mywcs,aperture=data_post.aperture)
        for k in range(len(radec)):
            aperture.add_aperture(id_mycatalog=id_mycatalog,ra=radec[k][0],dec=radec[k][1],name='eleanor_aperture',sector=int(sector))
            aperture.add_aperture(id_mycatalog=id_mycatalog,ra=radec[k][0],dec=radec[k][1],name='eleanor_aperture_current',sector=int(sector))

        data_frame=pd.DataFrame()
        data_frame['time']=data_post.time
        data_frame['corr_flux']=np.arange(0,len(data_frame))
        data_frame['pca_flux']=np.arange(0,len(data_frame))
        data_frame['psf_flux']=np.arange(0,len(data_frame))
        data_frame['flux_err']=np.arange(0,len(data_frame))
        data_frame['time_flag']=data_post.quality
        data_frame['sector']=star.sector


        filename_flux=filename(id_mycatalog=id_mycatalog,name='eleanor_flux',sector=int(sector))
        data_frame.to_csv(filename_flux)
        filename_flux_current=filename(id_mycatalog=id_mycatalog,name='eleanor_flux_current',sector=int(sector))
        data_frame.to_csv(filename_flux_current)


        filename_tpf=filename(id_mycatalog=id_mycatalog,name='eleanor_tpf',sector=int(sector))
        kk=data_post.tpf
        kk[:]=10
        kk[:,2]=30

        np.save(filename_tpf,kk)


        filename_header=filename(id_mycatalog=id_mycatalog,name='eleanor_header',sector=int(sector))
        data_post.header.totextfile(filename_header,overwrite=True)

        filename_time=filename(id_mycatalog=id_mycatalog,name='eleanor_time_flag',sector=int(sector))
        data_frame.to_csv(filename_time)
