from tessipack.My_catalog import mycatalog
from env import Environment
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.models import CustomJS, TextInput, Paragraph
from bokeh.models import Button, Select, RadioGroup  # for saving data
from bokeh.plotting import figure
import pandas as pd
from tessipack.functions import equations
import ast
from astropy.coordinates import SkyCoord
from astropy import units as unit
from tessipack import eleanor

from tessipack.functions import utils
from tessipack.functions import aperture

from pathlib import Path
import os
import ast

class Catalog(Environment):
    env=Environment
    def __init__(self):
        self.catalog_all=mycatalog.pointer(catalog='mycatalog')
        self.env.extra_flag_file=mycatalog.filename(name='extra_flag_file')

        all_clusters=self.catalog_all.cluster.unique().tolist()
        # catalog_all=catalog_all.query('flag_source==1 & flag_duplicate==0').reset_index()
        # catalog_all=catalog_all.sort_values(by=['Gmag']).reset_index()

        self.env.catalog_main=self.catalog_all.query('cluster==@self.env.default_cluster').reset_index(drop=True)

        # self.catalog_all=self.env.catalog_main.query('flag_source==1').reset_index()
        self.catalog_all=self.env.catalog_main


        self.env.text_cluster_query= TextInput(value=self.env.default_cluster, title="Select Cluster")
        self.env.text_cluster_query = Select(title='Cluster', options=all_clusters, value=all_clusters[0])

        self.env.update_cluster_button = Button(label="Update cluster", button_type="success",width=150)
        self.env.update_cluster_button.on_click(self.update_cluster)


        self.env.text_catalog_query= TextInput(value='', title="Catalog query")
        self.env.update_catalog_button = Button(label="Update catalog", button_type="success",width=150)
        self.env.update_catalog_button.on_click(self.update_catalog)

        self.env.update_id_mycatalog_button = Button(label="Update id_mycatalog", button_type="success",width=150)
        self.env.update_id_mycatalog_button.on_click(self.update_id_mycatalog)
        self.env.update_id_button = Button(label="Update id", button_type="success",width=150)
        self.env.update_id_button.on_click(self.update_id)

        self.env.selection_program = RadioGroup(labels=["Catalog", "Custom Star"],active=0,orientation='horizontal')
        self.env.selection_program.on_change('active',self.update_selection_program)

        self.env.text_custom_star_ra=TextInput(value='0', title="RA (deg)")
        self.env.text_custom_star_dec=TextInput(value='0', title="Dec (deg)")
        single_sector_list=list(['Check Sector'])
        self.env.text_custom_star_sector=Select(title='Sector', options=single_sector_list, value=single_sector_list[0])
        self.env.custom_star_download_button = Button(label="Download", button_type="danger",width=150)
        self.env.custom_star_download_button.on_click(self.download_custom_star)

        self.env.text_custom_star_sector.on_change('value',self.update_custom_sector)
        # self.env.custom_star_download_button.on_click(self.update_id)






        # catalog_all=catalog_all.query('flag_source>0').reset_index()
        # catalog_all=mycatalog.pointer(catalog='mycatalog').query('cluster=="NGC_6208" & flag_source==6').reset_index()
        # catalog_all=mycatalog.pointer(catalog='mycatalog').query('cluster=="NGC_2477"').sort_values(by=['Gmag']).reset_index()
        # catalog_all=mycatalog.pointer(catalog='mycatalog').query('cluster=="NGC_2477"').reset_index()
        self.id_mycatalog_all=list(self.catalog_all.id_mycatalog.values)
        # print('newwwwwwwwwww',id_mycatalog_all)
        self.id_all=np.arange(0,len(self.catalog_all))
        self.id=0
        self.id_mycatalog=self.id_mycatalog_all[self.id]


        mydata=mycatalog.pointer(id_mycatalog=self.id_mycatalog)
        sector_list=ast.literal_eval(mydata.Sector.values[0])
        sector_list=list([str(i) for i in sector_list])
        self.env.sector=sector_list[0]

        self.env.int_select_sector = Select(title='Sector', options=sector_list, value=sector_list[0])


        self.env.tb_source = ColumnDataSource(data=dict(id_all=self.id_all,id_mycatalog_all=self.id_mycatalog_all,id=[self.id],id_mycatalog=[self.id_mycatalog]))

        self.env.tb_catalog_main=ColumnDataSource(self.env.catalog_main.to_dict('list'))
        self.env.tb_catalog_all=ColumnDataSource(self.catalog_all.to_dict('list'))
        #self.env.tb_catalog_current=self.catalog_all.loc[self.id]


        self.env.text_id_mycatalog_query = TextInput(value=str(self.id_mycatalog), title="id_mycatalog")
        self.env.text_id_query = TextInput(value=str(self.id), title="id")

        self.env.tb_catalog_all.selected.on_change('indices',self.update_selected)
        self.env.tb_catalog_all.selected.indices=[self.id]

        self.env.generate_isochrone_button = Button(label="Generate Isochrone", button_type="success",width=150)
        self.env.generate_isochrone_button .on_click(self.generate_isochrone)
        self.env.delete_isochrone_button = Button(label="Delete Isochrone", button_type="success",width=150)
        self.env.delete_isochrone_button .on_click(self.delete_isochrone)

        self.env.catalog_find_from_isocrhone=self.find_from_isocrhone

        self.env.text_banner_Gmag = Paragraph(text='', width=1000, height=10)
        self.env.text_banner_bp_rp = Paragraph(text='', width=1000, height=10)
        self.env.text_banner_dmin= Paragraph(text='', width=1000, height=10)

        self.draw_hr_diagram()
        self.initiate_isochrone()


    def update_catalog(self):
        self.catalog_all=mycatalog.pointer(catalog='mycatalog')
        # catalog_all=catalog_all.query('flag_source==1 & flag_duplicate==0').reset_index()
        # catalog_all=catalog_all.sort_values(by=['Gmag']).reset_index()
        self.env.catalog_main=self.catalog_all.query('cluster==@self.env.default_cluster').reset_index()

        if not self.env.text_catalog_query.value=='':
            print('Query',self.env.text_catalog_query.value)
            self.catalog_all=self.env.catalog_main.reset_index(drop=True).query(self.env.text_catalog_query.value).reset_index(drop=True)
            print(self.catalog_all)
        elif self.env.text_catalog_query.value=='':
            self.catalog_all=self.env.catalog_main

        self.id_mycatalog_all=self.catalog_all.id_mycatalog
        # print('newwwwwwwwwww',id_mycatalog_all)
        self.id_all=np.arange(0,len(self.catalog_all))
        self.id=0
        self.id_mycatalog=self.id_mycatalog_all[self.id]
        # self.env.tb_source = ColumnDataSource(data=dict(id_all=self.id_all,id_mycatalog_all=self.id_mycatalog_all,id=[self.id],id_mycatalog=[self.id_mycatalog]))
        old=ColumnDataSource(data=dict(id_all=self.id_all,id_mycatalog_all=self.id_mycatalog_all,id=[self.id],id_mycatalog=[self.id_mycatalog]))
        self.env.tb_source.data = old.data
        self.env.tb_catalog_all.data=ColumnDataSource(self.catalog_all.to_dict('list')).data
        self.env.tb_catalog_all.selected.indices=[self.id]
        self.env.tb_catalog_main.data=ColumnDataSource(self.env.catalog_main.to_dict('list')).data

        mydata=mycatalog.pointer(id_mycatalog=self.id_mycatalog)
        sector_list=ast.literal_eval(mydata.Sector.values[0])
        sector_list=list([str(i) for i in sector_list])
        self.env.sector=sector_list[0]

    def update_cluster(self):
        #self.catalog_all=mycatalog.pointer(catalog='mycatalog')

        self.env.default_cluster=self.env.text_cluster_query.value

        mydata=mycatalog.pointer(id_mycatalog=self.id_mycatalog)
        print('Update cluster list',self.env.sector,mydata)

        sector_list=ast.literal_eval(mydata.Sector.values[0])
        sector_list=list([str(i) for i in sector_list])
        self.env.sector=sector_list[0]
        self.env.int_select_sector.options=sector_list
        print('Update cluster list',self.env.sector)
        #self.env.catalog_main=self.catalog_all.query('cluster==@self.env.default_cluster').reset_index()
        #self.env.tb_catalog_main.data=ColumnDataSource(self.env.catalog_main.to_dict('list')).data
        self.update_catalog()




    def draw_hr_diagram(self):

         self.env.fig_hr = figure(title='HR diagram',
                                  tools=["box_zoom", "wheel_zoom","lasso_select", "tap" ,"reset", "save"],
                                  tooltips=self.env.TOOLTIPS,)
         self.env.fig_hr.y_range.flipped = True

         # self.env.tb_hr = ColumnDataSource(data=dict(bp_rp=[0], Gmag=[1]))
         self.env.fig_hr.circle('bp_rp', 'Gmag',color='black', alpha=1,source=self.env.tb_catalog_main,nonselection_fill_alpha=1,nonselection_fill_color='black')
         self.se=self.env.fig_hr.circle('bp_rp', 'Gmag',color='blue', alpha=1,source=self.env.tb_catalog_all,nonselection_fill_alpha=1,**self.env.selection_2)
         self.draw_position_diagram()

    def update_id_mycatalog(self):
        query=self.catalog_all.query('id_mycatalog==@self.env.text_id_mycatalog_query.value')
        self.env.tb_source.data["id"][0]=query.index.values[0]
        #print(query.index.values[0])
        self.env.tb_source.trigger("data",0,1)

        # self.id_all=np.arange(0,len(self.catalog_all))
        # self.id=query.index.values[0]
        # self.id_mycatalog_all=self.catalog_all.id_mycatalog
        # self.id_mycatalog=self.id_mycatalog_all[self.id]
        # # self.env.tb_source = ColumnDataSource(data=dict(id_all=self.id_all,id_mycatalog_all=self.id_mycatalog_all,id=[self.id],id_mycatalog=[self.id_mycatalog]))
        # old=ColumnDataSource(data=dict(id_all=self.id_all,id_mycatalog_all=self.id_mycatalog_all,id=[self.id],id_mycatalog=[self.id_mycatalog]))
        # self.env.tb_source.data = old.data
        # self.env.tb_catalog_all.data=ColumnDataSource(self.catalog_all.to_dict('list')).data
        #

    def draw_position_diagram(self):
        self.env.fig_position = figure(title='Position diagram',
                                  tools=["box_zoom", "wheel_zoom","lasso_select", "tap" ,"reset", "save"],
                                  tooltips=self.env.TOOLTIPS,)
         # self.env.fig_hr.y_range.flipped = True

         # self.env.tb_hr = ColumnDataSource(data=dict(bp_rp=[0], Gmag=[1]))
        self.env.fig_position.circle('RA_ICRS', 'DE_ICRS',color='black', alpha=1,source=self.env.tb_catalog_main,nonselection_fill_alpha=1,nonselection_fill_color='black')
        self.se_pos=self.env.fig_position.circle('RA_ICRS', 'DE_ICRS',color='blue', alpha=1,source=self.env.tb_catalog_all,nonselection_fill_alpha=1,**self.env.selection_2)
        # print('Testtttttttttttttttttttttttttttttttttttttttttttttttttttttttttt')


    def update_id(self):
        # query=self.catalog_all.query('id_mycatalog==@self.env.text_id_query.value')
        # self.env.tb_source.data["id"][0]=w.index.values[0]

        self.env.tb_source.data["id"][0]=int(self.env.text_id_query.value)
        self.env.tb_source.trigger("data",0,1)


    def update_selected(self,attr,old,new):

        # print(self.se.nonselection_glyph.fill_color,self.se.nonselection_glyph.fill_alpha)
        # self.env.tb_catalog_all.selected.indices=[old[0]]

        if not new==[]:
            print(new,new[0])
            self.env.tb_source.data["id"][0]=new[0]
            # self.env.tb_catalog_all.selected.indices=[new[0]]

            self.env.tb_source.trigger("data",0,1)
        # if new==[]:
        #     self.env.tb_catalog_all.selected.indices=self.env.tb_source.data["id"]
            # self.env.tb_source.trigger("data",0,1)

    def initiate_isochrone(self):

        self.env.text_age=TextInput(value='5e6', title="Age")
        self.env.text_metallicity=TextInput(value='0.01', title="Z")
        self.env.text_extinction_av=TextInput(value='0.372', title="Av")
        self.env.text_distance=TextInput(value='400', title="Distance")


        self.env.tb_isochrone=ColumnDataSource(data=dict(Gmag=[],bp_rp=[]))
        self.env.fig_hr.line('bp_rp', 'Gmag',color='green', alpha=1,source=self.env.tb_isochrone)

    def generate_isochrone(self):
        print('Generating Isochrone Wait')
        from ezpadova import parsec

        age=float(self.env.text_age.value)
        z=float(self.env.text_metallicity.value)
        av=float(self.env.text_extinction_av.value)
        dist=float(self.env.text_distance.value)
        print('isochrone parameter',age,z,av)
        data_frame=pd.DataFrame()
        data=parsec.get_one_isochrone(age, z, model='parsec12s', phot='gaia',extinction_av=av).to_pandas()
        data['bp_rp']=data['G_BP']-data['G_RP']
        data['Gmag']=data.G+5*np.log10(dist/10)
        data_frame=data[['bp_rp','Gmag']]
        iso=ColumnDataSource(data=data_frame.to_dict('list'))
        self.env.tb_isochrone.data=iso.data
        self.env.isochrone_data=data
        print('Isochrone parameter Created')
        self.env.catalog_find_from_isocrhone()


    def delete_isochrone(self):
        self.env.tb_isochrone.data=ColumnDataSource(data=dict(Gmag=[],bp_rp=[])).data
        self.env.isochrone_data=None
        self.env.catalog_find_from_isocrhone()


    def find_from_isocrhone(self):


        if not type(self.env.isochrone_data)==type(None):
            # data=self.env.isochrone_data
            # id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
            # new=mycatalog.pointer(id_mycatalog=id_mycatalog)
            # min=abs(data['bp_rp']-new['bp_rp'].values)
            # bp_rp_data=data.iloc[min.argmin()]
            # min=abs(data['Gmag']-new['Gmag'].values)
            # Gmag_data=data.iloc[min.argmin()]
            #
            # Gmag_text='Gmag '+self.find_values(data=Gmag_data)
            # bp_rp_text='bp_rp '+self.find_values(data=bp_rp_data)
            #
            # print('Find Values')
            # print(Gmag_text)
            # print(bp_rp_text)
            # self.env.text_banner_Gmag.text = Gmag_text
            # self.env.text_banner_bp_rp.text = bp_rp_text
            data=self.env.isochrone_data
            id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]

            # data['bp_rp']=data['G_BP']-data['G_RP']
            # data['Gmag']=data.G+5*np.log10(dist/10)
            source=mycatalog.pointer(id_mycatalog=id_mycatalog)

            x1=source['bp_rp'].values
            y1=source['Gmag'].values
            x2=data['bp_rp']
            y2=data['Gmag']
            slope=(y1-y2)/(x1-x2)
            angle=np.arctan(slope)*(180/np.pi)
            data['angle']=angle.abs()
            data['d']=np.sqrt((data['Gmag']-source['Gmag'].values)**2+(data['bp_rp']-source['bp_rp'].values)**2)
            a=x2-x1
            zero_crossings = np.where(np.diff(np.sign(a)))[0]
            index=np.append(zero_crossings,zero_crossings+1)
            new=data.loc[index]
            v_bp_rp=new.loc[new.d.idxmin()]
            a=y2-y1
            zero_crossings = np.where(np.diff(np.sign(a)))[0]
            index=np.append(zero_crossings,zero_crossings+1)
            new=data.loc[index]
            h_gmag=new.loc[new.angle.idxmin()]

            dmin=data.loc[data.d.idxmin()]

            Gmag_text='Gmag '+self.find_values(data=h_gmag,text=True)
            bp_rp_text='bp_rp '+self.find_values(data=v_bp_rp,text=True)
            dmin_text='d_min '+self.find_values(data=dmin,text=True)


            self.env.text_banner_Gmag.text = Gmag_text
            self.env.text_banner_bp_rp.text = bp_rp_text
            self.env.text_banner_dmin.text = dmin_text

        else:
            self.env.text_banner_Gmag.text = 'None'
            self.env.text_banner_bp_rp.text = 'None'
            self.env.text_banner_dmin.text = 'None'




    def find_values(self,data='',text=False):

        if text==True:
            Mini=str(data['M_ini'].round(4))
            Mact=str(data['M_act'].round(4))
            temp=str((10**(data['logTe'])).round(2))
            L=str((10**(data['logL/Lo'])).round(2))
            gmag=str(data['Gmag'].round(4))
            bprp=str(data['bp_rp'].round(4))

            vmax=equations.vmax(M=data['M_ini'],teff=10**(data['logTe']),L=(10**(data['logL/Lo'])))
            vmax=str(vmax)
            delta_v=equations.delta_v(M=data['M_ini'],teff=10**(data['logTe']),L=(10**(data['logL/Lo'])))
            delta_v=str(delta_v)
            text='Mini:'+Mini+' '+'Mact:'+Mact+' '+'Temp:'+temp+' '+'L/Lo:'+L+' '+'Gmag:'+gmag+' '+'bp_rp:'+bprp+' delta_v:'+delta_v+' vmax:'+vmax

            return text

        else:
            new={}
            # new['Mini']=data['M_ini']
            # new['Mact']=data['M_act']
            # new['teff']=(10**(data['logTe'])
            # new['L']=(10**(data['logL/Lo'])
            # new['gmag']=data['Gmag'].round(4)
            # new['bprp']=data['bp_rp'].round(4)

            return new

    def update_selection_program(self,attr,old,new):
        self.env.selection_program_text = self.env.selection_program.labels[self.env.selection_program.active]
        print('Program ',self.env.selection_program_text)

        if self.env.selection_program_text=='Catalog':
            self.env.custom_star_download_button.button_type='danger'
            #self.id_mycatalog=self.env.text_id_mycatalog_query.value
            self.catalog_all=mycatalog.pointer(catalog='mycatalog')
            # catalog_all=catalog_all.query('flag_source==1 & flag_duplicate==0').reset_index()
            # catalog_all=catalog_all.sort_values(by=['Gmag']).reset_index()
            self.env.catalog_main=self.catalog_all.query('cluster==@self.env.default_cluster').reset_index()

            if not self.env.text_catalog_query.value=='':
                print('Query',self.env.text_catalog_query.value)
                self.catalog_all=self.env.catalog_main.reset_index(drop=True).query(self.env.text_catalog_query.value).reset_index(drop=True)
                print(self.catalog_all)
            elif self.env.text_catalog_query.value=='':
                self.catalog_all=self.env.catalog_main

            self.id_mycatalog_all=self.catalog_all.id_mycatalog


            #self.update_catalog()

            self.id_mycatalog=self.id_mycatalog_all[0]
            print('Current id_mycatalog',self.id_mycatalog)
            mydata=mycatalog.pointer(id_mycatalog=self.id_mycatalog)
            print('Current data',mydata)
            sector_list=ast.literal_eval(mydata.Sector.values[0])
            sector_list=list([str(i) for i in sector_list])
            print('Sector_list',sector_list)
            self.env.int_select_sector.options=sector_list

            self.env.sector=sector_list[0]
            self.update_catalog()
            #tb_source_new = ColumnDataSource(data=dict(id_all=self.id_all,id_mycatalog_all=self.id_mycatalog_all,id=[self.id],id_mycatalog=[self.id_mycatalog]))
            #self.env.tb_source.data=tb_source_new.data
            self.env.tb_source.trigger("data",0,1)

        if self.env.selection_program_text=='Custom Star':
            c=2
            self.env.sector=1
            self.env.custom_star_download_button.button_type='success'
            tb_source_new = ColumnDataSource(data=dict(id_all=[0],id_mycatalog_all=['custom_star'],id=[0],id_mycatalog=['custom_star']))
            self.env.tb_source.data=tb_source_new.data
            print('tb_source from update_selection_program',self.env.tb_source.data)
            self.env.tb_source.trigger("data",0,1)

    def download_custom_star(self):
        print('Download for custom_star')
        if self.env.selection_program_text=='Custom Star':
            ra=float(self.env.text_custom_star_ra.value)
            dec=float(self.env.text_custom_star_dec.value)

            center = SkyCoord(ra=ra, dec=dec, unit=(unit.deg, unit.deg))
            star_all = eleanor.multi_sectors(coords=center,sectors='all')
            all_sectors=[star.sector for star in star_all]
            # update(id_mycatalog,Sector=str(all_sectors))
        # else:
        #     all_sectors=ast.literal_eval(sec)


            id_mycatalog='custom_star'
            for sector in all_sectors:
                print(sector)

                filename_ap=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_aperture',sector=int(sector))
                filename_ap_c=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_aperture_current',sector=int(sector))
                filename_flux=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_flux',sector=int(sector))
                filename_flux_current=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_flux_current',sector=int(sector))
                filename_time=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_time_flag',sector=int(sector))
                filename_header=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_header',sector=int(sector))
                filename_tpf=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_tpf',sector=int(sector))
                verbose=True
                utils.remove_file_if_exist(filename_ap,verbose=verbose)
                utils.remove_file_if_exist(filename_ap_c,verbose=verbose)
                utils.remove_file_if_exist(filename_flux,verbose=verbose)
                utils.remove_file_if_exist(filename_flux_current,verbose=verbose)
                utils.remove_file_if_exist(filename_time,verbose=verbose)
                utils.remove_file_if_exist(filename_header,verbose=verbose)
                utils.remove_file_if_exist(filename_tpf,verbose=verbose)

                # if os.path.exists(filename_ap):
                #     os.remove(filename_ap)
                #     print('Removing old aperture for custom star')
                # if os.path.exists(filename_ap_c):
                #     os.remove(filename_ap_c)
                #     print('Removing old aperture for custom star')

                my_file = Path(filename_ap)
                print(filename_ap)
                center = SkyCoord(ra=ra, dec=dec, unit=(unit.deg, unit.deg))
                star = eleanor.Source(coords=center,sector=int(sector))

                    # for star in star_all:
                        # print('%%%%%%%%sector',star.sector)
                data_post=eleanor.TargetData(star,do_psf=True,do_pca=True)
                mywcs=utils.extract_essential_wcs_postcard(data_post)
                radec=utils.pixe2radec(wcs=mywcs,aperture=data_post.aperture)
                for k in range(len(radec)):
                    aperture.add_aperture(id_mycatalog=id_mycatalog,ra=radec[k][0],dec=radec[k][1],name='eleanor_aperture',sector=int(sector))
                    aperture.add_aperture(id_mycatalog=id_mycatalog,ra=radec[k][0],dec=radec[k][1],name='eleanor_aperture_current',sector=int(sector))


                filename_flux=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_flux',sector=int(sector))
                filename_flux_current=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_flux_current',sector=int(sector))


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

                filename_tpf=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_tpf',sector=int(sector))
                np.save(filename_tpf,data_post.tpf)

                filename_header=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_header',sector=int(sector))
                data_post.header.totextfile(filename_header,overwrite=True)

                filename_time=mycatalog.filename(id_mycatalog=id_mycatalog,name='eleanor_time_flag',sector=int(sector))

                print('time_computing',id_mycatalog)
                data_frame=pd.DataFrame()
                q=data_post.quality == 0
                data_frame['time_flag']=q
                data_frame.to_csv(filename_time)




            self.env.sector=sector


            self.env.text_custom_star_sector.options=list([str(i) for i in all_sectors])

            self.env.custom_star_download_button.button_type='success'
            # self.env.tb_source.data["id"]=[0]
            # self.env.tb_source.data["id_all"]=[0]
            # self.env.tb_source.data["id_mycatalog_all"]=['custom_star']
            # self.env.tb_source.data["id_mycatalog"]=['custom_star']

            tb_source_new = ColumnDataSource(data=dict(id_all=[0],id_mycatalog_all=['custom_star'],id=[0],id_mycatalog=['custom_star']))
            self.env.tb_source.data=tb_source_new.data
            print('Current tbsource',self.env.tb_source.data)
            self.env.tb_source.trigger("data",0,1)





    def update_custom_sector(self,attr,old,new):
        if self.env.selection_program_text=='Custom Star':
            print('Updating custom sector')
            self.env.sector=int(self.env.text_custom_star_sector.value)

            self.env.custom_star_download_button.button_type='success'
            # self.env.tb_source.data["id"]=[0]
            # self.env.tb_source.data["id_all"]=[0]
            # self.env.tb_source.data["id_mycatalog_all"]=['custom_star']
            # self.env.tb_source.data["id_mycatalog"]=['custom_star']

            tb_source_new = ColumnDataSource(data=dict(id_all=[0],id_mycatalog_all=['custom_star'],id=[0],id_mycatalog=['custom_star']))
            self.env.tb_source.data=tb_source_new.data
            self.env.tb_source.trigger("data",0,1)
