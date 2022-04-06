import sys
from tessipack import eleanor
from env import Environment
import pandas as pd
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from bokeh.models import LogColorMapper, Slider, RangeSlider, \
    Span, ColorBar, LogTicker, Range1d, LinearColorMapper, BasicTicker
from bokeh.models.formatters import PrintfTickFormatter
from bokeh.models import Button , Select ,CheckboxGroup # for saving data
from astropy.utils.exceptions import AstropyUserWarning
import warnings

from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import CustomJS, TextInput, Paragraph
import os.path

from astropy.io.fits import Header
from tessipack.functions import utils
from tessipack.functions import aperture
from tessipack.My_catalog import mycatalog
from tessipack.functions import maths

import astropy.coordinates as Coord
import astropy

class Control_panel(Environment):
        env=Environment
        def __init__(self):
            self.id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]

            self.env.Samp_status = CheckboxGroup(labels=['SAMP'],active=[1])
            self.env.Samp_selection = Select(title='Distribution', options=['Server', 'Display'], value='Server')
            self.env.Control_function=self.controller_main
            self.diff_id_mycatalog=utils.difference(self.id_mycatalog)
            self.receiver_variable=None
            self.server=None
            self.display=None

            self.env.gaia_Gmag_start_text=TextInput(value='', title="Gmag Start")
            self.env.gaia_Gmag_end_text=TextInput(value='', title="Gmag End")
            self.env.gaia_radius_text=TextInput(value='1', title="Radius(arcmin)")



            self.id_temp=utils.difference(self.id_mycatalog)

            self.catalog_all=mycatalog.pointer(catalog='mycatalog')
            self.prgm_ds9=None

            from astropy.samp import SAMPHubServer
            hub = SAMPHubServer()
            hub.start()


            self.initiate_ds9()


            self.env.gaia_update_button = Button(label="Update Gaia catalog", button_type="success",width=150)
            #print('testingggggggggg',self.env.gaia_update_button)
            self.env.gaia_update_button.on_click(self.setup_whole_gaia)

        def controller_main(self):
            self.id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]


            if self.env.ds9_status.active[0]==0 & self.env.Samp_status.active[0]==0:
                if self.diff_id_mycatalog.on_change(self.id_mycatalog):
                    self.ds9_command()
                    self.controller_samp_send()

            elif self.env.ds9_status.active[0]==0:
                if self.diff_id_mycatalog.on_change(self.id_mycatalog):
                    self.ds9_command()

            elif self.env.Samp_status.active[0]==0:
                print('Samp Active here')
                if self.diff_id_mycatalog.on_change(self.id_mycatalog):
                    print('Samp send here')

                    self.controller_samp_send()
            self.controller_samp_recieve()

            self.env.interactive_file_control=-1




        def controller_samp_send(self):
            if self.env.Samp_status.active[0]==0:
                # print('Samp Activated')
                if self.env.Samp_selection.value=='Server':
                    print('selection',self.env.Samp_selection.value)
                    if self.server==None:
                        from astropy.samp import SAMPIntegratedClient
                        self.server = SAMPIntegratedClient(name='Server')
                        self.server.connect()

                    self.message_send()


        def controller_samp_recieve(self):
            if self.env.Samp_status.active[0]==0:
                if self.env.Samp_selection.value=='Display':
                    if self.display==None:
                        print()
                        from astropy.samp import SAMPIntegratedClient
                        self.display = SAMPIntegratedClient(name='Display')
                        self.display.connect()
                        self.receiver_variable=utils.Receiver(self.display)
                        self.display.bind_receive_call("samp.id_mycatalog", self.receiver_variable.receive_call)
                        self.display.bind_receive_notification("samp.id_mycatalog", self.receiver_variable.receive_notification)

                    if self.receiver_variable.received:
                        if self.id_temp.on_change(self.receiver_variable.params['id_mycatalog']):
                            recieved_id_mycatalog=self.receiver_variable.params['id_mycatalog']
                            print('Display recieved id_mycatalog',self.receiver_variable.params['id_mycatalog'])

                            query=self.env.catalog_main.query('id_mycatalog==@recieved_id_mycatalog')
                            print('values_found',query.index.values[0])
                            self.env.tb_source.data["id"][0]=query.index.values[0]
                            self.env.tb_source.trigger("data",0,1)

                    #print(.received,self.display.get_registered_clients())
                    #self.message_recieve()

        def message_send(self):
            '''Send Message'''
            params = {}
            params["id_mycatalog"] = self.id_mycatalog

            message = {}
            message["samp.mtype"] = "samp.id_mycatalog"
            message["samp.params"] = params
            print(self.server.get_registered_clients())
            self.server.notify_all(message)
            print('Message Sent from Server')


        def message_recieve(self):
            r=utils.Receiver(self.display)
            self.display.bind_receive_call("samp.id_mycatalog", r.receive_call)
            self.display.bind_receive_notification("samp.id_mycatalog", r.receive_notification)
            print(r.received,self.display.get_registered_clients(),r.params)
            new=utils.difference(self.id_mycatalog)



        def initiate_ds9(self):
            self.env.ds9_status = CheckboxGroup(labels=['Ds9'],active=[1])
            self.env.ds9_command_button= Button(label="Send Command: DS9", button_type="success",width=150)
            self.env.ds9_command_button.on_click(self.ds9_command)
            self.env.ds9_catalog_status = CheckboxGroup(labels=['Catalog'],active=[1])



        def ds9_command(self):
            # if self.prgm_ds9==None:
            id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
            w=mycatalog.pointer(catalog='mycatalog',id_mycatalog=id_mycatalog)
            print('HIII',w.id_mycatalog.values[0],w.id_gaia.values[0])
            ra=str(w.RA_ICRS.values[0])
            dec=str(w.DE_ICRS.values[0])
            import pyds9
            self.prgm_ds9 = pyds9.DS9()
            filename=mycatalog.filename(name='cluster_ffi',id_mycatalog=id_mycatalog,sector=self.env.sector)[0]
            command='file '+filename
            # print(command)
            #self.prgm_ds9.set('file /media/dinilbose/Masterdisk/Works_data/FFI/IC_2714/tess2019089012934-s0010-3-1-0140-s_ffic.fits')
            self.prgm_ds9.set(command)
            #command='regions command "fk5;circle 168.18 -64.17 0.00291'' # color=red"'
            command='regions command "fk5;circle '+ra+' '+dec+' 0.00291'' # color=red"'
            #print(command)
            self.prgm_ds9.set(command)
            command='pan to '+ra+' '+dec+' wcs fk5'
            #print(command)
            self.prgm_ds9.set(command)
            #command='regions command "fk5;text '+ra+' '+dec+' '+id_mycatalog+'"'
            #self.prgm_ds9.set(command)

            #Think about changing to load using vot
            #new = self.env.catalog_main.rename(columns={'RA_ICRS':'_RAJ2000','DE_ICRS':'_DEJ2000'})
            #new.to_csv('/home/dinilbose/PycharmProjects/light_cluster/test.tsv',index=False,sep='\t')
            #print('hiiiiiiiiiiiiiiiiiiii',new)


            if type(self.env.catalog_whole_gaia)==type(None):
                # self.env.catalog_whole_gaia=pd.read_csv(mycatalog.filename(name='whole_gaia_data',id_mycatalog=id_mycatalog))
                self.setup_whole_gaia()


            if self.env.ds9_catalog_status.active[0]==0:
                self.prgm_ds9.set('catalog close')

                print('Catalog Active')
                id_coord=Coord.SkyCoord(w.RA_ICRS.values[0],w.DE_ICRS.values[0], frame='icrs', unit='deg')
                data_coord=Coord.SkyCoord(self.env.catalog_whole_gaia['_RAJ2000'].values, self.env.catalog_whole_gaia['_DEJ2000'].values, frame='icrs', unit='deg')
                array=data_coord.separation(id_coord) < float(self.env.gaia_radius_text.value)*astropy.units.arcmin
                filtered=self.env.catalog_whole_gaia[array].reset_index(drop=True)
                name=mycatalog.filename(name='temp',sector=self.env.sector).joinpath('whole_gaia.csv')
                filtered['id_mycatalog']=0
                # new=filtered.query('id_gaia==@self.catalog_all.id_gaia')
                # filtered[new.index]=new.id_gaia
                for i in filtered.index.values:
                    id_test=filtered.loc[i].id_gaia
                    # print('hiiiiiiiiiii',id_test)
                    # print('print',self.catalog_main.query('id_gaia==@id_test').id_gaia,self.catalog_main.query('id_gaia==@id_test').id_mycatalog)
                    #print(id_test)
                    k=self.catalog_main.query('id_gaia==@id_test').id_mycatalog.values
                    if len(k)>0:
                        #print('test',k[0])
                        filtered.loc[i,'id_mycatalog']=k[0]
                # print(filtered)
                self.env.whole_gaia_filter=filtered[['Gmag','bp_rp','id_mycatalog','_RAJ2000','_DEJ2000']]

                self.env.whole_gaia_filter.to_csv(name,index=False,sep='\t')
                command='catalog import tsv '+str(name)
                self.prgm_ds9.set(command)

            print('Ds9 Command finish')


        def setup_whole_gaia(self):
            '''Setup gaia catalog'''

            id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]

            self.env.catalog_whole_gaia=pd.read_csv(mycatalog.filename(name='whole_gaia_data',id_mycatalog=id_mycatalog,sector=self.env.sector))

            if not self.env.gaia_Gmag_start_text.value=='':
                value=float(self.env.gaia_Gmag_start_text.value)
                print('start',value)
                self.env.catalog_whole_gaia=self.env.catalog_whole_gaia.query('Gmag>@value')
            if not self.env.gaia_Gmag_end_text.value=='':
                value=float(self.env.gaia_Gmag_end_text.value)
                print('End',value)
                self.env.catalog_whole_gaia=self.env.catalog_whole_gaia.query('Gmag<@value')
            print('Updating gaia')
