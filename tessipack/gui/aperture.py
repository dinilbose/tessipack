import sys
import os
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
from bokeh.models import Button  # for saving data
from astropy.utils.exceptions import AstropyUserWarning
import warnings
from bokeh.models import CustomJS, TextInput, Paragraph,CheckboxGroup
from bokeh.events import ButtonClick
from bokeh.layouts import row,column


from astropy.io.fits import Header
from tessipack.My_catalog import mycatalog
from tessipack.functions import utils
from tessipack.functions import aperture
from tessipack.functions import flags
import os
from bokeh.models import ColumnDataSource, Label, LabelSet, Range1d

from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import Button, Select,CheckboxGroup  # for saving data

import astropy.coordinates as Coord
import astropy
import ast
class Aperture(Environment):
    env=Environment
    def __init__(self):
        env=Environment
        self.id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
        self.env.tb_nearby_star = ColumnDataSource(data=dict(pixel_x=[6],
                                            pixel_y=[6],
                                            id_mycatalog_names=[''],Gmag=[0]))


        self.env.tb_nearby_star_table=self.intiate_nearby_star_table()



        self.env.tb_lightcurve,self.env.fig_lightcurve=self.initiate_lk(self.id_mycatalog)
        self.env.tb_tpf,self.env.tpf_flux=self.initiate_tpf(self.id_mycatalog)
        self.env.fig_tpf,self.env.fig_stretch=self.make_tpf_figure_elements(self.env.tb_tpf,self.env.tpf_flux,fiducial_frame=self.env.fiducial_frame)
        self.env.tb_nearby=ColumnDataSource(data=dict(x=[],y=[]))
        self.env.fig_tpf.rect("x","y",1,1,source=self.env.tb_nearby,line_color='red',line_width=2,name='nearby',alpha=1,fill_color=None)


        # self.env.fig_tpf.rect(6.5,6.5,1,1,line_color='red',line_width=2,name='add',alpha=1,fill_color=None)

        self.time_default=self.env.tb_lightcurve.data["time"]
        tpf_index_lookup = {cad: idx for idx, cad in enumerate(self.time_default)}
        self.no_frame=np.arange(0,self.env.tpf_flux.shape[0])

        # Interactive slider widgets and buttons to select the cadence number

        self.env.cadence_slider,self.env.vertical_line=self.initiate_cadance_slider()

        self.env.fig_lightcurve.add_layout(self.env.vertical_line)
        self.env.Generate_lc_button = Button(label="Generate LC", button_type="success",width=150)
        self.env.Generate_lc_button.on_click(self.generate_lk)

        self.env.reset_axis_lc_button = Button(label="Reset LC Axes", button_type="success",width=150)
        self.env.reset_axis_lc_button.on_click(self.axes_reset_lc)
        self.env.reset_dflt_lc_button = Button(label="Reset LC", button_type="success",width=150)
        self.env.reset_dflt_lc_button.on_click(self.reset_default_lc)

        self.env.next_button = Button(label="Next Source", button_type="success",width=150)
        self.env.next_button.on_click(self.next)

        self.env.previous_button = Button(label="Previous Source", button_type="success",width=150)
        self.env.previous_button.on_click(self.previous)

        self.env.toggle_button = Button(label="Toggle", button_type="warning",width=60)
        self.env.toggle_button.on_event(ButtonClick,self.show_spinner)
        self.env.toggle_button.on_click(self.draw_nearby_button)


        self.env.save_userinput_button = Button(label="Save User Input", button_type="success",width=100)
        self.env.save_userinput_button.on_click(self.save_userinput)

        self.env.save_current_button = Button(label="Save Current Ap", button_type="success",width=100)
        self.env.save_current_button.on_click(self.save_current)

        # draw_nearby_button(self)
        #
        self.draw_nearbysource()
        self.text_format()
        self.env.text_banner = Paragraph(text=self.env.Message, width=1100, height=30)
        self.initiate_userinput()

        self.env.show_spinner_button = Button(label='Show Spinner', width=100)
        self.env.show_spinner_button.on_click(self.show_spinner)



        if self.env.aperture_setting==1:
            ap_label='Aperture_current'
        else:
            ap_label='Aperture_default'

        self.env.aperture_selection_button = Button(label=ap_label, button_type="warning",width=100)
        self.env.aperture_selection_button.on_click(self.aperture_selection_function)

        self.env.tb_source.on_change('data',self.update_all)

        self.update_tb_nearby_star() #drawing neabystars name
        self.initiate_errorbar()
        self.env.show_error_bar = CheckboxGroup(labels=['Errorbar'],active=[1],height=10,width=10)
        self.env.show_error_bar.on_change('active',self.update_lk_error_bar)
        self.env.tb_lightcurve.on_change('data',self.update_lk_error_bar)



        self.env.int_select_sector.on_change('value',self.update_sector_change)


        #print(sector_list)

        #new_star
        # self.env.tb_nearby_star=ColumnDataSource(data=dict(id_mycatalog=['test'],pixel_x=[6],pixel_y=[6]))



        # labels = LabelSet(x='pixel_x', y='pixel_y', text='id_mycatalog', level='glyph',source=self.env.tb_nearby_star, render_mode='canvas')
        # labels = Label(x=15, y=15, text='hii')

        # self.env.fig_tpf.add_layout(labels)

    def show_spinner(self):
        self.env.div_spinner.text = self.env.spinner_text
    def hide_spinner(self):
        self.env.div_spinner.text = ""

        #hide_spinner_button = Button(label='Hide Spinner', width=100)
        #hide_spinner_button.on_click(hide_spinner)
    def intiate_nearby_star_table(self):
        columns = [
            TableColumn(field="id_mycatalog_names", title="id_mycatalog"),
            TableColumn(field="pixel_x", title="pixel_x"),
            TableColumn(field="pixel_y", title="pixel_y"),
            TableColumn(field="Gmag", title="Gmag")
        ]
        table = DataTable(
            source=self.env.tb_nearby_star,
            columns=columns,
            width=300,
            height=300,
            sortable=True,
            selectable=True,
            editable=False,
        )
        return table


    def initiate_lk(self,id_mycatalog):
        # id_mycatalog=self.env.tb_source.data['id_mycatalog']
        if self.env.aperture_setting==-1:
        # print('#Setting eleanor_aperture_default')
            Data=pd.read_csv(mycatalog.filename(name='eleanor_flux',id_mycatalog=id_mycatalog,sector=self.env.sector))
            self.env.current_flux_dataframe=Data

        elif self.env.aperture_setting==1:
        # print('#Setting eleanor_aperture_current')
            self.create_current_files(name='eleanor_flux_current',id_mycatalog=id_mycatalog)
            Data=pd.read_csv(mycatalog.filename(name='eleanor_flux_current',id_mycatalog=id_mycatalog,sector=self.env.sector))
            self.env.current_flux_dataframe=Data

        details=mycatalog.pointer(catalog='mycatalog',id_mycatalog=id_mycatalog)
        clus=details.cluster.values[0]

        Data=flags.apply_flag(filename=self.env.extra_flag_file,data=Data,cluster=clus,apply_cluster=True)

        #print('Newwww Frameeeee',len(Data))

        flux_name='pca_flux'
        time=Data.time
        flux=Data[flux_name]
        flux_err=Data['flux_err']

        time_flag=Data['time_flag']==0

        #old version with time_flag as separate file
        #time_flag_frame=pd.read_csv(mycatalog.filename(name='eleanor_time_flag',id_mycatalog=id_mycatalog,sector=self.env.sector))
        #time_flag=time_flag_frame.time_flag.values

        data_new=utils.flux_filter_type(time_flag=time_flag,func='median',deviation='mad',sigma=self.env.sigma,time=time,flux=flux,flux_name='pca_flux',flux_err=flux_err).reset_index()
        # print(data_new)
        tb_lightcurve = ColumnDataSource(data=dict(time=list(data_new.time.values), flux=list(data_new.pca_flux.values),flux_err=list(data_new.flux_err.values)))
        TOOLTIPS = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
        ]
        fig_lightcurve = figure(
            plot_width=self.env.plot_width,
            plot_height=self.env.plot_height,
            tools=["box_zoom", "wheel_zoom","lasso_select", "tap" ,"reset", "save"],
            title="Light Curve",tooltips=TOOLTIPS,
        )
        fig_lightcurve.circle("time", "flux", source=tb_lightcurve, alpha=0.7,**self.env.selection,)
        fig_lightcurve.line("time","flux",source=tb_lightcurve,alpha=0.7,color="#1F77B4")

        #y_err_x, y_err_y=utils.bokeh_errorbar(x=data_new.time,y=data_new[flux_name],yerr=flux_err)


        return tb_lightcurve, fig_lightcurve









    def update_upon_cadence_change(self,attr, old, new):
        """Callback to take action when cadence slider changes"""
        frameno=new
        # print(self.pedestal)
        self.env.fig_tpf.select('tpfimg')[0].data_source.data['image'] = \
            [self.env.tpf_flux[frameno, :, :] + self.pedestal]
        self.env.vertical_line.update(location=self.time_default[frameno])

    def initiate_cadance_slider(self):
        cadence_slider = Slider(start=np.min(self.no_frame),
                                end=np.max(self.no_frame),
                                value=np.min(self.no_frame),
                                step=1,
                                title="Cadence Number",
                                width=490,sizing_mode='stretch_both')
        cadence_slider.on_change('value', self.update_upon_cadence_change)
        vertical_line = Span(location=self.env.tb_lightcurve.data["time"][0], dimension='height',
                            line_color='firebrick', line_width=4, line_alpha=0.5)
        return cadence_slider,vertical_line

    def initiate_tpf(self,id_mycatalog):
        tpf_default_filename=mycatalog.filename(id_mycatalog=id_mycatalog, name='eleanor_tpf',sector=self.env.sector)
        tpf_flux=np.load(tpf_default_filename)
        # time_default_filename=mycatalog.filename(id_mycatalog=id_mycatalog, name='eleanor_flux')
        # all_value=pd.read_csv(time_default_filename)
        header_default_filename=mycatalog.filename(name='eleanor_header',id_mycatalog=id_mycatalog,sector=self.env.sector)
        head=Header.fromtextfile(header_default_filename)
        tess_wcs=utils.extract_essential_wcs_postcard(head,header=True)
        # aperture_setting=0
        if self.env.aperture_setting==-1:
            # print('#Setting eleanor_aperture_default')
            aper=aperture.load_aperture(name='eleanor_aperture',id_mycatalog=id_mycatalog,sector=self.env.sector)
        elif self.env.aperture_setting==1:
            # print('#Setting eleanor_aperture_current')
            self.create_current_files(name='eleanor_aperture_current',id_mycatalog=id_mycatalog)
            aper=aperture.load_aperture(name='eleanor_aperture_current',id_mycatalog=id_mycatalog,sector=self.env.sector)

        pixels=utils.radec2pixel(wcs=tess_wcs,ra=aper.ra.values,dec=aper.dec.values)
        custom_aperture=aperture.create_mask(x=tpf_flux.shape[1],y=tpf_flux.shape[2])
        myaperture=[]
        for pix in pixels:
            # print(pix)
            aperture_mask=aperture.replace_mask(mask=custom_aperture,x=pix[0],y=pix[1],value=True)
        npix =tpf_flux[0, :, :].size
        pixel_index_array = np.arange(0, npix, 1).reshape(tpf_flux[0].shape)
        tools='tap,box_select,wheel_zoom,reset'
        big_flux_array=tpf_flux[0,:,:].reshape(-1)
        star=None
        column=1
        row=1
        npix = tpf_flux[0, :, :].size
        pixel_index_array = np.arange(0, npix, 1).reshape(tpf_flux[0].shape)
        xx = column + np.arange(tpf_flux.shape[1])
        yy = row + np.arange(tpf_flux.shape[2])
        xa, ya = np.meshgrid(xx, yy)
        tb_tpf = ColumnDataSource(data=dict(xx=xa-0.5, yy=ya-0.5))
        tb_tpf.selected.indices = pixel_index_array[aperture_mask].reshape(-1).tolist()
        tb_tpf_default=[]
        xx = column + np.arange(tpf_flux.shape[1])
        yy = row + np.arange(tpf_flux.shape[2])
        xa, ya = np.meshgrid(xx, yy)
        return tb_tpf,tpf_flux



    def make_tpf_figure_elements(self,tb_tpf,tpf_flux, pedestal=None, fiducial_frame=None,
                                 plot_width=370, plot_height=340, scale='log', vmin=None, vmax=None,
                                 cmap='Viridis256', tools='tap,box_select,wheel_zoom,reset'):
        if pedestal is None:
            self.pedestal = -np.nanmin(tpf_flux) + 1
        if scale == 'linear':
            self.pedestal = 0

        column=0
        row=0
        title = "Pixel data"
        fig = figure(plot_width=600, plot_height=600,
                     x_range=(column, column+tpf_flux.shape[2]),
                     y_range=(row, row+tpf_flux.shape[1]),
                     title=title, tools=tools,
                     toolbar_location="below",
                     border_fill_color="whitesmoke")

        fig.yaxis.axis_label = 'Pixel Row Number'
        fig.xaxis.axis_label = 'Pixel Column Number'

        vlo, lo, hi, vhi = np.nanpercentile(tpf_flux+ self.pedestal, [0.2, 1, 95, 99.8])
        if vmin is not None:
            vlo, lo = vmin, vmin
        if vmax is not None:
            vhi, hi = vmax, vmax

        if scale == 'log':
            vstep = (np.log10(vhi) - np.log10(vlo)) / 300.0  # assumes counts >> 1.0!
        if scale == 'linear':
            vstep = (vhi - vlo) / 300.0  # assumes counts >> 1.0!

        if scale == 'log':
            color_mapper = LogColorMapper(palette=cmap, low=lo, high=hi)
        elif scale == 'linear':
            color_mapper = LinearColorMapper(palette=cmap, low=lo, high=hi)
        else:
            raise ValueError('Please specify either `linear` or `log` scale for color.')


        fig.image([tpf_flux[fiducial_frame, :, :] + self.pedestal], x=column, y=row,
                  dw=tpf_flux.shape[2], dh=tpf_flux.shape[1], dilate=True,
                  color_mapper=color_mapper, name="tpfimg")

        if scale == 'log':
            ticker = LogTicker(desired_num_ticks=8)
        elif scale == 'linear':
            ticker = BasicTicker(desired_num_ticks=8)

        color_bar = ColorBar(color_mapper=color_mapper,
                             ticker=ticker,
                             label_standoff=-10, border_line_color=None,
                             location=(0, 0), background_fill_color='whitesmoke',
                             major_label_text_align='left',
                             major_label_text_baseline='middle',
                             title='e/s', margin=0)
        fig.add_layout(color_bar, 'right')

        color_bar.formatter = PrintfTickFormatter(format="%14i")

        if tb_tpf is not None:
            fig.rect('xx', 'yy', 1, 1, source=tb_tpf, fill_color='gray',
                    fill_alpha=0.4, line_color='white',name='new')
            # source = ColumnDataSource(data=dict(height=[6, 1, 7, 6, 8, 6],
            #                                     weight=[6, 1, 2, 4, 6, 7],
            #                                     names=['Mark', 'Amir', 'Matt', 'Greg',
            #                                            'Owen', 'Juan']))
            #
            # source = ColumnDataSource(data=dict(pixel_x=[6],
            #                                     pixel_y=[6],
            #                                     id_mycatalog_names=['Mark']))
            # labels = Label(x=10, y=10, text='hii')
            # labels = LabelSet(x='weight', y='height', text='names',x_offset=1, y_offset=1, source=source)
            # labels = LabelSet(x='pixel_x', y='pixel_y', x_offset=1, y_offset=1, text='id_mycatalog_names',source=self.env.tb_nearby_star)
            # print(self.env.tb_nearby_star.data)
            labels = LabelSet(x='pixel_x', y='pixel_y', x_offset=1, y_offset=1, text='id_mycatalog_names',source=self.env.tb_nearby_star)

            fig.add_layout(labels)

        # if tpf_source_default is not None:
        #     v=tpf_source_default.data['xxx']
        #     vv=tpf_source_default.data['yyy']
        #     fig.rect('xxx', 'yyy', 1, 1, source=tpf_source_default, fill_color='gray',
        #             fill_alpha=0.4, line_color='red',name='old')

        # Configure the stretch slider and its callback function
        if scale == 'log':
            start, end = np.log10(vlo), np.log10(vhi)
            values = (np.log10(lo), np.log10(hi))
        elif scale == 'linear':
            start, end = vlo, vhi
            values = (lo, hi)

        stretch_slider = RangeSlider(start=start,
                                     end=end,
                                     step=vstep,
                                     title='Screen Stretch ({})'.format(scale),
                                     value=values,
                                     orientation='horizontal',
                                     width=200,
                                     direction='ltr',
                                     show_value=True,
                                     sizing_mode='stretch_both',
                                     height=15,
                                     name='tpfstretch')

        def stretch_change_callback_log(attr, old, new):
            """TPF stretch slider callback."""
            fig.select('tpfimg')[0].glyph.color_mapper.high = 10**new[1]
            fig.select('tpfimg')[0].glyph.color_mapper.low = 10**new[0]

        def stretch_change_callback_linear(attr, old, new):
            """TPF stretch slider callback."""
            fig.select('tpfimg')[0].glyph.color_mapper.high = new[1]
            fig.select('tpfimg')[0].glyph.color_mapper.low = new[0]

        if scale == 'log':
            stretch_slider.on_change('value', stretch_change_callback_log)
        if scale == 'linear':
            stretch_slider.on_change('value', stretch_change_callback_linear)
        # print(fig)
        return fig, stretch_slider

    def generate_lk(self,star):
        print("Generate lk button is pressed")
        id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
        a,b=np.unravel_index(self.env.tb_tpf.selected.indices,[13,13])
        new_pixel=np.column_stack((a,b))
        custom_aperture=aperture.create_mask(x=self.env.tpf_flux.shape[1],y=self.env.tpf_flux.shape[2])
        myaperture=[]
        for pix in new_pixel:
            # print(pix)
            custom_aperture=aperture.replace_mask(mask=custom_aperture,y=pix[0],x=pix[1],value=True)
        default_ape=['New_aperture',custom_aperture]
        myaperture=[default_ape]

        if self.env.selection_program_text=='Custom Star':
            print('Custom Star is Selected')
            ra=float(self.env.text_custom_star_ra.value)
            dec=float(self.env.text_custom_star_dec.value)
            star=self.eleanor_lookup(ra=ra,dec=dec)
        else:
            star=self.eleanor_lookup()
        # if not star==None:
            # star=eleanor_lookup()
    #         if not eln_object==None:
    #             eln_object=eleanor.TargetData(star,do_psf=True,do_pca=True)
        print('Eleanor_version',eleanor.__file__)
        eln_object = eleanor.TargetData(star,do_psf=True,do_pca=True,other_aperture=myaperture)
        # print('Background########',eln_object.bkg_type)
        flux=utils.extract_flux_ap(data=eln_object,name='New_aperture',flux_name='pca_flux',bkg_type=eln_object.bkg_type)
        time=eln_object.time
        flux_name='pca_flux'


        data_frame=pd.DataFrame()
        data_frame['time']=time
        data_frame['corr_flux']=utils.extract_flux_ap(data=eln_object,name='New_aperture',flux_name='corr_flux',bkg_type=eln_object.bkg_type)
        data_frame['pca_flux']=utils.extract_flux_ap(data=eln_object,name='New_aperture',flux_name='pca_flux',bkg_type=eln_object.bkg_type)
        #psf_flux not avaialble for New_aperture
        data_frame['psf_flux']=np.nan
        data_frame['flux_err']=eln_object.flux_err
        # self.env.current_flux_dataframe=data_frame


        # time_flag_frame=pd.read_csv(mycatalog.filename(name='eleanor_time_flag',id_mycatalog=id_mycatalog))
        # time_flag=time_flag_frame.time_flag.values

        time_flag_frame=pd.DataFrame()
        time_flag=eln_object.quality == 0
        # data_frame['time_flag']=time

        # all_value_n=utils.flux_filter_type(time_flag=time_flag,func='median',deviation='mad',sigma=self.env.sigma,time=time,flux=flux,flux_name='pca_flux',flux_err=eln_object.flux_err).reset_index()
        # from importlib import reload
        # reload(utils)
        print('data_frame',data_frame)
        # print('time_frame',time_flag_frame)
        print('data_frame',data_frame)

        all_value_n=utils.flux_filter_type(time_flag=time_flag,func='median',deviation='mad',sigma=self.env.sigma,data=data_frame,flux_name='pca_flux').reset_index()

        # print('time',time_flag,len(time),len(all_value_n))
        time_n=list(all_value_n.time.values)
        flux_n=list(all_value_n.pca_flux.values)
        flux_err_n=list(all_value_n.flux_err.values)


        data_frame=data_frame.query('time==@time_n')
        details=mycatalog.pointer(catalog='mycatalog',id_mycatalog=id_mycatalog)
        clus=details.cluster.values[0]
        self.env.current_flux_dataframe=flags.apply_flag(filename=self.env.extra_flag_file,data=all_value_n,cluster=clus,apply_cluster=True)

        tb_lightcurve_n = ColumnDataSource(data=dict(time=time_n, flux=flux_n,flux_err=flux_err_n))
        self.env.tpf_flux= eln_object.tpf

        self.env.tb_lightcurve.data=tb_lightcurve_n.data

        npix = self.env.tpf_flux[0, :, :].size
        pixel_index_array = np.arange(0, npix, 1).reshape(self.env.tpf_flux[0].shape)
        tools='tap,box_select,wheel_zoom,reset'
        big_flux_array=self.env.tpf_flux[0,:,:].reshape(-1)
        star=None
        column=1
        row=1
        npix =self.env.tpf_flux[0, :, :].size
        pixel_index_array = np.arange(0, npix, 1).reshape(self.env.tpf_flux[0].shape)
        xx = column + np.arange(self.env.tpf_flux.shape[2])
        yy = row + np.arange(self.env.tpf_flux.shape[1])
        xa, ya = np.meshgrid(xx, yy)
        tpf_source_n = ColumnDataSource(data=dict(xx=xa-0.5, yy=ya-0.5))
        #Repalace aperture mask with custom aperture
        aperture_mask=custom_aperture
        tpf_source_n.selected.indices = pixel_index_array[aperture_mask].reshape(-1).tolist()
        xx = column + np.arange(self.env.tpf_flux.shape[2])
        yy = row + np.arange(self.env.tpf_flux.shape[1])
        xa, ya = np.meshgrid(xx, yy)
        tpf_source_default_n = ColumnDataSource(data=dict(xxx=xa-0.5, yyy=ya-0.5))
        tpf_source_default_n.selected.indices = pixel_index_array[aperture_mask].reshape(-1).tolist()
        self.env.tb_tpf.selected=tpf_source_default_n.selected
        self.env.tb_tpf.data=tpf_source_default_n.data
        print("Generating Light curve Finished")

        return 0


    def save_current(self):
        '''Saving the current settings to a file'''
        print('Save current button is pressed')

        if not type(self.env.current_flux_dataframe)==type(None) or self.env.aperture_setting==-1:
            self.env.current_flux_dataframe.to_csv(mycatalog.filename(name='eleanor_flux_current',id_mycatalog=self.id_mycatalog,sector=self.env.sector),index=False)
            a,b=np.unravel_index(self.env.tb_tpf.selected.indices,[13,13])
            new_pixel=np.column_stack((a,b))
            custom_aperture=aperture.create_mask(x=self.env.tpf_flux.shape[1],y=self.env.tpf_flux.shape[2])
            for pix in new_pixel:
                custom_aperture=aperture.replace_mask(mask=custom_aperture,y=pix[0],x=pix[1],value=True)
            header_default_filename=mycatalog.filename(name='eleanor_header',id_mycatalog=self.id_mycatalog,sector=self.env.sector)
            head=Header.fromtextfile(header_default_filename)
            tess_wcs=utils.extract_essential_wcs_postcard(head,header=True)
            radec=utils.pixe2radec(wcs=tess_wcs,aperture=custom_aperture)
            filename=mycatalog.filename(name='eleanor_aperture_current',id_mycatalog=self.id_mycatalog,sector=self.env.sector)
            if os.path.exists(filename):
                os.remove(filename)
            for k in range(len(radec)):
                aperture.add_aperture(name='eleanor_aperture_current',id_mycatalog=self.id_mycatalog,sector=self.env.sector,ra=radec[k][0],dec=radec[k][1])
            if self.env.aperture_setting==-1:
                print('Current aperture and lightcurve replaced by default')
            else:
                print('New aperture and lightcurve saved')

        # elif aperture_setting=-1:
            # print('Default and lightcurve saved')

            self.env.current_flux_dataframe.to_csv(mycatalog.filename(name='eleanor_flux_current',id_mycatalog=self.id_mycatalog,sector=self.env.sector),index=False)



        else:
            print('New light Curve is not generated: Cannot save aperture')

    def get_lightcurve_y_limits(self,tb_lightcurve):
        from astropy.stats import sigma_clip
        with warnings.catch_warnings():  # Ignore warnings due to NaNs
            warnings.simplefilter("ignore", AstropyUserWarning)
            # flux = sigma_clip(lc_source.data['flux'], sigma=5, masked=False)
            flux=tb_lightcurve.data['flux']
        low, high = np.nanpercentile(flux, (1, 99))
        margin = 0.30 * (high - low)
        high=tb_lightcurve.data['flux'].max()
        min=tb_lightcurve.data['flux'].min()
        high=high+high*0.10
        low=low-low*0.10
        return low, high

    def axes_reset_lc(self):
        ylims = self.get_lightcurve_y_limits(self.env.tb_lightcurve)
        self.fig_lightcurve.x_range.start=self.env.tb_lightcurve.data["time"].min()-1
        self.fig_lightcurve.x_range.end=self.env.tb_lightcurve.data["time"].max()+1
        self.fig_lightcurve.y_range.start = ylims[0]
        self.fig_lightcurve.y_range.end = ylims[1]

    def eleanor_lookup(self,ra=None,dec=None):
        id_mycatalog=self.env.tb_source.data['id_mycatalog']
        # print('##################Looking for',id_mycatalog)
        #Change of plan mycatalog contains all RA DEC infos
        # eleanor_lookup_catlog=mycatalog.pointer(catalog='gaia',id_mycatalog=id_mycatalog)
        print('')
        if type(ra)==type(None):

            eleanor_lookup_catlog=mycatalog.pointer(catalog='mycatalog',id_mycatalog=id_mycatalog)
            dec=eleanor_lookup_catlog.DE_ICRS.values
            ra=eleanor_lookup_catlog.RA_ICRS.values
        else:
            print('Custom aperture ra:',ra,' dec:',dec)
            ra=[ra]
            dec=[dec]

        center = SkyCoord(ra=ra[0], dec=dec[0], unit=(u.deg, u.deg))

        if not type(self.env.sector)==type(1):
            print('sector selected is', self.env.sector)



        star = eleanor.Source(coords=center,sector=int(self.env.sector))
        #star = eleanor.multi_sectors(coords=center,sectors='all')
        #least_sector=np.array([i.sector for i in star]).argmin()
        #star = star[least_sector]
        print('Generated light curve Current Sector',star.sector)
        return star

    def reset_default_lc(self):
        id_mycatalog=self.id_mycatalog
        tb_lightcurve,fig_lightcurve=self.initiate_lk(self.id_mycatalog)
        print('default###id_mycatalog_lightcurve',id_mycatalog)
        tb_tpf,tpf_flux=self.initiate_tpf(self.id_mycatalog)
        self.env.tb_lightcurve.data=tb_lightcurve.data
        self.env.tb_tpf.data=tb_tpf.data
        self.env.tb_tpf.selected=tb_tpf.selected
        self.env.tpf_flux=tpf_flux
        fig_tpf,fig_stretch=self.make_tpf_figure_elements(self.env.tb_tpf,self.env.tpf_flux,fiducial_frame=self.env.fiducial_frame)

        self.env.fig_tpf.select('tpfimg')[0].data_source.data['image']=fig_tpf.select('tpfimg')[0].data_source.data['image']
        # self.env.fig_stretch.select('tpfstretch')[0].data_source.data['start']=fig_stretch.select('tpfstretch')[0].data_source.data['start']
        self.env.fig_stretch.start=fig_stretch.start
        self.env.fig_stretch.end=fig_stretch.end
        self.env.fig_stretch.value=fig_stretch.value

        # self.env.fig_tpf.select('tpfimg')[0].data_source.data['image']=[self.env.tpf_flux[0, :, :] + self.pedestal]

    def next(self):
        # print(self.env.tb_source.data["id"])
        self.env.tb_source.data["id"][0]=int(self.env.tb_source.data["id"][0])+1
        id=self.env.tb_source.data["id"][0]
        self.env.tb_source.data["id_mycatalog"][0]=self.env.tb_source.data["id_mycatalog_all"][id]
        self.id_mycatalog=self.env.tb_source.data["id_mycatalog"][0]
        print(self.env.tb_source.data["id_mycatalog"][0],self.id_mycatalog)

        mydata=mycatalog.pointer(id_mycatalog=self.id_mycatalog)
        sector_list=ast.literal_eval(mydata.Sector.values[0])
        sector_list=list([str(i) for i in sector_list])
        #print(sector_list)
        self.env.sector=sector_list[0]



        self.update_all(0,0,0)
        # self.reset_default_lc()
        # self.update_nearbysource()
        #
        # self.text_format()
        # self.env.text_banner.text=self.env.Message
        # self.update_format()


        # self.draw_nearbysource()

    def update_all(self,attrname, old, new):
        # print('aperutre_function')
        self.env.tb_source.data["id"][0]=int(self.env.tb_source.data["id"][0])
        id=self.env.tb_source.data["id"][0]
        self.env.tb_source.data["id_mycatalog"][0]=self.env.tb_source.data["id_mycatalog_all"][id]
        self.id_mycatalog=self.env.tb_source.data["id_mycatalog"][0]
        print(self.env.tb_source.data["id_mycatalog"][0],self.id_mycatalog)
        # self.env.tb_catalog_all.selected.indices=[id]

        self.reset_default_lc()
        self.update_nearbysource()

        self.text_format()
        self.env.text_banner.text=self.env.Message
        self.update_format()
        self.env.text_id_mycatalog_query.value=str(self.id_mycatalog)
        self.env.text_id_query.value=str(self.env.tb_source.data["id"][0])
        self.env.tb_catalog_all.selected.indices=[id]
        self.update_tb_nearby_star()

        self.env.catalog_find_from_isocrhone()
        print('Ready')





    def text_format(self):
        '''Format text for display'''
        data=mycatalog.pointer(catalog='mycatalog',id_mycatalog=self.env.tb_source.data["id_mycatalog"][0])
        # apogee=mycatalog.pointer(catalog='apogee',id_mycatalog=self.env.tb_source.data["id_mycatalog"][0])
        apogee=pd.DataFrame()
        Gmag=str(data['Gmag'].values[0])
        bp_rp=str(data['bp_rp'].values[0])
        PMemb=str(data['PMemb'].values[0])
        id_apogee=str(data['id_apogee'].values[0])
        name=str(data['id_mycatalog'].values[0])
        print(name)

        if apogee.empty:
            self.env.Message='Soure: '+name+' Gmag:'+Gmag+' bp_rp:'+bp_rp+' PMemb:'+ PMemb+ ' id_apogee:'+id_apogee

        else:
            apogee_teff=str(apogee['Teff'].values[0])
            apogee_Fe_H=str(apogee['[Fe/H]'].values[0])
            apogee_logg=str(apogee['logg'].values[0])
            self.env.Message='Soure: '+name+' Gmag:'+Gmag+' bp_rp:'+bp_rp+' PMemb:'+ PMemb+ ' id_apogee:'+id_apogee+' apogee_teff:'+apogee_teff+' apogee_Fe_H:'+apogee_Fe_H+' apogee_logg:'+apogee_logg

        self.env.v_flag_duplicate=data['flag_duplicate'].values[0]
        self.env.v_flag_source=data['flag_source'].values[0]
        self.env.v_flag_check=data['flag_check'].values[0]
        self.env.v_text_Notes=data['Notes'].values[0]



    def save_userinput(self):
        id_mycatalog=self.env.tb_source.data["id_mycatalog"][0]
        mycatalog.update(id_mycatalog=id_mycatalog,flag_source=self.env.text_flag_source.value)
        mycatalog.update(id_mycatalog=id_mycatalog,flag_check=self.env.text_flag_check.value)
        mycatalog.update(id_mycatalog=id_mycatalog,flag_duplicate=self.env.text_flag_duplicate.value)
        mycatalog.update(id_mycatalog=id_mycatalog,Notes=self.env.text_Notes.value)

    def update_format(self):
        '''Update'''
        self.env.text_banner.text=self.env.Message
        self.env.text_flag_source.value=str(self.env.v_flag_source)
        self.env.text_flag_check.value=str(self.env.v_flag_check)
        self.env.text_flag_duplicate.value=str(self.env.v_flag_duplicate)
        self.env.text_Notes.value=str(self.env.v_text_Notes)

    def previous(self):
        # print(self.env.tb_source.data["id"])
        self.env.tb_source.data["id"][0]=int(self.env.tb_source.data["id"][0])-1
        id=self.env.tb_source.data["id"][0]
        self.env.tb_source.data["id_mycatalog"][0]=self.env.tb_source.data["id_mycatalog_all"][id]
        self.id_mycatalog=self.env.tb_source.data["id_mycatalog"][0]
        # print(self.env.tb_source.data["id_mycatalog"][0],self.id_mycatalog)


        # self.reset_default_lc()
        # self.update_nearbysource()
        #
        # self.text_format()
        # self.env.text_banner.text=self.env.Message
        # self.update_format()

        self.update_all(0,0,0)

    def update_sector_change(self,attr, old, new):
        self.env.sector=int(self.env.int_select_sector.value)
        self.update_all(0,0,0)

        # print(self.id_mycatalog,self.env.tb_source.data["id_mycatalog"])

    def nearby_source(self):
        '''Function to find out the pixels of other star'''
        id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
        print('id_mycatalog=',id_mycatalog)
        details=mycatalog.pointer(catalog='mycatalog',id_mycatalog=id_mycatalog)
        clus=details.cluster.values[0]
        ra0=details['RA_ICRS'].values[0]
        dec0=details['DE_ICRS'].values[0]
        all_memb=mycatalog.pointer(catalog='mycatalog').query('cluster==@clus')

        all_memb=all_memb.reset_index(drop=True).query(self.env.text_catalog_query.value).reset_index(drop=True)

        all_memb['distance']=all_memb.apply(self.distance,args=(ra0,dec0),axis=1)
        min_distance=(21*16)/60 #in arcminute
        all_memb=all_memb.query('abs(distance)<=@min_distance')
        all_id=all_memb.id_mycatalog.values
        header_default_filename=mycatalog.filename(name='eleanor_header',id_mycatalog=id_mycatalog,sector=self.env.sector)
        head=Header.fromtextfile(header_default_filename)
        tess_wcs=utils.extract_essential_wcs_postcard(head,header=True)
        data=pd.DataFrame(columns=['x','y','id_mycatalog'])
        for id in all_id:
            # print('aperture#######',id)
            aper=aperture.load_aperture(name='eleanor_aperture_current',id_mycatalog=id)
            pixels=utils.radec2pixel(wcs=tess_wcs,ra=aper.ra.values,dec=aper.dec.values)
            d=pd.DataFrame(pixels,columns=['x','y'])
            d['id_mycatalog']=id
            data=data.append(d)

        data=data.query('x<14 & y<14 & x>=0 &y>=0')
        #for removing the source and to show only others
        data=data.query('id_mycatalog!=@id_mycatalog')
        # print('dataset',len(data))
        return data


    def distance(self,row,ra0,dec0):
        ra=row['RA_ICRS']
        dec=row['DE_ICRS']
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        coord0 = SkyCoord(ra=ra0, dec=dec0, unit=(u.deg, u.deg))
        sep=coord.separation(coord0)
        return sep.arcmin

    def draw_nearbysource(self):
        if self.env.plot_nearby==1:
            data=self.nearby_source()
            data.x=data.x+0.5
            data.y=data.y+0.5
            x=data.x.to_list()
            y=data.y.to_list()
            extra=ColumnDataSource(data=dict(x=x,y=y))
            self.env.tb_nearby.data=extra.data
            # self.env.tb_nearby=ColumnDataSource(data=dict(x=x,y=y))
            # self.env.fig_tpf.rect("x","y",1,1,source=self.env.tb_nearby,line_color='red',line_width=2,name='nearby',alpha=1,fill_color=None)

    def update_nearbysource(self):
        if self.env.plot_nearby==1:
            self.env.toggle_button.button_type='warning'
            data=self.nearby_source()
            data.x=data.x+0.5
            data.y=data.y+0.5
            x=data.x.to_list()
            y=data.y.to_list()
            new=ColumnDataSource(data=dict(x=x,y=y))
            # print('test_data',new)
            self.env.tb_nearby.data=new.data

    def draw_nearby_button(self):
        self.show_spinner()
        print('Plot near by source intialized')
        self.env.toggle_button.button_type='warning'
        self.env.plot_nearby=self.env.plot_nearby*-1
        if self.env.plot_nearby==1:
            self.env.toggle_button.button_type='warning'
            self.update_nearbysource()
            print('Plot Near by True')
            self.env.toggle_button.button_type='success'
        if self.env.plot_nearby==-1:
            new=ColumnDataSource(data=dict(x=[],y=[]))
            self.env.tb_nearby.data=new.data
            print('Plot Near by False')
            self.env.toggle_button.button_type='danger'


    def aperture_selection_function(self):
        self.env.aperture_setting=self.env.aperture_setting*-1
        if self.env.aperture_setting==1:
            print('Aperture_Current_setting')
            self.env.aperture_selection_button.label='Aperture_current'
            self.env.aperture_selection_button.button_type='success'
            self.reset_default_lc()

        if self.env.aperture_setting==-1:
            print('Aperture_defautl_setting')
            self.env.aperture_selection_button.label='Aperture_default'
            self.env.aperture_selection_button.button_type='danger'
            self.reset_default_lc()


    def update_tb_nearby_star(self):
        '''Function to find out the pixels of other star'''
        print('status',self.env.draw_nearby_names,type(self.env.catalog_whole_gaia))
        if self.env.draw_nearby_names==1:

            if not type(self.env.catalog_whole_gaia)==type(None):
                id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
                # print('id_mycatalog=',id_mycatalog)
                w=mycatalog.pointer(catalog='mycatalog',id_mycatalog=id_mycatalog)
                # clus=details.cluster.values[0]
                # ra0=details['RA_ICRS'].values[0]
                # dec0=details['DE_ICRS'].values[0]
                # all_memb=mycatalog.pointer(catalog='mycatalog').query('cluster==@clus')
                # if not self.env.text_catalog_query.value=='':
                #     all_memb=all_memb.reset_index(drop=True).query(self.env.text_catalog_query.value).reset_index(drop=True)
                #
                # all_memb['distance']=all_memb.apply(self.distance,args=(ra0,dec0),axis=1)
                # min_distance=(21*16)/60 #in arcminute
                # all_memb=all_memb.query('abs(distance)<=@min_distance').reset_index(drop=True)
                # id_coord=Coord.SkyCoord(w.RA_ICRS.values[0],w.DE_ICRS.values[0], frame='icrs', unit='deg')
                # data_coord=Coord.SkyCoord(self.env.catalog_whole_gaia['_RAJ2000'].values, self.env.catalog_whole_gaia['_DEJ2000'].values, frame='icrs', unit='deg')
                # array=data_coord.separation(id_coord) < float(self.env.gaia_radius_text.value)*astropy.units.arcmin
                # filtered=self.env.catalog_whole_gaia[array].reset_index(drop=True)
                #
                # array=data_coord.separation(id_coord) < float(self.env.gaia_radius_text.value)*astropy.units.arcmin
                # filtered=self.env.catalog_whole_gaia[array].reset_index(drop=True)
                id_coord=Coord.SkyCoord(w.RA_ICRS.values[0],w.DE_ICRS.values[0], frame='icrs', unit='deg')
                data_coord=Coord.SkyCoord(self.env.catalog_whole_gaia['_RAJ2000'].values, self.env.catalog_whole_gaia['_DEJ2000'].values, frame='icrs', unit='deg')
                array=data_coord.separation(id_coord) < float(self.env.gaia_radius_text.value)*astropy.units.arcmin
                filtered=self.env.catalog_whole_gaia[array].reset_index(drop=True)
                filtered['id_mycatalog']=''

                for i in filtered.index.values:
                    id_test=filtered.loc[i].id_gaia
                    k=self.catalog_main.query('id_gaia==@id_test').id_mycatalog.values
                    if len(k)>0:
                        #print('test',k[0])
                        filtered.loc[i,'id_mycatalog']=k[0]

                header_default_filename=mycatalog.filename(name='eleanor_header',id_mycatalog=id_mycatalog,sector=self.env.sector)
                head=Header.fromtextfile(header_default_filename)
                tess_wcs=utils.extract_essential_wcs_postcard(head,header=True)
                data=pd.DataFrame(columns=['pixel_x','pixel_y','id_mycatalog_names','Gmag'])
        # print(all_memb.columns)
        # for id in all_id:
            # print('aperture#######',id)
        # aper=aperture.load_aperture(name='eleanor_aperture_current',id_mycatalog=id)
                data=pd.DataFrame()
                data['id_mycatalog_names']=''
                for i in range(len(filtered)):
                    new=filtered.loc[i]
                    pixels=utils.radec2pixel(wcs=tess_wcs,ra=[new['_RAJ2000']],dec=[new['_DEJ2000']],exact=True)
                    d=pd.DataFrame(pixels,columns=['pixel_x','pixel_y'])
                    d['id_mycatalog_names']=new['id_mycatalog']
                    d['Gmag']=new['Gmag']
                    data=data.append(d)

                print(data)
                if not data.empty:
                    data=data.query('pixel_x<14 & pixel_y<14 & pixel_x>=0 & pixel_y>=0')
                #data['id_mycatalog_names'] = data['id_mycatalog_names'].str.replace('N77_' , '')
                old = ColumnDataSource(data=data.to_dict('list'))
                self.env.tb_nearby_star.data=old.data
            # print(data)

    def initiate_userinput(self):
        self.env.text_flag_duplicate = TextInput(value=str(self.env.v_flag_duplicate), title="Flag duplicate",height=50)
        self.env.text_flag_source = TextInput(value=str(self.env.v_flag_source), title="Flag source",height=50)
        self.env.text_flag_check = TextInput(value=str(self.env.v_flag_check), title="Flag check",height=50)

        self.env.text_Notes = TextInput(value=str(self.env.v_text_Notes), title="",height=70,width=1500,align='start')
        title_ = Paragraph(text='Notes', align='center')
        self.env.text_Notes_w = row([title_, self.env.text_Notes])


    def create_current_files(self,name='',id_mycatalog=''):
        '''Function Creates Current files if it does not exist'''
        # self.env.current_flux_dataframe.to_csv(mycatalog.filename(name='eleanor_flux_current',id_mycatalog=self.id_mycatalog),index=False)
        filename=mycatalog.filename(name=name,id_mycatalog=id_mycatalog,sector=self.env.sector)
        if not os.path.exists(filename):
            print('Creating current file',id_mycatalog)
            name_old=name.replace('_current','')
            filename_old=mycatalog.filename(name=name_old,id_mycatalog=id_mycatalog,sector=self.env.sector)
            import shutil
            shutil.copy(filename_old,filename)

    def initiate_errorbar(self):

        self.env.tb_lightcurve_error=ColumnDataSource(data=dict(x=[], y=[]))
        self.fig_lightcurve.multi_line("x","y",source=self.env.tb_lightcurve_error,color='red')




    def update_lk_error_bar(self,attr, old, new):
        '''Update the error table with new error'''

        if self.env.show_error_bar.active[0]==0:

            x=self.env.tb_lightcurve.data['time']
            y=self.env.tb_lightcurve.data['flux']
            err=self.env.tb_lightcurve.data['flux_err']
            err_x, err_y=utils.bokeh_errorbar(x=x,y=y,yerr=err)

            tb=ColumnDataSource(data=dict(x=err_x, y=err_y))
            self.env.tb_lightcurve_error.data=tb.data

        if self.env.show_error_bar.active[0]==1:
            tb=ColumnDataSource(data=dict(x=[], y=[]))
            self.env.tb_lightcurve_error.data=tb.data
            #strange to find this here
            #self.env.tb_source.data["id_mycatalog"][0]='custom_star'
