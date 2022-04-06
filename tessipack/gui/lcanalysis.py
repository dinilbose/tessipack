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
from bokeh.models import Button,Select,CheckboxGroup  # for saving data
from astropy.utils.exceptions import AstropyUserWarning
import warnings
from bokeh.models import CustomJS, TextInput, Paragraph,CheckboxGroup
from bokeh.events import ButtonClick
from bokeh.models.annotations import Title


from astropy.io.fits import Header
from tessipack.My_catalog import mycatalog
from tessipack.functions import utils
from tessipack.functions import aperture
from tessipack.functions import flags
from tessipack.functions import maths
from tessipack.functions import io

import os
from bokeh.models import ColumnDataSource, Label, LabelSet, Range1d
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
import astropy.coordinates as Coord
import astropy
from astropy import units


class Lcanalysis(Environment):
    env=Environment
    def __init__(self):


        self.flux_name='pca_flux'
        self.flux_name2='pca_flux'
        self.fold_long=True

        self.id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
        print('Testttttttttttt',self.id_mycatalog)
        self.env.tb_lc_an1, self.env.fig_lc_an1= self.initiate_lk(id_mycatalog=self.id_mycatalog,
                                                                  flux_name=self.flux_name,
                                                                  tb_lc=self.env.tb_lc_an1,
                                                                  fig_lc=self.env.fig_lc_an1,
                                                                  onchange=False)

        self.env.tb_lc_an2, self.env.fig_lc_an2= self.initiate_lk(id_mycatalog=self.id_mycatalog,
                                                                  flux_name=self.flux_name2,
                                                                  tb_lc=self.env.tb_lc_an2,
                                                                  fig_lc=self.env.fig_lc_an2,
                                                                  onchange=False)

        self.tb_pr_se1=ColumnDataSource(data=dict(x=[], y=[]))
        self.tb_pr_se2=ColumnDataSource(data=dict(x=[], y=[]))


        self.env.tb_pr_an1=self.initiate_tb_pr(tb_lc=self.env.tb_lc_an1,tb_pr=self.env.tb_pr_an1)
        self.env.fig_pr_an1=self.initiate_fig_pr(tb_pr=self.env.tb_pr_an1,tb_pr_se=self.tb_pr_se1)

        self.env.tb_pr_an2=self.initiate_tb_pr(tb_lc=self.env.tb_lc_an2,tb_pr=self.env.tb_pr_an2)
        self.env.fig_pr_an2=self.initiate_fig_pr(tb_pr=self.env.tb_pr_an2,tb_pr_se=self.tb_pr_se2)

        self.tb_source.on_change('data',self.update)


        self.env.fold_select_1 = CheckboxGroup(labels=['Fold'],active=[1],height=20,width=40)
        self.env.flux_name_1 = Select(title='Flux', options=['pca_flux', 'psf_flux','corr_flux'],
                                       value='pca_flux')

        self.env.fold_select_2 = CheckboxGroup(labels=['Fold'],active=[1],height=20,width=40)
        self.env.flux_name_2 = Select(title='Flux', options=['pca_flux', 'psf_flux','corr_flux'],
                                      value='pca_flux')


        self.env.text_p1=TextInput(value='', title="P1",width=100)
        self.env.text_p0=TextInput(value='', title="Period",width=130)
        self.env.text_p2=TextInput(value='', title="P2",width=100)
        self.env.text_pe=TextInput(value='2', title="Pe",width=100)

        self.env.period_recompute_button = Button(label="Recompute", button_type="success",width=100)
        self.env.period_recompute_button.on_click(self.period_recompute)

        self.env.fold_button = Button(label="Fold", button_type="success",width=100)
        self.env.fold_button.on_click(self.fold)


        self.env.text2_p1=TextInput(value='', title="P1",width=100)
        self.env.text2_p0=TextInput(value='', title="Period",width=130)
        self.env.text2_p2=TextInput(value='', title="P2",width=100)
        self.env.text2_pe=TextInput(value='2', title="Pe",width=100)

        self.env.period_recompute_button2 = Button(label="Recompute2", button_type="success",width=100)
        self.env.period_recompute_button2.on_click(self.period_recompute2)

        self.env.fold_button2 = Button(label="Fold2", button_type="success",width=100)
        self.env.fold_button2.on_click(self.fold2)




        self.env.fold_select_1.on_change("active",self.figure1_update)
        self.env.flux_name_1.on_change("value",self.figure1_update)

        self.env.fold_select_2.on_change("active",self.figure2_update)
        self.env.flux_name_2.on_change("value",self.figure2_update)

        self.env.check_analysis = CheckboxGroup(labels=['Analysis window'],active=[0,1],height=20,width=60)
        self.env.check_analysis.on_change("active",self.update)
        self.env.extra_flag_file=mycatalog.filename(name='extra_flag_file')

    def update(self,attr, old, new):
        '''
        Update

        '''
        if self.env.check_analysis.active[0]==0:

                self.figure2_update(0,0,0)
                self.figure1_update(0,0,0)

        # print(0)
        return


    def figure1_update(self,attr, old, new):

        if self.env.check_analysis.active[0]==0:
            self.tb_pr_se1.data=ColumnDataSource(data=dict(x=[], y=[])).data


            self.id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]


            self.env.tb_lc_an1, self.env.fig_lc_an1= self.initiate_lk(id_mycatalog=self.id_mycatalog,
                                                                      flux_name=self.env.flux_name_1.value,
                                                                      tb_lc=self.env.tb_lc_an1,
                                                                      fig_lc=self.env.fig_lc_an1,
                                                                      onchange=True)

            if self.env.fold_select_1.active[0]==0:
                self.flux_name=self.env.flux_name_1.value
                if self.env.aperture_setting==-1:
                    self.data=io.read_lightcurve(name='eleanor_flux',sector=self.env.sector,id_mycatalog=self.id_mycatalog,
                                            extra_flag_file=self.env.extra_flag_file,sigma=5,flux_name=self.flux_name,
                                            filter=True,time_flag='default',keep_length=True,func='median',
                                            deviation='mad')
                elif self.env.aperture_setting==1:
                    self.data=io.read_lightcurve(name='eleanor_flux_current',id_mycatalog=self.id_mycatalog,sector=self.env.sector,
                                            extra_flag_file=self.env.extra_flag_file,sigma=5,flux_name=self.flux_name,
                                            filter=True,time_flag='default',keep_length=True,func='median',
                                            deviation='mad')

                # data_pr=io.read_prtable(id_mycatalog=self.id_mycatalog)
                # self.tb_pr_se1.data=ColumnDataSource(data=dict(x=data_pr.x, y=data_pr.y)).data
                data_pr=io.read_prtable(id_mycatalog=self.id_mycatalog,sector=self.env.sector)

                if not data_pr.empty:
                    p0=data_pr.loc[data_pr.y.argmax()].x
                    pe=self.env.text_pe.value
                    p0=maths.minimise_period(data=self.data,x1=float(p0)-float(pe),x2=float(p0)+float(pe))
                    self.env.text_p0.value=str(p0)
                    self.env.text_p1.value=str(p0-float(pe))
                    self.env.text_p2.value=str(p0+float(pe))
                    fold_data=maths.folding(data=self.data,flux_name=self.flux_name,
                                                  period=p0*units.microhertz,duplicate=True,addition=0.5,append=self.fold_long)
                    self.env.tb_pr_an1.data=ColumnDataSource(data=dict(x=fold_data.phase,y=fold_data[self.flux_name])).data

                else:
                    p0=5
                    pe=self.env.text_pe.value
                    self.env.text_p0.value=str(p0)
                    self.env.text_p1.value=str(p0-float(pe))
                    self.env.text_p2.value=str(p0+float(pe))

                self.env.fig_pr_an1.select('line').visible=False
                self.env.fig_pr_an1.title.text='Folded Light Curve'

            else:

                self.env.fig_pr_an1.select('line').visible=True
                self.env.fig_pr_an1.title.text='Periodogram'
                data_pr=io.read_prtable(id_mycatalog=self.id_mycatalog)
                self.tb_pr_se1.data=ColumnDataSource(data=dict(x=data_pr.x, y=data_pr.y)).data
                self.env.tb_pr_an1=self.initiate_tb_pr(tb_lc=self.env.tb_lc_an1,tb_pr=self.env.tb_pr_an1,onchange=True)




    def figure2_update(self,attr, old, new):
        if self.env.check_analysis.active[0]==0:

            self.id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
            self.tb_pr_se2.data=ColumnDataSource(data=dict(x=[], y=[])).data



            self.env.tb_lc_an2, self.env.fig_lc_an2= self.initiate_lk(id_mycatalog=self.id_mycatalog,
                                                                      flux_name=self.env.flux_name_2.value,
                                                                      tb_lc=self.env.tb_lc_an2,
                                                                      fig_lc=self.env.fig_lc_an2,
                                                                      onchange=True)


            if self.env.fold_select_2.active[0]==0:
                self.flux_name2=self.env.flux_name_2.value
                if self.env.aperture_setting==-1:
                    self.data2=io.read_lightcurve(name='eleanor_flux',id_mycatalog=self.id_mycatalog,
                                            extra_flag_file=self.env.extra_flag_file,sigma=5,flux_name=self.flux_name2,
                                            filter=True,time_flag='default',keep_length=True,func='median',
                                            deviation='mad',sector=self.env.sector)
                elif self.env.aperture_setting==1:
                    self.data2=io.read_lightcurve(name='eleanor_flux_current',id_mycatalog=self.id_mycatalog,
                                            extra_flag_file=self.env.extra_flag_file,sigma=5,flux_name=self.flux_name2,
                                            filter=True,time_flag='default',keep_length=True,func='median',
                                            deviation='mad',sector=self.env.sector)
                # data_pr=io.read_prtable(id_mycatalog=self.id_mycatalog)
                # self.tb_pr_se2.data=ColumnDataSource(data=dict(x=data_pr.x, y=data_pr.y)).data
                data_pr=io.read_prtable(id_mycatalog=self.id_mycatalog,sector=self.env.sector)

                if not data_pr.empty:
                    p0=data_pr.loc[data_pr.y.argmax()].x
                    pe=self.env.text2_pe.value
                    p0=maths.minimise_period(data=self.data2,x1=float(p0)-float(pe),x2=float(p0)+float(pe))
                    self.env.text2_p0.value=str(p0)
                    self.env.text2_p1.value=str(p0-float(pe))
                    self.env.text2_p2.value=str(p0+float(pe))
                    fold_data=maths.folding(data=self.data2,flux_name=self.flux_name2,
                                                  period=p0*units.microhertz,duplicate=True,addition=0.5,append=self.fold_long)
                    self.env.tb_pr_an2.data=ColumnDataSource(data=dict(x=fold_data.phase,y=fold_data[self.flux_name2])).data

                else:

                    p0=5
                    pe=self.env.text2_pe.value

                    self.env.text2_p0.value=str(p0)
                    self.env.text2_p1.value=str(p0-float(pe))
                    self.env.text2_p2.value=str(p0+float(pe))

                self.env.fig_pr_an2.select('line').visible=False
                self.env.fig_pr_an2.title.text='Folded Light Curve'


            else:

                self.env.fig_pr_an2.select('line').visible=True
                self.env.fig_pr_an2.title.text='Periodogram'
                data_pr=io.read_prtable(id_mycatalog=self.id_mycatalog,sector=self.env.sector)
                self.tb_pr_se2.data=ColumnDataSource(data=dict(x=data_pr.x, y=data_pr.y)).data
                self.env.tb_pr_an2=self.initiate_tb_pr(tb_lc=self.env.tb_lc_an2,tb_pr=self.env.tb_pr_an2,onchange=True)

        return 0


    def initiate_lk(self,id_mycatalog='',flux_name='',tb_lc='',fig_lc='',onchange=False):
        '''
        New

        '''

        if self.env.aperture_setting==-1:

            data=io.read_lightcurve(name='eleanor_flux',sector=self.env.sector,id_mycatalog=id_mycatalog,extra_flag_file=self.env.extra_flag_file,sigma=5,flux_name=flux_name,filter=True,time_flag='default',keep_length=True,func='median',deviation='mad')

        elif self.env.aperture_setting==1:

            data=io.read_lightcurve(name='eleanor_flux_current',sector=self.env.sector,id_mycatalog=id_mycatalog,extra_flag_file=self.env.extra_flag_file,sigma=5,flux_name=flux_name,filter=True,time_flag='default',keep_length=True,func='median',deviation='mad')


        if not onchange:

            tb_lc = ColumnDataSource(data=dict(time=data.time, flux=data[flux_name],flux_err=data.flux_err))

            TOOLTIPS = [
                ("index", "$index"),
                ("(x,y)", "($x, $y)"),
            ]
            fig_lc = figure(
                plot_width=self.env.plot_width,
                plot_height=self.env.plot_height,
                tools=["box_zoom", "wheel_zoom","lasso_select", "tap" ,"reset", "save"],
                title="Light Curve",tooltips=TOOLTIPS,
            )
            fig_lc.circle("time", "flux", source=tb_lc, alpha=0.7,**self.env.selection,)
            fig_lc.line("time","flux",source=tb_lc,alpha=0.7,color="#1F77B4")

        else:

            tb_lc.data =  ColumnDataSource(data=dict(time=data.time, flux=data[flux_name],flux_err=data.flux_err)).data

        return tb_lc, fig_lc

    def initiate_tb_pr(self,tb_lc='',tb_pr='',tb_pr_se='',onchange=False):

        flux=tb_lc.data["flux"]
        time=tb_lc.data["time"]
        flux_err=tb_lc.data["flux_err"]
        f,p=maths.lomb_scargle(flux=flux,time=time,flux_err=flux_err)

        if not onchange:
            # print('onnn',onchange)
            tb_pr=ColumnDataSource(data=dict(x=f, y=p))

        else:
            tb_pr.data = ColumnDataSource(data=dict(x=f, y=p)).data

        return tb_pr

    def initiate_fig_pr(self,x_range=None,y_range=None,tb_pr='',tb_pr_se=''):

        alpha=0.7
        if x_range==None:
            fig_pr = figure(
                plot_width=self.env.plot_width,
                plot_height=self.env.plot_height,
                tools=["box_zoom", "wheel_zoom","lasso_select", "tap" ,"reset", "save"],
                title="Periodogram",tooltips=self.env.TOOLTIPS,
            )
        else:
            fig_pr = figure(
                plot_width=self.env.plot_width,
                plot_height=self.env.plot_height,x_range=x_range,y_range=y_range,
                tools=["box_zoom", "wheel_zoom","lasso_select", "tap" ,"reset", "save"],
                title="Periodogram",tooltips=self.env.TOOLTIPS,
            )
        fig_pr.circle("x", "y", source=tb_pr, alpha=alpha,**self.env.selection)
        fig_pr.line("x","y",source=tb_pr,alpha=alpha,name='line',**self.env.selection_l)
        fig_pr.ray(x="x",y=0,source=tb_pr_se,name='selection_line',length=300, angle=np.pi/2,color='red')

        return fig_pr


    def period_recompute(self):

        if self.env.fold_select_1.active[0]==0:

            p1=float(self.env.text_p1.value)
            p2=float(self.env.text_p2.value)
            p0=float(self.env.text_p0.value)
            pe=float(self.env.text_pe.value)
            p0=maths.minimise_period(data=self.data,x1=p1,x2=p2)

            self.env.text_p0.value=str(p0)

            fold_data=maths.folding(data=self.data,flux_name=self.flux_name,
                                          period=p0*units.microhertz,duplicate=True,addition=0.5,append=self.fold_long)

            self.env.tb_pr_an1.data=ColumnDataSource(data=dict(x=fold_data.phase,y=fold_data[self.flux_name])).data

        return

    def fold(self):
        if self.env.fold_select_1.active[0]==0:

            p0=float(self.env.text_p0.value)
            p1=float(self.env.text_p1.value)
            p2=float(self.env.text_p2.value)
            p0=float(self.env.text_p0.value)
            pe=float(self.env.text_pe.value)
            self.env.text_p1.value=str(p0-pe)
            self.env.text_p2.value=str(p0+pe)
            fold_data=maths.folding(data=self.data,flux_name=self.flux_name,
                                          period=p0*units.microhertz,duplicate=True,addition=0.5,append=self.fold_long)
            self.env.tb_pr_an1.data=ColumnDataSource(data=dict(x=fold_data.phase,y=fold_data[self.flux_name])).data

        return


    def fold2(self):
        print('fold_period_2')

        if self.env.fold_select_2.active[0]==0:

            p1=float(self.env.text2_p1.value)
            p2=float(self.env.text2_p2.value)
            p0=float(self.env.text2_p0.value)
            pe=float(self.env.text2_pe.value)
            self.env.text2_p1.value=str(p0-pe)
            self.env.text2_p2.value=str(p0+pe)
            fold_data=maths.folding(data=self.data2,flux_name=self.flux_name2,
                                          period=p0*units.microhertz,duplicate=True,addition=0.5,append=self.fold_long)
            self.env.tb_pr_an2.data=ColumnDataSource(data=dict(x=fold_data.phase,y=fold_data[self.flux_name2])).data

        return


    def period_recompute2(self):

        if self.env.fold_select_2.active[0]==0:


            p1=float(self.env.text2_p1.value)
            p2=float(self.env.text2_p2.value)
            p0=float(self.env.text2_p0.value)
            pe=float(self.env.text2_pe.value)
            p0=maths.minimise_period(data=self.data2,x1=p1,x2=p2)

            self.env.text2_p0.value=str(p0)

            fold_data=maths.folding(data=self.data2,flux_name=self.flux_name2,
                                          period=p0*units.microhertz,duplicate=True,addition=0.5,append=self.fold_long)

            self.env.tb_pr_an2.data=ColumnDataSource(data=dict(x=fold_data.phase,y=fold_data[self.flux_name2])).data

        return
