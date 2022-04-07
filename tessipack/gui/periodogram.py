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
from bokeh.models import Button  # for saving data
from astropy.utils.exceptions import AstropyUserWarning
import warnings
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import CustomJS, TextInput, Paragraph
import os.path

from astropy.io.fits import Header
from tessipack.functions import utils
from tessipack.functions import aperture
from tessipack.functions import maths
from tessipack.My_catalog import mycatalog

class Periodogram(Environment):
    env=Environment
    def __init__(self):
        env=Environment
        self.env.tb_oscillation_modell0 = ColumnDataSource(data=dict(x=[], y=[]))
        self.env.tb_oscillation_modell1 = ColumnDataSource(data=dict(x=[], y=[]))
        self.env.tb_oscillation_modell2 = ColumnDataSource(data=dict(x=[], y=[]))


        self.env.tb_periodogram=self.initiate_periodogram()
        self.initiate_all_fig()
        self.env.tb_lightcurve.on_change('data',self.update)
        self.env.reset_axes_prd_button = Button(label="Reset Axes", button_type="success",width=150)
        self.env.reset_axes_prd_button.on_click(self.reset_axis)
        self.env.table_periodogram,self.env.tb_periodogram_se_tb=self.initiate_table()
        self.intiate_table_selection()
        self.initiate_savebutton()
        self.env.tb_periodogram.selected.on_change('indices',self.write_to_table)
        self.load_table()
        self.initiate_table_buttons()
        self.initiate_ray()


    def initiate_periodogram(self):

        flux=self.env.tb_lightcurve.data["flux"]
        time=self.env.tb_lightcurve.data["time"]
        flux_err=self.env.tb_lightcurve.data["flux_err"]
        f,p=maths.lomb_scargle(flux=flux,time=time,flux_err=flux_err)
        tb_periodogram = ColumnDataSource(data=dict(x=f.value, y=p.value))

        return tb_periodogram

    def initiate_fig(self,x_range=None,y_range=None):

        alpha=0.7
        if x_range==None:
            fig_periodogram = figure(
                plot_width=self.env.plot_width,
                plot_height=self.env.plot_height,
                tools=["box_zoom", "wheel_zoom","lasso_select", "tap" ,"reset", "save"],
                title="Periodogram",tooltips=self.env.TOOLTIPS,
            )
        else:
            fig_periodogram = figure(
                plot_width=self.env.plot_width,
                plot_height=self.env.plot_height,x_range=x_range,y_range=y_range,
                tools=["box_zoom", "wheel_zoom","lasso_select", "tap" ,"reset", "save"],
                title="Periodogram",tooltips=self.env.TOOLTIPS,
            )
        fig_periodogram.circle("x", "y", source=self.env.tb_periodogram, alpha=alpha,**self.env.selection)
        fig_periodogram.line("x","y",source=self.env.tb_periodogram,alpha=alpha,**self.env.selection_l)

        fig_periodogram.ray(x="x",y=0,source=self.env.tb_oscillation_modell0,length=300, angle=np.pi/2,color='green')
        fig_periodogram.ray(x="x",y=0,source=self.env.tb_oscillation_modell1,length=300, angle=np.pi/2,color='yellow')
        fig_periodogram.ray(x="x",y=0,source=self.env.tb_oscillation_modell2,length=300, angle=np.pi/2,color='black')

        return fig_periodogram

    def initiate_all_fig(self):
        self.env.fig_periodogram=self.initiate_fig()
        self.env.fig_periodogram1=self.initiate_fig(x_range=(0,50),y_range=(0,100))
        self.env.fig_periodogram2=self.initiate_fig(x_range=(50,100),y_range=(0,100))
        self.env.fig_periodogram3=self.initiate_fig(x_range=(100,150),y_range=(0,100))
        self.env.fig_periodogram4=self.initiate_fig(x_range=(150,270),y_range=(0,100))

    def update(self,attr,old,new):
        tb_periodogram=self.initiate_periodogram()
        self.env.tb_periodogram.data=tb_periodogram.data
        self.reset_axis()
        self.load_table()

    def reset_axis(self):
        ll,hh=self.find_yrange(data=self.env.tb_periodogram,xlimit=[0,50])
        self.env.fig_periodogram1.y_range.start = 0
        self.env.fig_periodogram1.y_range.end = hh
        ll,hh=self.find_yrange(data=self.env.tb_periodogram,xlimit=[50,100])
        self.env.fig_periodogram2.y_range.start = 0
        self.env.fig_periodogram2.y_range.end = hh
        ll,hh=self.find_yrange(data=self.env.tb_periodogram,xlimit=[100,150])
        self.env.fig_periodogram3.y_range.start = 0
        self.env.fig_periodogram3.y_range.end = hh
        ll,hh=self.find_yrange(data=self.env.tb_periodogram,xlimit=[150,270])
        self.env.fig_periodogram4.y_range.start = 0
        self.env.fig_periodogram4.y_range.end = hh

    def find_yrange(self,data='',xlimit=[],percent=10):
        # xlimit=[100,150]
        tb_periodogram=data
        xx=tb_periodogram.data['x']
        yy=tb_periodogram.data['y']
        x_a=(xx>xlimit[0])&(xx<xlimit[1])
        yy_l=yy[x_a]
        yy_l=yy_l.max()+yy_l.max()*(percent/100)
        return 0,yy_l

    def initiate_table(self):
        tb_periodogram_se_tb = ColumnDataSource(data=dict(x=[], y=[]))
        columns = [
            TableColumn(field="x", title="Frequency"),
            TableColumn(field="y", title="Amplitude"),
        ]
        table_periodogram = DataTable(
            source=tb_periodogram_se_tb,
            columns=columns,
            width=300,
            height=300,
            sortable=True,
            selectable=True,
            editable=True,
        )
        return table_periodogram,tb_periodogram_se_tb

    def intiate_table_selection(self):
        self.env.tb_periodogram.selected.js_on_change(
            "indices",
            CustomJS(
                args=dict(s1=self.env.tb_periodogram, s2=self.env.tb_periodogram_se_tb, table=self.env.table_periodogram),
                code="""
                var inds = cb_obj.indices;
                var d1 = s1.data;
                var d2 = s2.data;
                //d2['x'] = []
                //d2['y'] = []
                for (var i = 0; i < inds.length; i++) {
                    d2['x'].push(d1['x'][inds[i]])
                    d2['y'].push(d1['y'][inds[i]])
                }
                s2.change.emit();
                table.change.emit();

                var inds = source_data.selected.indices;
                var data = source_data.data;
                var out = "x, y\\n";
                for (i = 0; i < inds.length; i++) {
                    out += data['x'][inds[i]] + "," + data['y'][inds[i]] + "\\n";
                }
                var file = new Blob([out], {type: 'text/plain'});

            """,
            ),
        )

    def initiate_savebutton(self):
        self.env.saveas_prd_tb_button = Button(label="Save As", button_type="success")
        self.env.saveas_prd_tb_button.callback = CustomJS(args=dict(source=self.env.tb_periodogram_se_tb,name=self.env.tb_source), code="""
                var data = source.data;
                var filename=name.data['id_mycatalog'][0]
                value1=data['x'];
                value2=data['y'];
                var out = "";
                for (i = 0; i < value1.length; i++) {
                    out += value1[i]+','+value2[i]+"\\n";
                }
                var file = new Blob([out], {type: 'texpip install pscriptt/plain'});
                var elem = window.document.createElement('a');
                elem.href = window.URL.createObjectURL(file);
                elem.download = filename;
                document.body.appendChild(elem);
                elem.click();
                document.body.removeChild(elem);
                """)
        self.env.save_prd_tb_button = Button(label="Save File", button_type="success")
        self.env.save_prd_tb_button.on_click(self.save_file)

    def write_to_table(self,attr,old,new):
        pd_periodogram=self.env.tb_periodogram.to_df()
        pd_periodogram_se_tb=self.env.tb_periodogram_se_tb.to_df()
        df2=pd_periodogram.loc[new]
        final=pd.concat([pd_periodogram_se_tb,df2]).drop_duplicates().dropna().reset_index(drop=True)
        tb_final=ColumnDataSource(data=dict(x=final.x, y=final.y))
        self.env.tb_periodogram_se_tb.data=tb_final.data

    def save_file(self):
        source=self.env.tb_source.data['id_mycatalog'][0]
        filename=mycatalog.filename(id_mycatalog=source,name='bokeh_periodogram_table',sector=self.env.sector)
        data=self.env.table_periodogram.source.to_df()
        data=data.query('x>0').reset_index()
        data.to_csv(filename,index=False)
        print('File Saved to',filename,data,self.env.table_periodogram.source.data)

    def reset_table(self):
        tb_final=ColumnDataSource(data=dict(x=[], y=[]))
        self.env.tb_periodogram_se_tb.data=tb_final.data

    def load_table(self):
        '''Load Selected Period file'''
        source=self.env.tb_source.data['id_mycatalog'][0]
        filename=mycatalog.filename(id_mycatalog=source,name='bokeh_periodogram_table',sector=self.env.sector)
        if os.path.isfile(filename):
            final=pd.read_csv(filename)
            tb_final=ColumnDataSource(data=dict(x=final.x, y=final.y))
            self.env.tb_periodogram_se_tb.data=tb_final.data
        else:
            print('Intial selected Frequency does not exist for:',source)
            tb_final=ColumnDataSource(data=dict(x=[], y=[]))
            self.env.tb_periodogram_se_tb.data=tb_final.data

    def initiate_table_buttons(self):
        self.env.reset_prd_tb_button = Button(label="Reset Table", button_type="success",width=120)
        self.env.reset_prd_tb_button.on_click(self.reset_table)

        self.env.load_prd_tb_button = Button(label="Load Table", button_type="success")
        self.env.load_prd_tb_button.on_click(self.load_table)


    def initiate_ray(self):
        self.env.fig_periodogram4.ray(x="x",y=0,source=self.env.tb_periodogram_se_tb,length=300, angle=np.pi/2,color='red')
        self.env.fig_periodogram3.ray(x="x",y=0,source=self.env.tb_periodogram_se_tb,length=300, angle=np.pi/2,color='red')
        self.env.fig_periodogram2.ray(x="x",y=0,source=self.env.tb_periodogram_se_tb,length=300, angle=np.pi/2,color='red')
        self.env.fig_periodogram1.ray(x="x",y=0,source=self.env.tb_periodogram_se_tb,length=300, angle=np.pi/2,color='red')
        self.env.fig_periodogram.ray(x="x",y=0,source=self.env.tb_periodogram_se_tb,length=300, angle=np.pi/2,color='red')




        # self.env.table_periodogram.source.data=self.env.tb_periodogram_se_tb.data
        # print(self.env.tb_periodogram_se_tb.data)
        # new_dataframe=pd.



        # self.env.save_prd_tb_button = Button(label="Save File", button_type="success")
        # self.env.save_prd_tb_button.on_click(self.save_file)

    # def save_file(self):
    #     source=self.env.tb_source.data['id_mycatalog'][0]
    #     filename=mycatalog.filename(id_mycatalog=source,name='bokeh_periodogram_table')
    #     data=pd.DataFrame(self.env.table_periodogram.source.data)
    #     # data.to_csv(filename,data,index=False)
    #     print('File Saved to',filename,data,self.env.table_periodogram.source.data)
