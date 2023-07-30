"""Defines the Seismology class."""
import logging
import warnings
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from bokeh.models import ColumnDataSource

from astropy import units as u
from astropy.units import cds

from lightkurve import MPLSTYLE
from lightkurve.seismology import utils, stellar_estimators
from lightkurve.periodogram import SNRPeriodogram
from lightkurve.utils import LightkurveWarning, validate_method
from lightkurve.seismology import SeismologyQuantity

from astropy import units
from tessipack.functions import maths
from tessipack.functions import io
from tessipack.My_catalog import mycatalog

import lightkurve
import Periodo
from env import Environment
from astropy import units
from bokeh.models import CustomJS, TextInput, Paragraph
# for saving data
from bokeh.models import Button, Select, CategoricalColorMapper, CheckboxGroup, TableColumn, DataTable
from lightkurve import periodogram as lk_prd_module

# Import the optional Bokeh dependency required by ``interact_echelle```,
# or print a friendly error otherwise.
try:
    import bokeh  # Import bokeh first so we get an ImportError we can catch
    from bokeh.io import show, output_notebook
    from bokeh.plotting import figure
    from bokeh.models import LogColorMapper, Slider, RangeSlider, Button
    from bokeh.layouts import layout, Spacer
except:
    # Nice error will be raised when ``interact_echelle``` is called.
    pass
from bokeh.colors import RGB
import bokeh.palettes

log = logging.getLogger(__name__)


class Interactive(Environment):
    env = Environment

    def __init__(self):


        self.mode_color_map = {
                    '-1': "blue",
                    '0': "green",
                    '1': "yellow",
                    '2': "orange",
                    }

        self.tb_constants_val = ColumnDataSource(
            data=dict(
                other_prd_cuttoff=list([500]),
                minimum_frequency=list([0])
            )
        )

        self.mode_shape_selection = 'circle'
        self.id_mycatalog = self.env.tb_source.data['id_mycatalog'][0]
        #self.palette = "hot"

        self.env.text_osc_query = TextInput(
            value='n_pg>=0', title="Select Cluster")

        self.env.tb_other_periodogram, self.env.fig_other_periodogram = self.initialize_dnu_periodogram()
        self.env.frequency_minimum_text = TextInput(
            value=str(self.env.minimum_frequency), title="Freq min", width=80)
        self.env.frequency_maximum_text = TextInput(
            value=str(self.env.maximum_frequency), title="Freq max", width=80)
        self.env.frequency_maxdnu_text = TextInput(
            value=str(self.env.maxdnu), title="Maxdnu", width=80)
        self.env.echelle_noise_cuttoff_text = TextInput(
            value=str(0), title="Threshold", width=80)

        self.env.dnu_text = TextInput(
            value=str(self.env.dnu_val), title="Delta Nu", width=100)
        self.env.update_int_button = Button(
            label="Update Plot", button_type="success", width=150)
        self.env.update_int_button.on_click(self.update_value)
        
        self.env.select_mode_menu = Select(title='Mode Select', 
                                         options=['-1','0','1','2'], 
                                         value='-1',width=150)
        self.env.mode_apply_button = Button(
                                        label="Select Mode", 
                                        button_type="success", width=150)
        self.env.mode_apply_button.on_click(self.click_mode_apply_button)

        self.env.move_se_1_2_button = Button(
                                        label="Move 1 -> 2", 
                                        button_type="success", width=150)
        self.env.move_se_1_2_button.on_click(self.click_move_se_1_2_button)
        self.env.move_se_2_1_button = Button(
                                        label="Move 2 -> 1", 
                                        button_type="success", width=150)
        self.env.move_se_2_1_button.on_click(self.click_move_se_2_1_button)

        self.env.save_table_2_button = Button(
                                        label="Save Table 2", 
                                        button_type="success", width=150)
        self.env.save_table_2_button.on_click(self.save_table_2)

        self.env.load_table_2_button = Button(
                                        label="Load Values", 
                                        button_type="success", width=150)
        self.env.load_table_2_button.on_click(self.load_table)




        self.env.select_color_palette = Select(title='Color Palette', 
                                         options=self.env.color_palette_options, 
                                         value=self.env.default_color_palette,
                                         width=150)
        self.palette = getattr(bokeh.palettes, self.env.default_color_palette)[9]
        self.env.check_reverse_color_palette = CheckboxGroup(
                                                            labels=['Reverse'],
                                                            active=[1],
                                                            height=10,
                                                            width=10)




        self.make_tb_echelle_diagram()
        self.interact_echelle()
        self.make_grid()
        self.initialize_selection_tables()

        self.env.test_button = Button(
            label="Test", button_type="success", width=150)
        self.env.test_button.on_click(self.get_all_selection_button)
        
        self.env.clear_se_grid_prd_button = Button(
            label="Clear Selection", button_type="success", width=150)
        self.env.clear_se_grid_prd_button.on_click(self.clear_se_grid_prd)

        self.env.clear_se_table1_button = Button(
            label="Clear Table 1", button_type="success", width=150)
        self.env.clear_se_table1_button.on_click(self.clear_se_table1)
        
        self.env.find_peaks_button = Button(
            label="Find Peak", button_type="success", width=150)
        self.env.find_peaks_button.on_click(self.find_peak_frequencies)
        #self.update_selection_tables()
        self.env.tb_catalog_all.selected.on_change('indices',self.update_source)

        self.plot_vertical_lines()




    def update_source(self,attr,old,new):
        ff, pp = self.read_fits_get_fp()
        mm = ['-1']*(len(pp))


        f = (ff*u.Hz).to(self.env.frequency_unit)
        p = pp*self.env.power_unit
        period = lk_prd_module.Periodogram(f, p)

        self.periodogram = period
        # self.env.minimum_frequency =   # 1
        # self.env.maximum_frequency =   # 8000
        # self.env.maxdnu = 50
        # self.env.dnu_val = 24.8
        old_data= ColumnDataSource(
            data=dict(
                frequency=list(self.periodogram.frequency.value),
                power=list(self.periodogram.power.value),
                Mode = mm,
                # cuttoff=list(np.array([0])),
            ))
        self.env.tb_other_periodogram.data=old_data.data
        #self.env.tb_other_periodogram.selected = list([])
        self.update_plot(0, 0, 0)



    def initialize_dnu_periodogram(self):
        '''
        Function initialise the the dnu periodogram table source and also the periodogram graph
        '''
        ff, pp = self.read_fits_get_fp()
        mm = ['-1']*(len(pp))

        f = (ff*u.Hz).to(self.env.frequency_unit)
        p = pp*self.env.power_unit
        period = lk_prd_module.Periodogram(f, p)

        self.periodogram = period
        self.env.minimum_frequency = 1  # 1
        self.env.maximum_frequency = 800  # 8000
        self.env.maxdnu = 50
        self.env.dnu_val = 24.8
        tb_other_periodogram = ColumnDataSource(
            data=dict(
                frequency=list(self.periodogram.frequency.value),
                power=list(self.periodogram.power.value),
                Mode = mm,
                # cuttoff=list(np.array([0])),
            ))

        # Intialize other periodgram
        fig_other_periodogram = figure(
            plot_width=1400,
            plot_height=600,
            tools=["box_zoom", "wheel_zoom",
                   "lasso_select", "tap", "reset", "save"],
            title="Other Periodogram", tooltips=self.env.TOOLTIPS,
        )


        color_map = self.mode_color_map
        color_mapper = CategoricalColorMapper(
            factors=list(color_map.keys()), 
            palette=list(color_map.values()))

        fig_other_periodogram.circle("frequency", "power",
                                     source=tb_other_periodogram,
                                     alpha=0.7, 
                                     color = {'field': 'Mode', 
                                                'transform': color_mapper},
                                     #**self.env.selection,
                                     )

        fig_other_periodogram.line("frequency", "power",
                                   source=tb_other_periodogram,
                                   alpha=0.7, color="#1F77B4")

        fig_other_periodogram.ray(
            y="other_prd_cuttoff",
            x="minimum_frequency",
            source=self.tb_constants_val,
            name='threshold_line',
            length=8000,
            angle=0,
            color='red'
        )





        

        # fig_other_periodogram.circle("freq_values","power_values",
        #                             source=self.env.tb_grid_source,
        #                             alpha=0.7,**self.env.selection,)

        # fig_other_periodogram.line("freq_values","power_values",
        #                             source=self.env.tb_grid_source,
        #                             alpha=0.7,color="#1F77B4")

        #fig_other_periodogram.x_range.start = int(self.env.minimum_frequency)
        #fig_other_periodogram.x_range.end = int(self.env.maximum_frequency)

        #fig_other_periodogram.x_range.start = 200
        #fig_other_periodogram.x_range.end = 800

        return tb_other_periodogram, fig_other_periodogram

    def plot_vertical_lines(self):


        color_map = self.mode_color_map
        color_mapper = CategoricalColorMapper(
            factors=list(color_map.keys()), 
            palette=list(color_map.values()))

        # Reversed
        self.env.fig_other_periodogram.ray(x="Frequency",
                                           y=-300,
                        source= self.tb_se_second_source,
                        length=300, 
                        angle=np.pi/2,
                        color={'field': 'Mode', 
                                                'transform': color_mapper})

        self.env.fig_other_periodogram.ray(x="Frequency",y=0,
                        source= self.tb_se_second_source,
                        length=2137, 
                        angle=np.pi/2,
                        color={'field': 'Mode', 
                                                'transform': color_mapper})


    def make_tb_echelle_diagram(self):
        '''
        Width & Height
        '''
        if self.env.tb_echelle_diagram == None:
            print('Creating an echelle diagram, Creating column source')
            self.env.tb_echelle_diagram = ColumnDataSource(
                data=dict(image=[],
                          x_f=[],
                          y_f=[],
                          dw=[],
                          dh=[],
                          xmin=[],
                          ymin=[],
                          y_original=[],  # value without
                          # xx=[],
                          # yy=[],
                          # freq_values=[],
                          ))
        else:
            print('Refereshing everything')

        # Load the values
        frequency = self.env.tb_other_periodogram.data['frequency'] * \
            self.env.frequency_unit
        power = self.env.tb_other_periodogram.data['power']*self.env.power_unit
        (ep, self.x_echelle, self.y_echelle, y_original,self.xx, self.yy, 
         self.freq_values, self.power_values) = self._clean_echelle(
                            deltanu=self.dnu_val,
                            #minimum_frequency = self.env.minimum_frequency*self.env.frequency_unit,
                            #maximum_frequency = self.env.maximum_frequency*self.env.frequency_unit
                            )

        x_f = self.x_echelle
        y_f = self.y_echelle
        dw = (x_f.flatten().max() - x_f.flatten().min()).value
        dh = (y_f.flatten().max() - y_f.flatten().min()).value
        xmin = x_f.flatten().min().value
        ymin = y_f.flatten().min().value

        new_data = ColumnDataSource(
            data=dict(image=[ep.value],
                      x_f=[x_f.value],
                      y_f=[y_f.value],
                      dw=[dw],
                      dh=[dh],
                      xmin=[xmin],
                      ymin=[ymin],
                      y_original=[y_original],
                      # xx=[xx],
                      # yy=[yy],
                      # freq_values=[freq_values],
                      )
        )

        self.env.tb_echelle_diagram.data = new_data.data

    def initialize_selection_tables(self):

        # self.tb_se_first_source=ColumnDataSource(data=dict(x=[], y=[],z=[]))
        # self.tb_se_second_source=ColumnDataSource(data=dict(x=[], y=[], z=[]))

        self.tb_se_first_source = ColumnDataSource(
            data=dict(Slicefreq=[], Frequency=[], Power=[], Mode=[], xx=[]))
        self.tb_se_second_source = ColumnDataSource(
            data=dict(Slicefreq=[], Frequency=[], Power=[], Mode=[], xx=[], mode_color=[]))
        columns = [
            TableColumn(field="Slicefreq", title="Slice Freq"),
            TableColumn(field="Frequency", title="Frequency"),
            TableColumn(field="Power", title="Power"),
            TableColumn(field="Mode", title="Mode"),

        ]

        self.env.table_se_first = DataTable(
            source=self.tb_se_first_source,
            columns=columns,
            width=300,
            height=300,
            sortable=True,
            selectable=True,
            editable=True,
        )

        self.env.table_se_second = DataTable(
            source=self.tb_se_second_source,
            columns=columns,
            width=300,
            height=300,
            sortable=True,
            selectable="checkbox",
            editable=True,
        )




        # Figure 
        # self.env.fig_tpfint.circle(x='xx', 
        #                            y='Slicefreq',
        #                            size=2, 
        #                            fill_alpha=1,
        #                            color={'field': 'Mode', 
        #                                   'transform': color_mapper}, 
        #                            source=self.tb_se_second_source )



        # self.mode_color_map
        # self.env.fig_tpfint.circle(x='xx', y='Slicefreq',size=2, 
        #                            fill_alpha=0.2,
        #                            color='mode_color', 
        #                            source=self.tb_se_second_source )

        # from bokeh.transform import linear_cmap
        # from bokeh.palettes import Spectral4
        # self.env.fig_tpfint.circle(x='xx', y='Slicefreq', source=self.tb_se_second_source, 
        #                            size=10, 
        #                            color=linear_cmap('Mode', 
        #                                              palette=Spectral4, 
        #                                              factors=list(color_map.keys())), 
        #                                              legend_field='Mode', 
        #                                              fill_alpha=0.6)



        # self.env.fig_tpfint.circle(x='xx', y='Slicefreq', source=self.tb_se_second_source, 
        #                         size=10, 
        #                         color=linear_cmap('Mode', 
        #                                             palette=list(color_map.values()), 
        #                                             factors=list(color_map.keys())), 
        #                         legend_field='Mode', 
        #                         fill_alpha=0.6)

    def update_selection_tables(self):
        # self.env.tb_grid_source.selected.js_on_change(
        #     "indices",
        #     CustomJS(
        #         args=dict(s1=self.env.tb_grid_source,
        #                   dnu=self.env.dnu_val,
        #                   s2=self.tb_se_first_source,
        #                   table=self.env.table_se_first),

        #         code="""
        #         var inds = cb_obj.indices;
        #         var d1 = s1.data;
        #         var d2 = s2.data;
        #         //d2['SliceFreq'] = []
        #         //d2['Frequency'] = []
        #         for (var i = 0; i < inds.length; i++) {
        #             d2['Slicefreq'].push(d1['yy'][inds[i]])
        #             d2['Frequency'].push(d1['freq_values'][inds[i]])
        #             d2['Power'].push(d1['power_values'][inds[i]])
        #         }
        #         s2.change.emit();
        #         table.change.emit();

        #         var inds = source_data.selected.indices;
        #         var data = source_data.data;
        #         var out = "x, y\\n";
        #         for (i = 0; i < inds.length; i++) {
        #             out += data['x'][inds[i]] + "," + data['y'][inds[i]] + "\\n";
        #         }
        #         var file = new Blob([out], {type: 'text/plain'});

        #     """,
        #     ),
        # )

        self.env.tb_grid_source.selected.on_change(
            'indices', self.selection_grid_to_prd_fig)

        self.env.tb_other_periodogram.selected.on_change(
            'indices', self.selection_prd_to_grid_fig)
        
        self.env.tb_grid_source.selected.on_change(
            'indices', self.selection_grid_to_table_fig)


        # self.env.tb_grid_source.selected.on_change(
        #     'indices', self.selection_grid_to_table_fig)

        # self.env.tb_grid_source.selected.on_change(
        #     'indices', self.selection_table_to_prd_fig)

        # self.env.tb_other_periodogram.selected.on_change(
        #     'indices', self.selection_prd_to_grid_fig)
        # print('Nothing')



        




    def make_grid(self):
        '''

        Initialize grid source

        '''

        if self.env.tb_grid_source == None:
            self.env.tb_grid_source = ColumnDataSource(
                data=dict(
                    xx=[],
                    yy=[],
                    freq_values=[],
                    power_values=[],
                    Mode=[],
                )
            )

            if not self.mode_shape_selection == 'circle':
                # To be done properly
                self.env.fig_tpfint.rect('center_x', 'center_y', width.flatten(), height.flatten(), fill_color='gray',
                                         fill_alpha=0.2, line_color='blue', name='grid', source=self.env.tb_grid_source)
            else:
                

                color_map = self.mode_color_map
                color_mapper = CategoricalColorMapper(
                    factors=list(color_map.keys()), 
                    palette=list(color_map.values()))

                self.env.fig_tpfint.circle(x = 'xx', 
                                        y = 'yy',
                                        size = 2, 
                                        fill_alpha = 0.2, 
                                        line_color = {'field': 'Mode', 
                                                'transform': color_mapper}, 
                                        source=self.env.tb_grid_source)





                # color_mapper = CategoricalColorMapper(
                #     factors=list(color_map.keys()), 
                #     palette=list(color_map.values()))
                
                # self.env.fig_tpfint.circle(x='xx', y='yy',
                #                            size=2, 
                #                            fill_alpha=0.2, 
                #                            line_color='blue', 
                #                            source=self.env.tb_grid_source)

                # self.env.fig_tpfint.circle(x = 'xx', 
                #                            y = 'yy',
                #                            size = 2, 
                #                            fill_alpha = 0.2, 
                #                            color = {'field': 'Mode', 
                #                                   'transform': color_mapper}, 
                #                            source=self.env.tb_grid_source)



        cutt_off = float(self.env.echelle_noise_cuttoff_text.value)
        self.tb_constants_val.data['other_prd_cuttoff'] = list([cutt_off])
        
        val = float(self.env.frequency_minimum_text.value)
        self.tb_constants_val.data['minimum_frequency'] = list([val])
        print('Threshold', val)

        # self.freq_values = self.env.tb_grid_source.data['freq_values']
        # self.power_values = self.env.tb_grid_source.data['power_values']

        ind = np.array(self.power_values) >= cutt_off
        xx = np.array(self.xx)[ind]
        yy = np.array(self.yy)[ind]
        ff = np.array(self.freq_values)[ind]
        pp = np.array(self.power_values)[ind]
        # mm = list(np.ones(len(pp))*-1)
        # mm = list(map(str,mm))
        mm = ['-1']*(len(pp))
        #print(mm)
        # old_data = ColumnDataSource(
        #     data=dict(
        #         # center_x=center_x.flatten(),
        #         # center_y=center_y.flatten(),
        #         xx=self.xx,
        #         yy=self.yy,
        #         freq_values=self.freq_values,
        #         power_values=self.power_values,
        #     )
        # )

        old_data = ColumnDataSource(
            data=dict(
                # center_x=center_x.flatten(),
                # center_y=center_y.flatten(),
                xx=list(xx),
                yy=list(yy),
                freq_values=list(ff),
                power_values=list(pp),
                Mode = mm,
            )
        )

        self.env.tb_grid_source.data = old_data.data

    def read_fits_get_fp(self):
        '''
        Test function: Read fits file and get f and p"
        '''

        id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
        self.id_mycatalog = id_mycatalog
        filename=mycatalog.filename(
            id_mycatalog=id_mycatalog,
            name='other_psd')

        print('Running read fits',filename)
        from astropy.io import fits
        import pandas
        with fits.open(
            #'/Users/dp275303/work/tessipack_developement_test/PSD_ _no_gap.fits'
            filename
            ) as data:
            df = pandas.DataFrame(data[0].data)

        ff = df[0].values
        pp = df[1].values
        ff = ff.byteswap().newbyteorder()
        pp = pp.byteswap().newbyteorder()
        return ff, pp

    def update_value(self):
        self.env.minimum_frequency = float(
            self.env.frequency_minimum_text.value)
        self.env.maximum_frequency = float(
            self.env.frequency_maximum_text.value)
        self.env.maxdnu = float(self.env.frequency_maxdnu_text.value)
        self.env.dnu_val = float(self.env.dnu_text.value)
        self.env.dnu_slider.value = self.env.dnu_val

        print('Values transfering', self.env.minimum_frequency,
              self.env.maximum_frequency, self.env.maxdnu)
        
        #print('range', self.env.fig_other_periodogram.x_range.end)

        #print(self.env.fig_other_periodogram.x_range.end)
        self.update_plot(0, 0, 0)
        #self.env.fig_other_periodogram.x_range.start = int(self.env.minimum_frequency)
        #self.env.fig_other_periodogram.x_range.end = int(self.env.maximum_frequency)
        start = int(self.env.minimum_frequency)
        end = int(self.env.maximum_frequency)
       
        #self.env.fig_other_periodogram.x_range=(start, end)
        #print(self.env.fig_other_periodogram.x_range.end)

    def update_plot(self, attr, old, new):
        
        self.trim_frequency()

        self.id_mycatalog = self.env.tb_source.data['id_mycatalog'][0]
        dnu = self.dnu_val
        self.env.fig_tpfint.xaxis.axis_label = r'Frequency / {:.3f} Mod. 1'.format(
            dnu)

        self.make_tb_echelle_diagram()
        ep = self.env.tb_echelle_diagram.data['image']*self.env.power_unit

        lo, hi = np.nanpercentile(ep.value, [0.1, 99.9])
        vlo, vhi = 0.3 * lo, 1.7 * hi
        vstep = (lo - hi)/500
         
        self.palette = getattr(bokeh.palettes, 
                               self.env.select_color_palette.value)[9]

        if self.env.check_reverse_color_palette.active[0]!=1:
            self.palette = list(reversed(self.palette))
            

        color_mapper = LogColorMapper(palette=self.palette, low=lo, high=hi)
        self.env.fig_tpfint.select(
            'img').glyph.color_mapper.low = color_mapper.low
        self.env.fig_tpfint.select(
            'img').glyph.color_mapper.high = color_mapper.high
        
        self.env.fig_tpfint.select(
            'img').glyph.color_mapper.palette = getattr(color_mapper,'palette')
        
        self.env.stretch_sliderint.start = vlo
        self.env.stretch_sliderint.end = vhi
        self.env.stretch_sliderint.value = (lo, hi)
        self.env.dnu_slider.start = 0.01
        self.env.dnu_slider.end = self.env.maxdnu
        self.make_grid()

    def _validate_numax(self, numax):
        """
        Raises exception if `numax` is None and `self.numax` is not set.
        """
        if numax is None:
            try:
                return self.numax
            except AttributeError:
                raise AttributeError(
                    "You need to call `Seismology.estimate_numax()` first.")
        return numax

    def _validate_deltanu(self, deltanu):
        """
        Raises exception if `deltanu` is None and `self.deltanu` is not set.
        """
        if deltanu is None:
            try:
                print('Check here')
                return self.dnu_val
            except AttributeError:
                raise AttributeError(
                    "You need to call `Seismology.estimate_deltanu()` first.")
        return deltanu

    def _clean_echelle(self, deltanu=None, numax=None,
                       minimum_frequency=None, maximum_frequency=None,
                       smooth_filter_width=None, scale='linear'):
        """
        Takes input seismology object and creates the necessary arrays for an echelle

        diagram. Validates all the inputs.

        Parameters
        ----------
        deltanu : float
            Value for the large frequency separation of the seismic mode
            frequencies in the periodogram. Assumed to have the same units as
            the frequencies, unless given an Astropy unit.
            Is assumed to be in the same units as frequency if not given a unit.
        numax : float
            Value for the frequency of maximum oscillation. If a numax is
            passed, a suitable range one FWHM of the mode envelope either side
            of the will be shown. This is overwritten by custom frequency ranges.
            Is assumed to be in the same units as frequency if not given a unit.
        minimum_frequency : float
            The minimum frequency at which to display the echelle
            Is assumed to be in the same units as frequency if not given a unit.
        maximum_frequency : float
            The maximum frequency at which to display the echelle.
            Is assumed to be in the same units as frequency if not given a unit.
        smooth_filter_width : float
            If given a value, will smooth periodogram used to plot the echelle
            diagram using the periodogram.smooth(method='boxkernel') method with
            a filter width of `smooth_filter_width`. This helps visualise the
            echelle diagram. Is assumed to be in the same units as the
            periodogram frequency.
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.
        scale: str
            Set z axis to be "linear" or "log". Default is linear.

        Returns
        -------
        ep : np.ndarray
            Echelle diagram power
        x_f : np.ndarray
            frequencies for X axis
        y_f : np.ndarray
            frequencies for Y axis
        """
        # if (minimum_frequency is None) & (maximum_frequency is None):
        #     numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)

        # if (not hasattr(numax, 'unit')) & (numax is not None):
        #     numax = numax * self.periodogram.frequency.unit
        # if (not hasattr(deltanu, 'unit')) & (deltanu is not None):
        #     deltanu = deltanu * self.periodogram.frequency.unit
        deltanu = deltanu * self.periodogram.frequency.unit

        # freq = self.env.tb_other_periodogram.data['frequency'] * \
        #     self.env.frequency_unit
        # power = self.env.tb_other_periodogram.data['power']*self.env.power_unit

        # fmin = freq[0]
        # fmax = freq[-1]

        # # Check for any superfluous input
        # if (numax is not None) & (any([a is not None for a in [minimum_frequency, maximum_frequency]])):
        #     warnings.warn("You have passed both a numax and a frequency limit. "
        #                   "The frequency limit will override the numax input.",
        #                   LightkurveWarning)

        # # Ensure input numax is in the correct units (if there is one)
        # if numax is not None:
        #     numax = u.Quantity(numax, freq.unit).value
        #     if numax > freq[-1].value:
        #         raise ValueError("You can't pass in a numax outside the"
        #                          "frequency range of the periodogram.")

        #     fwhm = utils.get_fwhm(self.periodogram, numax)

        #     fmin = numax - 2*fwhm
        #     if fmin < freq[0].value:
        #         fmin = freq[0].value

        #     fmax = numax + 2*fwhm
        #     if fmax > freq[-1].value:
        #         fmax = freq[-1].value

        # # Set limits and set them in the right units
        # if minimum_frequency is not None:
        #     fmin = u.Quantity(minimum_frequency, freq.unit).value
        #     if fmin > freq[-1].value:
        #         raise ValueError('Fmin', fmin, "You can't pass in a limit outside the "
        #                          "frequency range of the periodogram.")

        # if maximum_frequency is not None:
        #     fmax = u.Quantity(maximum_frequency, freq.unit).value
        #     if fmax > freq[-1].value:
        #         raise ValueError('Fmax', fmax, "You can't pass in a limit outside the "
        #                          "frequency range of the periodogram.")

        # # Make sure fmin and fmax are Quantities or code below will break
        # fmin = u.Quantity(fmin, freq.unit)
        # fmax = u.Quantity(fmax, freq.unit)

        # # Add on 1x deltanu so we don't miss off any important range due to rounding
        # if fmax < freq[-1] - 1.5*deltanu:
        #     fmax += deltanu

        # fs = np.median(np.diff(freq))
        # x0 = int(freq[0] / fs)

        # ff = freq[int(fmin/fs)-x0:int(fmax/fs)-x0]  # Selected frequency range
        # pp = power[int(fmin/fs)-x0:int(fmax/fs)-x0]  # Power range

        # # Reshape the power into n_rows of n_columns
        # # When modulus ~ zero, deltanu divides into frequency without remainder
        # mod_zeros = find_peaks(-1.0*(ff % deltanu))[0]

        # # The bottom left corner of the plot is the lowest frequency that
        # # divides into deltanu with almost zero remainder
        # start = mod_zeros[0]

        # # The top left corner of the plot is the highest frequency that
        # # divides into deltanu with almost zero remainder.  This index is the
        # # approximate end, because we fix an integer number of rows and columns
        # approx_end = mod_zeros[-1]

        # # The number of rows is the number of times you can partition your
        # # frequency range into chunks of size deltanu, start and ending at
        # # frequencies that divide nearly evenly into deltanu
        # n_rows = len(mod_zeros) - 1

        # # The number of columns is the total number of frequency points divided
        # # by the number of rows, floor divided to the nearest integer value
        # n_columns = int((approx_end - start) / n_rows)

        # # The exact end point is therefore the ncolumns*nrows away from the start
        # end = start + n_columns*n_rows

        # ep = np.reshape(pp[start: end], (n_rows, n_columns))

        # if scale == 'log':
        #     ep = np.log10(ep)

        # # Reshape the freq into n_rowss of n_columnss & create arays
        # ef = np.reshape(ff[start: end], (n_rows, n_columns))
        # x_f = ((ef[0, :]-ef[0, 0]) % deltanu)
        # # Test : Scaling
        # x_f = ((ef[0, :]) % deltanu)
        # # print('x_f max',x_f.max(),deltanu)
        # y_f = (ef[:, 0])

        freq = self.env.tb_other_periodogram.data['frequency']
        power = self.env.tb_other_periodogram.data['power']

        data_frame = pd.DataFrame()
        data_frame['freq'] = freq
        data_frame['power'] = power

        minf = int(self.env.minimum_frequency)
        maxf = int(self.env.maximum_frequency)
        data_frame = data_frame.query('freq<@maxf & freq>@minf')

        ep, x_f, y_f, xx, yy, freq_values, power_values = self.apollinaire_echelle(
            data_frame.freq.values,
            data_frame.power.values,
            deltanu.value
        )

        y_original = y_f
        mean_diff = np.mean(np.diff(y_f))
        y_f_extra = y_f-(mean_diff/2)
        val = y_f[-1]+(mean_diff/2)
        center_y = np.append(y_f_extra, val)
        y_f = center_y

        ep = ep*self.env.power_unit
        x_f = x_f*self.env.frequency_unit
        y_f = y_f*self.env.frequency_unit
        return ep, x_f, y_f, y_original, xx, yy, freq_values, power_values

    def _make_echelle_elements(self, deltanu, cmap='hot',
                               minimum_frequency=None, maximum_frequency=None, smooth_filter_width=None,
                               scale='linear', plot_width=490, plot_height=340, title='Echelle'):
        """
        Helper function to make the elements of the echelle diagram for bokeh plotting.
        """

        freq = self.env.tb_other_periodogram.data['frequency'] * \
            self.env.frequency_unit
        ep = self.env.tb_echelle_diagram.data['image']*self.env.power_unit
        x_f = self.env.tb_echelle_diagram.data['x_f']*self.env.frequency_unit
        y_f = self.env.tb_echelle_diagram.data['y_f']*self.env.frequency_unit

        fig = figure(plot_width=800, plot_height=800,
                     # x_range=(0, 1), y_range=(y_f[0].value, y_f[-1].value),
                     title=title, tools='pan,box_zoom,reset,lasso_select',
                     toolbar_location="above",
                     border_fill_color="white", tooltips=self.env.TOOLTIPS)

        fig.yaxis.axis_label = r'Frequency [{}]'.format(freq.unit.to_string())
        fig.xaxis.axis_label = r'Frequency / {:.3f} Mod. 1'.format(deltanu)

        lo, hi = np.nanpercentile(ep.value, [0.1, 99.9])
        vlo, vhi = 0.3 * lo, 1.7 * hi
        vstep = (lo - hi)/500
        color_mapper = LogColorMapper(palette=self.palette, low=lo, high=hi)

        fig.image(image='image',
                  x='xmin',
                  y='ymin',
                  dw='dw',
                  dh='dh',
                  color_mapper=color_mapper,
                  name='img',
                  source=self.env.tb_echelle_diagram)

        stretch_slider = RangeSlider(start=vlo,
                                     end=vhi,
                                     step=vstep,
                                     title='',
                                     value=(lo, hi),
                                     orientation='vertical',
                                     width=10,
                                     height=230,
                                     direction='rtl',
                                     show_value=False,
                                     sizing_mode='fixed',
                                     name='stretch')

        def stretch_change_callback(attr, old, new):
            """TPF stretch slider callback."""
            fig.select('img')[0].glyph.color_mapper.high = new[1]
            fig.select('img')[0].glyph.color_mapper.low = new[0]

        stretch_slider.on_change('value', stretch_change_callback)

        return fig, stretch_slider

    def interact_echelle(self, notebook_url="localhost:8888", **kwargs):
        """
        Display an interactive Jupyter notebook widget showing an Echelle 
        diagram.

        This feature only works inside an active Jupyter Notebook, and
        requires an optional dependency, ``bokeh`` (v1.0 or later).
        This dependency can be installed using e.g. `conda install bokeh`.

        Parameters
        ----------
        notebook_url : str
            Location of the Jupyter notebook page (default: "localhost:8888")
            When showing Bokeh applications, the Bokeh server must be
            explicitly configured to allow connections originating from
            different URLs. This parameter defaults to the standard notebook
            host and port. If you are running on a different location, you
            will need to supply this value for the application to display
            properly. If no protocol is supplied in the URL, e.g. if it is
            of the form "localhost:8888", then "http" will be used.
        """
        try:
            import bokeh
            if bokeh.__version__[0] == '0':
                warnings.warn(
                    "interact() requires Bokeh version 1.0 or later", LightkurveWarning)
        except ImportError:
            log.error("The interact() tool requires the `bokeh` Python package; "
                      "you can install bokeh using e.g. `conda install bokeh`.")
            return None

        maximum_frequency = kwargs.pop(
            'maximum_frequency', self.periodogram.frequency.max().value)
        minimum_frequency = kwargs.pop(
            'minimum_frequency', self.periodogram.frequency. min().value)

        dnu = SeismologyQuantity(quantity=self.env.dnu_val*u.microhertz,
                                 name='deltanu', method='echelle')

        def create_interact_ui():
            self.env.fig_tpfint, self.env.stretch_sliderint = self._make_echelle_elements(deltanu=dnu,
                                                                                          maximum_frequency=maximum_frequency,
                                                                                          minimum_frequency=minimum_frequency,
                                                                                          **kwargs)
            # maxdnu = self.periodogram.frequency.max().value/5
            maxdnu = self.env.maxdnu
            # Interactive slider widgets
            self.env.dnu_slider = Slider(start=0.01,
                                         end=maxdnu,
                                         value=dnu.value,
                                         step=0.01,
                                         title="Delta Nu",
                                         width=290)
            self.env.r_button = Button(
                label=">", button_type="default", width=30)
            self.env.l_button = Button(
                label="<", button_type="default", width=30)
            self.env.rr_button = Button(
                label=">>", button_type="default", width=30)
            self.env.ll_button = Button(
                label="<<", button_type="default", width=30)

            def update(attr, old, new):
                """Callback to take action when dnu slider changes"""
                dnu = SeismologyQuantity(quantity=self.env.dnu_slider.value*self.env.frequency_unit,
                                         name='deltanu',
                                         method='echelle')

                self.make_tb_echelle_diagram()
                self.make_grid()

                self.env.fig_tpfint.xaxis.axis_label = r'Frequency / {:.3f} Mod. 1'.format(
                    dnu)
                self.env.dnu_val = self.env.dnu_slider.value
                self.env.dnu_text.value = str(self.env.dnu_slider.value)

            def go_right_by_one_small():
                """Step forward in time by a single cadence"""
                existing_value = self.env.dnu_slider.value
                if existing_value < self.env.maxdnu:
                    self.env.dnu_slider.value = existing_value + 0.1

            def go_left_by_one_small():
                """Step back in time by a single cadence"""
                existing_value = self.dnu_slider.value
                if existing_value > 0:
                    self.env.dnu_slider.value = existing_value - 0.1

            def go_right_by_one():
                """Step forward in time by a single cadence"""
                existing_value = self.env.dnu_slider.value
                if existing_value < self.env.maxdnu:
                    self.env.dnu_slider.value = existing_value + 1

            def go_left_by_one():
                """Step back in time by a single cadence"""
                existing_value = self.env.dnu_slider.value
                if existing_value > 0:
                    self.env.dnu_slider.value = existing_value - 1

            self.env.dnu_slider.on_change('value', update)
            self.env.r_button.on_click(go_right_by_one_small)
            self.env.l_button.on_click(go_left_by_one_small)
            self.env.rr_button.on_click(go_right_by_one)
            self.env.ll_button.on_click(go_left_by_one)

        create_interact_ui()

    def apollinaire_echelle(self, freq, PSD, dnu, modes, twice=False, fig=None, index=111,
                            figsize=(16, 16), title=None,
                            smooth=10, cmap='cividis', cmap_scale='linear',
                            mode_freq=None, mode_freq_err=None,
                            vmin=None, vmax=None, scatter_color='white', fmt='+', ylim=None,
                            shading='gouraud', mfc='none', ms=20, index_offset=None,
                            mec=None, xlabel=None, ylabel=None, **kwargs):
        '''
        Build the echelle diagram of a given PSD.  

        :param freq: input vector of frequencies.
        :type freq: ndarray

        :param PSD: input vector of power. Must be of same size than freq.
        :type PSD: ndarray

        :param dnu: the large frequency separation use to cut slices 
            into the diagram. 
        :type dnu: float

        :param twice: slice using 2 x *dnu* instead of *dnu*, default False.
        :type twice: bool

        :param fig: figure on which the echelle diagram will be plotted. If ``None``, a new figure 
            instance will be created. Optional, default ``None``. 
        :type fig: matplotlib Figure

        :param index: position of the echelle diagram Axe in the figure. Optional, default ``111``.
        :type index: int

        :param figsize: size of the echelle diagram to plot.
        :type figsize: tuple

        :param title: title of the figure. Optional, default ``(16, 16)``
        :type title: str

        :param smooth: size of the rolling window used to smooth the PSD. Default 10.
        :type smooth: int

        :param cmap: select one available color map provided by matplotlib, default ``cividis``
        :type cmap: str

        :param cmap_scale: scale use for the colormap. Can be 'linear' or 'logarithmic'.
            Optional, default 'linear'.
        :type cmap_scale: str

        :param mode_freq: frequency array of the modes to represent on the diagram. It can be single 
            array or a tuple of array.
        :type mode_freq: ndarray or tuple of array

        :param mode_freq_err: frequency uncertainty of the modes to represent on the diagram. It can be
            a single array or a tuple of array.
        :type mode_freq_err: ndarray or tuple of array

        :param vmin: minimum value for the colormap.
        :type vmin: float

        :param vmax: maximum value for the colormap.
        :type vmax: float

        :param scatter_color: color of the scatter point of the mode frequencies. Optional, default ``white``.
        :type scatter_color: str

        :param fmt: the format of the errorbar to plot. Can be a single string or a tuple of string with the same
            dimension that ``mode_freq``.
        :type fmt: str or tuple

        :param ylim: the y-bounds of the echelle diagram.
        :type ylim: tuple

        :param mew: marker edge width. Optional, default 1.
        :type mew: float

        :param markersize: size of the markers used for the errorbar plot. Optional, default 10.
        :type markersize: float

        :param capsize: length of the error bar caps. Optional, default 2.
        :type capsize: float

        :return: the matplotlib Figure with the echelle diagram.
        :rtype: matplotlib Figure
        '''

        # if cmap_scale not in ['linear', 'logarithmic'] :
        #     raise Exception ("cmap_scale should be set to 'linear' or 'logarithmic'.")

        # if cmap_scale=='logarithmic':
        #     norm = colors.LogNorm (vmin=vmin, vmax=vmax)
        # elif cmap_scale=='linear' :
        #     norm = colors.Normalize (vmin=vmin, vmax=vmax)

        # if smooth != 1 :
        #     PSD = pd.Series(data=PSD).rolling (window=smooth, min_periods=1,
        #                             center=True).mean().to_numpy()

        # if twice==True :
        #     dnu = 2.*dnu
        # print('freq',freq)
        res = freq[2]-freq[1]

        # if index_offset is not None :
        #     PSD = PSD[index_offset:]
        #     freq = freq[index_offset:]

        n_slice = int(np.floor_divide(freq[-1]-freq[0], dnu))
        len_slice = int(np.floor_divide(dnu, res))

        if (n_slice*len_slice > PSD.size):
            len_slice -= 1

        ed = PSD[:len_slice*n_slice]
        ed = np.reshape(ed, (n_slice, len_slice))

        freq_ed = freq[:len_slice*n_slice]
        freq_ed = np.reshape(freq_ed, (n_slice, len_slice))
        # x_freq = freq_ed[0,:] - freq_ed[0,0]
        x_freq = freq_ed[0, :]
        y_freq = freq_ed[:, 0]

        # if fig is None :
        #     fig = plt.figure (figsize=figsize)
        # ax = fig.add_subplot (index)
        # ax.pcolormesh (x_freq, y_freq, ed, cmap=cmap,
        #                 norm=norm, shading=shading)

        # if mode_freq is not None :
        #     if type (mode_freq) is not tuple :
        #     mode_freq = (mode_freq,)
        #     mode_freq_err = (mode_freq_err,)
        #     if type (scatter_color) is str :
        #     scatter_color = np.repeat (scatter_color, len (mode_freq))
        #     if type (fmt) is str :
        #     fmt = np.repeat (fmt, len (mode_freq))
        #     if type (ms) in [float, int] :
        #     ms = np.repeat (ms, len (mode_freq))
        #     if type (mfc) is str :
        #     mfc = np.repeat (mfc, len (mode_freq))
        #     if mec is None :
        #     mec = scatter_color
        #     if type (mec) is str :
        #     mec = np.repeat (mec, len (mode_freq))

        # for m_freq, m_freq_err, color, m_fmt, s, fc, edge in zip (mode_freq, mode_freq_err,
        #                                                         scatter_color, fmt, ms, mfc, mec) :
        # x_mode = np.zeros (m_freq.size)
        # y_mode = np.zeros (m_freq.size)
        # for ii, elt in enumerate (m_freq) :
        #     aux_1 = elt - y_freq
        #     aux_2 = y_freq[aux_1>0]
        #     aux_1 = aux_1[aux_1>0]
        #     jj = np.argmin (aux_1)
        #     x_mode[ii] = aux_1[jj]
        #     y_mode[ii] = aux_2[jj]
        # print (x_mode[ii], y_mode[ii], x_mode[ii]+y_mode[ii], elt)
        # ax.errorbar (x=x_mode, y=y_mode, xerr=m_freq_err, fmt=m_fmt, color=color, barsabove=True,
        #             mfc=fc, ms=s, mec=edge, **kwargs)

        # if xlabel is None :
        #     xlabel = r'$\nu$ mod. {:.1f} $\mu$Hz'.format (len_slice*res)
        # if ylabel is None :
        #     ylabel = r'$\nu$ ($\mu$Hz)'
        # ax.set_xlabel (xlabel)
        # ax.set_ylabel (ylabel)

        # ax.set_xlim (left=0, right=x_freq[-1])

        # if ylim is not None :
        #     ax.set_ylim (ylim[0], ylim[1])

        # if title is not None :
        #     ax.set_title (title)

        value = freq_ed
        x = x_freq
        y = y_freq
        xx, yy = np.meshgrid(x, y, sparse=False)
        xx = xx.reshape(-1, 1).flatten().tolist()
        yy = yy.reshape(-1, 1).flatten().tolist()
        freq_values = value.reshape(-1, 1).flatten().tolist()

        power_values = ed.reshape(-1, 1).flatten().tolist()

        return ed, x_freq, y_freq, xx, yy, freq_values, power_values

    def clear_se_table1(self):
        '''This function clear table1 of the program'''

        old_data = ColumnDataSource(
            data=dict(Slicefreq=[], Frequency=[], Power=[],Mode=[], xx=[]))
        self.tb_se_first_source.data = old_data.data

    def find_peak_frequencies(self):
        '''
        Find peak of the frequencies
        '''

        self.clear_se_table1()
        df_se = pd.DataFrame()
        yy = self.env.tb_grid_source.data['yy']
        xx = self.env.tb_grid_source.data['xx']
        freq_values = self.env.tb_grid_source.data['freq_values']
        power_values = self.env.tb_grid_source.data['power_values']

        df_se['freq_values'] = freq_values
        df_se['power_values'] = power_values
        df_se['xx'] = xx
        df_se['yy'] = yy

        se_indices = self.env.tb_grid_source.selected.indices
        real_freq = df_se.freq_values.loc[se_indices]
        real_power = df_se.power_values.loc[se_indices]
        xx_f = df_se.xx.loc[se_indices]
        yy_f = df_se.yy.loc[se_indices]
        df_se_freq = pd.DataFrame()
        df_se_freq['indices'] = se_indices
        df_se_freq['real_freq'] = real_freq.values
        df_se_freq['real_power'] = real_power.values
        df_se_freq['xx'] = xx_f.values
        df_se_freq['yy'] = yy_f.values

        # print(' All selected freq', df_se_freq)
        max_indices = df_se_freq.groupby('yy')['real_power'].idxmax()
        # print(' Max indices', max_indices)
        df_se_freq = df_se_freq.loc[max_indices]
        # print(' filetered', df_se_freq)

        self.tb_other_periodogram.selected.indices = list([])


        self.env.tb_grid_source.selected.indices = df_se_freq['indices'].to_list(
        )
        
        #print('Selected indices', self.env.tb_grid_source.selected.indices)

        # Clear prd selections

        # self.selection_grid_to_prd_fig(0,0,0)
        # self.update_plot(0, 0, 0)

        #self.get_all_selection_button()


    def selection_prd_to_grid_fig(self, attrname, old, new):
        """
        Selected frequencies from prd to grid
        """
        
        se_indices = self.tb_other_periodogram.selected.indices

        df_prd = self.env.tb_other_periodogram.to_df()
        df_prd = df_prd.loc[se_indices]

        selected_freq = list(np.round(df_prd.frequency.values, self.env.freq_round))

        se_indices = self.env.tb_grid_source.selected.indices
        df_grid = self.env.tb_grid_source.to_df()

        df_grid['freq_values'] = df_grid['freq_values'].round(self.env.freq_round)
        
        df_grid = df_grid.query('freq_values==@selected_freq')
        
        all_list = df_grid.index.to_list() + se_indices 

        #print('all list', all_list)
        if set(all_list) != set(se_indices):
            self.tb_grid_source.selected.indices = list(set(all_list))
            self.update_plot(0, 0, 0)
        self.env.fig_other_periodogram.x_range.start = int(self.env.minimum_frequency)
        self.env.fig_other_periodogram.x_range.end = int(self.env.maximum_frequency)


    def selection_grid_to_table_fig(self, attrname, old, new):
        """
        Selection from grid to table 
        """
        
        se_indices = self.env.tb_grid_source.selected.indices
        # print('Grid2Table se_indices', se_indices)
        df_grid = self.env.tb_grid_source.to_df()
        df_grid = df_grid.loc[se_indices]
        df_table = self.tb_se_first_source.to_df()
        df_grid.rename(columns={'yy':'Slicefreq', 
                       'freq_values':'Frequency',
                       'power_values':'Power'},
                       inplace=True,)
        df_grid = df_grid[['Slicefreq', 'Frequency', 'Power', 'Mode','xx']]
        all_data = pd.concat([df_table, df_grid],ignore_index=True)
        # print(all_data)
        all_data['Frequency'] = all_data['Frequency'].round(self.env.freq_round)
        all_data = all_data.drop_duplicates(subset=['Frequency'])
        #print(all_data.to_dict())
        #old_data = ColumnDataSource.from_df(all_data)
        df=all_data
        old_data = ColumnDataSource(
                    data=dict(Slicefreq = df['Slicefreq'].to_list(), 
                            Frequency = df['Frequency'].to_list(), 
                            Power = df['Power'].to_list(),
                            Mode = df['Mode'].to_list(),
                            xx = df['xx'].to_list()
                            ))

        self.tb_se_first_source.data = old_data.data




    def selection_grid_to_prd_fig(self, attrname, old, new):
        '''
        This function will not be used, slowing the system
        This function moves selected indices in grid to periodogram

        '''

        df_se = pd.DataFrame()
        yy = self.env.tb_grid_source.data['yy']
        xx = self.env.tb_grid_source.data['xx']
        freq_values = self.env.tb_grid_source.data['freq_values']

        df_se['xx'] = xx
        df_se['yy'] = yy
        df_se['freq_values'] = freq_values

        se_indices = self.env.tb_grid_source.selected.indices
        slice_freq = df_se.loc[se_indices]['yy']
        mod_val = df_se.loc[se_indices]['xx']

        real_freq = slice_freq+(self.dnu_val*mod_val)
        real_freq = real_freq.round(self.env.freq_round)
        real_freq = mod_val

        real_freq = df_se.freq_values.loc[se_indices]

        real_freq = np.round(real_freq, self.env.freq_round)
        real_freq = real_freq.tolist()

        df_prd = self.env.tb_other_periodogram.to_df()
        df_prd['frequency'] = df_prd['frequency'].round(self.env.freq_round)
        df_prd = df_prd.query('frequency==@real_freq')
        all_list = self.env.tb_other_periodogram.selected.indices + df_prd.index.to_list()
        
        
        if set(all_list) != set(self.env.tb_other_periodogram.selected.indices):
            self.env.tb_other_periodogram.selected.indices = list(set(all_list))

        

    def selection_table_to_prd_fig(self, attrname, old, new):
        """
        Table to prd selection
        """

        df_table = self.tb_se_first_source.to_df()
        selected_freq = df_table['Frequency'].round(self.env.freq_round)
        selected_freq = selected_freq.to_list()

        df_prd = self.env.tb_other_periodogram.to_df()
        df_prd['frequency'] = df_prd['frequency'].round(self.env.freq_round)  
        df_prd = df_prd.query('frequency==@selected_freq')

        all_list = self.env.tb_other_periodogram.selected.indices + df_prd.index.to_list()
        
        
        if set(all_list) != set(self.env.tb_other_periodogram.selected.indices):
            self.env.tb_other_periodogram.selected.indices = list(set(all_list))


    def selection_table2_to_prd_fig(self, attrname, old, new):
        """
        Table to prd selection
        """

        df_table = self.tb_se_second_source.to_df()
        selected_freq = df_table['Frequency'].round(self.env.freq_round)
        selected_freq = selected_freq.to_list()

        df_prd = self.env.tb_other_periodogram.to_df()
        df_prd['frequency'] = df_prd['frequency'].round(self.env.freq_round)  
        df_prd = df_prd.query('frequency==@selected_freq')

        all_list = self.env.tb_other_periodogram.selected.indices + df_prd.index.to_list()
        
        
        if set(all_list) != set(self.env.tb_other_periodogram.selected.indices):
            self.env.tb_other_periodogram.selected.indices = list(set(all_list))



    def get_all_selection_button(self):
        """
        Move selection around
        """

       
        # self.selection_grid_to_prd_fig(0,0,0)

        # self.selection_prd_to_grid_fig(0,0,0)
        
        # self.selection_grid_to_table_fig(0,0,0)
        # self.selection_table_to_prd_fig(0,0,0)
        # self.selection_prd_to_grid_fig(0,0,0)
        self.selection_grid_to_table_fig(0,0,0)
        self.selection_table_to_prd_fig(0,0,0)
        self.selection_prd_to_grid_fig(0,0,0)
        
        self.selection_grid_to_table_fig(0,0,0)
        self.selection_table_to_prd_fig(0,0,0)
        self.selection_prd_to_grid_fig(0,0,0)

    def clear_se_grid_prd(self):
        """
        Clear all selections
        """
        self.tb_other_periodogram.selected.indices = list([])
        self.env.tb_grid_source.selected.indices = list([])
        self.update_plot(0, 0, 0)



    def click_mode_apply_button(self):
        """
        Change mode selection
        """
        df_first=self.tb_se_first_source.to_df()
        df_first['Frequency']=df_first['Frequency'].round(
            self.env.freq_round)
        list_freq = df_first['Frequency'].to_list()
        
        print('Drop down value is', self.env.select_mode_menu.value)
        #ind = self.env.tb_grid_source.selected.indices 
        
        df_grid = self.env.tb_grid_source.to_df()
        df_grid['freq_values'] = df_grid['freq_values'].round(
            self.env.freq_round)
        ind = df_grid.query('freq_values == @list_freq').index

        df_grid.loc[ind,'Mode'] = self.env.select_mode_menu.value
        old_data = ColumnDataSource(df_grid.to_dict('list'))
        self.env.tb_grid_source.data = old_data.data

        #ind = self.env.tb_other_periodogram.selected.indices

        df_other = self.env.tb_other_periodogram.to_df()
        df_other['frequency'] = df_other['frequency'].round(
            self.env.freq_round)
        ind = df_other.query('frequency == @list_freq').index

        df_other.loc[ind,'Mode'] = self.env.select_mode_menu.value
        old_data = ColumnDataSource(df_other.to_dict('list'))
        self.env.tb_other_periodogram.data = old_data.data
        
        self.clear_se_table1()
        self.get_all_selection_button()


    # def get_index_from_frequency(self,df=None,name='Frequency',val_list=''):
        
    #     df['name']=df['name'].round(self.env.freq_round)






    def click_move_se_1_2_button(self):
        """
        Move frequencies from table 1 to table 2
        """

        mode = self.env.select_mode_menu.value
        df_table1 = self.tb_se_first_source.to_df()
        df_table1_se = df_table1.query('Mode == @mode')
        print(df_table1_se)
        #df_table1_left = df_table1[~df_table1.index.isin(df_table1_se.index)]
        df_table1_left = df_table1.query('Mode != @mode')
        print(df_table1_left.to_dict('list'))
        old_data = ColumnDataSource(df_table1_left.to_dict('list'))
        self.tb_se_first_source.data = old_data.data
        # Second table
        df_table2 = self.tb_se_second_source.to_df()
        #df_table1_se = df_table1.query('Mode == @mode')
        all_data = pd.concat([df_table2, df_table1_se], ignore_index=True)

        #all_data['mode_color']= self.assign_mode_color(all_data.Mode.values)
        #df_table2
        old_data = ColumnDataSource(all_data.to_dict('list'))
        self.tb_se_second_source.data = old_data.data

    def click_move_se_2_1_button(self):
        """
        Move frequencies from table 1 to table 2
        """
        # self.selection_table2_to_prd_fig(0,0,0)
        # self.selection_prd_to_grid_fig(0,0,0)
        # self.selection_grid_to_table_fig(0,0,0)

        mode = self.env.select_mode_menu.value
        df_table2 = self.tb_se_second_source.to_df()
        df_table2_se = df_table2.query('Mode == @mode')
        # print(df_table2_se)
        #df_table1_left = df_table1[~df_table1.index.isin(df_table1_se.index)]
        df_table2_left = df_table2.query('Mode != @mode')
        # print(df_table2_left.to_dict('list'))

        old_data = ColumnDataSource(df_table2_left.to_dict('list'))
        self.tb_se_second_source.data = old_data.data
        # Second table
        df_table1 = self.tb_se_first_source.to_df()
        #df_table1_se = df_table1.query('Mode == @mode')
        
        selected_freq = list(np.round(df_table2_se.Frequency.values, self.env.freq_round))
        df_grid = self.env.tb_grid_source.to_df()
        df_grid['freq_values'] = df_grid['freq_values'].round(self.env.freq_round)
        df_grid = df_grid.query('freq_values==@selected_freq')
        df_grid.rename(columns={'yy':'Slicefreq', 
                        'freq_values':'Frequency',
                        'power_values':'Power'},
                        inplace=True,)
        
        df_final = df_grid[['Slicefreq', 'Frequency', 'Power', 'Mode','xx']]
        df_final['mode_color'] = '' 
        
        all_data = pd.concat([df_table1, df_final], ignore_index=True)
        all_data =all_data.drop('mode_color',axis=1)
        print(all_data)
        all_data = all_data.drop_duplicates(subset=['Frequency'],
                                            ignore_index=True)
        #df_table2
        print(all_data.to_dict('list'))
        old_data = ColumnDataSource(all_data.to_dict('list'))
        self.tb_se_first_source.data = old_data.data

        #self.selection_table_to_prd_fig(0,0,0)
        #self.selection_prd_to_grid_fig(0,0,0)
        # self.selection_grid_to_table_fig(0,0,0)


    def assign_mode_color(self,modes):
        val=list(modes)
        val= map(str, val)

        val = list(map(lambda x: x.replace('-1', self.mode_color_map[-1]),val))
        val = list(map(lambda x: x.replace('0', self.mode_color_map[0]), val))
        val = list(map(lambda x: x.replace('1', self.mode_color_map[1]), val))
        val = list(map(lambda x: x.replace('2', self.mode_color_map[2]), val))
        return val
    
    def trim_frequency(self):

        ff, pp = self.read_fits_get_fp()
        #mm = ['-1']*(len(pp))
        
        min_freq = int(self.env.minimum_frequency)*self.env.frequency_unit
        max_freq = int(self.env.maximum_frequency)*self.env.frequency_unit
        #print('minimum',min_freq,ff)
        # ff = ff[(ff <= max_freq)]
        # pp = pp[(ff <= max_freq)]
        # print('maximum',max_freq,ff)

        ff = (ff*u.Hz).to(self.env.frequency_unit).value
        pp = pp*self.env.power_unit
        pp = pp.value
        
        #print('F and P with unit',ff, pp)
        
        ind=(ff <= max_freq.value) & (ff >= min_freq.value)

        f = ff[ind]
        p = pp[ind]
        f = f*self.env.frequency_unit
        p = p*self.env.power_unit
        mm = ['-1']*(len(p))

        #self.env.minimum_frequency = f.min().value
        #self.env.maximum_frequency = f.max().value

        #print(f,p)
        period = lk_prd_module.Periodogram(f, p)
        self.periodogram = period

        old_data= ColumnDataSource(
            data=dict(
                frequency=list(self.periodogram.frequency.value),
                power=list(self.periodogram.power.value),
                Mode = mm,
                # cuttoff=list(np.array([0])),
            ))  
        self.env.tb_other_periodogram.data=old_data.data

    def save_table_2(self):

        df=self.tb_se_second_source.to_df()
        filename=mycatalog.filename(
            id_mycatalog=self.id_mycatalog,
            name='other_psd_save_freq')
        
        print('Save Table 2 values',filename,self.tb_se_second_source.data)

        df.to_csv(filename,index=False)
        
    def load_table(self):

        filename=mycatalog.filename(
            id_mycatalog=self.id_mycatalog,
            name='other_psd_save_freq')
        df = pd.read_csv(filename)
        df['Mode']=df['Mode'].astype(str)
        df['Frequency']=df['Frequency'].astype(float)
        df['Frequency']=df['Frequency'].astype(float)
        #print(df)    
        old_data = ColumnDataSource(df.to_dict('list'))

        self.tb_se_second_source.data = old_data.data
        print('Load saved Table 2 values',
              filename,
              self.tb_se_second_source.data)

        #df['Freq']
        
        df_second=self.tb_se_second_source.to_df()
        df_second['Frequency']=df_second['Frequency'].round(
            self.env.freq_round)
        list_freq = df_second['Frequency'].to_list()

        for mode in df_second.Mode.unique():

            
            df_grid = self.env.tb_grid_source.to_df()
            df_grid['freq_values'] = df_grid['freq_values'].round(
                self.env.freq_round)
            ind = df_grid.query('freq_values == @list_freq').index

            df_grid.loc[ind,'Mode'] = mode
            old_data = ColumnDataSource(df_grid.to_dict('list'))
            self.env.tb_grid_source.data = old_data.data

            #ind = self.env.tb_other_periodogram.selected.indices

            df_other = self.env.tb_other_periodogram.to_df()
            df_other['frequency'] = df_other['frequency'].round(
                self.env.freq_round)
            ind = df_other.query('frequency == @list_freq').index

            df_other.loc[ind,'Mode'] = mode
            old_data = ColumnDataSource(df_other.to_dict('list'))
            self.env.tb_other_periodogram.data = old_data.data
        

    
