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
from lightkurve.utils  import LightkurveWarning, validate_method
from lightkurve.seismology  import  SeismologyQuantity

from astropy import units
from tessipack.functions import maths
from tessipack.functions import io
import lightkurve
import Periodo
from env import Environment
from astropy import units
from bokeh.models import CustomJS, TextInput, Paragraph
from bokeh.models import Button, Select,CheckboxGroup,TableColumn,DataTable  # for saving data

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

log = logging.getLogger(__name__)



class Interactive(Environment):
    env=Environment

    def __init__(self):
        self.mode_shape_selection='circle'
        self.id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
        self.palette="Spectral11"
        self.env.text_osc_query= TextInput(value='n_pg>=0', title="Select Cluster")

        self.env.tb_other_periodogram,self.env.fig_other_periodogram=self.initialize_dnu_periodogram()
        self.env.frequency_minimum_text = TextInput(value=str(self.env.minimum_frequency), title="Frequency_min",width=100)
        self.env.frequency_maximum_text = TextInput(value=str(self.env.maximum_frequency), title="Frequency_max",width=100)
        self.env.frequency_maxdnu_text = TextInput(value=str(self.env.maxdnu), title="Maxdnu",width=100)
                
        self.env.dnu_text = TextInput(value=str(self.env.dnu_val), title="Delta Nu",width=100)
        self.env.update_int_button = Button(label="Update Plot", button_type="success",width=150)
        self.env.update_int_button.on_click(self.update_value)


        self.make_tb_echelle_diagram()
        self.interact_echelle()
        self.initialize_selection_tables()



        self.env.test_button = Button(label="Test", button_type="success",width=150)
        self.env.test_button.on_click(self.selection_table_to_prd_fig)



    def initialize_dnu_periodogram(self):
        '''
        Function initialise the the dnu periodogram table source and also the periodogram graph
        '''
        ff,pp=self.read_fits_get_fp()
        f=(ff*u.Hz).to(self.env.frequency_unit)
        p=pp*self.env.power_unit

        from lightkurve import periodogram as lk_prd_module
        period=lk_prd_module.Periodogram(f,p)
        #period=Periodo.Periodogram(frequency=f,power=p)

        self.periodogram = period
        self.env.minimum_frequency=1
        self.env.maximum_frequency=8000
        self.env.maxdnu=50
        self.env.dnu_val=20
        tb_other_periodogram = ColumnDataSource(
            data=dict(frequency=list(self.periodogram.frequency.value), 
                    power=list(self.periodogram.power.value)))

        #intialize other periodgram
        fig_other_periodogram = figure(
            plot_width=self.env.plot_width,
            plot_height=self.env.plot_height,
            tools=["box_zoom", "wheel_zoom","lasso_select", "tap" ,"reset", "save"],
            title="Other Periodogram",tooltips=self.env.TOOLTIPS,
        )
        fig_other_periodogram.circle("frequency", "power", source=tb_other_periodogram, alpha=0.7,**self.env.selection,)
        fig_other_periodogram.line("frequency","power",source=tb_other_periodogram,alpha=0.7,color="#1F77B4")


        fig_other_periodogram.x_range.start = 200
        fig_other_periodogram.x_range.end = 800
        return tb_other_periodogram,fig_other_periodogram


    def make_tb_echelle_diagram(self):


        if self.env.tb_echelle_diagram==None:
            print('Creating an echelle diagram, Creating column source')
            self.env.tb_echelle_diagram=ColumnDataSource(
                                        data=dict(image=[],
                                        x_f=[],
                                        y_f=[],
                                        dw=[],
                                        dh=[],
                                        ))
        else: 
            print('Refereshing everything')
        #Load the values
        frequency=self.env.tb_other_periodogram.data['frequency']*self.env.frequency_unit
        power=self.env.tb_other_periodogram.data['power']*self.env.power_unit
        ep, self.x_echelle, self.y_echelle = self._clean_echelle(deltanu=self.dnu_val,
                                minimum_frequency=self.env.minimum_frequency*self.env.frequency_unit,
                                maximum_frequency=self.env.maximum_frequency*self.env.frequency_unit)
        x_f=self.x_echelle 
        y_f=self.y_echelle
        dw=(x_f.flatten().max()-x_f.flatten().min()).value
        dh=(y_f.flatten().max()-y_f.flatten().min()).value
        
        new_data=ColumnDataSource(
                                data=dict(image=[ep.value],
                                x_f=[x_f.value],
                                y_f=[y_f.value],
                                dw=[dw],
                                dh=[dh],
                                ))
        self.env.tb_echelle_diagram.data = new_data.data
        
    def make_grid_fig(self): 
        

        self.env.fig_tpfint.rect('x_f', 'y_f', 'dw', 'dh', fill_color='gray',
            fill_alpha=0.2, line_color='blue',name='grid',source=self.env.tb_echelle_diagram)


    def initialize_selection_tables(self):
    
        self.tb_se_first_source=ColumnDataSource(data=dict(x=[], y=[]))
        self.tb_se_second_source=ColumnDataSource(data=dict(x=[], y=[]))
        
        columns = [
            TableColumn(field="x", title="Slice Freq"),
            TableColumn(field="y", title="Freq"),
            TableColumn(field="z", title="Mod"),
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
            source=self.tb_se_first_source,
            columns=columns,
            width=300,
            height=300,
            sortable=True,
            selectable=True,
            editable=True,
        )
    
    
    def selection_table_to_prd_fig(self):
        '''
        This function moves selected indices in grid to periodogram
        '''

        df_se=pd.DataFrame()
        yy=self.env.tb_grid_source.data['yy']
        xx=self.env.tb_grid_source.data['xx']
        df_se['xx']=xx
        df_se['yy']=yy
        se_indices=self.env.tb_grid_source.selected.indices

        slice_freq=df_se.loc[se_indices]['yy']
        mod_val=df_se.loc[se_indices]['xx']


        real_freq=slice_freq+(self.dnu_val*mod_val)
        real_freq=real_freq.round(self.env.freq_round)
        real_freq=real_freq.to_list()
        print(real_freq)
        df_prd=self.env.tb_other_periodogram.to_df()
        df_prd['frequency']=df_prd['frequency'].round(self.env.freq_round)
        df_prd=df_prd.query('frequency==@real_freq')
        print(df_prd)
        self.env.tb_other_periodogram.selected.indices=df_prd.index.to_list()
        print(df_prd.index.to_list())


    def update_selection_tables(self):
        self.env.tb_grid_source.selected.js_on_change(
            "indices",
            CustomJS(
                args=dict(s1=self.env.tb_grid_source,dnu=self.env.dnu_val, s2=self.tb_se_first_source, table=self.env.table_se_first),
                code="""
                var inds = cb_obj.indices;
                var d1 = s1.data;
                var d2 = s2.data;
                //d2['x'] = []
                //d2['y'] = []
                for (var i = 0; i < inds.length; i++) {
                    d2['x'].push(d1['center_y'][inds[i]])
                    d2['y'].push(d1['center_y'][inds[i]]+(dnu*d1['center_x'][inds[i]]))
                    d2['z'].push(d1['center_x'][inds[i]])
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
    

    def initialize_grid_delete(self,update=None):
        '''
        Initialize grid source
        '''
        
        x_f=self.x_echelle 
        y_f=self.y_echelle

        x_f=x_f.value
        y_f=y_f.value
        x_2d = np.array(x_f).reshape(-1, 1)
        y_2d = np.array(y_f).reshape(-1, 1)
        xx, yy = np.meshgrid(x_2d, y_2d)

        width_mean = np.mean(np.diff(x_f))
        height_mean =np.mean(np.diff(y_f))
        width = np.diff(xx, axis=1)
        height = np.diff(yy, axis=0)

        center_x = xx[:-1, :-1] + width_mean / 2
        center_y = yy[:-1, :-1] + height_mean / 2

        xx=xx[:-1, :-1]
        yy=yy[:-1, :-1]
        print('y_f', y_f)
        if not update:
            self.env.tb_grid_source = ColumnDataSource(
                data=dict(center_x=center_x.flatten(), 
                          center_y=center_y.flatten(),
                          xx=xx.flatten(),
                          yy=yy.flatten())
                          )
            
            if not self.mode_shape_selection=='circle':
                # To be done properly
                self.env.fig_tpfint.rect(center_x.flatten(), center_y.flatten(), width.flatten(), height.flatten(), fill_color='gray',
                    fill_alpha=0.2, line_color='blue',name='grid',source=self.env.tb_grid_source)
            else:
                self.env.fig_tpfint.circle(x='x_f', y='y_f',
                                         size=2,fill_alpha=0.2, line_color='blue',source=self.env.tb_grid_source)
        else:
            tb_old=ColumnDataSource(data=dict(center_x=center_x.flatten(),
                                              center_y=center_y.flatten(),
                                              xx=xx.flatten(),
                                              yy=yy.flatten())
                                              )
            self.env.tb_grid_source.data = tb_old.data


    def read_fits_get_fp(self):

        
        '''
        Test function: Read fits file and get f and p"
        '''
        print('Running read fits')
        from astropy.io import fits
        import pandas
        with fits.open('/Users/dp275303/work/tessipack_developement_test/PSD_003429205_no_gap.fits') as data:
            df = pandas.DataFrame(data[0].data)

        ff=df[0].values
        pp=df[1].values
        ff=ff.byteswap().newbyteorder()
        pp=pp.byteswap().newbyteorder()
        return ff,pp

        

    def update_value(self):
        self.env.minimum_frequency=float(self.env.frequency_minimum_text.value)
        self.env.maximum_frequency=float(self.env.frequency_maximum_text.value)
        self.env.maxdnu=float(self.env.frequency_maxdnu_text.value)
        self.env.dnu_val=float(self.env.dnu_text.value)
        self.env.dnu_slider.value=self.env.dnu_val

        print('Values transfering',self.env.minimum_frequency,self.env.maximum_frequency,self.env.maxdnu)
        self.update_plot(0,0,0)


    def update_plot(self,attr,old,new):

        self.id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
        dnu=self.dnu_val
        self.env.fig_tpfint.xaxis.axis_label = r'Frequency / {:.3f} Mod. 1'.format(dnu)


        self.make_tb_echelle_diagram()
        ep= self.env.tb_echelle_diagram.data['image']*self.env.power_unit

        lo, hi = np.nanpercentile(ep.value, [0.1, 99.9])
        vlo, vhi = 0.3 * lo, 1.7 * hi
        vstep = (lo - hi)/500
        color_mapper = LogColorMapper(palette=self.palette, low=lo, high=hi)

        self.env.fig_tpfint.select('img').glyph.color_mapper.low=color_mapper.low
        self.env.fig_tpfint.select('img').glyph.color_mapper.high=color_mapper.high
        self.env.stretch_sliderint.start=vlo
        self.env.stretch_sliderint.end=vhi
        self.env.stretch_sliderint.value=(lo,hi)

        self.env.dnu_slider.start=0.01
        self.env.dnu_slider.end=self.env.maxdnu


    def _validate_numax(self, numax):
        """
        Raises exception if `numax` is None and `self.numax` is not set.
        """
        if numax is None:
            try:
                return self.numax
            except AttributeError:
                raise AttributeError("You need to call `Seismology.estimate_numax()` first.")
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
                raise AttributeError("You need to call `Seismology.estimate_deltanu()` first.")
        return deltanu


    def _clean_echelle(self, deltanu=None, numax=None,
                         minimum_frequency=None, maximum_frequency=None,
                         smooth_filter_width=None, scale='linear'):
        """Takes input seismology object and creates the necessary arrays for an echelle
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
        if (minimum_frequency is None) & (maximum_frequency is None):
            numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)

        if (not hasattr(numax, 'unit')) & (numax is not None):
            numax = numax * self.periodogram.frequency.unit
        if (not hasattr(deltanu, 'unit')) & (deltanu is not None):
            deltanu = deltanu * self.periodogram.frequency.unit

        # if smooth_filter_width:
        #     print('Smooth filter applied')
        #     pgsmooth = self.periodogram.smooth(filter_width=smooth_filter_width)
        #     freq = pgsmooth.frequency  # Makes code below more readable below
        #     power = pgsmooth.power     # Makes code below more readable below
        # else:
        #     freq = self.periodogram.frequency  # Makes code below more readable
        #     power = self.periodogram.power     # Makes code below more readable
        freq = self.env.tb_other_periodogram.data['frequency']*self.env.frequency_unit
        power = self.env.tb_other_periodogram.data['power']*self.env.power_unit

        fmin = freq[0]
        fmax = freq[-1]

        # Check for any superfluous input
        if (numax is not None) & (any([a is not None for a in [minimum_frequency, maximum_frequency]])):
            warnings.warn("You have passed both a numax and a frequency limit. "
                          "The frequency limit will override the numax input.",
                          LightkurveWarning)

        # Ensure input numax is in the correct units (if there is one)
        if numax is not None:
            numax = u.Quantity(numax, freq.unit).value
            if numax > freq[-1].value:
                raise ValueError("You can't pass in a numax outside the"
                                "frequency range of the periodogram.")

            fwhm = utils.get_fwhm(self.periodogram, numax)

            fmin = numax - 2*fwhm
            if fmin < freq[0].value:
                fmin = freq[0].value

            fmax = numax + 2*fwhm
            if fmax > freq[-1].value:
                fmax = freq[-1].value

        # Set limits and set them in the right units
        if minimum_frequency is not None:
            fmin =  u.Quantity(minimum_frequency, freq.unit).value
            if fmin > freq[-1].value:
                raise ValueError('Fmin',fmin,"You can't pass in a limit outside the "
                                 "frequency range of the periodogram.")

        if maximum_frequency is not None:
            fmax = u.Quantity(maximum_frequency, freq.unit).value
            if fmax > freq[-1].value:
                raise ValueError('Fmax',fmax,"You can't pass in a limit outside the "
                                 "frequency range of the periodogram.")

        # Make sure fmin and fmax are Quantities or code below will break
        fmin = u.Quantity(fmin, freq.unit)
        fmax = u.Quantity(fmax, freq.unit)

        # Add on 1x deltanu so we don't miss off any important range due to rounding
        if fmax < freq[-1] - 1.5*deltanu:
            fmax += deltanu

        fs = np.median(np.diff(freq))
        x0 = int(freq[0] / fs)

        ff = freq[int(fmin/fs)-x0:int(fmax/fs)-x0] # Selected frequency range
        pp = power[int(fmin/fs)-x0:int(fmax/fs)-x0] # Power range

        # Reshape the power into n_rows of n_columns
        # When modulus ~ zero, deltanu divides into frequency without remainder
        mod_zeros = find_peaks( -1.0*(ff % deltanu) )[0]

        # The bottom left corner of the plot is the lowest frequency that
        # divides into deltanu with almost zero remainder
        start = mod_zeros[0]

        # The top left corner of the plot is the highest frequency that
        # divides into deltanu with almost zero remainder.  This index is the
        # approximate end, because we fix an integer number of rows and columns
        approx_end = mod_zeros[-1]

        # The number of rows is the number of times you can partition your
        # frequency range into chunks of size deltanu, start and ending at
        # frequencies that divide nearly evenly into deltanu
        n_rows = len(mod_zeros) - 1

        # The number of columns is the total number of frequency points divided
        # by the number of rows, floor divided to the nearest integer value
        n_columns =  int( (approx_end - start) / n_rows )

        # The exact end point is therefore the ncolumns*nrows away from the start
        end = start + n_columns*n_rows

        ep = np.reshape(pp[start : end], (n_rows, n_columns))

        if scale=='log':
            ep = np.log10(ep)

        # Reshape the freq into n_rowss of n_columnss & create arays
        ef = np.reshape(ff[start : end], (n_rows, n_columns))
        x_f = ((ef[0,:]-ef[0,0]) % deltanu)
        #Test : Scaling 
        x_f = ((ef[0,:]) % deltanu)
        #print('x_f max',x_f.max(),deltanu)
        y_f = (ef[:,0])
        return ep, x_f, y_f

    def _make_echelle_elements(self, deltanu, cmap='viridis',
        minimum_frequency=None, maximum_frequency=None, smooth_filter_width=None,
        scale='linear', plot_width=490, plot_height=340, title='Echelle'):
        """
        Helper function to make the elements of the echelle diagram for bokeh plotting.
        """
        # if not hasattr(deltanu, 'unit'):
        #     deltanu = deltanu * self.periodogram.frequency.unit

        # if smooth_filter_width:
        #     pgsmooth = self.periodogram.smooth(filter_width=smooth_filter_width)
        #     freq = pgsmooth.frequency  # Makes code below more readable below
        # else:
        #     freq = self.periodogram.frequency  # Makes code below more readable

        # ep, self.x_echelle, self.y_echelle = self._clean_echelle(deltanu=deltanu,
        #                                    minimum_frequency=minimum_frequency,
        #                                    maximum_frequency=maximum_frequency,
        #                                    smooth_filter_width=smooth_filter_width,
        #                                    scale=scale)

        freq = self.env.tb_other_periodogram.data['frequency']*self.env.frequency_unit
        ep= self.env.tb_echelle_diagram.data['image']*self.env.power_unit
        x_f=self.env.tb_echelle_diagram.data['x_f']*self.env.frequency_unit
        y_f=self.env.tb_echelle_diagram.data['y_f']*self.env.frequency_unit
        # dw=self.env.tb_echelle_diagram.data['dw']
        # dh=self.env.tb_echelle_diagram.data['dh']


        fig = figure(plot_width=plot_width, plot_height=plot_height,
                     #x_range=(0, 1), y_range=(y_f[0].value, y_f[-1].value),
                     title=title, tools='pan,box_zoom,reset,lasso_select',
                     toolbar_location="above",
                     border_fill_color="white",tooltips=self.env.TOOLTIPS)

        fig.yaxis.axis_label = r'Frequency [{}]'.format(freq.unit.to_string())
        fig.xaxis.axis_label = r'Frequency / {:.3f} Mod. 1'.format(deltanu)

        lo, hi = np.nanpercentile(ep.value, [0.1, 99.9])
        vlo, vhi = 0.3 * lo, 1.7 * hi
        vstep = (lo - hi)/500
        color_mapper = LogColorMapper(palette=self.palette, low=lo, high=hi)

        # dw=(x_f.flatten().max()-x_f.flatten().min()).value
        # dh=(y_f.flatten().max()-y_f.flatten().min()).value


        # print('y_f',y_f)
        # print('print',dw,dh)

        # fig.image(image=[ep.value], x=x_f.min().value, y=y_f.min().value,
        #           dw=dw, dh=dh,
        #           color_mapper=color_mapper, name='img')

        fig.image(image='image', 
                  x=x_f.min().value, 
                  y=y_f.min().value,
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
        """Display an interactive Jupyter notebook widget showing an Echelle diagram.

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
                warnings.warn("interact() requires Bokeh version 1.0 or later", LightkurveWarning)
        except ImportError:
            log.error("The interact() tool requires the `bokeh` Python package; "
                      "you can install bokeh using e.g. `conda install bokeh`.")
            return None

        maximum_frequency = kwargs.pop('maximum_frequency', self.periodogram.frequency.max().value)
        minimum_frequency = kwargs.pop('minimum_frequency', self.periodogram.frequency. min().value)

        # if not hasattr(self, 'deltanu'):
        #     dnu = SeismologyQuantity(quantity=self.periodogram.frequency.max()/30,
        #                              name='deltanu', method='echelle')
        # else:
        #     dnu = self.deltanu
        #print('dnu',self.deltanu)
        # self.deltanu=dnu
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
            self.env.r_button = Button(label=">", button_type="default", width=30)
            self.env.l_button = Button(label="<", button_type="default", width=30)
            self.env.rr_button = Button(label=">>", button_type="default", width=30)
            self.env.ll_button = Button(label="<<", button_type="default", width=30)





            def update(attr, old, new):
                """Callback to take action when dnu slider changes"""
                dnu = SeismologyQuantity(quantity=self.env.dnu_slider.value*self.env.frequency_unit,
                                         name='deltanu', 
                                         method='echelle')
                # ep, self.x_echelle, self.y_echelle = self._clean_echelle(deltanu=dnu,
                #                                minimum_frequency=self.env.minimum_frequency*self.env.frequency_unit,
                #                                maximum_frequency=self.env.maximum_frequency*self.env.frequency_unit,
                #                                smooth_filter_width=None,
                #                                **kwargs)
                self.make_tb_echelle_diagram()

                # self.env.fig_tpfint.select('img')[0].data_source.data['image'] = [ep.value]
                
                # x_f=self.x_echelle 
                # y_f=self.y_echelle
                # dw=(x_f.flatten().max()-x_f.flatten().min()).value
                # dh=(y_f.flatten().max()-y_f.flatten().min()).value
                # self.env.fig_tpfint.select('img')[0].glyph.dw = dw
                # self.env.fig_tpfint.select('img')[0].glyph.dh = dh
                
                
                self.env.fig_tpfint.xaxis.axis_label = r'Frequency / {:.3f} Mod. 1'.format(dnu)
                self.env.dnu_val=self.env.dnu_slider.value
                self.env.dnu_text.value=str(self.env.dnu_slider.value)

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

            # widgets_and_figures = layout([fig_tpf, [Spacer(height=20), stretch_slider]],
            #                              [ll_button, Spacer(width=30), l_button,
            #                               Spacer(width=25), dnu_slider, Spacer(width=30),
            #                               r_button, Spacer(width=23), rr_button])
            #doc.add_root(widgets_and_figures)
        create_interact_ui()




