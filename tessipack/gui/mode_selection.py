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
from bokeh.models import Button, Select,CheckboxGroup  # for saving data

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

        self.id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
        self.palette="Spectral11"
        self.env.text_osc_query= TextInput(value='n_pg>=0', title="Select Cluster")

        #extra_flag_file='/home/dinilbose/PycharmProjects/light_cluster/cluster/Collinder_69/Data/extra_flag.flag'
        #extra_flag_file=self.env.extra_flag_file
        #lc=io.read_lightcurve(name='eleanor_flux',id_mycatalog=self.id_mycatalog,sector=self.env.sector,extra_flag_file=extra_flag_file,sigma=5)
        #f,p=maths.lomb_scargle(flux=lc['pca_flux'],time=lc.time,flux_err=lc.flux_err)
        # self.tb_source.on_change('data',self.update_id)
        #self.env.tb_periodogram.on_change('data',self.update_plot)
        self.env.tb_other_peridogram,self.env.fig_other_periodogram=self.initialize_dnu_periodogram()
        self.env.frequency_minimum_text=TextInput(value=str(self.env.minimum_frequency), title="Frequency_min",width=100)
        self.env.frequency_maximum_text=TextInput(value=str(self.env.maximum_frequency), title="Frequency_max",width=100)
        self.env.frequency_maxdnu_text=TextInput(value=str(self.env.maxdnu), title="Maxdnu",width=100)
        self.env.dnu_text=TextInput(value=str(self.env.dnu_val), title="Delta Nu",width=100)

        self.env.update_int_button = Button(label="Update Plot", button_type="success",width=150)
        self.env.update_int_button.on_click(self.update_value)
        self.interact_echelle()
        #self.mesa_interactive()


    def initialize_dnu_periodogram(self):
        '''Function initialise the the dnu periodogram table source and also the periodogram graph
        '''
        ff,pp=self.read_fits_get_fp()
        f=(ff*u.Hz).to(u.uHz)
        p=pp*u.electron/u.s

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




    def read_fits_get_fp(self):

        "Test function: Read fits file and get f and p"
        print('Running read fits')
        from astropy.io import fits
        import pandas
        with fits.open('/Users/dp275303/work/tessipack_developement_test/PSD_003429205_no_gap.fits') as data:
            df = pandas.DataFrame(data[0].data)
        return df[0].values,df[1].values

        

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



        #dnu = self.deltanu for current purpose
        dnu=self.dnu_val

        # f=self.env.tb_periodogram.data['x']*units.microhertz
        # p=self.env.tb_periodogram.data['y']*units.microhertz
        #extra_flag_file='/home/dinilbose/PycharmProjects/light_cluster/cluster/Collinder_69/Data/extra_flag.flag'
        # extra_flag_file=self.env.extra_flag_file
        # lc=io.read_lightcurve(name='eleanor_flux',id_mycatalog=self.id_mycatalog,sector=self.env.sector,extra_flag_file=extra_flag_file,sigma=5)
        # f,p=maths.lomb_scargle(flux=lc['pca_flux'],time=lc.time,flux_err=lc.flux_err)


        # ff,pp=self.read_fits_get_fp()
        # f=(ff*u.Hz).to(u.uHz)
        # p=pp*u.electron/u.s
        # from lightkurve import periodogram as lk_prd_module
        # period=lk_prd_module.Periodogram(f,p)

        # period=Periodo.Periodogram(frequency=f,power=p)

        # self.periodogram = period


        ep, x_f, y_f = self._clean_echelle(deltanu=dnu,
                                       minimum_frequency=self.env.minimum_frequency,
                                       maximum_frequency=self.env.maximum_frequency)
        self.env.fig_tpfint.select('img')[0].data_source.data['image'] = [ep.value]
        self.env.fig_tpfint.xaxis.axis_label = r'Frequency / {:.3f} Mod. 1'.format(dnu)





        #print('Test##',self.env.fig_tpfint.select('img').glyph.y)


        lo, hi = np.nanpercentile(ep.value, [0.1, 99.9])
        vlo, vhi = 0.3 * lo, 1.7 * hi
        vstep = (lo - hi)/500
        color_mapper = LogColorMapper(palette=self.palette, low=lo, high=hi)

        self.env.fig_tpfint.y_range.start = y_f[0].value
        self.env.fig_tpfint.y_range.end = y_f[-1].value


        self.env.fig_tpfint.select('img').glyph.y=y_f[0].value
        self.env.fig_tpfint.select('img').glyph.dh=y_f[-1].value
        self.env.fig_tpfint.select('img').glyph.color_mapper.low=color_mapper.low
        self.env.fig_tpfint.select('img').glyph.color_mapper.high=color_mapper.high
        self.env.stretch_sliderint.start=vlo
        self.env.stretch_sliderint.end=vhi
        self.env.stretch_sliderint.value=(lo,hi)

        self.env.dnu_slider.start=0.01
        self.env.dnu_slider.end=self.env.maxdnu


    def _validate_numax(self, numax):
        """Raises exception if `numax` is None and `self.numax` is not set."""
        if numax is None:
            try:
                return self.numax
            except AttributeError:
                raise AttributeError("You need to call `Seismology.estimate_numax()` first.")
        return numax

    def _validate_deltanu(self, deltanu):
        """Raises exception if `deltanu` is None and `self.deltanu` is not set."""
        if deltanu is None:
            try:
                return self.deltanu
            except AttributeError:
                raise AttributeError("You need to call `Seismology.estimate_deltanu()` first.")
        return deltanu


    def _clean_echelle(self, deltanu=None, numax=None,
                         minimum_frequency=None, maximum_frequency=None,
                         smooth_filter_width=.1, scale='linear'):
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

        if smooth_filter_width:
            pgsmooth = self.periodogram.smooth(filter_width=smooth_filter_width)
            freq = pgsmooth.frequency  # Makes code below more readable below
            power = pgsmooth.power     # Makes code below more readable below
        else:
            freq = self.periodogram.frequency  # Makes code below more readable
            power = self.periodogram.power     # Makes code below more readable

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
        #  When modulus ~ zero, deltanu divides into frequency without remainder
        mod_zeros = find_peaks( -1.0*(ff % deltanu) )[0]

        # The bottom left corner of the plot is the lowest frequency that
        # divides into deltanu with almost zero remainder
        start = mod_zeros[0]

        # The top left corner of the plot is the highest frequency that
        # divides into deltanu with almost zero remainder.  This index is the
        # approximate end, because we fix an integer number of rows and columns
        approx_end = mod_zeros[-1]

        # The number of rows is the number of times you can partition your
        #  frequency range into chunks of size deltanu, start and ending at
        #  frequencies that divide nearly evenly into deltanu
        n_rows = len(mod_zeros) - 1

        # The number of columns is the total number of frequency points divided
        #  by the number of rows, floor divided to the nearest integer value
        n_columns =  int( (approx_end - start) / n_rows )

        # The exact end point is therefore the ncolumns*nrows away from the start
        end = start + n_columns*n_rows

        ep = np.reshape(pp[start : end], (n_rows, n_columns))

        if scale=='log':
            ep = np.log10(ep)

        # Reshape the freq into n_rowss of n_columnss & create arays
        ef = np.reshape(ff[start : end], (n_rows, n_columns))
        x_f = ((ef[0,:]-ef[0,0]) % deltanu)
        y_f = (ef[:,0])
        return ep, x_f, y_f

    def _make_echelle_elements(self, deltanu, cmap='viridis',
        minimum_frequency=None, maximum_frequency=None, smooth_filter_width=None,
        scale='linear', plot_width=490, plot_height=340, title='Echelle'):
        """Helper function to make the elements of the echelle diagram for bokeh plotting.
        """
        if not hasattr(deltanu, 'unit'):
            deltanu = deltanu * self.periodogram.frequency.unit

        if smooth_filter_width:
            pgsmooth = self.periodogram.smooth(filter_width=smooth_filter_width)
            freq = pgsmooth.frequency  # Makes code below more readable below
        else:
            freq = self.periodogram.frequency  # Makes code below more readable

        ep, x_f, y_f = self._clean_echelle(deltanu=deltanu,
                                           minimum_frequency=minimum_frequency,
                                           maximum_frequency=maximum_frequency,
                                           smooth_filter_width=smooth_filter_width,
                                           scale=scale)
        

        fig = figure(plot_width=plot_width, plot_height=plot_height,
                     x_range=(0, 1), y_range=(y_f[0].value, y_f[-1].value),
                     title=title, tools='pan,box_zoom,reset,lasso_select',
                     toolbar_location="above",
                     border_fill_color="white",tooltips=self.env.TOOLTIPS)

        fig.yaxis.axis_label = r'Frequency [{}]'.format(freq.unit.to_string())
        fig.xaxis.axis_label = r'Frequency / {:.3f} Mod. 1'.format(deltanu)

        lo, hi = np.nanpercentile(ep.value, [0.1, 99.9])
        vlo, vhi = 0.3 * lo, 1.7 * hi
        vstep = (lo - hi)/500
        color_mapper = LogColorMapper(palette=self.palette, low=lo, high=hi)

        fig.image(image=[ep.value], x=x_f[0].value, y=y_f[0].value,
                  dw=1, dh=y_f[-1].value,
                  color_mapper=color_mapper, name='img')



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

        if not hasattr(self, 'deltanu'):
            dnu = SeismologyQuantity(quantity=self.periodogram.frequency.max()/30,
                                     name='deltanu', method='echelle')
        else:
            dnu = self.deltanu
        self.deltanu=dnu
        def create_interact_ui():
            self.env.fig_tpfint, self.env.stretch_sliderint = self._make_echelle_elements(dnu,
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
                dnu = SeismologyQuantity(quantity=self.env.dnu_slider.value*u.microhertz,
                                         name='deltanu', method='echelle')
                ep, x_f, y_f = self._clean_echelle(deltanu=dnu,
                                               minimum_frequency=self.env.minimum_frequency,
                                               maximum_frequency=self.env.maximum_frequency,
                                               smooth_filter_width=None,
                                               **kwargs)
                self.env.fig_tpfint.select('img')[0].data_source.data['image'] = [ep.value]
                #test
                # print('x===========',x_f[0].value)
                # print('y===========',y_f[0].value)
                # self.env.fig_tpfint.select('img')[0].data_source.data['x'] = x_f.value
                # self.env.fig_tpfint.select('img')[0].data_source.data['y'] = y_f.value



                print('y_f',y_f)
                print('x_f',x_f)


                # rows, cols = len(y_f), len(x_f)
                # width = np.diff(x_f).repeat(rows).reshape(cols-1, rows).T
                # height = np.diff(y_f).repeat(cols).reshape(rows, cols-2)

                x_f=x_f.value/x_f.value.max()
                y_f=y_f.value

                # Reshape x and y into 2D arrays
                x_2d = np.array(x_f).reshape(-1, 1)
                y_2d = np.array(y_f).reshape(-1, 1)

                # Create a grid from x_2d and y_2d
                xx, yy = np.meshgrid(x_2d, y_2d)

                # Create a figure

                width_mean = np.mean(np.diff(x_f))
                height_mean =np.mean(np.diff(y_f))
                width = np.diff(xx, axis=1)
                height = np.diff(yy, axis=0)


                center_x = xx[:-1, :-1] + width_mean / 2
                center_y = yy[:-1, :-1] + height_mean / 2

                # # Create a figure
                # x_coords = np.repeat(x_f[:-1], rows).reshape(cols-1, rows).T
                # y_coords = np.repeat(y_f[:-1], cols).reshape(rows, cols-1)
                # self.env.fig_tpfint.rect(xx.flatten(), yy.flatten(), width.flatten(), height.flatten(), fill_color='gray',
                #     fill_alpha=0.2, line_color='blue',name='new')

                self.env.fig_tpfint.rect(center_x.flatten(), center_y.flatten(), width.flatten(), height.flatten(), fill_color='gray',
                    fill_alpha=0.2, line_color='blue',name='new')


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




