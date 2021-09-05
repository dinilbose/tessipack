import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import warnings, os, requests
import lightkurve as lk
from bs4 import BeautifulSoup
from pylab import *
from astropy.timeseries import LombScargle

from .ffi import use_pointing_model, load_pointing_model
from .mast import *

__all__ = []


class Visualize(object):
    """
    The main class for creating figures, movies, and interactive plots.
    Allows the user to have a grand ole time playing with their data!

    Parameters
    ----------
    obj :
        Object must have minimum attributes of 2D array of flux.
        Will allow for plotting of both postcards & tpfs.
    obj_type :
        Object type can be set to "tpf" or "postcard". Default is "tpf".
    """

    def __init__(self, object, obj_type="tpf"):
        self.obj      = object
        self.obj_type = obj_type.lower()
        self.get_youtube_links()

        if self.obj_type == "tpf":
            self.flux   = self.obj.tpf
            self.center = (np.nanmedian(self.obj.centroid_xs),
                             np.nanmedian(self.obj.centroid_ys))
            self.dimensions = self.obj.tpf[0].shape
        else:
            self.flux   = self.obj.flux
            self.center = self.obj.center_xy
            self.dimensions = self.obj.dimensions


    def get_youtube_links(self):
        """
        Scrapes the YouTube links to Ethan Kruse's TESS: The Movie videos.

        Parameters
        ---------- 

        Attributes
        ----------
        youtube : dict

        """
        url = "https://www.youtube.com/user/ethank18/videos"
        paths = BeautifulSoup(requests.get(url).text, "lxml").find_all('a')

        videos = {}

        for direct in paths:
            name = str(direct.get('title'))
            if 'TESS: The Movie.' in name:
                sector = int(name.split(',')[0].split('.')[-1].split(' ')[-1])
                link = direct.get('href')
                link_path = "https://www.youtube.com/" + link
                videos[sector] = link_path
        self.youtube = videos


    def aperture_contour(self, aperture=None, ap_color='w', ap_linewidth=4, **kwargs):
        """
        Overplots the countour of an aperture on a target pixel file.
        Contribution from Gijs Mulders. 

        Parameters
        ---------- 
        aperture : np.2darray, optional
            A 2D mask the same size as the target pixel file. Default
            is the eleanor default aperture.
        ap_color : str, optional
            The color of the aperture contour. Takes a matplotlib color.
            Default is red.
        ap_linewidth : int, optional
            The linewidth of the aperture contour. Default is 4.
        """

        fig = plt.figure()

        if aperture is None:
            aperture = self.obj.aperture

        plt.imshow(self.obj.tpf[0], **kwargs)

        f = lambda x,y: aperture[int(y),int(x) ]
        g = np.vectorize(f)
        
        x = np.linspace(0,aperture.shape[1], aperture.shape[1]*100)
        y = np.linspace(0,aperture.shape[0], aperture.shape[0]*100)
        X, Y= np.meshgrid(x[:-1],y[:-1])
        Z = g(X[:-1],Y[:-1])
        
        plt.contour(Z[::-1], [0.5], colors=ap_color, linewidths=[ap_linewidth],
                    extent=[0-0.5, x[:-1].max()-0.5,0-0.5, y[:-1].max()-0.5])
        
        return fig



    def pixel_by_pixel(self, colrange=None, rowrange=None, cmap='viridis',
                       data_type="corrected", mask=None, xlim=None,
                       ylim=None, color_by_pixel=False, freq_range=[1/20., 1/0.1]):
        """
        Creates a pixel-by-pixel light curve using the corrected flux.
        Contribution from Oliver Hall.

        Parameters
        ----------
        colrange : np.array, optional
             A list of start column and end column you're interested in
             zooming in on.
        rowrange : np.array, optional
             A list of start row and end row you're interested in zooming
             in on.
        cmap : str, optional
             Name of a matplotlib colormap. Default is 'viridis'.
        data_type : str, optional
             The type of flux used. Either: 'raw', 'corrected', 'amplitude',
             or 'periodogram'. If not, default set to 'corrected'.
        mask : np.array, optional
             Specifies the cadences used in the light curve. If not, default
             set to good quality cadences.
        xlim : np.array, optional
             Specifies the xlim on the subplots. If not, default is set to 
             the entire light curve.
        ylim : np.array, optional
             Specifies the ylim on the subplots, If not, default is set to 
             the entire light curve flux range.
        color_by_pixel : bool, optional
             Colors the light curve given the color of the pixel. If not,
             default is set to False.
        freq_range : list, optional
             List of minimum and maximum frequency to search in Lomb Scargle
             periodogram. Only used if data_type = 'periodogram'. If None,
             default = [1/20., 1/0.1].
        """
        if colrange is None:
            colrange = [0, self.dimensions[1]]

        if rowrange is None:
            rowrange = [0, self.dimensions[0]]

        nrows = int(np.round(colrange[1]-colrange[0]))
        ncols = int(np.round(rowrange[1]-rowrange[0]))

        if (colrange[1] > self.dimensions[1]) or (rowrange[1] > self.dimensions[0]):
            raise ValueError("Asking for more pixels than available in the TPF.")


        figure = plt.figure(figsize=(20,8))
        outer = gridspec.GridSpec(1,2, width_ratios=[1,4])

        inner = gridspec.GridSpecFromSubplotSpec(ncols, nrows, hspace=0.1, wspace=0.1,
                                                 subplot_spec=outer[1])

        i, j = rowrange[0], colrange[0]

        if mask is None:
            q = self.obj.quality == 0
        else:
            q = mask == 0


        ## PLOTS TARGET PIXEL FILE ##
        ax = plt.subplot(outer[0])
        
        c = ax.imshow(self.flux[100, rowrange[0]:rowrange[1],
                                colrange[0]:colrange[1]],
                      vmax=np.percentile(self.flux[100], 95),
                      cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        plt.colorbar(c, cax=cax, orientation='vertical')

        ## PLOTS PIXEL LIGHT CURVES ##
        for ind in range( int(nrows * ncols) ):
            ax = plt.Subplot(figure, inner[ind])

            flux = self.flux[:,i,j]
            time = self.obj.time
            corr_flux = self.obj.corrected_flux(flux=flux)

            if data_type.lower() == 'corrected':
                y = corr_flux[q]/np.nanmedian(corr_flux[q])
                x = time[q]

            elif data_type.lower() == 'amplitude':
                lc = lk.LightCurve(time=time, flux=corr_flux)
                pg = lc.normalize().to_periodogram()
                x = pg.frequency.value
                y = pg.power.value

            elif data_type.lower() == 'raw':
                y = flux[q]/np.nanmedian(flux[q])
                x = time[q]
            
            elif data_type.lower() == 'periodogram':
                freq, power = LombScargle(time, corr_flux).autopower(minimum_frequency=freq_range[0],
                                                                     maximum_frequency=freq_range[1],
                                                                     method='fast')
                y = power
                x = 1/freq

            if color_by_pixel is False:
                color = 'k'
            else:
                rgb = c.cmap(c.norm(self.flux[100,i,j]))
                color = matplotlib.colors.rgb2hex(rgb)

            ax.plot(x, y, c=color)

            j += 1
            if j == colrange[1]:
                i += 1
                j  = colrange[0]

            if ylim is None:
                ax.set_ylim(np.percentile(y, 1), np.percentile(y, 99))
            else:
                ax.set_ylim(ylim[0], ylim[1])

            if xlim is None:
                ax.set_xlim(np.min(x)-0.1, np.max(x)+0.1)
            else:
                ax.set_xlim(xlim[0], xlim[1])

            if data_type.lower() == 'amplitude':
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.set_ylim(y.min(), y.max())
                ax.set_xlim(np.min(x),
                            np.max(x))
#                ax.set_xticks([])
#                ax.set_yticks([])

            ax.set_xticks([])
            ax.set_yticks([])

            figure.add_subplot(ax)

        return figure


    def tess_the_movie(self):
        """
        Opens the link to Ethan Kruse's TESS: The Movie YouTube videos for
        the sector your target is observed in.

        Parameters
        ---------- 

        Attributes
        ----------
        movie_url : str

        """
        def type_of_script():
            try:
                ipy_str = str(type(get_ipython()))
                if 'zmqshell' in ipy_str:
                    return 'jupyter'
                if 'terminal' in ipy_str:
                    return 'ipython'
            except:
                return 'terminal'

        sector = self.obj.source_info.sector
        self.movie_url = self.youtube[sector]

        call_location = type_of_script()

        if (call_location == 'terminal') or (call_location == 'ipython'):
            os.system('python -m webbrowser -t "{0}"'.format(self.movie_url))

        elif (call_location == 'jupyter'):
            from IPython.display import YouTubeVideo
            id = self.movie_url.split('=')[-1]
            return YouTubeVideo(id=id, width=900, height=500)

            
