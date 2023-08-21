import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from photutils import CircularAperture, RectangularAperture, aperture_photometry
from photutils import MMMBackground
from lightkurve import lightcurve
from lightkurve.correctors import SFFCorrector
from scipy.optimize import minimize
from astropy import time, coordinates as coord, units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table, Column
from astropy.wcs import WCS
from astropy.stats import SigmaClip, sigma_clip
from time import strftime
from astropy.io import fits
from scipy.stats import mode
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
from urllib.request import urlopen
import os, sys, copy
import os.path
import warnings
import pickle
from tqdm import tqdm
from eleanor.ffi import use_pointing_model, load_pointing_model, centroid_quadratic
from eleanor.postcard import Postcard, Postcard_tesscut
from eleanor.models import Gaussian, Moffat
from eleanor.utils import *
from eleanor import TargetData 
from eleanor import Source
from tessipack import config
from eleanor.source import *
from eleanor.maxsector import maxsector

class TargetData(TargetData):
    print('Running Patched Eleanor')
    def get_lightcurve(self, aperture=None,bkg_type=None):
        """Extracts a light curve using the given aperture and TPF.
        Can pass a user-defined aperture mask, otherwise determines which of a
        set of pre-determined apertures
        provides the lowest scatter in the light curve.
        Produces a mask, a numpy.ndarray object of the same shape as the target
        pixel file, which every pixel assigned
        a weight in the range [0, 1].

        Parameters
        ----------
        aperture : numpy.ndarray
            (`height`, `width`) array of floats in the range [0,1] with desired
            weights for each pixel to
            create a light curve. If not set, ideal aperture is inferred
            automatically. If set, uses this
            aperture at the expense of all other set apertures.
        """

        def apply_mask(mask):
            lc     = np.zeros(len(self.tpf))
            lc_err = np.zeros(len(self.tpf))
            for cad in range(len(self.tpf)):
                lc[cad]     = np.nansum( self.tpf[cad] * mask)
                lc_err[cad] = np.sqrt( np.nansum( self.tpf_err[cad]**2 * mask))
            self.raw_flux   = np.array(lc)
            self.corr_flux  = self.corrected_flux(flux=lc, skip=0.25)
            self.flux_err   = np.array(lc_err)
            return

        self.flux_err = None
        if aperture is not None:
            self.aperture = aperture

        if self.language == 'Australian':
            print("G'day Mate! ʕ •ᴥ•ʔ Your light curves are being translated ...")


        if self.aperture is not None:
            if np.shape(self.all_apertures[0]) != np.shape(self.aperture):
                raise ValueError(
                    "Passed aperture does not match the size of the TPF. Please \
                    correct and try again. "
                    "Or, create a custom aperture using the function \
                    TargetData.custom_aperture(). See documentation for inputs.")

            self.all_apertures = np.zeros((1, np.shape(self.tpf[0])[0], np.shape(self.tpf[0])[1]))
            self.all_apertures[0] = self.aperture

        self.all_flux_err  = None

        all_raw_lc_pc_sub  = np.zeros((len(self.all_apertures), len(self.tpf)))
        all_lc_err  = np.zeros((len(self.all_apertures), len(self.tpf)))
        all_corr_lc_pc_sub = np.copy(all_raw_lc_pc_sub)

        # TPF background subtracted light curves
        all_raw_lc_tpf_sub = np.zeros((len(self.all_apertures), len(self.tpf)))
        all_corr_lc_tpf_sub = np.copy(all_raw_lc_tpf_sub)

        # 2D background subtracted light curves
        all_raw_lc_tpf_2d_sub = np.copy(all_raw_lc_tpf_sub)
        all_corr_lc_tpf_2d_sub = np.copy(all_raw_lc_tpf_sub)

        if self.source_info.tc == True:
            for epoch in range(len(self.time)):
                self.tpf[epoch] -= self.tpf_flux_bkg[epoch]

        pc_stds  = np.ones(len(self.all_apertures))
        tpf_stds = np.ones(len(self.all_apertures))
        stds_2d  = np.ones(len(self.all_apertures))

        ap_size = np.nansum(self.all_apertures, axis=(1,2))

        bkg_subbed = self.tpf + (self.flux_bkg - self.tpf_flux_bkg)[:, None, None]
        if not self.source_info.tc:
            bkg_subbed_2 = self.tpf - self.bkg_tpf

        for a in range(len(self.all_apertures)):
            try:
                all_lc_err[a] = np.sqrt( np.nansum(self.tpf_err**2 * self.all_apertures[a], axis=(1,2)))
                all_raw_lc_pc_sub[a] = np.nansum( (self.tpf * self.all_apertures[a]), axis=(1,2) )
                all_raw_lc_tpf_sub[a]  = np.nansum( (bkg_subbed * self.all_apertures[a]), axis=(1,2) )

                if self.source_info.tc == False:
                    all_raw_lc_tpf_2d_sub[a] = np.nansum( bkg_subbed_2 * self.all_apertures[a],
                                                            axis=(1,2))
            except ValueError:
                continue

            ## Remove something from all_raw_lc before passing into jitter_corr ##
            try:
                norm = np.nansum(self.all_apertures[a], axis=1)
                all_corr_lc_pc_sub[a] = self.corrected_flux(flux=all_raw_lc_pc_sub[a]/np.nanmedian(np.abs(all_raw_lc_pc_sub[a])),
                                                            bkg=self.flux_bkg[:, None] * norm)
                all_corr_lc_tpf_sub[a]= self.corrected_flux(flux=all_raw_lc_tpf_sub[a]/np.nanmedian(np.abs(all_raw_lc_tpf_sub[a])),
                                                            bkg=self.tpf_flux_bkg[:, None] * norm)

                if self.source_info.tc == False:
                    all_corr_lc_tpf_2d_sub[a] = self.corrected_flux(flux=all_raw_lc_tpf_2d_sub[a]/np.nanmedian(np.abs(all_raw_lc_tpf_2d_sub[a])),
                                                                    bkg=np.nansum(self.bkg_tpf*self.all_apertures[a], axis=(1,2)))


            except IndexError:
                continue

            q = self.quality == 0

            tpf_stds[a] = get_flattened_sigma(all_corr_lc_tpf_sub[a][q][self.cal_cadences[0]:self.cal_cadences[1]])
            pc_stds[a] = get_flattened_sigma(all_corr_lc_pc_sub[a][q][self.cal_cadences[0]:self.cal_cadences[1]])

            if self.source_info.tc == False:
                stds_2d[a] = get_flattened_sigma(all_corr_lc_tpf_2d_sub[a][q][self.cal_cadences[0]:self.cal_cadences[1]])
                all_corr_lc_tpf_2d_sub[a] = all_corr_lc_tpf_2d_sub[a] * np.nanmedian(all_raw_lc_tpf_2d_sub[a])

            all_corr_lc_pc_sub[a]  = all_corr_lc_pc_sub[a]  * np.nanmedian(all_raw_lc_pc_sub[a])
            all_corr_lc_tpf_sub[a] = all_corr_lc_tpf_sub[a] * np.nanmedian(all_raw_lc_tpf_sub[a])


        if self.aperture_mode == 1:
            tpf_stds[ap_size > 8] = 10.0
            pc_stds[ap_size > 8] = 10.0
            if self.source_info.tc == False:
                stds_2d[ap_size > 8] = 10.0

        if self.aperture_mode == 2:
            tpf_stds[ap_size < 8] = 10.0
            pc_stds[ap_size < 8] = 10.0
            if self.source_info.tc == False:
                stds_2d[ap_size < 8] = 10.0

        if self.aperture_mode == 0:
            if self.source_info.tess_mag < 7:
                tpf_stds[ap_size < 15] = 10.0
                pc_stds[ap_size < 15]  = 10.0
                if self.source_info.tc == False:
                    stds_2d[ap_size < 15] = 10.0


        best_ind_tpf = np.where(tpf_stds == np.nanmin(tpf_stds))[0][0]
        best_ind_pc  = np.where(pc_stds == np.nanmin(pc_stds))[0][0]

        if not self.source_info.tc:
            if np.isfinite(stds_2d).any():
                best_ind_2d = np.where(stds_2d == np.nanmin(stds_2d))[0][0]
            else:
                best_ind_2d = None
        else:
            best_ind_2d = None

        if best_ind_2d is not None:
            stds = np.array([pc_stds[best_ind_pc],
                                tpf_stds[best_ind_tpf],
                                stds_2d[best_ind_2d]])
            std_inds = np.array([best_ind_pc, best_ind_tpf, best_ind_2d])
            types = np.array(['PC_LEVEL', 'TPF_LEVEL', '2D_BKG_MODEL'])

        else:
            stds = np.array([pc_stds[best_ind_pc],
                                tpf_stds[best_ind_tpf]])
            std_inds = np.array([best_ind_pc, best_ind_tpf])
            types =np.array(['PC_LEVEL', 'TPF_LEVEL'])

        #print('stds','standards_for_selection',stds)
        best_ind = std_inds[np.argmin(stds)]
        if bkg_type==None:
            self.bkg_type = types[np.argmin(stds)]
        else:
            print('User Selection of bkg')
            self.bkg_type=bkg_type
            
        if self.bkg_type == 'PC_LEVEL':
            self.all_raw_flux  = np.array(all_raw_lc_pc_sub)
            self.all_corr_flux = np.array(all_corr_lc_pc_sub)
            self.tpf += self.tpf_flux_bkg[:, None, None]

        elif self.bkg_type == 'TPF_LEVEL':
            self.all_raw_flux  = np.array(all_raw_lc_tpf_sub)
            self.all_corr_flux = np.array(all_corr_lc_tpf_sub)

        elif self.bkg_type == '2D_BKG_MODEL':
            self.all_raw_flux  = np.array(all_raw_lc_tpf_2d_sub)
            self.all_corr_flux = np.array(all_corr_lc_tpf_2d_sub)
            self.flux_bkg = np.array(np.nansum(self.bkg_tpf*self.all_apertures[best_ind], axis=(1,2)))

        if self.language == 'Australian':
            for i in range(len(self.all_raw_flux)):
                med = np.nanmedian(self.all_raw_flux[i])
                self.all_raw_flux[i] = (med-self.all_raw_flux[i]) + med

                med = np.nanmedian(self.all_corr_flux[i])
                self.all_corr_flux[i] = (med-self.all_corr_flux[i]) + med

        self.all_flux_err    = np.array(all_lc_err)

        self.corr_flux= self.all_corr_flux[best_ind]
        self.raw_flux = self.all_raw_flux[best_ind]
        self.aperture = self.all_apertures[best_ind]
        self.flux_err = self.all_flux_err[best_ind]
        self.aperture_size = np.nansum(self.aperture)
        self.best_ind = best_ind

        return 0

from eleanor.mast import *
from tess_stars2px import tess_stars2px_function_entry as tess_stars2px


def multi_sectors(sectors, tic=None, gaia=None,
                  coords=None, name=None, tc=False, local=False,
                  post_dir=config.post_dir, pm_dir=None,
                  metadata_path=None, tesscut_size=31):
    """Obtain a list of Source objects for a single target, for each of
       multiple sectors for which the target was observed.

    Parameters
    ----------
    sectors : list or str
        The list of sectors for which data should be returned, or `'all'` to
        return all sectors for which there are data.
    tic : int, optional
        The TIC ID of the source.
    gaia : int, optional
        The Gaia DR2 source_id.
    coords : tuple, optional
        The (RA, Dec) coords of the object in degrees.
    tc : bool, optional
        If True, use a TessCut cutout to produce postcards rather than
        downloading the eleanor postcard data products.
    tesscut_size : int, array-like, astropy.units.Quantity
        The size of the cutout array, when tc is True. Details can be seen in
        astroquery.mast.TesscutClass.download_cutouts
    """

    objs = []

    if sectors == 'all':
        if coords is None:
            if tic is not None:
                coords, _, _, _ = coords_from_tic(tic)
            elif gaia is not None:
                coords = coords_from_gaia(gaia)
            elif name is not None:
                coords = coords_from_name(name)

        if coords is not None:
            if type(coords) is SkyCoord:
                coords = (coords.ra.degree, coords.dec.degree)
            result = tess_stars2px(8675309, coords[0], coords[1])
            sector = result[3][result[3] < maxsector + 0.5]
            sectors = sector.tolist()

        if len(sectors) == 0 or sectors[0] < 0:
            raise SearchError("Your target is not observed by TESS, or maybe you need to run eleanor.Update()")
        else:
            print('Found star in Sector(s) ' +" ".join(str(x) for x in sectors))

    if (type(sectors) == list) or (type(sectors) == np.ndarray):
        for s in sectors:
            star = Source(tic=tic, gaia=gaia, coords=coords, sector=int(s), tc=tc,
                          local=local, post_dir=post_dir, pm_dir=pm_dir,
                          metadata_path=metadata_path, tesscut_size=tesscut_size)
            if star.sector is not None:
                objs.append(star)
        if len(objs) < len(sectors):
            warnings.warn('Only {} targets found instead of desired {}. Your '
                          'target may not have been observed yet in these sectors.'
                          ''.format(len(objs), len(sectors)))
        return objs







class Source(Source):

    Source.post_dir=config.post_dir











def norm(l, q):
    l = l[q]
    l /= np.nanmedian(l)
    l -= 1
    return l

def fhat(xhat, data):
    return np.dot(data, xhat)

def xhat(mat, lc):
    return np.linalg.lstsq(mat, lc, rcond=None)[0]

def rotate_centroids(centroid_col, centroid_row):
    centroids = np.array([centroid_col, centroid_row])
    _, eig_vecs = np.linalg.eigh(np.cov(centroids))
    return np.dot(eig_vecs, centroids)

def get_flattened_sigma(y, maxiter=100, window_size=51, nsigma=4):
    y = np.copy(y[np.isfinite(y)], order="C")
    y[:] /= savgol_filter(y, window_size, 2)
    y[:] -= np.mean(y)
    sig = np.std(y)
    m = np.ones_like(y, dtype=bool)
    n = len(y)
    for _ in range(maxiter):
        sig = np.std(y[m])
        m = np.abs(y) < nsigma * sig
        if m.sum() == n:
            break
        n = m.sum()
    return sig