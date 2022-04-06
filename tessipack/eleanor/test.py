import sys
sys.path.append("/home/dinilbose/github/eleanor")
print(sys.path)
from tessipack import eleanor
star = eleanor.Source(name='WASP-100', sector=1)
data = eleanor.TargetData(star, height=15, width=15, bkg_size=31,do_psf=True, do_pca=True)

m1=data.build_custom_aperture(name='dinil',shape='rectangle',w=2,h=2)
m2=data.build_custom_aperture(name='dinil2',shape='rectangle',w=3,h=3)
#m3=['default',default]
myaperture=[m1,m2]

data2 = eleanor.TargetData(star, height=15, width=15, bkg_size=31,do_psf=True, do_pca=True,other_aperture=myaperture)

#best_corr = data2.info_aperture[data2.aperture_names[data2.best_ind]][data2.bkg_type]['corr_flux']
#default_corr = data2.info_aperture['default'][data2.bkg_type]['corr_flux']


#first_key=
#sc
