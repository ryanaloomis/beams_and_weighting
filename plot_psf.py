import numpy as np
cc = 2.99792458e10 # [cm s^-1]
from numpy import pi
arcsec = pi / (180. * 3600) # [radians] = 1/206265 radian/arcsec
from numpy.fft import fftfreq, rfftfreq, fftshift, fft2, ifftshift, ifft2
import scipy.optimize as optimize
from scipy.signal import convolve2d
from scipy.interpolate import interpn
import numpy.linalg as linalg
import casatools
from numba import njit, prange

# set up mpl params for pretty plots
import matplotlib.pylab as pl
from matplotlib.pyplot import *
import matplotlib.colors as col
from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_family("sans-serif")
from matplotlib import rc
rcParams['mathtext.default']='regular'
rcParams['axes.linewidth'] = 1.0
pl.style.use('classic')
pl.rcParams['hatch.linewidth'] = 0.5
pl.rc('text', usetex=True)
rcParams['font.size'] = 17
rcParams['font.family'] = 'serif'
rcParams['font.weight']='light'
rcParams['mathtext.bf'] = 'serif:normal'
pl.rcParams['xtick.major.pad']='2'
pl.rcParams['ytick.major.pad']='2'


prefix = '/lustre/cv/users/rloomis/research_tickets/mod_pcwd/'


filenames=[prefix + 'r-2_c0.02_pcwdT.psf', prefix + 'r0_c0.02_pcwdT.psf', prefix + 'r0.5_c0.02_pcwdT.psf', prefix + 'r1_c0.02_pcwdT.psf', prefix + 'r2_c0.02_pcwdT.psf', prefix + 'r-2_c0.02_pcwdMOD.psf', prefix + 'r0_c0.02_pcwdMOD.psf', prefix + 'r0.5_c0.02_pcwdMOD.psf', prefix + 'r1_c0.02_pcwdMOD.psf', prefix + 'r2_c0.02_pcwdMOD.psf', prefix + 'r-2_c0.02_pcwdF.psf', prefix + 'r0_c0.02_pcwdF.psf', prefix + 'r0.5_c0.02_pcwdF.psf', prefix + 'r1_c0.02_pcwdF.psf', prefix + 'r2_c0.02_pcwdF.psf']

#filenames=[prefix + 'r2_c0.02_pcwdF.psf']

ia = casatools.image()
npix_window = 31


# ellipse fitting code from: nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
# moded a number of typos from original source
def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  linalg.eig(np.dot(linalg.inv(S), C))
    n =  np.argmax(E)
    a = V[:,n]
    if a[0] < 0:
        a = -a
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a < c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

def ellipse_axis_length(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*(np.sqrt((a-c)**2 + 4*b*b)-(a+c))
    down2=(b*b-a*c)*(-np.sqrt((a-c)**2 + 4*b*b)-(a+c))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def find_ellipse(x, y):
    xmean = x.mean()
    ymean = y.mean()
    x = x - xmean
    y = y - ymean
    a = fitEllipse(x,y)
    center = ellipse_center(a)
    center[0] += xmean
    center[1] += ymean
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    x += xmean
    y += ymean
    return center, phi, axes


def gaussian2D(params, nrow):
    width_x, width_y, rotation = params
    rotation = 90-rotation

    rotation = np.deg2rad(rotation)
    x, y = np.indices((nrow*2+1,nrow*2+1)) - nrow

    xp = x * np.cos(rotation) - y * np.sin(rotation)
    yp = x * np.sin(rotation) + y * np.cos(rotation)
    g = 1.*np.exp(-(((xp)/width_x)**2+((yp)/width_y)**2)/2.)
    return g


def beam_chi2(params, psf, nrow):
    psf_ravel = psf[~np.isnan(psf)]
    gaussian = gaussian2D(params, nrow)[~np.isnan(psf)]
    chi2 = np.sum((gaussian-psf_ravel)**2)
    return chi2



def fit_beam(psf_data_raw, cell_size):
    delta = cell_size
    npix = psf_data_raw.shape[0]         # Assume image is square

    # Check if image cube, or just single psf; this example doesn't handle the full polarization case - implicitly assumes we can drop Stokes
    # If single psf, add an axis so we can use a single loop
    psf_data = np.squeeze(psf_data_raw)
    if len(psf_data.shape) == 2:
        psf_data = np.expand_dims(psf_data, axis=2)

    # Roll the axes to make looping more straightforward
    psf_rolled = np.rollaxis(psf_data,2)

    major = np.zeros(psf_rolled.shape[0])
    minor = np.zeros(psf_rolled.shape[0])
    phi_grid = np.zeros(psf_rolled.shape[0])

    # Loop through the channels and fit a psf to each one
    for chan in np.arange(psf_rolled.shape[0]):
        psf_currchan = psf_rolled[chan]

        # Window out the central 21 pixels - this seems to be wide enough in general, but could be modified
        psf_windowed = psf_currchan[int(npix/2-(npix_window-1)/2):int(npix/2+(npix_window-1)/2 + 1), int(npix/2-(npix_window-1)/2):int(npix/2+(npix_window-1)/2 + 1)]

        # make pixel coordinates for the interpolation - x, y are the native grid
        x = np.arange(psf_windowed.shape[0])-(npix_window-1)/2
        y = np.arange(psf_windowed.shape[0])-(npix_window-1)/2

        # xp, yp are the interpolated grid. Here I use 1001 points, for a factor of ~50 refinement to the fit.
        # This might be overkill, but seems reasonable computationally in my tests.
        xp, yp = np.meshgrid(np.linspace(-(npix_window-1)/2, (npix_window-1)/2, 401), np.linspace(-(npix_window-1)/2, (npix_window-1)/2, 401))
        points = np.vstack((np.ravel(xp), np.ravel(yp))).T

        # interpolate the windowed psf onto this new fine grid. A spline method seems to work best,
        # as linear interpolation starts to perform poorly at low numbers of pixels per beam.
        interpolated = np.reshape(interpn((x, y), psf_windowed, points, method="splinef2d"), (401,401)).T

        # find all points where the interpolated image is close to half power. Anything within .3% seemed to work well
        # in my tests, but this value could probably be tweaked.
        ellipse_pts = np.argwhere(np.abs(interpolated - 0.5) < 0.003)

        # fit an ellipse to these points
        center, phi, axes = find_ellipse(ellipse_pts[:,0], ellipse_pts[:,1])

        # convert to useful units
        phi = np.degrees(phi) - 90.                     # Astronomers use east of north
        if phi < -90.:
            phi += 180.
        phi_grid[chan] = phi
        major[chan] = axes[0]/400.*(npix_window-1)*delta*2
        minor[chan] = axes[1]/400.*(npix_window-1)*delta*2

    return phi_grid, major, minor




def fit_beam_CASA(psf_data_raw, cell_size):
    delta = cell_size
    npix = psf_data_raw.shape[0]         # Assume image is square

    # Check if image cube, or just single psf; this example doesn't handle the full polarization case - implicitly assumes we can drop Stokes
    # If single psf, add an axis so we can use a single loop
    psf_data = np.squeeze(psf_data_raw)
    if len(psf_data.shape) == 2:
        psf_data = np.expand_dims(psf_data, axis=2)

    # Roll the axes to make looping more straightforward
    psf_rolled = np.rollaxis(psf_data,2)

    major = np.zeros(psf_rolled.shape[0])
    minor = np.zeros(psf_rolled.shape[0])
    phi_grid = np.zeros(psf_rolled.shape[0])

    nrow = 4

    # Loop through the channels and fit a psf to each one
    for chan in np.arange(psf_rolled.shape[0]):
        psf_currchan = psf_rolled[chan]

        # Window out the central 11 pixels - hardcoded for CASA
        psf_windowed = psf_currchan[int(npix/2-(nrow*2)/2):int(npix/2+(nrow*2)/2 + 1), int(npix/2-(nrow*2)/2):int(npix/2+(nrow*2)/2 + 1)]

        # Set threshold (a=0.35 by default for CASA)
        threshold = 0.35

        # Only consider pixels over the threshold
        psf_thresh = np.copy(psf_windowed)
        psf_thresh[psf_thresh<threshold] = np.nan

        #pl.imshow(psf_thresh)
        #pl.show()
        
        # Fit a gaussian to the thresholded points
        p0 = [2.5, 2.5, 0.]
        res = optimize.minimize(beam_chi2, p0, args=(psf_thresh, nrow))

        # convert to useful units
        phi = res.x[2] - 90.
        if phi < -90.:
            phi += 180.

        phi_grid[chan] = phi
        major[chan] = np.max(np.abs(res.x[0:2]))*delta*2.355
        minor[chan] = np.min(np.abs(res.x[0:2]))*delta*2.355

    return phi_grid, major, minor


fig = pl.figure(figsize=(4,2.7), dpi=300)
ax = pl.axes([0.12,0.13,0.83,0.84])
ax.minorticks_on()

r = [-2, 0., 0.5, 1., 2, -2, 0., 0.5, 1., 2., -2, 0., 0.5, 1., 2.]


for i, filename in enumerate(filenames):
    ia = casatools.image()
    ia.open(filename)
    psf_data_raw = ia.getregion()
    hdr = ia.summary(list=False)
    ia.close()

    delta = np.abs(hdr['incr'][0]*206265)

    major_CASA = np.zeros(118)
    minor_CASA = np.zeros(118)
    phi_CASA = np.zeros(118)

    for j in range(118):
        chan = j+5
        major_CASA[j] = hdr['perplanebeams']['beams']['*'+str(chan)]['*0']['major']['value']
        minor_CASA[j] = hdr['perplanebeams']['beams']['*'+str(chan)]['*0']['minor']['value']
        phi_CASA[j] = hdr['perplanebeams']['beams']['*'+str(chan)]['*0']['positionangle']['value']

    psf_data_raw = psf_data_raw[:,:,:,5:123]

    phi, major, minor = fit_beam_CASA(psf_data_raw, 0.02)

    if i < 5:
        color = 'darkslateblue'
        img1, = pl.plot(major, linewidth=1.5, color=color, alpha=0.4+0.15*i)
        img1_C, = pl.plot(major_CASA, linewidth=1.5, color=color, alpha=0.4+0.15*i, linestyle='dashed')
        ax.annotate('{:.2f}'.format(r[i]), (0., major[0]), horizontalalignment='right', verticalalignment='center', size=7, color=color, xytext=(-2,0), textcoords='offset points')
    
    elif i < 10:
        color = 'darkorange'
        img2, = pl.plot(major, linewidth=1.5, color=color, alpha=0.4+0.15*(i-5))
        img2_C, = pl.plot(major_CASA, linewidth=1.5, color=color, alpha=0.4+0.15*(i-5), linestyle='dashed')
        if i < 9:
            ax.annotate('{:.2f}'.format(r[i]), (118., major[117]), horizontalalignment='left', verticalalignment='center', size=7, color=color, xytext=(2,0), textcoords='offset points')
    else:
        color = 'firebrick'
        img3, = pl.plot(major, linewidth=1.5, color=color, alpha=0.4+0.15*(i-10))
        img3_C, = pl.plot(major_CASA, linewidth=1.5, color=color, alpha=0.4+0.15*(i-10), linestyle='dashed')
        ax.annotate('{:.2f}'.format(r[i]), (0., major[0]), horizontalalignment='right', verticalalignment='center', size=7, color=color, xytext=(-2,0), textcoords='offset points')
    
    print("plotted")


pl.ylabel(r'$\Theta_{maj}$ ["]', size=9)
ax.yaxis.set_label_coords(-0.09, 0.5)
pl.xlabel(r'Channel', size=9)
ax.xaxis.set_label_coords(0.5, -0.09)


pl.xlim(-20,170)
pl.ylim(0.1,0.185)

ax.yaxis.set_major_locator(MultipleLocator(0.02))
ax.yaxis.set_minor_locator(MultipleLocator(0.005))
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(10))

pl.setp(ax.get_xticklabels(), size='9')
pl.setp(ax.get_yticklabels(), size='9')

ax.text(0.37, 0.94, "PSF from CASA imaging", size='9', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

prop=FontProperties(size=6)
legend = pl.legend([img1, img2, img3], ["PCWD=T", "PCWD=T+mod", "PCWD=F"], prop=prop, loc=1, borderaxespad=0.35)
legend.draw_frame(False)

pl.savefig("spectral_curvature_CASA.pdf")
