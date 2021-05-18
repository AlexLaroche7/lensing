import numpy as np 
import copy
import scalelens as sl
from lenstronomy.LensModel.lens_model import LensModel
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from scipy import integrate
from pyHalo.Halos.lens_cosmo import LensCosmo
from lenstronomy.LensModel.Profiles.nfw import NFW 
from lenstronomy.LensModel.Profiles.uldm import Uldm 
from num2tex import num2tex
from mpl_toolkits.axes_grid1 import make_axes_locatable

def rescale_NFW(Rmax,kwargs_nfw,kwargs_uldm):
    """
    computes the rescaling factor required to conserve mass between a NFW profile and a NFW + ULDM profile
    within a radius Rmax.
    
    :param Rmax: rescaling radius
    :param nfw: NFW profile
    :param uldm: ULDM profile
    :param kwargs_nfw: NFW params
    :param kwargs_ULDM: ULDM params
    """
    # lens profiles
    nfw = NFW()
    uldm = Uldm()
    # M_nfw = (scaling factor)*M_nfw + M_uldm [contained within Rmax]
    M_nfw = nfw.mass_3d_lens(Rmax,kwargs_nfw['Rs'],kwargs_nfw['alpha_Rs'])
    M_uldm = uldm.mass_3d_lens(Rmax,kwargs_uldm['kappa_0'],kwargs_uldm['theta_c'])
    scaling_factor = (M_nfw - M_uldm) / M_nfw
    return scaling_factor

def kappa_ref(nfwModel,kwargs_nfw,rad_ref=0.01):
    """
    computes kappa_ref, the reference density, which will then be used to set central density normalization kappa_0
    for susbequent ULDM profiles.
    
    :param nfwModel: NFW lens model
    :param kwargs_nfw: NFW params
    :param rad_ref: reference radius
    """
    integral = integrate.quad(kappa_ref_integrand,0,rad_ref,(nfwModel,kwargs_nfw))
    kappa_ref = integral[0]*(2/rad_ref**2)
    
    return kappa_ref

def kappa_ref_integrand(r,nfwModel,kwargs_nfw):
    """
    computes integrand for kappa_ref integral.
    
    :param r: integration variable
    :param kwargs_nfw: NFW params
    :param rad_ref: reference radius
    """
    return nfwModel.kappa(r,0,[kwargs_nfw])*r

def tune_ULDM_mass(frac,Rmax,kwargs_nfw,kwargs_uldm):
    """
    Computes necessary kappa_0 scale factor for ULDM profile to account for (100*frac)% of the NFW + ULDM profile.
    
    :param frac: desired decimal ULDM mass fraction
    :param Rmax: rescaling radius
    :param kwargs_nfw: NFW params
    :param kwargs_ULDM: ULDM params
    """
    # lens profiles
    nfw = NFW()
    uldm = Uldm()
    
    # M_{NFW+ULDM} = M_{NFW} 
    total_mass = nfw.mass_3d_lens(Rmax,kwargs_nfw['Rs'],kwargs_nfw['alpha_Rs'])
    uldm_mass = uldm.mass_3d_lens(Rmax,kwargs_uldm['kappa_0'],kwargs_uldm['theta_c'])
    scale_factor = frac*(total_mass/uldm_mass)
    
    return scale_factor

def nfw_uldm_profile(Rmax,uldm_frac,kwargs_nfw,kwargs_uldm):
    """
    Computes ULDM & NFW parameters for a composite profile with total mass, within Rmax, 
    equal to the given NFW profile and a ULDM/NFW mass ratio specified by uldm_frac.
    
    :param Rmax: rescaling radius
    :param uldm_frac: ULDM mass fraction of composite profile
    :param kwargs_nfw: NFW params
    :param kwargs_uldm: ULDM params
    """
    # copy kwargs
    kwargs_nfw_cp = kwargs_nfw.copy()
    kwargs_uldm_cp = kwargs_uldm.copy()
    
    # scale ULDM
    uldm_factor = tune_ULDM_mass(uldm_frac,Rmax,kwargs_nfw_cp,kwargs_uldm_cp)
    kwargs_uldm_cp['kappa_0'] *= uldm_factor
    
    # scale NFW
    nfw_factor = rescale_NFW(Rmax,kwargs_nfw_cp,kwargs_uldm_cp)
    kwargs_nfw_cp['alpha_Rs'] *= nfw_factor
    
    return [kwargs_uldm_cp,kwargs_nfw_cp]

def plot_stuff(r,comp_data,nfw_data,var1,var2,ylabel,var2name,name):
    c = var2
    norm = colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.cividis)
    cmap.set_array([])

    fig,ax = plt.subplots(2,3,figsize=(30,20),sharex=True,sharey=True,constrained_layout=True)

    for i in range(len(var1)):
        title = r'$R_{max}/R_s=' + str('{:0.3e}'.format(num2tex(var1[i]))) + '$'
        ax.flatten()[i].set_title(title,fontsize=30)
        if i>2:
            ax.flatten()[i].set_xlabel(r'$r/R_s$',fontsize=30)
        ax.flatten()[i].grid(True)
        ax.flatten()[i].set_xlim(min(r),max(r))
        ax.flatten()[i].loglog(r,nfw_data,'k--', label='NFW')
        for j in range(len(var2)):
            ax.flatten()[i].loglog(r,comp_data[i][j],c=cmap.to_rgba(c[j]))
            for tick in ax.flatten()[i].xaxis.get_major_ticks():
                    tick.label.set_fontsize(20)
            for tick in ax.flatten()[i].yaxis.get_major_ticks():
                    tick.label.set_fontsize(20)
    cax = fig.add_axes([1.02, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(cmap, cax=cax)
    cbar.set_label(label=var2name,fontsize=30,rotation=360,labelpad=40)
    cbar.ax.tick_params(labelsize=25)
    cbar.set_ticks(var2)
    fig.tight_layout()
    text = ax.flatten()[0].yaxis.get_offset_text()
    text.set_size(20)
    ax.flatten()[0].set_ylabel(ylabel,fontsize=30)
    ax.flatten()[3].set_ylabel(ylabel,fontsize=30)
    ax.flatten()[-1].legend(fontsize=30)
    save = 'figs/' + name + '.pdf'
    plt.savefig(save)
    plt.show()
    
    return 