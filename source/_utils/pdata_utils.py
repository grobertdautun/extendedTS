import numpy as np
from scipy.constants import e

def _get_data(ts, species, iteration, waist, verbose=False, e_lim=50, sim_3D=False):
    pdata = ts.get_particle(species=species, var_list=["x", "z", "ux", "uy", "uz", "w"], iteration=iteration)
    if not sim_3D:
        pdata[5] *= waist*1e-6
    if verbose:
        print("species : {}, charge above {} MeV : {:.3f} pC".format(
            species, 
            e_lim, 
            np.sum(pdata[5][pdata[4] > e_lim]) * e / 1e-12
            ))
    return pdata

def _get_e_w(pdata, ang_lim=1.0):
    en = (np.sqrt(1.0 + pdata[2]**2 + pdata[3]**2 + pdata[4]**2)-1)*0.511
    if ang_lim==None:
        return [en, pdata[5]]
    ang = np.arctan2(np.sqrt(pdata[2]**2 + pdata[3]**2), pdata[4])*180/np.pi
    fil = ang < ang_lim
    return [en[fil], pdata[5][fil]]

def _get_e_w_ang(pdata, ang_lim=1.0):
    en = (np.sqrt(1.0 + pdata[2]**2 + pdata[3]**2 + pdata[4]**2)-1)*0.511
    ang = np.sign(pdata[2]) * np.arctan2(np.sqrt(pdata[2]**2 + pdata[3]**2), pdata[4])*180/np.pi
    if ang_lim==None:
        return [en, pdata[5], ang]
    fil = np.abs(ang) < ang_lim
    return [en[fil], pdata[5][fil], ang[fil]]

def _get_hist (pdata, nbins=50, maxE=200, minE=50, ang_lim=1.0):
    en, w = _get_e_w (pdata, ang_lim=ang_lim)
    hh, bins = np.histogram(en, weights=w, range=(minE,maxE), bins=nbins)
    return hh, bins

def _get_peak(hist, bins, peak_height=50):
    bc = (bins[:-1] + bins[1:]) / 2

    peak_en = np.argmax(hist)
    peak_en_val = bc[peak_en]
    peak_en_h = hist[peak_en]
    HM = peak_height / 100 * peak_en_h
    left = peak_en
    right = peak_en +1
    while hist[left] > HM and left !=0:
        left -= 1
    while right < len(hist)-1 and hist[right] > HM:
        right += 1
    if right==len(hist): right -= 1
    peak_range = (bc[left], bc[right])
    
    return peak_en_val, peak_range, peak_en_h, (left, right)