import numpy as np
import matplotlib.pyplot as plt

from openpmd_viewer.addons import LpaDiagnostics
from prettytable import PrettyTable
from scipy.constants import c, e

import os
import seaborn
import subprocess
import warnings

from ._utils.pdata_utils import _get_data, _get_e_w, _get_e_w_ang, _get_hist, _get_peak

def _adaptTableValues(x):
    return list(map(lambda s: "{:.2f}".format(float(s)), x))

def scanPath(path, only_finalised=False, verbose=False):
    """
    Scans a path recursively to attempt to find simulations 
    (basically looks for a submit.sh file atm not too concerned about compatibility etc)

    Parameters
    -----------------
    path : str

    Returns
    -----------------
    sim_paths : array of str
    """
    all_path_diags = []
    in_dir = os.listdir(path)
    if "submit.sh" in in_dir and "output.txt" in in_dir:
        if verbose: print(path+" is sim")
        if only_finalised:
            last_lines = subprocess.check_output(['tail', '-n', '500', path+"/output.txt"]).decode().splitlines()
            for l in last_lines[::-1]:
                if "finalized" in l:
                    if verbose: print(path+" finalised !")
                    return [path]
            return []
        return [path]
    if len(in_dir)==0:
        return []
    for entry in in_dir:
        if verbose:
            print(f'Checking {path+entry}')
        if os.path.isdir(path+entry):
            all_path_diags += scanPath(path+entry, only_finalised, verbose)
    return all_path_diags

def checkSims(sims_paths, output_file_name="output.txt"):
    """
    Checks whether the simulations given are finished and their last step

    Parameters
    ------------------
    - sims_paths: list of str, all paths with simulations to check
    - output_file_name: str, name of the warpx output file, default "output.txt"

    Returns
    ------------------
    None
    """
    table = PrettyTable()
    table.field_names = ["simulation", "last iteration", "finalised"]
    for sim in sims_paths:
        sim_name = sim.split("/")[-1]
        if not os.path.isfile(sim+f'/{output_file_name}'):
            table.add_row([sim_name, "FILE DOES NOT EXIST", "no"])
            continue
        with open(sim+f'/{output_file_name}', "rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() ==0:
                table.add_row([sim_name, "FILE IS EMPTY", "no"])
                continue
        last_lines = subprocess.check_output(['tail', '-n', '500', sim+f'/{output_file_name}']).decode().splitlines()
        # print(last_lines)
        finalized = "no"
        last_step = "not found"
        for l in last_lines[::-1]:
            if "finalized" in l: finalized = "yes"
            if "STEP"in l:
                last_step = l.split(' ')[1]
                break
        table.add_row([sim_name, last_step, finalized])
    print(table)



class ExtendedTS:
    """
    OpenPMDTimeSeries inspired object but based on the simulation directory
    """
    path: str
    """relative path to the simulation directory"""
    name: str
    """simulation name to display in outputs, directory name by default but can be customised"""
    available_diags: list[str]
    """Names of all the diags available"""
    diags: dict[str, LpaDiagnostics]
    """available timeseries"""
    dt: float
    """simulation timestep (if output.txt exists)"""
    dz_mov: float
    """if moving window moving @ c, z displacement per time step"""
    is3D: bool
    """self explanatory"""
    def __init__(self, sim_path, customName=None, sim_3D=False):
        """
        Parameters
        ---------------
        - sim_path: str, SE

        Optional
        ---------------
        - customName: str, SE

        Returns
        ---------------
        - ExtendedTS
        """
        self.path = sim_path
        self.is3D = sim_3D
        if customName==None:
            self.name = (sim_path.split("/"))[-1]
        else:
            self.name = customName
        if not os.path.exists(sim_path):
            raise OSError(2,"Simulation path not found")
        if not os.path.isfile(sim_path + "/warpx_used_inputs"):
            warnings.warn("No used inputs file founds. Did the simulation run ?",UserWarning)
        if not (os.path.isdir(sim_path+"/diags") and len(os.listdir(sim_path+"/diags/"))!=0 ):
            warnings.warn("/diags directory not found or is empty. Did the simulation run ?",UserWarning)
            self.available_diags = None
            self.diags = None
        else:
            self.available_diags = os.listdir(sim_path+"/diags/")
            self.diags = {}
            for diag in self.available_diags:
                if "checkpoint" in diag or "reducedfiles" in diag:
                    self.diags[diag] = None # cannot create LpaDiagnostics on checkpoint diag or reduced diags
                    continue
                self.diags[diag] = LpaDiagnostics(sim_path+"/diags/"+diag)
        self.dt = 0
        self.dz_mov = 0
        if not os.path.isfile(sim_path + '/output.txt'):
            print(f'output.txt file not found. Time step defaulted to 0')
        else:
            with open(sim_path + "/output.txt", 'r') as f:
                for line in f:
                    if "Level 0" in line:
                        split_line = line.split(" ")
                        self.dt = float(split_line[4])
                        self.dz_mov = c * self.dt
                        break

    def _prepareDiagIterationSpecies(self, diag, iteration, species):
        if type(diag)==str:
            diag = self.diags[diag]
        if type(species)==str:
            if species=='all':
                species = diag.avail_species
            else:
                species = [species] # iterable
        if iteration=='last':
            iteration = diag.iterations[-1]
        if iteration not in diag.iterations:
            raise ValueError("Iteration {} is not available. Available iterations : {}".format(iteration, diag.iterations))
        return diag, iteration, species

    def displayTSInfos(self):
        """
        Prints out the found informations on the simulation
        """
        message = ""
        message += 30 * '-'
        if self.name!=None:
            message += "ExtendedTS \"" + self.name + "\" @ "
        message += "Simulation path : " + self.path
        message += "\n" + 30 * '-'
        message += "\n" + "Available diagnostics : "
        for diag in self.available_diags:
            message += "\n + " + diag
            diag_ts = self.diags[diag]
            if diag_ts is None:
                message += "\n | " + "Unavailable"
                continue
            else:
                message += "\n | + " + "Fields"
                if diag_ts.avail_fields == None:
                    message += "\n | | " + "None"
                else:
                    for field in diag_ts.avail_fields:
                        message += "\n | | " + field
                message += "\n | + " + "Particles"
                if diag_ts.avail_species == None:
                    message += "\n | | " + "None"
                else:
                    for particle in diag_ts.avail_species:
                        message += "\n | | " + particle
                message += "\n | Iterations :"
                message += "\n | + " + str(diag_ts.iterations)
        print(message)

    def getParticlesSummary(self, diag, iteration, waist, species='all', cumulative=False, verbose=False, print_table=False, nbins=150, minE=50, maxE=None, peak_height=50, ang_lim=1.0):
        """
        Returns the peak energy, the charge (50%) and energy spread (50%)

        Parameters
        -----------------------
        **diag** : str or LpaDiagnostics

            diag name or LpaDiagnostics

        **iteration** : int or 'last'

            iteration to analyse

        **waist** : float

            in µm ! 
            TEMPORARY: retrieve from input file, for now it's manual

        **species**: 'all' or list of str, optional

            required species, default is 'all'

        **cumulative** : boolean, optional

            output each species separately or sum them, default is False

        **verbose** : boolean, optional

            prints more infos. Default is False

        **print_table** : boolean, optional

            print a summary table, splitting each species. Does nothing when cumulative is True. Default is False

        **nbins** : int, optional

            number of bins to compute the histogram for the 50% peak. Default is 150

        **minE** and **maxE** : float, optional

            energy limits of the spectrum. Defaults are 50, None (auto computed)

        **peak_height** : float, optional

            percentage of the peak height to compute the energy spread. Default is 50 (%)

        **ang_lim** : float, optional

            divergence angle limit in degrees to consider particles for the analysis. Default is 1.0°

        Returns
        -------------------------
        - E : array of float or float if cumulative = True
        - Q : array of float or float if cumulative = True
        - dE : array of float or float if cumulative = True
        - species_list : array of str
        """
        diag, iteration, species = self._prepareDiagIterationSpecies(diag, iteration, species)

        if cumulative:
            for i,spec in enumerate(species):
                if spec not in diag.avail_species:
                    print(f'Species {spec} does not exist. Skipping this one')
                    continue
                if i==0:
                    pdata = _get_data(diag, spec, iteration, waist, verbose=verbose, sim_3D=self.is3D)
                else:
                    pdata_sp = _get_data(diag, spec, iteration, waist, verbose=verbose, sim_3D=self.is3D)
                    pdata = np.concatenate((pdata, pdata_sp), axis=1)
            en, _ = _get_e_w(pdata, ang_lim=ang_lim)
            if en.shape[0]==0: # no energy data -> skip
                E = 0
                Q = 0
                dE = 0
                print("Warning : no electrons in the given energy range")
                return E,Q,dE,species
            if maxE==None:
                maxE = np.max(en)
            hist, bins = _get_hist(pdata, maxE=maxE, nbins=nbins, minE=minE, ang_lim=ang_lim)

            peak_en_val, peak_range, _, lr = _get_peak(hist, bins, peak_height)

            total_charge = np.sum(hist[lr[0]:lr[1]]) * e * 1e12
            dEsE = 100 * (peak_range[1] - peak_range[0]) / peak_en_val

            E = peak_en_val
            Q = total_charge
            return E, Q, dEsE, species
        else:
            E = np.zeros((len(species),))
            Q = np.zeros((len(species),))
            dE = np.zeros((len(species),))
            for i,spec in enumerate(species):
                if spec not in diag.avail_species:
                    print(f'Species {spec} does not exist. Skipping this one')
                    continue

                pdata = _get_data(diag, spec, iteration, waist, verbose=verbose, e_lim=10, sim_3D=self.is3D)
                en, _ = _get_e_w(pdata, ang_lim=ang_lim)
                if en.shape[0]==0: # no energy data -> skip
                    E[i] = 0
                    Q[i] = 0
                    dE[i] = 0
                    continue
                if maxE==None:
                    maxE = np.max(en)
                hist, bins = _get_hist(pdata, maxE=maxE, nbins=nbins, minE=minE, ang_lim=ang_lim)
                
                peak_en_val, peak_range, _, lr = _get_peak(hist, bins, peak_height)

                total_charge = np.sum(hist[lr[0]:lr[1]]) * e * 1e12
                dEsE = 100 * (peak_range[1] - peak_range[0]) / peak_en_val
                
                E[i] = peak_en_val
                Q[i] = total_charge
                dE[i] = dEsE

            if print_table:
                table = PrettyTable()
                table.add_column("Species", species)
                table.add_column("E peak (MeV)", _adaptTableValues(E))
                table.add_column("Q 50% (pC)", _adaptTableValues(Q))
                table.add_column("dE 50% (%)", _adaptTableValues(dE))

                print(f'Diagnostic {diag} at iteration {iteration}')
                print(table)

            return E, Q, dE, species

    def makeSpectrumOnAx(self, ax, diag, iteration, waist, species='all', nbins=150, verbose=False, cumulative=False, minE=50, maxE=None, ang_lim=1.0, label=None, set_ax=True):
        """
        Plots the spectrum on a given matplotlib ax

        Parameters 
        -------------
        **ax** : matplotlib.pyplot.Ax

            ax to plot on
        
        **diag** : str or LpaDiagnostics

            diag name or LpaDiagnostics

        **iteration** : int or 'last'

            iteration to analyse

        **waist** : float

            in µm ! 
            TEMPORARY: retrieve from input file, for now it's manual

        **species** : 'all' or str or list of str, optional

            which species to plot. Default is 'all'

        **verbose** : boolean, optional

            More infos. Default is False

        **cumulative** : boolean, optional

            output each species separately or sum them, default is False

        **nbins** : int, optional

            number of bins to compute the histogram for the 50% peak. Default is 150

        **minE** and **maxE** : float, optional

            energy limits of the spectrum. Defaults are 50, None (auto computed)
            !! maxE not working yet

        **ang_lim** : float, optional

            divergence angle limit in degrees to consider particles for the analysis. Default is 1.0°

        **label** : str, optional

            label for the plot legend, default is species name

        **set_ax** : boolean, optional

            whether to set axis labels and title, default is True

        """
        diag, iteration, species = self._prepareDiagIterationSpecies(diag, iteration, species)
        maxE = 0 
        maxQ = 0
        if cumulative:
            for i,spec in enumerate(species):
                if spec not in diag.avail_species:
                    print(f'Species {spec} does not exist. Skipping this one')
                    continue
                if i==0:
                    pdata = _get_data(diag, spec, iteration, waist, verbose=verbose, sim_3D=self.is3D)
                else:
                    pdata_sp = _get_data(diag, spec, iteration, waist, verbose=verbose, sim_3D=self.is3D)
                    pdata = np.concatenate((pdata, pdata_sp), axis=1)
            en, _ = _get_e_w(pdata, ang_lim=ang_lim)
            if en.shape[0]==0:
                print("no particles found")
                return
            hist, bins = _get_hist(pdata, maxE=np.max(en), minE=minE, nbins=nbins, ang_lim=ang_lim)

            bb = (bins[1:] + bins[:-1]) / 2
            coeff = e * 1e12 / (bb[1] - bb[0])

            ax.plot(bb, hist*coeff, label=spec if label==None else label)
            if i==0: 
                maxE = np.max(en)
                maxQ = np.max(hist * coeff)
            else: # if current species has a higher maxE than currently set, expand x-y axis to match
                maxE = max(maxE, np.max(en))
                maxQ = max(maxQ, np.max(hist * coeff))
        else:
            for i,spec in enumerate(species):
                if spec not in diag.avail_species:
                    print(f'Species {spec} does not exist. Skipping this one')
                    continue

                pdata = _get_data(diag, spec, iteration, waist, verbose=verbose, e_lim=minE, sim_3D=self.is3D)
                en, _ = _get_e_w(pdata, ang_lim=ang_lim)
                if en.shape[0]==0:
                    print("no charge for species {}".format(spec))
                    continue
                hist, bins = _get_hist(pdata, maxE=np.max(en), minE=minE, nbins=nbins, ang_lim=ang_lim)

                bb = (bins[1:] + bins[:-1]) / 2
                coeff = e * 1e12 / (bb[1] - bb[0])

                ax.plot(bb, hist*coeff, label=spec if label==None else label)
                if i==0: 
                    maxE = np.max(en)
                    maxQ = np.max(hist * coeff)
                else: # if current species has a higher maxE than currently set, expand x-y axis to match
                    maxE = max(maxE, np.max(en))
                    maxQ = max(maxQ, np.max(hist * coeff))

        # post plot limits and labels
        AD_minE, AD_maxE = ax.get_xlim() # already defined axis limits
        AD_minQ, AD_maxQ = ax.get_ylim()
        maxE = max(maxE+10, AD_maxE) # resize if new spectrum does not fit
        maxQ = max(maxQ+1, AD_maxQ)
        minE = min(minE, AD_minE) 
        minQ = min(0, AD_minQ)
        ax.set(
            xlim = (minE,maxE),
            ylim = (minQ,maxQ),
        )

        if set_ax:
            ax.set(
                xlabel = "E (MeV)",
                ylabel = "dQ/dE (pC/MeV)",
                title = "Spectre @ {:.2f} mm".format((iteration-13333) * self.dz_mov * 1e3)
            )
            ax.legend()

    def makeSpectrum(self, diag, iteration, waist, species='all', nbins=150, minE=50, verbose=False, save_path='./results/'):
        """
        Plots and saves the figure

        obsolete
        """
        diag, iteration, species = self._prepareDiagIterationSpecies(diag, iteration, species)

        fig, ax = plt.subplots()
        
        self.makeSpectrumOnAx(ax, diag, iteration, waist, species, nbins, verbose, minE=minE)

        plt.tight_layout()

        if not os.path.isdir("./results"):
            os.makedirs("./results")
        plt.savefig(f'{save_path}/spectrum_{self.name}.svg')

    def makeEDiv(self, diag, iteration, species, waist, ang_lim=1.0, verbose=False, bw_adjust=0.7, minE=50, maxE=None):
        diag, iteration, species = self._prepareDiagIterationSpecies(diag, iteration, species)

        if len(species) != 1:
            print("only 1 species")
            return

        pdata = _get_data(diag, species[0], iteration, waist, verbose=verbose, e_lim=minE, sim_3D=self.is3D)
        en, w , ang = _get_e_w_ang(pdata, ang_lim)

        fig, ax = plt.subplots(figsize = (8,4))
        seaborn.kdeplot(
            x=en, 
            y=ang,
            weights=w, 
            ax=ax, 
            fill=True, 
            levels=100, 
            cmap='turbo', 
            thresh=0,
            cbar=True,
            bw_adjust=bw_adjust,
            antialiased=False
        )

        for c in ax.collections:
            c.set_edgecolor("face")

        if maxE==None:
            _, maxE = ax.get_xlim()
        ax.set_xlim(minE,maxE)

        fig.tight_layout()

        if not os.path.isdir("./results"):
            os.makedirs("./results")
        plt.savefig("results/EDivMap.svg")