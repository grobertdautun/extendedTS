import os
import subprocess

from prettytable import PrettyTable

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