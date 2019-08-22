"""Wraps BART functions."""
import sys 
import numpy as np
import subprocess
import random
#import fileio
sys.path.append('/home/davidyzeng/machinelearning/cones/python/packages')
#from fileio import Fileio
#import fileio
import fileio.cfl as cfl
from timeit import default_timer as timer


def bart_cs(ks_input, verbose=False,
              sensemap=None,
              lamda=1e-3,
              num_iter=100,
              filename_ks_tmp="/mnt/ramdisk/ks.tmp",
              filename_map_tmp="/mnt/ramdisk/map.tmp",
              filename_im_tmp="/mnt/ramdisk/im.tmp",
              filename_ks_out_tmp="/mnt/ramdisk/ks_out.tmp"):
    """BART PICS reconstruction."""
    if verbose:
        print("CS (TV) reconstruction...")

    # write out kspace data and sensitivity maps
    cfl.write(filename_ks_tmp, ks_input)
    if sensemap is None:
        sensemap = np.ones_like(ks_input)
    cfl.write(filename_map_tmp, sensemap)

    # PICS flags
    flags = ""
    if lamda > 0:
        flags = flags + ("-R T:3:0:%f " % lamda)
    if num_iter > 0:
        flags = flags + ("-i %d " % num_iter)

    # perform PICS recon
    cmd = "bart pics %s -S %s %s %s" % (flags,
                                        filename_ks_tmp,
                                        filename_map_tmp,
                                        filename_im_tmp)
    if verbose:
        print("  %s" % cmd)
    subprocess.check_output(['bash', '-c', cmd])

    # read image
    im_pics = np.squeeze(cfl.read(filename_im_tmp))

    return im_pics
