# """VLA pipeline helper functions"""

import numpy as np
import os
import scipy.constants

# import some stuff from CASA
if os.getenv('CASAPATH') is not None:
    # import casadef
    from taskinit import *
    msmd = msmdtool() # need for metadata


def flagtemplate_add(sdm="", flagcmds=[], outfile=""):
    """
    Append additional flagging commands to the flagtemplate if they are not present already.
    Useful for L-band VLA HI obs: e.g. flagcmds=["mode='manual' spw='1,2:0~64,4'"]

    :param sdm: path to visibility data, will append .flagtemplate.txt to it
    :param flagcmds: list of commands to append into the template file
    :param outfile: override naming to custom filename
    :return: None
    """
    if outfile == "":
        outfile = sdm + ".flagtemplate.txt"

    if os.path.exists(outfile):
        with open(outfile, "r") as f:
            content = f.read().splitlines()
    else:
        content = [""]

    with open(outfile, "a") as f:
        for cmd in flagcmds:
            if cmd not in content:
                f.write("\n" + cmd + "\n")
    return


def partition_cont_range(line_freqs=[], line_widths=[], spw_start=1, spw_end=2):
    """
    Cuts one continuum range into smaller ones to avoid lines.

    :param line_freqs: line frequencies in GHz
    :param line_widths: widths of lines in GHz to cut from the continuum
    :param spw_start: start of the SPW in GHz
    :param spw_end: end of the SPW in GHz
    :return: list of continuum chunks, each defined as a dictionary with start and end freqs in GHz.
    """

    # make sure lists are treaded as float vectors
    line_freqs = np.array(line_freqs)
    line_widths = np.array(line_widths)

    # define line ranges that will be excluded
    line_starts = line_freqs - 0.5 * line_widths
    line_ends = line_freqs + 0.5 * line_widths

    # start with the whole spw as one continuum chunk
    cont_chunks = [dict(start=spw_start, end=spw_end)]

    for i in range(len(line_freqs)):
        # for each line loop over the continuum chunk collection and modify it in the process
        j = 0
        while j < len(cont_chunks):
            # line outside chunk, skip
            if line_ends[i] < cont_chunks[j]["start"] or line_starts[i] > cont_chunks[j]["end"]:
                pass

            # line covers whole chunk, delete it
            elif line_starts[i] <= cont_chunks[j]["start"] and line_ends[i] >= cont_chunks[j]["end"]:
                cont_chunks.pop(j)
                j = j - 1

            # line covers left edge only, edit cont chunk start
            elif line_starts[i] < cont_chunks[j]["start"] and line_ends[i] >= cont_chunks[j]["start"]:
                cont_chunks[j]["start"] = line_ends[i]

            # line covers right edge only, edit cont chunk end
            elif line_starts[i] <= cont_chunks[j]["end"] and line_ends[i] > cont_chunks[j]["end"]:
                cont_chunks[j]["end"] = line_starts[i]

            # line in the middle, splits chunk into two
            elif line_starts[i] > cont_chunks[j]["start"] and line_ends[i] < cont_chunks[j]["end"]:
                cont_chunks.insert(j + 1, dict(start=line_ends[i], end=cont_chunks[j]["end"]))
                cont_chunks[j]["end"] = line_starts[i]
                j = j + 1

            # other non-implemented scenarios (are all useful cases covered? any pathological edge case?)
            else:
                pass

            # progress to the next chunk
            j = j + 1

    return cont_chunks


def build_cont_dat(vis="", line_freqs=[], line_widths=[], fields=[], outfile="cont.dat", overwrite=False, append=False):
    """
    Creates a cont.dat file for the VLA pipeline. Must be run in CASA (uses msmetadata).
    It currently reads SPW edges in the original observed frame (usually TOPO),
    but writes them down as LSRK. Should not matter much, edges should be flagged anyway.
    Example of cont.dat content from NRAO online documentation:
    https://science.nrao.edu/facilities/vla/data-processing/pipeline/#section-25

    :param vis: path to the measurement set
    :param line_freqs: line frequencies (obs frame, LSRK) in GHz
    :param line_widths: widths of lines (obs frame, LSRK) in GHz to cut from the continuum
    :param fields: science target fields. If empty, TARGET intent fields are used.
    :param outfile: path to the output cont.dat file
    :param overwrite: if True and the outfile exists, it will be overriten
    :param append: add at the end of existing cont.dat file, useful for optimising lines per field
    :return: None
    """

    # if no fields are provided use observe_target intent
    # I saw once a calibrator also has this intent so check carefully
    msmd.open(vis)
    if len(fields) < 1:
        # fields = msmd.fieldsforintent("*OBSERVE_TARGET*", True)
        fields = msmd.fieldsforintent("*TARGET*", True)

    if len(fields) < 1:
        print("ERROR: no fields!")
        return

    if os.path.exists(outfile) and not overwrite and not append:
        print("ERROR: file already exists!")
        return

    # generate a dictonary containing continuum chunks for every spw of every field
    cont_dat = {}
    for field in fields:
        spws = msmd.spwsforfield(field)
        cont_dat_field = {}

        for spw in spws:
            # Get freq range of the SPW
            chan_freqs = msmd.chanfreqs(spw)
            # SPW edges are reported in whichever frame was used for observing (usually TOPO)
            # TODO: implement some transformations to LSRK for the edges?
            spw_start = np.min(chan_freqs) * 1e-9  # GHz
            spw_end = np.max(chan_freqs) * 1e-9  # GHz
            cont_chunks = partition_cont_range(line_freqs, line_widths, spw_start, spw_end)
            cont_dat_field.update({spw: cont_chunks})
        # print(spw, cont_chunks)
        # print(spw_start, spw_end)

        cont_dat.update({field: cont_dat_field})

    msmd.close()

    # write the dictionary into a file usable by the CASA VLA pipeline
    access_mode = "a" if append else "w"
    with open(outfile, access_mode) as f:
        for field in cont_dat.keys():
            f.write("\nField: " + field + "\n")
            for spw in cont_dat[field].keys():
                if len(cont_dat[field][spw]) > 0:
                    f.write("\nSpectralWindow: " + str(spw) + "\n")
                    for chunk in cont_dat[field][spw]:
                        f.write(str(chunk["start"]) + "~" + str(chunk["end"]) + "GHz LSRK\n")
            f.write("\n")

    print("DONE: written in " + outfile)
    return


def lines_rest2obs(line_freqs_rest, line_widths_kms, vrad0):
    """
    Get observed frame frequencies and widths of the lines.

    :param line_freqs_rest: list of rest-frame line frequencies
    :param line_widths_kms: list of rest-frame linewidths in km/s
    :param vrad0: systemic velocity of the galaxy in km/s
    :return: line_freqs, line_widths, both in GHz
    """
    ckms = scipy.constants.c / 1000.
    line_freqs = np.array(line_freqs_rest) * (1 - vrad0 / ckms)
    line_widths = np.array(line_freqs) * line_widths_kms / ckms

    return line_freqs, line_widths

