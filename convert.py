#!/usr/bin/env python

# Based off https://gist.github.com/iamaziz/1d87582adcb4450b2f66

import os
import argparse


def get_path(dir):
    if os.path.exists(dir):
        return os.path.realpath(dir)
    else:
        print("No folder named {}!".format(dir))
        exit(0)


def collect_ipynbs(wdir):
    nbooks = os.popen('find {} -name "*.ipynb" -not -path "*/\.*"'.format(wdir)).read()
    if nbooks:
        ##___ clean up ipynb file names
        nbooks = nbooks.replace(" ", "\ ")  # avoid white spaces in a file name
        ipynb_files = nbooks.split('\n')    # split files by new line
        ipynb_files = filter(None, ipynb_files)
        return ipynb_files
    else:
        print("No jupyter notebook(s) found in {}!".format(wdir))
        exit(0)


def change_to_output_dir(wdir):
    ##___ output directory
    outd = os.path.join(wdir, "python")
    if not os.path.exists(outd):
        os.makedirs(outd)
    os.chdir(outd)
    return outd


def convert_ipynb(ipynb_files, outputdir):
    files = list(ipynb_files)
    print("Converting {} notebooks into python ... \n\n".format( len(files)))
    for nb in files:
        temp = nb.split("/")
        print(temp[len(temp)-1].split(".")[0])
        name = temp[len(temp)-1].split(".")[0]
        convert_cmd = 'jupyter nbconvert {} --to="python" --output="python/{}"'.format(nb, name)
        os.system(convert_cmd)
    print("\nSee output at: {}".format(outputdir))


def main():
    #___ parse input
    desc = 'Convert all jupyter notebook(s) in a given directory into the selected format and place output in a separate folder. Using: jupyter nbconvert and find command (Unix-like OS only).'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("dir", help="Name of the folder where the `*.ipynb` files are (or in its sub-folders).", type=str)
    parser.add_argument("-to", help="Convert to what format. (default: pdf) other formats: html, latex, markdown, python, rst, or slides.", default='pdf', type=str)
    args = parser.parse_args()

    #___ validate args
    wdir = get_path(args.dir)
    #___ start
    ipynb_files = collect_ipynbs(wdir)
    output_dir = change_to_output_dir(wdir)
    convert_ipynb(ipynb_files, output_dir)
if __name__ == '__main__':
    main()
