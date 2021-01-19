#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, glob, codecs, logging, getopt, subprocess
from fnmatch import fnmatch
from praat import textgrid
from praat import intervaltier
from praat import interval
logging.basicConfig(format="%(levelname)-10s %(asctime)s %(message)s", level=logging.INFO)

# This script converts a directory of CTM files made with a Kaldi based
# recognition system to Praat TextGrid files in long syntax. It also merges in the
# manual transcription tier.
# The following 1 arguments are required by the script: full path to a Kaldi CTM file.
# Author: Mario Ganzeboom
# Modified: Emre Yilmaz
# Last modification: June 12, 2017

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            help()                 
            sys.exit()

    transcription_root = None
    ctm_file = argv[1]
    wav_file = argv[2]
    results_subdir = "/".join(ctm_file.split('/')[:-1])
    logging.info("Processing file '" + ctm_file + "'...")
    file_name_base = ctm_file[:-len(".ctm")]
    textgrid_file = textgrid.Textgrid()
    textgrid_file.etime = float("{:.2f}".format(float(subprocess.check_output("soxi -D "+wav_file, shell="True")[:-1])))
    word_tier_intervals = create_tier_intervals_from_ctm(os.path.join(ctm_file),textgrid_file)
    word_tier = intervaltier.IntervalTier("Words", 0.0, word_tier_intervals[len(word_tier_intervals)-1].etime, len(word_tier_intervals), word_tier_intervals)
    textgrid_file.tiers.append(word_tier)
    textgrid_file.nr_tiers = 1
    output_tg_file = os.path.join(file_name_base+".tg")
    textgrid_file.write(output_tg_file)
    logging.info("Converted CTM for utterance '" + ctm_file + "' to TextGrid file '" + output_tg_file + "'...")

def create_tier_intervals_from_ctm(path_to_ctm_file,textgrid_file):
    ctm_file = open(path_to_ctm_file, "r")
    intervals = []
    ctm_lines = ctm_file.readlines()
    new_interval = interval.Interval(0.0, float(ctm_lines[0].split()[2]) , "")
    intervals.append(new_interval)
    for line in ctm_lines:
        line_comps = line.split()
        btime = float(line_comps[2])
        new_interval = interval.Interval(btime, btime+float(line_comps[3]), str(line_comps[4]))
        intervals.append(new_interval)
        utt_id = line_comps[0]
    new_interval = interval.Interval(intervals[-1].etime, textgrid_file.etime, "")
    intervals.append(new_interval)
    return intervals

def help():
    logging.info(get_script_name() + " - Merge a directory of Kaldi alignments optionally with corresponding manual transcriptions " + \
		"to a single Praat TextGrid file (in long notation format).\n")
    usage()
    logging.info("\nParameters:\n" + \
            "-h    --help        Show this help message")

def usage():
    logging.info("Usage: " + get_script_name() + " <dir. with CTM files ending on .sym.ctm> [<root dir. where the man. transcriptions and SPRAAK results are located>]")		
		
""" Error handler printing custom_msg through logging.error() and exception.message through logging.exception().
At the end sys.exit(2) is called to abort the script.

@param exception: The exception object containing specific codes and/or messages.
@param custom_msg: Message provided by the script/application explaining the exception.
"""
def handle_exception(exception, custom_msg):
    logging.error("An error occurred during execution, see messages below:")
    logging.error(custom_msg)
    if hasattr(exception, 'message'):
        logging.exception("" + exception.message)
    logging.shutdown()
    sys.exit(2);

""" Utility function to get the 'pretty printed' name of this script from the sys.argv[0] array.

@return: String containing the file name of this script (e.g. name-of-script.py, examplescript.py, etc.)
"""
def get_script_name():
    script_name = sys.argv[0]
    last_path_sep = script_name.rfind(os.sep)
    if last_path_sep > -1:
        script_name = script_name[last_path_sep+1:]
    
    return script_name

if __name__ == '__main__':
    main(sys.argv)
