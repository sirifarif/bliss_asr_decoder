#!/bin/bash
#
# Copyright 2018  Radboud University (Author: Emre Yilmaz)
# Adapted from the decode.sh of https://github.com/opensource-spraakherkenning-nl/Kaldi_NL
# Some cleanup by Maarten van Gompel (2020)
# Adopted to BLISS ASR by Arif Khan (2021)
# Author: Laurens van der Werff (University of Twente)
# Apache 2.0

set -a

cmd="utils/run.pl" # "slurm_pony.pl -p short --time 00:59:00 --mem 10G"
nj=8                  # maximum number of simultaneous jobs used for feature generation and decoding
stage=0

file_types="wav mp3"			# file types to include for transcription
splittext=true
dorescore=true			# rescore with largeLM as default
copyall=false			# copy all source files (true) or use symlinks (false)
overwrite=true			# overwrite the 1st pass output if already present
multichannel=false
inv_acoustic_scale="11"    # used for 1-best and N-best generation
nbest=0					  # if value >0, generate NBest.ctm with this amount of transcription alternatives
word_ins_penalty="-1.0"   # word insertion penalty
beam=11
decode_mbr=true
miac=
mwip=

model=bliss_models/AM/online
lmodel=bliss_models/LM/LM.gz
lpath=bliss_models/Lang
llpath=bliss_models/LM/rnn_folder
extractor=bliss_models/AM/online/ivector_extractor

symtab=$lpath/words.txt
wordbound=$lpath/phones/word_boundary.int
[ "$lpath" ] && symtab=$lpath/words.txt && wordbound=$lpath/phones/word_boundary.int

[ -f ./path.sh ] && . ./path.sh; # source the path.
[ -f ./cmd.sh ] && . ./cmd.sh; # source the path.

die() {
    echo "-------------- fatal error ----------------" >&2
    echo "$1" >&2
    echo "-------------------------------------------" >&2
    exit 2
}

export train_cmd=run.pl
export decode_cmd=run.pl
export cuda_cmd=run.pl
export mkgraph_cmd=run.pl

. parse_options.sh || exit 1;

if [ $# -lt 2 ]; then
    echo "Wrong #arguments ($#, expected 2)"
    echo "Usage: decode.sh [options] <source-dir|source files|txt-file list of source files> <decode-dir>"
    echo "  "
    echo "main options (for others, see top of script file)"
    echo "  --config <config-file>             # config containing options"
    echo "  --nj <nj>                          # maximum number of parallel jobs"
    echo "  --cmd <cmd>                        # Command to run in parallel with"
	if [ ! -z ${acwt+x} ]; then
    	echo "  --acwt <acoustic-weight>                 # value is ${acwt} ... used to get posteriors"
    fi
    echo "  --inv-acoustic-scale               # used for 1-best and N-best generation, may have multiple values, value is $inv_acoustic_scale"
    echo "  --word-ins-penalty                 # used for 1-best generation, may have multiple values, value is $word_ins_penalty"
    echo "  --num-threads <n>                  # number of threads to use, value is 1."
    echo "  --file-types <extensions>          # include audio files with the given extensions, default \"wav mp3\" "
    echo "  --copyall <true/false>             # copy all source files (true) or use symlinks (false), value is $copyall"
    echo "  --splittext <true/false>           # split resulting 1Best.txt into separate .txt files for each input file, value is $splittext"
    exit 1;
fi

result=${!#}
inter="${result}/intermediate"
data="${inter}/data"
logging="${inter}/log"
rescore=$inter/decode

[ `echo $inv_acoustic_scale | wc -w` -gt 1 ] && miac=true
[ `echo $word_ins_penalty | wc -w` -gt 1 ] && mwip=true

set +a

mkdir -p $inter
timer="$(which time)"
if [ -z "$timer" ]; then
    echo "GNU time not found  (apt install time)">&2
    exit 2
fi
timer="$timer -o $inter/time.log -f \"%e %U %S %M\""
cp recognize.sh $inter			# Make a copy of this file and..
echo "$0 $@" >$logging      # ..print the command line for logging

## data prep
if [ $stage -le 3 ]; then
	local/decode_prepdata.sh $@ || die "data preparation failed"
fi

for f in $data/ALL/spk2utt $data/ALL/segments; do
  if [ ! -f $f ]; then
    echo "$0: No speech found, exiting."
    exit 1
  fi
done

# determine maximum number of jobs for this feature generation and decoding
numspeak=$(cat $data/ALL/spk2utt | wc -l)
if (( $numspeak == 0 )); then echo "No speech found, exiting."; exit
elif (( $nj > $numspeak )); then this_nj=$numspeak; echo "Number of speakers is less than $nj, reducing number of jobs to $this_nj"
else this_nj=$nj
fi

## feature generation
if [ $stage -le 5 ]; then
	echo "Feature generation" >$inter/stage
	[ -e $model/conf/mfcc.conf ] && cp $model/conf/mfcc.conf $inter 2>/dev/null
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $this_nj --mfcc-config $inter/mfcc.conf $data/ALL $data/ALL/log $inter/mfcc >>$logging 2>&1 || die "Feature generation failed (make_mfcc.sh)"
    steps/compute_cmvn_stats.sh $data/ALL $data/ALL/log $inter/mfcc >>$logging 2>&1 || "Feature generation failed (compute_cmvn_stats.sh)"

fi

## decode
if [ $stage -le 6 ]; then
	echo "Decoding" >$inter/stage
	echo -n "Duration of speech: "
	cat $data/ALL/segments | awk '{s+=$4-$3} END {printf("%.0f", s)}' | local/convert_time.sh
	totallines=$(cat $data/ALL/segments | wc -l)
	rm -r -f ${inter}/decode
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $this_nj --beam $beam $data/ALL $extractor $data/ALL/ivectors_hires >>$logging 2>&1 || die "Extacting vectors failed (extract_ivectors_online.sh)"
	tmp_decode=$result/tmp/ && mkdir -p $tmp_decode
# 	eval $timer steps/nnet3/decode_looped.sh --nj $this_nj --beam $beam --acwt 1.0 --post-decode-acwt 10.0 --skip-scoring true --skip_diagnostics true --frames-per-chunk 30 --online-ivector-dir $data/ALL/ivectors_hires bliss_models/AM/graph $data/ALL $tmp >>$logging 2>&1 &
	tmp=`mktemp -d -p $tmp_decode`
	cp -r bliss_models/AM/online/* $tmp_decode
    eval $timer steps/online/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --skip-scoring true --nj $this_nj $model/graph $data/ALL $tmp >>$logging 2>&1 &
	pid=$!
	while kill -0 $pid 2>/dev/null; do
		linesdone=$(cat $tmp/log/decode.*.log 2>/dev/null | grep "Log-like per frame for utterance" | wc -l)
		local/progressbar.sh $linesdone $totallines 50 "Chain Decoding" || die "Unable to render progress bar"
		sleep 2
	done
	tail -1 $inter/time.log | awk '{printf( "Chain decoding completed in %d:%02d:%02d (CPU: %d:%02d:%02d), Memory used: %d MB                \n", int($1/3600), int($1%3600/60), int($1%3600%60), int(($2+$3)/3600), int(($2+$3)%3600/60), int(($2+$3)%3600%60), $4/1000) }'

	mv -f $tmp ${inter}/decode
	rm -r $tmp_decode
	mv $inter/time.log $inter/time.decode.log
fi

# ## rescore
# if [ $stage -le 7 ] && [ $llpath ] && [ -e $inter/decode/num_jobs ]; then
# 	echo "Rescoring" >$inter/stage
# 	numjobs=$(< $inter/decode/num_jobs)
#         eval $timer steps/rnnlmrescore.sh --rnnlm_ver faster-rnnlm --N 1000 --skip-scoring true --inv-acwt 10 0.75 $lpath $llpath $data/ALL $inter/decode $inter/rescore >>$logging 2>&1 &
# 	pid=$!
# 	spin='-\|/'
# 	i=0
# 	while kill -0 $pid 2>/dev/null; do
# 		i=$(( (i+1) %4 ))
#   		printf "\rRescoring.. ${spin:$i:1}"
#   		sleep .2
# 	done
# 	cat $inter/time.log | awk '{printf("\rRescoring completed in %d:%02d:%02d (CPU: %d:%02d:%02d), Memory used: %d MB                \n", int($1/3600), int($1%3600/60), int($1%3600%60), int(($2+$3)/3600), int(($2+$3)%3600/60), int(($2+$3)%3600%60), $4/1000) }'
# 	rescore=$inter/rescore
# 	mv $inter/time.log $inter/time.rescore.log
# fi

[ $llpath ] && rescore=$inter/decode

## create readable output
if [ $stage -le 8 ] && [ -e $rescore/num_jobs ]; then
	echo -e "Producing output" >$inter/stage

	frame_shift_opt=
	rm -f $data/ALL/1Best.* $result/1Best* $rescore/1Best.*

	numjobs=$(< $rescore/num_jobs)

	if [ -f $model/frame_shift ]; then
		frame_shift_opt="--frame-shift=$(cat $model/frame_shift)"
	elif [ -f $model/frame_subsampling_factor ]; then
		factor=$(cat $model/frame_subsampling_factor) || exit 1
		frame_shift_opt="--frame-shift=0.0$factor"
	fi

	# produce 1-Best with confidence
	for iac in $inv_acoustic_scale; do
		for wip in $word_ins_penalty; do
			ident=
			[ $mwip ] && ident="$wip."
			[ $miac ] && ident="$ident$iac."
			$cmd --max-jobs-run $nj JOB=1:$numjobs $inter/l2c_log/lat2ctm.${ident}JOB.log \
				gunzip -c $rescore/lat.JOB.gz \| \
				lattice-push ark:- ark:- \| \
				lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
				lattice-align-words $wordbound $model/final.mdl ark:- ark:- \| \
				lattice-to-ctm-conf $frame_shift_opt --inv-acoustic-scale=$iac ark:- - \| utils/int2sym.pl -f 5 $symtab \| \
				local/ctm_time_correct.pl $data/ALL/segments \| sort \> $rescore/1Best.${ident}JOB.ctm | exit 1;
			cat $rescore/1Best.${ident}*.ctm >$rescore/1Best_raw.${ident}ctm

			cat $rescore/1Best_raw.${ident}ctm | sort -k1,1 -k3,3n | \
				perl local/combine_numbers.pl | sort -k1,1 -k3,3n | local/compound-restoration.pl 2>>$logging | \
				grep -E --text -v 'uh|<unk>' >$result/1Best.${ident}ctm
			[ -s $data/ALL/all.glm ] && mv $result/1Best.${ident}ctm $rescore/1Best_prefilt.${ident}ctm && \
				cat $rescore/1Best_prefilt.${ident}ctm | csrfilt.sh -s -i ctm -t hyp $data/ALL/all.glm >$result/1Best.${ident}ctm

			local/ctmseg2sent.pl $result $splittext $ident
                        cat $result/1Best.txt | cut -d'(' -f 1 > $result/$(basename $result).txt
                        begin_line=$(cut -d' ' -f 1-2 $result/1Best.ctm | head -n1)
                        tail -n +2 $result/1Best.txt | cut -d'(' -f2 | cut -d' ' -f2 | sed 's/)//g' > $result/temp1
			cat $result/temp1 | sed "s/^/$begin_line /g" | sed 's/$/ 0.00 <eos> 1.00/g' > $result/eos.ctm
			cat $result/1Best.ctm $result/eos.ctm | awk '{print $0, $3+$4}'| sort -nk7 | cut -d' ' -f 1-6 | awk '{printf "%s %s %.2f %.2f %s %.2f\n", $1, $2, $3, $4, $5, $6}' > $result/$(basename $result).ctm

		done
	done
fi
