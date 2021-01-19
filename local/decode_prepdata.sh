#!/bin/bash

#
#    Prepare source data for Kaldi ASR.
#    This includes processing the command line for source materials and starting the diarization through flist2scp.sh
#

die() {
    echo "-------------- fatal error ----------------" >&2
    echo "$1" >&2
    echo "-------------------------------------------" >&2
    exit 2
}

timer="$(which time)"
if [ -z "$timer" ]; then
    die "GNU time not found  (apt install time)">&2
fi
timer="$timer -o $inter/time.log -f \"%e %U %S %M\""

mkdir -p $data/ALL/liumlog
echo "Data preparation (dir=$data)" >&2
echo "Data preparation (dir=$data)" >$inter/stage

# handle input types: either directory, separate files, or list of files to process
while [[ $# > 1 ]] ; do
	i=$1
	shift
	if [ -f $i ]; then
		filetype=$(file -ib $i)
		if [[ $filetype =~ .*audio.* ]]; then
			echo "Argument $i is a sound file, using it as audio"
			if $copyall; then
				cp $i $data/ || die "unable to copy data to $data"
			else
				fn=$(readlink -f $i)
				ln -s -f $fn $data || die "unable to link data to $data"
			fi
		elif [[ $filetype =~ .*text.* ]]; then
			echo "Argument $i is a text file, using it as list of files to copy"
			if $copyall; then
				xargs -a $i cp -t $data || die "unable to copy data to $data"
			else
				xargs -a $i ln -s -t $data || die "unable to link data to $data"
			fi
		fi
	elif [ -d $i ]; then
		echo -n "Argument $i is a directory, copying contents..  "
		if $copyall; then
			cp -a $i/* $data || die "unable to copy data to $data"
		else
			find $i -mindepth 1 -depth -type d -printf "%P\n" | while read dir; do mkdir -p "$data/$dir"; done
			find $i -type f -printf "%P\n" | while read file; do rm -f $data/$file; ln -s "$i/$file" "$data/$file"; done
		# ln -s -f $i/* $data
		fi
		echo "done"
	else
		echo "Argument $i cannot be processed - skipping"
	fi
done

if ! which find > /dev/null; then
    die "GNU find not found  (apt install findutils)"
fi

## Process source directory
# create file list to process, only use audio files whose type was specified in file_types
findcmd="find $data "
for type in $file_types; do
	findcmd="$findcmd -iname '*.${type}' -o "
done
findcmd=${findcmd%????}
eval $findcmd >$data/test.flist

# prepare data & do diarization
>$data/ALL/liumlog/done.log
eval $timer local/flist2scp.sh $data >>$logging 2>&1 &
pid=$!
numfiles=$(cat $data/test.flist | wc -l)
while kill -0 $pid 2>/dev/null; do
	numsegmented=$(cat $data/ALL/liumlog/done.log | wc -l)
	local/progressbar.sh $numsegmented $numfiles 50 "Diarization"
	sleep 1
done
cat $inter/time.log | awk '{printf( "Diarization completed in %d:%02d:%02d (CPU: %d:%02d:%02d), Memory used: %d MB                \n", int($1/3600), int($1%3600/60), int($1%3600%60), int(($2+$3)/3600), int(($2+$3)%3600/60), int(($2+$3)%3600%60), $4/1000) }'

numsegments=$(cat $data/ALL/segments | wc -l)
plu1=
plu2=
[ $numfiles -gt 1 ] && plu1="s"
[ $numsegments -gt 1 ] && plu2="s"
echo "Split $numfiles source file${plu1} into $numsegments segment${plu2}                              "
cat $data/*.glm 2>/dev/null >$data/ALL/all.glm								# copy any .glm's
utils/fix_data_dir.sh $data/ALL >>$logging 2>&1 || die "fix data dir failed"
cp -r $data/ALL/liumlog $result

if [ $numfiles -eq 0 ]; then
    die "No files prepared (no input found)"
fi
if [ $numsegments -eq 0 ]; then
    die "No segments extracted (no speech found)"
fi
