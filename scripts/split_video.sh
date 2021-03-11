#!/bin/bash
# Short script to split videos by filesize using ffmpeg by LukeLR
#Features:
#Written by LukeLR stackoverflow user
#Use ffmpeg's -fs-flag to limit filesize
#Check length of resulting video part and calculate where to start next part
#Enumerate video parts
#proceed with as many parts as needed to contain the whole source file.
#Allow for custom ffmpeg arguments to also reduce resolution and quality in one pass.
#It takes only three arguments: The source filename, the desired filesize of each part and the desired arguments for ffmpeg.
#
#Example call to split a file into 64MB parts:
#
#./split-video.sh huge-video.mov 64000000 "-c:v libx264 -crf 23 -c:a copy -vf scale=960:-1"


if [ $# -ne 4 ]; then
    echo 'Illegal number of parameters. Needs 3 parameters:'
    echo 'Usage:'
    echo './split-video.sh FILE SIZELIMIT "FFMPEG_ARGS'
    echo
    echo 'Parameters:'
    echo '    - FILE:        Name of the video file to split'
    echo '    - SIZELIMIT:   Maximum file size of each part (in bytes)'
    echo '    - FFMPEG_ARGS: Additional arguments to pass to each ffmpeg-call'
    echo '                   (video format and quality options etc.)'
    echo '    - EXTENSION:   Video output extension.'
    exit 1
fi

FILE="$1"
SIZELIMIT="$2"
FFMPEG_ARGS="$3"

# Duration of the source video
DURATION=$(ffprobe -i "$FILE" -show_entries format=duration -v quiet -of default=noprint_wrappers=1:nokey=1)

# Duration that has been encoded so far
CUR_DURATION=0.0

# Filename of the source video (without extension)
BASENAME="${FILE%.*}"

# Extension for the video parts
#EXTENSION="${FILE##*.}"
EXTENSION="$4"

# Number of the current video part
i=1

# Filename of the next video part
NEXTFILENAME="$BASENAME-$i.$EXTENSION"

echo "Duration of source video: $DURATION"

# Until the duration of all partial videos has reached the duration of the source video
#if (`fcomp $CUR_DURATION '<=' $DURATION`); then true; else echo false; fi
#A=`awk "BEGIN {print $CUR_DURATION + $CUR_DURATION}"`


#while [[ $CUR_DURATION -lt $DURATION ]]; do
#while awk 'BEGIN { if ('$CUR_DURATION'>'$DURATION') {exit 1}}'; do
while [ "$(bc <<< "$CUR_DURATION < $DURATION")" == "1" ]; do
    # Encode next part
    echo ffmpeg -i "$FILE" -ss "$CUR_DURATION" -fs "$SIZELIMIT" $FFMPEG_ARGS "$NEXTFILENAME"
    ffmpeg -ss "$CUR_DURATION" -i "$FILE" -fs "$SIZELIMIT" $FFMPEG_ARGS "$NEXTFILENAME"

    #Duration of the new part
    NEW_DURATION=$(ffprobe -i "$NEXTFILENAME" -show_entries format=duration -v quiet -of default=noprint_wrappers=1:nokey=1)
    # NEW_DURATION='0.1'
    # Total duration encoded so far
    CUR_DURATION=`awk "BEGIN {print $CUR_DURATION + $NEW_DURATION}"`

    i=$((i + 1))

    echo "Duration of $NEXTFILENAME: $NEW_DURATION"
    echo "Part No. $i starts at $CUR_DURATION"

    NEXTFILENAME="$BASENAME-$i.$EXTENSION"
done

echo "DONE"