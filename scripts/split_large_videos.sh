#!/bin/bash
# by Ariel Iporre

if [ $# -ne 3 ]; then
    echo 'Illegal number of parameters. Needs 3 parameters:'
    echo 'Usage:'
    echo './split_large_videos.sh VIDEOSPATH SIZELIMIT EXTENSION_OUTPUT'
    echo
    echo 'Parameters:'
    echo '    - VIDEOSPATH:  Path to the videos. '
    echo '    - SIZELIMIT:   Maximum file size of each part (in bytes)'
    echo '    - EXTENSION_OUTPUT:   Video output extension.'
    exit 1
fi
VIDEOSPATH=$1
SIZELIMIT=$2
EXTENSION=$3

for f in $(ls -R $VIDEOSPATH | awk '/:$/&&f{s=$0;f=0}; /:$/&&!f{sub(/:$/,"");s=$0;f=1;next}; NF&&f{ print s"/"$0 };'); do
  if [[ $f =~ ^.*-[0-9]+\.mpg$ ]]; then
    echo "splited video: $f";
  else
    echo "large videos: $f";
    bash split_video.sh $f $SIZELIMIT "-hide_banner -loglevel error -c:v copy -c:a copy" $EXTENSION
  fi

done
