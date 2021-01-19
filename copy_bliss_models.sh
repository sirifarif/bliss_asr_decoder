#!/bin/bash
h=`hostname`
prefix="${h:0:3}"
if [ $prefix == "mlp" ]; then
  cp /vol/tensusers5/arifkhan/mod.tar.gz /var/www/applejack/live/htdocs/downloads/bliss_ASR/ || exit 1;
  tar -xvzf mod.tar.gz || exit 1;
  echo "models copied and extracted to downloads folder"
else
  echo "not on ponylany machines"
fi
