#!/bin/bash
username=$1
if [ $username == "" ]; then
  echo "provivde your username to upload files"
else
scp bliss_models.tar.gz ${username}@applejack.science.ru.nl:/var/www/applejack/live/htdocs/downloads/bliss_ASR/bliss_models.tar.gz
fi