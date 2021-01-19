#!/bin/bash

rm bliss_models.tar.gz 2>/dev/null
wget https://applejack.science.ru.nl/downloads/bliss_ASR/bliss_models.tar.gz && tar -xvzf bliss_models.tar.gz
exit $?
