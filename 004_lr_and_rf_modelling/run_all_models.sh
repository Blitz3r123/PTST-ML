#!/bin/bash

rm -rf lr_models/;
rm -rf rf_models/;

python3 lr_and_rf_modelling.py lr;
python3 lr_and_rf_modelling.py rf;
python3 model_results_analysis.py;
