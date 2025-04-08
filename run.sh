#!/bin/bash

#load datasets
python 'scripts\download.py'

#run GA-EVLRU

python 'scripts\ev-load-open-data\demand.py'
python 'scripts\ev-load-open-data\bss.py'

python 'scripts\st-evcdp\demand.py'
python 'scripts\st-evcdp\bss.py'

python 'scripts\urbanev\demand.py'
python 'scripts\urbanev\bss.py'

python predict.py 'ev-load-open-data'
python predict.py 'st-evcdp'
python predict.py 'urbanev'

python run_ab.py 'ev-load-open-data'
python run_ab.py 'st-evcdp'
python run_ab.py 'urbanev'

python ga_convergence.py 'st-evcdp'
python ga_convergence.py 'urbanev'


python run_ga_evlru.py 'st-evcdp'
python run_ga_evlru.py 'urbanev'

#Evaluation
python 'scripts\ev-load-open-data\evaluation.py'
python 'scripts\st-evcdp\evaluation.py'
python 'scripts\urbanev\evaluation.py'

python predict_score.py 'ev-load-open-data'
python predict_score.py 'st-evcdp'
python predict_score.py 'urbanev'

python ga_evaluation.py 'st-evcdp'
python ga_evaluation.py 'urbanev'


