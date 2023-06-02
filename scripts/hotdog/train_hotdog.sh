#!/bin/sh
### General options
### ?- specify queue --
#BSUB -q 02514sh
### -- set the job Name --
#BSUB -J HotdogClassifier
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- set span if number of cores is more than 1
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 10GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s183920@student.dtu.dk
### -- send notification at start --
### BSUB -B
### -- send notification at completion--
###BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Hotdog_%J.out
#BSUB -e Hotdog_%J.err
# -- end of LSF options --



### Load modules
# module load python3
module load cuda/11.8

### Run setup
# sh setup.sh $run_dir || exit 1
source ../DLCI-venv/bin/activate
echo $PWD

### Run python script
python hotdog/classifier.py