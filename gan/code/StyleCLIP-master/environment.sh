#!/bin/bash

# module load python3/3.8.4
module load python3/3.11.3
module load cuda/11.0


python3 -m venv style_clip_env
source style_clip_env/bin/activate
# python -m pip install site
python -m pip install -r requirements.txt
