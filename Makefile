test:
	@echo "Hello World"


PORT_NUMBER_TO_SERVE_JUPYTER_NOTEBOOK=8811
USER_NAME=s183920

jupyter: 
	02514sh
	source DLCI-venv/bin/activate
	# CUDA_VISIBLE_DEVICES=0 jupyter notebook --no-browser --port=8811 --ip=$HOSTNAME
	jupyter notebook --notebook-dir=exercises --no-browser --port={$PORT_NUMBER_TO_SERVE_JUPYTER_NOTEBOOK}

port_forward:
	ssh s183920@login1.hpc.dtu.dk -g -L8000:n-62-20-5:8000 –N