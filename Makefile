test:
	@echo "Hello World"


jupyter: 
	02514sh
	source DLCI-venv/bin/activate
	CUDA_VISIBLE_DEVICES=0 jupyter notebook --no-browser --port=8888 --ip=$HOSTNAME

port_forward:
	ssh s183920@login1.hpc.dtu.dk -g -L8888:n-00-00-00:8888 –N