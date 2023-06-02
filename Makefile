test:
	@echo "Hello World"


PORT_NUMBER_TO_SERVE_JUPYTER_NOTEBOOK=8811
USER_NAME=s183920

# jupyter: 
# 	ssh s183920@login1.gbar.dtu.dk
# 	sleep 2
# 	02514sh
# 	cd /work3/s183920/02514-DLCI/02514-Deep-Learning-In-Computer-Vision/
# 	source ../DLCI-venv/bin/activate
# 	jupyter notebook --no-browser --port=8811 --ip=$HOSTNAME
# bkill -u s183920
hpc1:
	ssh s183920@login1.gbar.dtu.dk 'cd /work3/s183920 && ls && x-terminal-emulator -e'
	wait
	echo helle
hpc2:
	02514sh
	cd /work3/s183920/02514-deep-learning-with-PyTorch

port_forward:
	ssh s183920@login1.hpc.dtu.dk -g -L8811:n-62-20-15:8811 -N