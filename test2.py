import subprocess

# Define server and user
server = "login1.gbar.dtu.dk"
user = "s183920"
port = "8100"

# SSH into server
subprocess.run(["ssh", f"{user}@{server}"])

# Activate GPU node
# subprocess.run(["ssh", f"{user}@{server}", "02514sh"])
subprocess.run(["02514sh"])

# Activate virtual environment
subprocess.run(["source", "/work3/s183920/02514-DLCI/DLCI-venv/bin/activate"])

# Run Jupyter instance
subprocess.Popen(["ssh", f"{user}@{server}", "jupyter", "notebook", "--no-browser", f"--port={port}", "--ip=$HOSTNAME"])

# Open new terminal and port forward
subprocess.Popen(["ssh", f"{user}@{server}", "-g", f"-L{port}:$HOSTNAME:{port}", "-N", "gnome-terminal"])

# ssh USER@login1.hpc.dtu.dk -g  –N