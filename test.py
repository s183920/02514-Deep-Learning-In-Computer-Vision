# import torch
# print(torch.cuda.is_available())


import subprocess

# Define server and user
user = "s183920"
server = "login1.gbar.dtu.dk"

p = subprocess.Popen(["ssh", f"{user}@{server}"], shell=True)

# Activate GPU node
# p = subprocess.Popen(["ssh", f"{user}@{server}", "02514sh"])