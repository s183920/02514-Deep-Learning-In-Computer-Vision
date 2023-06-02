# import subprocess

user = "s183920"
# host = "login1.gbar.dtu.dk"
# cmd = "02415sh"

# subprocess.Popen(f"ssh {user}@{host} {cmd}", shell=True).communicate()
from paramiko import SSHClient
from getpass import getpass

# Connect
client = SSHClient()
client.load_system_host_keys()
username = input("Username: ")
pw = getpass("Password: ")
client.connect('login1.gbar.dtu.dk', username=username, password=pw)
