import subprocess

# Start Jupyter server
subprocess.Popen(["jupyter", "notebook", "--no-browser"])

# Get URL and token
result = subprocess.run(["jupyter", "notebook", "list"], capture_output=True, text=True)
output = result.stdout.strip().split("\n")[1]
url = output.split(" ")[0]
token = output.split(" ")[-1]

# Output URL and token
print(f"Jupyter server running at {url}")
print(f"Token: {token}")