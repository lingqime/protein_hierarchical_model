import platform

# Get the machine type
machine_type = platform.machine()
print(f"Machine type: {machine_type}")

# Get the operating system name and version
os_name = platform.system()
os_version = platform.version()
print(f"Operating system: {os_name} {os_version}")