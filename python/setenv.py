# This takes a bash script file as input and sets the environment variables in the running Python environment

import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("bash_script", nargs="?", help="path to bash script file")
args = parser.parse_args()

# Check if the path to the bash script is provided
if args.bash_script:
    # Read in the bash script and set the environment variables
    with open(args.bash_script, "r") as bash_file:
        for line in bash_file:
            line = line.strip()
            if line.startswith("#"):
                continue
            elif line.startswith("export"):
                parts = line.split("=")
                key = parts[0].split()[1]
                value = "=".join(parts[1:])
                os.environ[key] = value
            elif line.startswith("unset"):
                key = line.split()[1]
                if key in os.environ:
                    os.environ.pop(key)

    # Your code can now reference the environment variables as usual
else:
    print("Error: path to bash script file not provided")
