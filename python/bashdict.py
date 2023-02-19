"""
Reads environment variables from a bash script file and returns them as a dictionary
"""

def bash_to_dict(bash_script):
    """
    Converts a bash script file to a config file
    
    There is a special key called "unset" that contains a list of environment variables that were unset in the bash script
    """
    config = {}
    unset = []
    with open(bash_script, "r") as bash_file:
        for line in bash_file:
            line = line.strip()
            if line.startswith("#"):
                continue
            elif line.startswith("export"):
                parts = line.split("=")
                key = parts[0].split()[1]
                value = "=".join(parts[1:])
                config[key] = value
            elif line.startswith("unset"):
                key = line.split()[1]
                unset.append(key)
    config['unset'] = unset
    return config
    