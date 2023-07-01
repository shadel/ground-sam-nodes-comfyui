package_list = None
def update_package_list():
    import sys
    import subprocess

    global package_list
    package_list = [r.decode().split('==')[0] for r in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).split()]