import subprocess
import sys

package_list = None
def update_package_list():
    import sys
    import subprocess

    global package_list
    package_list = [r.decode().split('==')[0] for r in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).split()]

def ensure_package(package_name, import_path):
    global package_list
    if package_list == None:
        update_package_list()

    if package_name not in package_list:
        print("(First Run) Installing missing package %s" % package_name)
        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', import_path])
        update_package_list()