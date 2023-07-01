import subprocess
import sys

from .update_package_list import update_package_list, package_list


def ensure_package(package_name, import_path):
    global package_list
    if package_list == None:
        update_package_list()

    if package_name not in package_list:
        print("(First Run) Installing missing package %s" % package_name)
        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', import_path])
        update_package_list()