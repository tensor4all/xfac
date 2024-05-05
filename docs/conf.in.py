## to be called from parent folder
import subprocess
subprocess.call('doxygen', shell=True, cwd='docs/')
