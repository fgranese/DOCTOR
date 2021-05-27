import math
import sys

class GUI_tools:
    def __init__(self):
        pass

    def print_progress(self, i, slop, n, progress_bar):
        if i == math.ceil(n / progress_bar):
            slop = slop.replace('>', '=>', 1)
        if i == n - 1:
            slop = slop.replace('>', '=', 1)
            slop = slop.replace('.', '=')
        else:
            slop = slop.replace('>.', '=>', 1)
        sys.stdout.write("\r" + slop)
        sys.stdout.flush()
        return slop

    def print_status(self, i, n):
        sys.stdout.write("\r" + '{} of {}'.format(i, n))
        sys.stdout.flush()