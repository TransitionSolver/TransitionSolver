# Taken from https://stackoverflow.com/a/45669280.
# Temporarily switches the output of printing to a null destination, effectively disabling printing. To use this print
# suppression feature, use "with PrintSuppressor.PrintSuppressor: <code here>".

import os
import sys


class PrintSuppressor:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
