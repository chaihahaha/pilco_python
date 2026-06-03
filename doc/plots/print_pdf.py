# print_pdf  Save a figure to PDF with matplotlib's savefig.
#
# Examples:
#   print_pdf('filename')
#   print_pdf('filename', fig_handle)
#
# Copyright (C) Oliver Woodford 2008
# Python translation: use matplotlib's savefig instead of Ghostscript pipeline.

import matplotlib.pyplot as plt
import os


def print_pdf(name, fig=None):
    if fig is None:
        fig = plt.gcf()

    if not name.endswith('.pdf'):
        name = name + '.pdf'

    fig.savefig(name, format='pdf', bbox_inches='tight', pad_inches=0.1)
    print('pdf successfully printed')
