# ! pip install pygcode
# ! pip install regex

import os.path
from pygcode import Line
from re import search
import numpy as np

layers = []
layer = 0
lines_list = []


def stats(path):
    """Obtain stats on gcode files."""
    for file in os.listdir(path):
        with open(file, 'r') as fh:
            # NUMBER OF LINES
            lines_list.append(len(fh.readlines()))

            for line_text in fh.readlines():
                try:
                    line = Line(line_text)
                    if line.comment:
                        # LAYER COUNT IN PRUSA SLICER
                        if line.comment.text == 'AFTER_LAYER_CHANGE':
                            layer += 1
                        # LAYER COUNT IN CURA SLICER
                        elif search('LAYER_COUNT', line.comment.text):
                            layer_line = line.comment.text
                            layer = layer_line.split(':')[-1]
                            print(layer)
                        layers.append(int(layer))
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    stats(path)
    print(f'Layers: {layers}')
    print(f'Lines: {np.mean(lines_list)}')

