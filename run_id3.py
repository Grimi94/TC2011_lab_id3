import pandas as pd
import numpy as np
import sys
from id3 import DecisionTree


if __name__ == '__main__':
    feature_names = []
    data = []
    state = 0
    # Read each input line
    for line in sys.stdin:
        # Ignore comments
        line = line.strip("\n")
        if line.startswith("%"):
            continue
        if state == 0:
            # Parse attributes
            if line.startswith("@attribute"):
                feature_names.append(line.split()[1])
            # Parse data
            elif line.startswith("@data"):
                state = 1
        else:
            data.append(line.split(','))

    data = np.array(data)
    y = data[:, -1]
    X = data[:, 0:-1]

    tree = DecisionTree()
    tree.fit(X, y, feature_names[0:-1])
    print str(tree)
