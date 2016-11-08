import pandas as pd
import numpy as np
import sys
from id3 import DecisionTree


if __name__ == '__main__':
    feature_names = []
    feature_values = {}
    data = []
    state = 0
    # Read each input line
    for line in sys.stdin:
        line = line.strip("\n")

        # Ignore comments
        if line.startswith("%"):
            continue

        if state == 0:
            # Parse attributes
            if line.startswith("@attribute"):
                # Parse attribute name
                contents = line.replace("{", "").replace("}", "").replace(",", "").split()
                feature_names.append(contents[1])
                # Parse possible attribute values
                feature_values[contents[1]] = contents[2:]

            # Parse data
            elif line.startswith("@data"):
                state = 1
        else:
            data.append(line.split(','))

    data = np.array(data)
    y = data[:, -1]
    X = data[:, 0:-1]

    tree = DecisionTree()
    tree.fit(X, y, feature_names[0:-1], feature_values)
    print str(tree).strip("\n")
