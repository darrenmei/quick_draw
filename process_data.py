import numpy as np
import pandas as pd
import sys


def CreateLabels(inputfile, outfile):
    data = pd.read_csv(inputfile, sep = ',', header = 0)

    print('dim of data: ', data.shape)
    data = data.values
    m, n = data.shape
    #print('data: ', data)

    # Key: col 1: airplane, col 2: campfire, col 3: key, col 4: moon, col 5: palm tree
    #print('data: ', data)
    print('m: ', m)
    labels = np.zeros((m, 5))
    for row in range(m):
        if row == 0:
            pass
        #print('data: ', data[row, 0])
        if data[row, 2] == 'airplane':
            labels[row, 0] = 1
        elif data[row, 2] == 'campfire':
            labels[row, 1] = 1
        elif data[row, 2] == 'key':
            labels[row, 2] = 1
        elif data[row, 2] == 'moon':
            labels[row, 3] = 1
        else:
            labels[row, 4] = 1
    print('labels: ', labels)
    df = pd.DataFrame(labels)
    df.to_csv(outfile, index = False)


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python CreateLabels.py <infile>.csv <outfile>.csv")

    print('infile: ', sys.argv[1])
    infile= sys.argv[1]
    outfile = sys.argv[2]

    CreateLabels(infile, outfile)

if __name__ == '__main__':
	main()
