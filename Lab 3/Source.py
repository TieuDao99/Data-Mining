import pandas as pd
import numpy as np
import sys


def init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def group_data(X, centers):
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        d = X[i] - centers
        d = np.linalg.norm(d, axis=1)
        y[i] = np.argmin(d)
    return y

def update_centers(X, y, k):
    centers = np.zeros((k, X.shape[1]))
    for i in range(k):
        X_i = X[y == i, :]
        centers[i] = np.mean(X_i, axis=0)
    return centers

def kmeans(X, k):
    centers = init_centers(X, k)
    y = []
    i = 0
    while True:
        y_old = y
        y = group_data(X, centers)
        if np.array_equal(y, y_old):
            break
        centers = update_centers(X, y, k)
        i += 1
    return centers, y

def writeOut(data, centers):
    #np.set_printoptions(threshold=sys.maxsize)
    pd.set_option('display.max_rows', 2000)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 2000)

    f = open('output.txt', mode='w')
    for i in range(k):
        cluster = data[data.cluster == float(i)]
        f.write('Cluster {}: {}\n'.format((i + 1), len(cluster)))
        f.write('{} \n'.format(cluster['Phase'].value_counts()))
        f.write('{} \n'.format(centers[i]))
    for i in range(k):
        cluster = data[data.cluster == float(i)]
        f.write('##### {} \n'.format(i + 1))
        f.write('{}\n'.format(cluster.iloc[:, :-1]))
    f.close()


if __name__ == '__main__':
    while (True):
        print('Follow this syntax:\t\t <Labname> <K> <Input file> <Output file> \n'
              'With:'
              '\n\t\t- <Labname>     = {Lab3}'
              '\n\t\t- <K>           = number of wanted clusters'
              '\n\t\t- <Input file>  = {a1_va3.csv;'
              '\n\t\t                   a2_va3.csv;'
              '\n\t\t                   a3_va3.csv;'
              '\n\t\t                   b1_va3.csv;'
              '\n\t\t                   b3_va3.csv;'
              '\n\t\t                   c1_va3.csv'
              '\n\t\t                   c3_va3.csv;'
              '\n\t\t- <Output file> = {output.txt}')
        lab, k, infile, outfile = input().split()
        k = int(k)
        with open('gesture_phase_dataset\\' + infile) as f:
            data = pd.read_csv(f)

        X = np.array(data.iloc[:, :-1])
        centers, y = kmeans(X, k)
        data['cluster'] = y

        writeOut(data, centers)

        restart = input('Successfully compiled! Do you want to continue?(y/n)- ')
        if restart == 'y':
            continue
        else:
            break