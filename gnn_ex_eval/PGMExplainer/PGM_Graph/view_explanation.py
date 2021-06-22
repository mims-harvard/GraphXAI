import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def arg_parse():
    parser = argparse.ArgumentParser(description="PGM Explainer arguments.")
    parser.add_argument(
            "--index", dest="index", type=int, help="Index of explanation to view."
        )
    parser.set_defaults(
        index = None
    )
    
    return parser.parse_args()

prog_args = arg_parse()

filenames = ['result/explanations.txt']

dataframes = []
for f in filenames:
    dataframes.append(pd.read_csv(f, delimiter=r'\t', 
                   names = ["data","ID","Label","Nodes","X_cor","Y_cor"], 
                   header=None, engine='python'))

Ex = pd.concat(dataframes)
Ex["data"] = Ex["data"].map(lambda x: x.replace('[', ':'))
Ex["data"] = Ex["data"].map(lambda x: x.replace(']', ''))
new = Ex["data"].str.split(":",expand = True)
new_ = new[1].str.split(",",expand = True)
Ex["ID"] = new_[0]
Ex["Label"] = new_[1]
Ex["Nodes"] = new[2]
Ex["X_cor"] = new[3]
Ex["Y_cor"] = new[4]
del new
del new_

Ex.drop(["data"], axis = 1, inplace=True)
Ex = Ex.astype({'ID':'int32'})
Ex = Ex.astype({'Label':'int32'})

# print(Ex)

if prog_args.index == None:
    print("View all")
    result_X = {}
    result_Y = {}

    for label in range(10):
        Xs = []
        Ys = []
        for index, row in Ex.loc[Ex["Label"] ==  label].iterrows():           
            if len(row["Nodes"]) > 2:
                X_cor = row["X_cor"].split(',')
                X_cor = X_cor[:-1]
                X_cor = [float(i) for i in X_cor]
                for x in X_cor:
                    Xs.append(1 - x)
                Y_cor = row["Y_cor"].split(',')
                Y_cor = [float(i) for i in Y_cor]
                for y in Y_cor:
                    Ys.append(y)

        result_X[label] = Xs
        result_Y[label] = Ys
        
    plt.figure(figsize=(15, 6))
    plt.subplot(2,5,1)
    plt.scatter(result_Y[0], result_X[0], s = 1, alpha = 0.8, marker = '.')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 0')

    plt.subplot(2,5,2)
    plt.scatter(result_Y[1], result_X[1], s = 1, alpha = 0.8, marker = '.')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 1')

    plt.subplot(2,5,3)
    plt.scatter(result_Y[2], result_X[2], s = 1, alpha = 0.8, marker = '.')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 2')

    plt.subplot(2,5,4)
    plt.scatter(result_Y[3], result_X[3], s = 1, alpha = 0.8, marker = '.')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 3')

    plt.subplot(2,5,5)
    plt.scatter(result_Y[4], result_X[4], s = 1, alpha = 0.8, marker = '.')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 4')

    plt.subplot(2,5,6)
    plt.scatter(result_Y[5], result_X[5], s = 1, alpha = 0.8, marker = '.')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 5')

    plt.subplot(2,5,7)
    plt.scatter(result_Y[6], result_X[6], s = 1, alpha = 0.8, marker = '.')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 6')

    plt.subplot(2,5,8)
    plt.scatter(result_Y[7], result_X[7], s = 1, alpha = 0.8, marker = '.')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 7')

    plt.subplot(2,5,9)
    plt.scatter(result_Y[8], result_X[8], s = 1, alpha = 0.8, marker = '.')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 8')

    plt.subplot(2,5,10)
    plt.scatter(result_Y[9], result_X[9], s = 1, alpha = 0.8, marker = '.')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 9')
    
    plt.savefig('result/all_loc.png')
    
    resolution = 10
    xedges = [i / resolution for i in list(range(resolution+1))]
    yedges = xedges
    
    plt.figure(figsize=(15, 6))
    plt.subplot(2,5,1)
    H, xedges, yedges = np.histogram2d([1-i for i in result_X[0]], result_Y[0], bins=(xedges, yedges))
    plt.imshow(H)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 0')

    plt.subplot(2,5,2)
    H, xedges, yedges = np.histogram2d([1-i for i in result_X[1]], result_Y[1], bins=(xedges, yedges))
    plt.imshow(H)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 1')

    plt.subplot(2,5,3)
    H, xedges, yedges = np.histogram2d([1-i for i in result_X[2]], result_Y[2], bins=(xedges, yedges))
    plt.imshow(H)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 2')

    plt.subplot(2,5,4)
    H, xedges, yedges = np.histogram2d([1-i for i in result_X[3]], result_Y[3], bins=(xedges, yedges))
    plt.imshow(H)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 3')

    plt.subplot(2,5,5)
    H, xedges, yedges = np.histogram2d([1-i for i in result_X[4]], result_Y[4], bins=(xedges, yedges))
    plt.imshow(H)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 4')

    plt.subplot(2,5,6)
    H, xedges, yedges = np.histogram2d([1-i for i in result_X[5]], result_Y[5], bins=(xedges, yedges))
    plt.imshow(H)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 5')

    plt.subplot(2,5,7)
    H, xedges, yedges = np.histogram2d([1-i for i in result_X[6]], result_Y[6], bins=(xedges, yedges))
    plt.imshow(H)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 6')

    plt.subplot(2,5,8)
    H, xedges, yedges = np.histogram2d([1-i for i in result_X[7]], result_Y[7], bins=(xedges, yedges))
    plt.imshow(H)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 7')

    plt.subplot(2,5,9)
    H, xedges, yedges = np.histogram2d([1-i for i in result_X[8]], result_Y[8], bins=(xedges, yedges))
    plt.imshow(H)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 8')

    plt.subplot(2,5,10)
    H, xedges, yedges = np.histogram2d([1-i for i in result_X[9]], result_Y[9], bins=(xedges, yedges))
    plt.imshow(H)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Label 9')
    
    plt.savefig('result/all_dense.png')

else:
    Xs = []
    Ys = []
    Ex_nodes = []
    for index, row in Ex.loc[Ex["ID"] == prog_args.index].iterrows():
        if len(row["Nodes"]) > 2:
            Nodes = row["Nodes"].split(',')
            Nodes = Nodes[:-1]
            Nodes = [int(i) for i in Nodes]
            Ex_nodes = Nodes
            X_cor = row["X_cor"].split(',')
            X_cor = X_cor[:-1]
            X_cor = [float(i) for i in X_cor]
            for x in X_cor:
                Xs.append(1 - x)
            Y_cor = row["Y_cor"].split(',')
            Y_cor = [float(i) for i in Y_cor]
            for y in Y_cor:
                Ys.append(y)

        
    



