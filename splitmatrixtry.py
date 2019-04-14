from scipy.sparse import dok_matrix
import numpy as np
import tensorflow as tf
import time
start = time.time()
outlinks = {}
directed_graph = {}
matrix_dimension = 0
epsilon = 0.0001
damping_factor = 0.85

def createStructure(from_node,to_node):
    #print(f'{from_node} {to_node}')
    if from_node not in outlinks.keys():
        outlinks[from_node] = 0
        directed_graph[from_node] = set()
    if to_node not in outlinks.keys():
        outlinks[to_node] = 0
        directed_graph[to_node] = set()


with open('C://Users//anand//Documents//input.txt') as fp:
    for line in fp:
        if "#" not in line:
            if('\t' in line):
                edge_info = (line.strip().split('\t'))
                from_node = int(edge_info[0])
                to_node = int(edge_info[1])
                createStructure(from_node,to_node)
                if from_node == to_node:
                    continue
                elif to_node not in directed_graph[from_node]:
                    directed_graph[from_node].add(to_node)
                    outlinks[from_node]+=1
            else:
                edge_info = (line.strip().split(' '))
                from_node = int(edge_info[0])
                to_node = int(edge_info[1])
                createStructure(from_node,to_node)
                if from_node == to_node:
                    continue
                elif to_node not in directed_graph[from_node]:
                    directed_graph[from_node].add(to_node)
                    outlinks[from_node]+=1

    #for key, value in sorted(directed_graph.items()):
        #print(key, ":", value)
    #print(sorted(outlinks.items()))
    matrix_dimension = len(directed_graph.keys())
    print(matrix_dimension)
    print(time.time()-start)

#Create the A matrix i.e. N X N matrix that has the 1/L information
M = dok_matrix((matrix_dimension,matrix_dimension),dtype=np.float32)

for k,v in directed_graph.items():
    for val in v:
        M[val-1,k-1] = 1. / outlinks[k] * damping_factor

#print(M)
print("Here1")
print(time.time()-start)
coo = M.tocsr()
coo = M.tocoo()
indices = np.mat([coo.row, coo.col]).transpose()
