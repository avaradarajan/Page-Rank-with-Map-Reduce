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
def csr_matrix_equal2(a1, a2):
    print(np.array_equal(a1.shape, a2.shape))
    print(np.array_equal(a1.indices,a2.indices))
    print(np.array_equal(a1.data, a2.data))
def createStructure(from_node,to_node):
    #print(f'{from_node} {to_node}')
    if from_node not in outlinks.keys():
        outlinks[from_node] = 0
        directed_graph[from_node] = set()

    if to_node not in outlinks.keys():
        outlinks[to_node] = 0
        directed_graph[to_node] = set()


with open('C://Users//anand//Documents//email.txt') as fp:
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
    print(directed_graph[1004])
    #print(len(directed_graph[0]))
#    print(time.time()-start)

#Create the A matrix i.e. N X N matrix that has the 1/L information
M = dok_matrix((matrix_dimension,matrix_dimension),dtype=np.float32)
N = dok_matrix((matrix_dimension,matrix_dimension),dtype=np.float32)
for k,v in directed_graph.items():
    for val in v:
        #print(f'{val},{k}')
        if val!=k:
            M[val,k] = 1. / outlinks[k] * damping_factor
#print("nreak")
#Handling deadends
'''for i in range(0,matrix_dimension):
    for j in range(0,matrix_dimension):
         if i+1 in directed_graph[j+1] and i+1 != j+1:
             print(f'{i}{j}')
             M[i,j] = 1./outlinks[j+1] * damping_factor'''

#print(M)
for i in range(0,matrix_dimension):
    for j in range(0,matrix_dimension):
         if i in directed_graph[j] and i != j:
             N[i,j] = 1./outlinks[j] * damping_factor

#print(M)
print("Here1")
coo = M.tocsr()
coo2 = N.tocsr()


'''indices = np.mat([coo.row, coo.col]).transpose()
ind2 = np.mat([coo2.row, coo2.col]).transpose()

print(indices)
print(ind2)'''

print(csr_matrix_equal2(coo,coo2))

