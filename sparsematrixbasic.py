from scipy.sparse import dok_matrix
import numpy as np
import tensorflow as tf

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
        edge_info = (line.strip().split(' '))
        from_node = int(edge_info[0])
        to_node = int(edge_info[1])
        createStructure(from_node,to_node)
        if from_node == to_node:
            continue
        elif to_node not in directed_graph[from_node]:
            directed_graph[from_node].add(to_node)
            outlinks[from_node]+=1
    for key, value in sorted(directed_graph.items()):
        print(key, ":", value)
    #print(sorted(outlinks.items()))
    matrix_dimension = len(directed_graph.keys())
    print(matrix_dimension)

#Create the A matrix i.e. N X N matrix that has the 1/L information
M = dok_matrix((matrix_dimension,matrix_dimension),dtype=np.float32)

#Handling deadends
for i in range(0,matrix_dimension):
    for j in range(0,matrix_dimension):
         if i+1 in directed_graph[j+1] and i+1 != j+1:
             print(f'{i}{j}')
             M[i,j] = 1./outlinks[j+1] * damping_factor

print(M)
coo = M.tocoo()
indices = np.mat([coo.row, coo.col]).transpose()
Mtf = tf.SparseTensor(indices, coo.data, coo.shape)

print(Mtf)
init_rank = np.multiply(np.ones([matrix_dimension,1]),1./matrix_dimension)
tensor_rank = tf.convert_to_tensor(init_rank,dtype=tf.float32)
print(tensor_rank)
Mdash = tf.sparse.sparse_dense_matmul(Mtf,tensor_rank) + (1-damping_factor)/matrix_dimension
#Mdash = M.dot(init_rank) + (1-damping_factor)/matrix_dimension
#
print(Mdash)

print()
#WM weighted matrix
WM = tf.placeholder(dtype=tf.float32,shape=(matrix_dimension,matrix_dimension))
x = tf.placeholder(dtype=tf.float32,shape=(matrix_dimension,1))
pagerank = tf.matmul(WM,x)
convergence = tf.norm((pagerank - x),ord='euclidean')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    ans = sess.run(Mdash)
    print(ans)
