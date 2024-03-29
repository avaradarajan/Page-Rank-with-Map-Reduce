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
    print(time.time()-start)

#Create the A matrix i.e. N X N matrix that has the 1/L information
M = dok_matrix((matrix_dimension,matrix_dimension),dtype=np.float32)

for k,v in directed_graph.items():
    for val in v:
        M[val,k] = 1. / outlinks[k] * damping_factor

#Handling deadends
'''for i in range(0,matrix_dimension):
    for j in range(0,matrix_dimension):
         if i+1 in directed_graph[j+1] and i+1 != j+1:
             print(f'{i}{j}')
             M[i,j] = 1./outlinks[j+1] * damping_factor'''

print(M)
'''for i in range(0,matrix_dimension):
    for j in range(0,matrix_dimension):
         if i in directed_graph[j] and i != j:
             #print(f'{i}{j}')
             M[i,j] = 1./outlinks[j] * damping_factor'''

#print(M)
print("Here1")
print(time.time()-start)
coo = M.tocoo()
indices = np.mat([coo.row, coo.col]).transpose()
Mtf = tf.SparseTensor(indices, coo.data, coo.shape)

print("Here2")
print(time.time()-start)
#print(Mtf.dtype)
init_rank = np.multiply(np.ones([matrix_dimension,1]),1./matrix_dimension)
tensor_rank = tf.convert_to_tensor(init_rank,dtype=tf.float32)
#print(tensor_rank)
#Mdash = tf.sparse.sparse_dense_matmul(Mtf,tensor_rank) + (1-damping_factor)/matrix_dimension
#Mdash = M.dot(init_rank) + (1-damping_factor)/matrix_dimension
#
#print(Mdash)
print("Here3")
print()
print(time.time()-start)
#WM weighted matrix

WM = tf.sparse.placeholder(dtype=Mtf.dtype,shape=Mtf.shape)
x = tf.placeholder(dtype=tensor_rank.dtype,shape=tensor_rank.shape)
pagerank = tf.sparse.sparse_dense_matmul(WM,x) + (1-damping_factor)/matrix_dimension
convergence = tf.norm((pagerank - x),ord='euclidean')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #writer = tf.summary.FileWriter("output", sess.graph)
    objective_func = float('inf')
    while(objective_func > epsilon):
        newrank = sess.run(pagerank,feed_dict={WM:tf.SparseTensorValue(indices, coo.data, coo.shape),x:init_rank})
        objective_func = sess.run(convergence,feed_dict={pagerank:newrank,x:init_rank})
        init_rank = newrank
        print(objective_func)
        #writer.close()
    print("Output")
    print(newrank)

resu = sorted(newrank.tolist(),reverse=True)
nodeids = np.argsort(newrank.flatten().tolist())[::-1]

print(resu)
print(nodeids)

print("Top Ranks")
print(f'NODE IDS\tRANK')
for i in range(0,20):
    print(f'{nodeids[i]}\t\t{resu[i][0]}')
print("Bottom Ranks")
print(f'NODE IDS\tRANK')
for i in range(1,21):
    print(f'{nodeids[-i]}\t\t{resu[-i][0]}')
'''print(type(coo.data))
x = tf.sparse.placeholder(tf.float32)
z = tf.placeholder(tf.float32,shape=tensor_rank.shape)
y = tf.sparse.sparse_dense_matmul(x,z)

with tf.Session() as sess:
  #indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
  #values = np.array([1.0, 2.0], dtype=np.float32)
  #shape = np.array([7, 9, 2], dtype=np.int64)
  print(sess.run(y, feed_dict={x: tf.SparseTensorValue(indices, coo.data, coo.shape),z:init_rank}))'''