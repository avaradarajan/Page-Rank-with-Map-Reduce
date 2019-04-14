import numpy as np
import tensorflow as tf
outlinks = {}
directed_graph = {}
matrix_dimension = 0
epsilon = 0.0001
damping_factor = 0.85
#get filename as arg
#read from hdfs
#top 20 node ids with rank
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
    #for key, value in sorted(directed_graph.items()):
    #    print(key, ":", value)
    #print(sorted(outlinks.items()))
    matrix_dimension = len(directed_graph.keys())
    print(matrix_dimension)

#Create the A matrix i.e. N X N matrix that has the 1/L information
A = [[0 for i in range(matrix_dimension)] for j in range(matrix_dimension)]

#Handling deadends
for i in range(0,matrix_dimension):
    for j in range(0,matrix_dimension):
         if i+1 in directed_graph[j+1] and i+1 != j+1:
             A[i][j] = 1./outlinks[j+1]

A = np.array(A)

dangling = np.sum(A,axis=0)
ind = [i for i, e in enumerate(dangling) if e == 0]
for column in ind:
    A[:,column] = 1./matrix_dimension

#print(A)

#rand_prob creates the 1-d/N matrix which is the prob part when user randomly opens a page
rand_prob = np.multiply(np.ones([matrix_dimension,matrix_dimension]),(1-damping_factor)/matrix_dimension)

#M is the weighted matrix. This has to be multiplied with rank matrix till convergence is met. Avoiding spider traps with damping factor
M = np.add(np.multiply(A,damping_factor),rand_prob)

#print(M)

rank = np.multiply(np.ones([matrix_dimension,1]),1./matrix_dimension)

#print(rank)
#print()
#WM weighted matrix
WM = tf.placeholder(dtype=tf.float32,shape=(matrix_dimension,matrix_dimension))
x = tf.placeholder(dtype=tf.float32,shape=(matrix_dimension,1))
pagerank = tf.matmul(WM,x)
convergence = tf.norm((pagerank - x),ord='euclidean')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    objective_func = float('inf')
    while(objective_func > epsilon):
        newrank = sess.run(pagerank,feed_dict={WM:M,x:rank})
        objective_func = sess.run(convergence,feed_dict={pagerank:newrank,x:rank})
        rank = newrank
    print(newrank)
    #print(list(sorted(newrank,reverse=True)))
