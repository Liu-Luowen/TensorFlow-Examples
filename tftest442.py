#an example to calculate the L2 loss function

import tensorflow as tf

#get the weight of one layer and make it by L2 adding to losses
def get_weight(shape,lambdaa):
   #generate an variable
   var=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
   #add_to_collection is used to add the var to L2 losses
   tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambdaa)(var))
   return var


x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))
batch_size=8

#define the node's number of each layer
layer_dimension=[2,10,10,10,1]

#the number of neural network layers
n_layers=len(layer_dimension)

#this variable keep the nodes, which is the input layer in the begining
cur_layer=x

#the node's number of the current layer
in_dimension=layer_dimension[0]

#generate a whole conneciton network of 5 layers
for i in range(1,n_layers):
   out_dimension=layer_dimension[i]
   weight=get_weight([in_dimension,out_dimension],0.001)
   bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]))
   cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias) #use the ReLU activition funciton
   in_dimension=layer_dimension[i]

#having add all L2 loss to the concept before define the farward 
#there just need to calaulate the model in the training data loss function
mse_loss=tf.reduce_mean(tf.square(y_-cur_layer))

#add the square loss function to the loss concept
tf.add_to_collection('loss',mse_loss)

#get_collection return a list, the list is the element in the concept
#in this example,these element are the different part of the loss function 
#the final loss function is add them together
loss=tf.add_n(tf.get_collection('losses'))
print loss

