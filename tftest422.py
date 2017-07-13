#this an example of a simple neural network
import tensorflow as tf
from numpy.random import RandomState
import matplotlib.pyplot as plt
#the size of training data batch
batch_size=8

#None is used to match the batch
x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
#regression problem usually have one output point
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#define the parameters of the neural network
w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y=tf.matmul(x,w1)

#define the chengben 
loss_less=10
loss_more=1
loss=tf.reduce_sum(tf.select(tf.greater(y,y_),(y-y_)*loss_more,(y_-y)*loss_less))
#cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

#random generate a simulation dataset
rdm=RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)

Y=[[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

#create a session to run the program
with tf.Session() as sess:
   init_op=tf.initialize_all_variables()
   sess.run(init_op)
   print sess.run(w1)
   STEPS=5000
   eachloss=[]
   for i in range(STEPS):
      start=(i*batch_size)%dataset_size
      end=min(start+batch_size,dataset_size)
      sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
      if i%200==0:
         t_loss=sess.run(loss,feed_dict={x:X,y_:Y})
         eachloss.append(t_loss)
         print 'training step(s) -->%d, loss on all data is %f'%(i,t_loss)
   print sess.run(w1)
   xl=range(len(eachloss))
   plt.plot(xl,eachloss,'--ro')
   plt.xlabel('Number of training step')
   plt.ylabel('Loss of all dataset')
   plt.show()

