#an exanple for the moving average

import tensorflow as tf

#define a variable to calculate the movingAverage, the initial is 0, 
#the typedef is tf.float32, because all the variable needed to calculate is real number
v1=tf.Variable(0, dtype=tf.float32)

#step is the iter number,can control the decay parameter
step=tf.Variable(0,trainable=False)

#define a moving average class the initail decay rate is 0.99 and the control variable is step
ema=tf.train.ExponentialMovingAverage(0.99, step)

#define an operate to update the variable with a list, all the element will be updated.
maintain_averages_op=ema.apply([v1])

with tf.Session() as sess:
   init_op = tf.initialize_all_variables() #initial all the variable
   sess.run(init_op)

   print sess.run([v1,ema.average(v1)])

   sess.run(tf.assign(v1,5))
   sess.run(maintain_averages_op)
   print sess.run([v1,ema.average(v1)])

   sess.run(tf.assign(step,10000)) #update step is 10000
   sess.run(tf.assign(v1,10))
   sess.run(maintain_averages_op)
   print sess.run([v1,ema.average(v1)])

   sess.run(maintain_averages_op)
   print sess.run([v1,ema.average(v1)])
