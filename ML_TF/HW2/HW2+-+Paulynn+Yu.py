
# coding: utf-8

# In[2]:

# import packages
import tensorflow as tf
import numpy as np


# In[3]:

# 1: Create placeholder for an input array with dtype float32 and shape None
a = tf.placeholder(tf.float32, shape=[None],name = "input_a")


# In[4]:

# 2: Scopes for the input, middle section and final node f

with tf.name_scope("input_placeholder"):
    a = a
    
with tf.name_scope("middle_section"):
    b = tf.reduce_prod(a, name = "prod_b")
    c = tf.reduce_mean(a, name = "mean_c")
    d = tf.reduce_sum(a, name = "sum_d")
    e = tf.add(b,c, name = "add_e")
    
with tf.name_scope("final_node"):
    f = tf.multiply(e,d, name = "output")


# In[5]:

# 3: Feed placeholder with an array of A consisting of 100 normally distributed
# random  numbers with m ean = 1 and std = 2

# open a tensorflow session
sess = tf.Session()

# create a dict to pass into feed_dict
np.random.seed(3)
random_list = np.random.normal(1.0,2.0,100)
input_dict = {a: random_list}

# fetch the value of 'a', feeding values of the 'input vector'
sess.run(f, feed_dict=input_dict)


# In[ ]:

# 4: Save your graph and show it on TensorBoard
writer = tf.summary.FileWriter('./HW2',graph = tf.get_default_graph())
writer.close()


# In[15]:

# 5: plot your input array on a separate figure

import matplotlib.pyplot as plt

plt.plot(random_list, linestyle="",marker="o")
plt.xlabel('position in list')
plt.ylabel('values')
plt.title('Scatter plot of random list')
plt.show()

plt.hist(random_list,facecolor='green', bins=25)
plt.xlabel('position in list')
plt.ylabel('values')
plt.title('Histogram of random list')
plt.show()


# In[ ]:



