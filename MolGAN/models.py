from layers import multi_graph_convolution_layer as mgcl
from layers import multi_dense_layer as mdl

import tensorflow as tf
from tensorflow.keras import layers

class Generator(keras.Model):
    def __init__(self, units=(128,256,512), vertices=8, edges=5, nodes=5,activation='tanh', dropout_rate=0., name='', **kwargs):
        super(Generator, self).__init__(name=name, **kwargs)
        
        self.mdl = MultiDenseLayer(units=units, activation=activation, dropout_rate=dropout_rate)
        
        self.edges_dense = layers.Dense(units=edges*vertices*vertices, activation=None)
        self.edges_reshape = layers.Reshape((edges,vertices, vertices))
        self.edges_transpose = layers.Permute((2,3,1))
        self.edges_dropout = layers.Dropout(dropout_rate)
        
        self.nodes_dense = layers.Dense(units=vertices*nodes, activation=None)
        self.nodes_reshape = layers.Reshape((vertices,nodes))
        self.nodes_dropout = layers.Dropout(dropout_rate)
                   
    def call(self, inputs, training=False):
        output  = self.mdl(inputs)
        #edges logits
        edges_logits = self.edges_dense(output)
        edges_logits = self.edges_reshape(edges_logits)
        edges_logits = (edges_logits + tf.linalg.matrix_transpose(edges_logits))/2
        edges_logits = self.edges_transpose(edges_logits)
        edges_logits = self.edges_dropout(edges_logits)
        
        #nodes logits
        nodes_logits = self.nodes_dense(output)
        nodes_logits = self.nodes_reshape(nodes_logits)
        nodes_logits = self.nodes_dropout(nodes_logits)
        
        return edges_logits,nodes_logits
    
class Discriminator(keras.Model):
    def __init__(self, units=[(128,64),128,(128,64)], activation='tanh', dropout_rate=.0, edges=5, name='', batch_discriminator=True, **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(units=units[:-1], activation=activation, dropout_rate=dropout_rate, edges= edges)
        self.mdl = MultiDenseLayer(units=units[-1], activation=activation, dropout_rate = dropout_rate)
        self.batch_discriminator = batch_discriminator
        self.dense =  layers.Dense(units=1)
        
        if batch_discriminator:
            self.d1 =  layers.Dense(units = units[-2]//8, activation=activation)
            self.d2 =  layers.Dense(units= units[-2]//8, activation=activation)

    def call(self, inputs, training=False):
        outputs0 = self.encoder(inputs)
        outputs1 = self.mdl(outputs0)
               
        if self.batch_discriminator:
            outputs_batch = self.d1(outputs0)
            outputs_batch = tf.reduce_mean(outputs_batch, 0 , keepdims=True)
            outputs_batch = self.d2(outputs_batch)
            outputs_batch = tf.tile(outputs_batch, (tf.shape(outputs0)[0],1))
            outputs1 = tf.concat((outputs1, outputs_batch), -1)
            
        outputs = self.dense(outputs1)
        
        return outputs, outputs1

class Estimator(keras.Model):
    def __init__(self, units, activation='tanh', dropout_rate = 0., edges = 5, name='', **kwargs):
        super(Estimator,self).__init__(name=name, **kwargs)
        self.encoder = Encoder(units=units[:-1], activation=activation, dropout_rate=dropout_rate, edges= edges)
        self.mdl = MultiDenseLayer(units=units[-1], activation=activation, dropout_rate = dropout_rate)
        self.dense = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        outputs = self.encoder(inputs)
        outputs = self.mdl(outputs)
        outputs = self.dense(outputs)
        
        return outputs
        
