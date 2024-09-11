import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend  as K
from tensorflow.keras import layers
import spektral

# STAGE 1's component of GraphRNAFold (Sequential). --> predict dot-bracket notations and use them as embedding for STAGE 2
def create_GNN(n_labels=8, hidden_dim=128, n_layers=2, activation="gelu"):
    pre_dense = layers.Conv1D(hidden_dim, 3, padding='same')
    gnn_layers = [spektral.layers.GCSConv(hidden_dim, activation=activation) for _ in range(n_layers)]
    d0 = layers.Dropout(0.2)
    dense = layers.TimeDistributed(layers.Dense(n_labels, activation="softmax"))

    node_inputs = keras.Input(shape=(None, 4), name='Nodes')
    edge_inputs = keras.Input(shape=(None, None), name='Edges')

    node_embed = pre_dense(node_inputs)
    for i in range(n_layers):
        node_embed = gnn_layers[i]([node_embed, edge_inputs])
        node_embed = layers.Bidirectional(layers.LSTM(hidden_dim//2, dropout=0.25, return_sequences=True))(node_embed)
    y = dense(d0(node_embed))
    model = keras.models.Model([node_inputs, edge_inputs], y, name="GNN_1")
    return model

def acc(y_true, y_pred):
    y = tf.argmax(y_true, axis=-1)
    y_ = tf.argmax(y_pred, axis=-1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

# STAGE 2's component of GraphRNAFold (Mapping). --> predict base pairing probability matrix. 
def gcn_mapping(embed_model, conv1D_filters=64, conv2D_filters=64, activation='swish', 
                act_after_dot=True, freeze_embed=True,):
    '''
    embed_model: the pretrained model that has been trained on the first stage.
    (i.e., to predict a simple dot-bracket structure sequence of an input RNA) 
    '''

    mask = tf.keras.Input(shape=(None, None, 1), name='Mask Input') # embed_model.input[1][...,1]
    embed_model.input.append(mask)
    x = embed_model.output #(b,None,512) --> #  (b, None, 8)
    x = layers.Conv1D(conv1D_filters, kernel_size=3, padding='same', name='con1d_1')(x)
    x = layers.Dropout(0.3, name='dropout_in')(layers.Activation(activation)(x)) 
    xT = layers.Permute((2, 1))(x)
    xe = layers.Dot(axes=(2,1))([x, xT]) # x*x^T
    xe = tf.expand_dims(xe ,axis=-1) #add axis to make it (None, None, None, 1)
    if act_after_dot:
      xe = layers.Activation(activation)(xe)

    # construct 2d map
    # Conv2d
    x = tf.keras.layers.Conv2D(conv2D_filters, kernel_size=3, padding='same')(xe)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_conv2d_1')(x)
    x = tf.keras.layers.Conv2D(conv2D_filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_conv2d_2')(x)
    x = layers.Add()([x, xe]) # WLOG to the previous version
    x = tf.keras.layers.Activation(activation)(x) 

    y = tf.keras.layers.Conv2D(1, kernel_size=1, padding='same')(x)
    y = tf.keras.layers.BatchNormalization()(y)  
    y = tf.keras.layers.Multiply()([y, mask]) # element-wise with mask
    y = tf.keras.layers.Activation('sigmoid')(y)
    yT = tf.keras.layers.Permute((2, 1, 3))(y)
    y = (y + yT)/2
    model = tf.keras.Model(embed_model.input, y)

    if freeze_embed:
        for layer in model.layers[0:9]: 
            layer.trainable = False

    return model

def weighted_BCE_loss(y_true, y_pred, positive_weight=300):  
    # y_true: (None,None,None,None)     y_pred: (None,512,512,1)
    y_pred = K.clip(y_pred, min_value=1e-12, max_value=1 - 1e-12)
    weights = K.ones_like(y_pred)  # (None,512,512,1)
    weights = tf.where(y_pred < 0.5, positive_weight * weights, weights)
    out = keras.losses.binary_crossentropy(y_true, y_pred)  # (None,512,512)
    out = out = K.expand_dims(out, axis=-1) * weights  # (None,512,512,1)* (None,512,512,1)
    return K.mean(out)

# from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def create_f1():
    def f1_function(y_true, y_pred):
        y_pred_binary = tf.where(y_pred>0.5, 1., 0.)
        y_true = tf.dtypes.cast(y_true, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred_binary)
        predicted_positives = tf.reduce_sum(y_pred_binary)
        possible_positives = tf.reduce_sum(y_true)
        return tp, predicted_positives, possible_positives
    return f1_function

class F1_score(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.f1_function = create_f1()
        self.tp_count = self.add_weight("tp_count", initializer="zeros")
        self.all_predicted_positives = self.add_weight('all_predicted_positives', initializer='zeros')
        self.all_possible_positives = self.add_weight('all_possible_positives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, predicted_positives, possible_positives = self.f1_function(y_true, y_pred)
        self.tp_count.assign_add(tp)
        self.all_predicted_positives.assign_add(predicted_positives)
        self.all_possible_positives.assign_add(possible_positives)

    def result(self):
        ## Just add epsilon for 0 division
        precision = self.tp_count / (self.all_predicted_positives + K.epsilon())
        recall = self.tp_count / (self.all_possible_positives + K.epsilon())
        f1 = 2*(precision*recall)/(precision+recall + K.epsilon())        
        return f1
    
