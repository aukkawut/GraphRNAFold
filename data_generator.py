import numpy as np
from spektral.data import Graph, Dataset, BatchLoader
import collections
import os
import _pickle as cPickle
import tensorflow as tf
from tensorflow import keras

class RNAGraphDataset(Dataset):
    """
    RNA grpah data loader
    """

    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        super().__init__(**kwargs)
    
    def extract_label(self, seq_id, **kwarg):
        raise NotImplementedError
    
    def read(self):        
        graph_list = []
        bpRNA_data = collections.namedtuple('bpRNA_data', 'name node_feat bpps mask dot pairs')
        with open(os.path.join(self.data_path), 'rb') as f:
            data = cPickle.load(f)


        node_feats = np.array([instance[1] for instance in data])
        edge_feats = np.array([np.where(instance[2] > 0.1, instance[2], 0) for instance in data]) # np.array([instance[3] for instance in data]) 
        dot_brackets = np.array([instance[4] for instance in data])

        dot_dict = ['.','(',')','[',']','{','}']
        tokenizer_decoder = keras.preprocessing.text.Tokenizer(char_level=True)
        tokenizer_decoder.fit_on_texts(dot_dict)
        target_data = tokenizer_decoder.texts_to_sequences(dot_brackets)
        #target_data = keras.preprocessing.sequence.pad_sequences(target_data, maxlen=maxlen_seq, padding='post')

        for i in range(len(node_feats)):          
            graph_list.append(Graph(x=node_feats[i].astype('float32'), a=None, e=edge_feats[i][:, :, tf.newaxis].astype('float32'), 
                                    y=keras.utils.to_categorical(target_data[i], num_classes=len(dot_dict)+1).astype('float32')))
                    
        return graph_list
    

class RNAGraphToMapDataset(Dataset):
    """
    RNA grpah data loader: Use for training GCNfold to predict the base pairing probability matrix (BPP) 
    for a given RNA input sequence and prior base-pairing probabilities calculated using McCaskill’s partition function. 
    (The RNA graph data loader is employed in the 2nd stage of the training procedure for GCNfold)
    """

    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        super().__init__(**kwargs)
    
    def extract_label(self, seq_id, **kwarg):
        raise NotImplementedError
    
    def pairs2map(self, pairs, map_len):
        contact = np.zeros([map_len, map_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
            contact[pair[1], pair[0]] = 1
        return contact
 

    def read(self):        
        graph_list = []
        bpRNA_data = collections.namedtuple('bpRNA_data', 'name node_feat bpps mask dot pairs')
        with open(os.path.join(self.data_path), 'rb') as f:
            data = cPickle.load(f)

        node_feats = [instance[1] for instance in data]
        bpps = [np.where(instance[2] > 0.1, instance[2], 0) for instance in data]
        masks = [instance[3] for instance in data]
        pairs = [instance[5] for instance in data]

        score_maps = []
        for i in range(len(pairs)):
            score_maps.append(self.pairs2map(pairs[i], len(node_feats[i])))

        edge_feats = []
        for i in range(len(node_feats)):
            edge_feats.append(np.concatenate((bpps[i][:, :, tf.newaxis], masks[i][:, :, tf.newaxis]), axis=2).astype('float32'))

        for i in range(len(node_feats)):          
            graph_list.append(Graph(x=node_feats[i].astype('float32'), a=None, e=edge_feats[i], y=score_maps[i].astype('float32')))
                    
        return graph_list
    

def get_dimensions(array, level=0):
    yield level, len(array)
    try:
        for row in array:
            yield from get_dimensions(row, level + 1)
    except TypeError: #not an iterable
        pass

def get_max_shape(array):
    dimensions = collections.defaultdict(int)
    for level, length in get_dimensions(array):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]

def iterate_nested_array(array, index=()):
    try:
        for idx, row in enumerate(array):
            yield from iterate_nested_array(row, (*index, idx))
    except TypeError: # final level
        for idx, item in enumerate(array):
            yield (*index, idx), item

def pad(array, fill_value):
    dimensions = get_max_shape(array)
    result = np.full(dimensions, fill_value)
    for index, value in iterate_nested_array(array):
        result[index] = value
    return result

def state1generator(data_set):
  while True:
    for x, y in data_set:
      yield ([x[0], tf.squeeze(x[1])] , pad(y, fill_value=0))

def state2generator(data_set):
  while True:
    for x, y in data_set:
      yield ([x[0], x[1][...,0], x[1][...,1][:, :, :, tf.newaxis]], 
             pad(y, fill_value=0)[:, :, :, tf.newaxis])
      

# data_path = 'your_path'

# bpRNA_data = collections.namedtuple('bpRNA_data', 'name node_feat bpps mask dot pairs')
# train_path = data_path + 'TR0_graph.pickle'
# val_path = data_path + 'VL0_graph.pickle'

# train_data = RNAGraphToMapDataset(train_path)
# val_data = RNAGraphToMapDataset(val_path)

# bs = 8 #16
# train_loader = BatchLoader(train_data, batch_size=bs, epochs=None, shuffle=True)
# val_loader = BatchLoader(val_data, batch_size=bs, epochs=None, shuffle=False)