from model import create_GNN, acc
from data_generator import state1generator, RNAGraphDataset 
from spektral.data import BatchLoader
import tensorflow as tf
import collections

data_path = './RNA_graph_data/'

bpRNA_data = collections.namedtuple('bpRNA_data', 'name node_feat bpps mask dot pairs')
train_path = data_path + 'TR0_graph.pickle'
val_path = data_path + 'VL0_graph.pickle'

train_data = RNAGraphDataset(train_path)
val_data = RNAGraphDataset(val_path)


batch_size = 128
epochs = 80
learning_rate = 0.01
train_loader = BatchLoader(train_data, batch_size=batch_size, epochs=None, shuffle=True)
val_loader = BatchLoader(val_data, batch_size=batch_size, epochs=None, shuffle=False)

save_path = './model/'
model_name = "stage1_model.h5"
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=0.001)
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=save_path+model_name, monitor='val_loss', 
                                                mode='min', verbose=1, save_best_only=True, save_weights_only=True), reduce_lr]

gnn1 = create_GNN(n_labels=8, hidden_dim=128, n_layers=4, activation="gelu")
gnn1.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=[acc])
history = gnn1.fit(x = state1generator(train_loader.load()), validation_data = state1generator(val_loader.load()), 
                   epochs=epochs, steps_per_epoch = train_loader.steps_per_epoch, validation_steps = val_loader.steps_per_epoch,
                   verbose=1, callbacks=callbacks)
