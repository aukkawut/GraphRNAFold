from model import gcn_mapping, weighted_BCE_loss, F1_score
from data_generator import state2generator, RNAGraphToMapDataset 
from spektral.data import BatchLoader
import tensorflow as tf
from tensorflow import keras
import spektral
import collections

data_path = './RNA_graph_data/'

bpRNA_data = collections.namedtuple('bpRNA_data', 'name node_feat contra_map pos_pair dot pairs')
train_path = data_path + 'TR0_graph_map.pickle'
val_path = data_path + 'VL0_graph_map.pickle'


batch_size = 16
train_data = RNAGraphToMapDataset(train_path)
val_data = RNAGraphToMapDataset(val_path)
train_loader = BatchLoader(train_data, batch_size=batch_size, epochs=None, shuffle=True)
val_loader = BatchLoader(val_data, batch_size=batch_size, epochs=None, shuffle=False)


epochs = 60
learning_rate = 0.001
stage1_model_path = '/model/stage1_model.h5'  # pretrained model from stage 1

save_path = './model/'
model_name = "GraphRNAFold-V.h5"
model = keras.models.load_model(stage1_model_path, custom_objects={"GCSConv":spektral.layers.GCSConv})
embed_model = keras.Model(model.input, model.layers[10].output)
gcnfold = gcn_mapping(embed_model, conv1D_filters=64, freeze_embed=True)

gcnfold.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=weighted_BCE_loss, metrics=['binary_crossentropy', F1_score()])
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.0004)
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=save_path+model_name, monitor='val_loss', 
                                                mode='min', verbose=1, save_best_only=True, save_weights_only=True), reduce_lr]
history = gcnfold.fit(x = state2generator(train_loader.load()), validation_data = state2generator(val_loader.load()), epochs=epochs, 
                   steps_per_epoch = train_loader.steps_per_epoch, validation_steps = val_loader.steps_per_epoch, verbose=1, callbacks=callbacks)


