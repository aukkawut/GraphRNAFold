from model import weighted_BCE_loss, F1_score
from utils import * #extract_rna_node_features, possible_pairs, conditional_argmax_pp, pairs_to_dot, map2pairs, pairing_idx_to_dot, map2idx
from tensorflow import keras
import spektral
import numpy as np
from arnie.bpps import bpps


def GraphRNAFold_prediction(rna_seq, GraphRNAFold, one_type=True, threshold=0.35):
    edge_feature = bpps(rna_seq, "vienna_2")
    node_feature = extract_rna_node_features(rna_seq)
    mask = possible_pairs(rna_seq)
    out = GraphRNAFold.predict([node_feature[np.newaxis,...], edge_feature[np.newaxis,...], mask[np.newaxis, :, :, np.newaxis]])
    map_pred = np.squeeze(conditional_argmax_pp(out, threshold=threshold)[0], axis=-1)
    
    if one_type:
        dot_pred = pairs_to_dot(map2pairs(map_pred), len(rna_seq))
    else:
        dot_pred = pairing_idx_to_dot(map2idx(map_pred))
    return dot_pred

model_path = './model/GCNfold-V.h5'
GraphRNAFold = keras.models.load_model(model_path, custom_objects={"GCSConv":spektral.layers.GCSConv, 
                                                              "weighted_BCE_loss":weighted_BCE_loss, "F1_score": F1_score})

# example use case!
rna_seq = 'CACACAAGGCAGAUGGGCUAUAUAAACGUUUUCGCUUUUCCGUUUACGAUAUAUAGUCUACUCUUGUGCAGAAUGAAUUCUCGUAACUACAUAGCACAAGUAGAUGUAGUUAACU'

predict_structure = GraphRNAFold_prediction(rna_seq, GCNfold, threshold=0.35)
print(predict_structure)
