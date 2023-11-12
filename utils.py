import numpy as np

def extract_rna_node_features(rna_string):    
    # onehot AGCU features
    # base_vocab = 'AUCG'
    seq_dict = {
    'A': np.array([1,0,0,0]),
    'U': np.array([0,1,0,0]),
    'C': np.array([0,0,1,0]),
    'G': np.array([0,0,0,1]),
    'N': np.array([0,0,0,0]),
    'T': np.array([0,1,0,0]),}

    def seq_encoding(seq):
        base_list = list(seq)
        encoding = list(map(lambda x: seq_dict[x] if x in seq_dict else [0,0,0,0], base_list))
        return np.stack(encoding, axis=0)
    
    return seq_encoding(rna_string)

def pairs2map(pairs, map_len):
    contact = np.zeros([map_len, map_len])
    for pair in pairs:
        contact[pair[0], pair[1]] = 1
        contact[pair[1], pair[0]] = 1
    return contact

def map2pairs(map):
    # Input: LxL pairing map (shape [L,L])
    # output: a list of base pair indices (index start from 0 to L-1, e.g. [[0,10],[1,9], [4,7], [9,1]]).
    pairs = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i,j] == 1:
                pairs.append([i,j])
    return pairs

def pairs_to_dot(pairs, seq_len):
    # Input: a list of base pairs for a RNA (e.g. [[0,10],[1,9], [4,7], [9,1]])
    # Output: dot-bracket notations (i.e., "(,.,)")
    dot_seq = ['.']*seq_len
    for pair in pairs:
        if pair[0] < pair[1]:
            dot_seq[pair[0]] = '('
            dot_seq[pair[1]] = ')'
    return ''.join(dot_seq)

def mask_pairing(seq):
    # create a mask for hard-constraints (i) and (ii), refer to the papaer.
    # mask map --> size L*L
    pairing_list = []
    pos_pairs = [("A","U"), ("U","A"), ("C","G"), ("G","C"), ("G","U"), ("U","G")]
    for i in range(len(seq)):
        for j in range(len(seq)):
            if ((seq[i],seq[j]) in pos_pairs) and np.abs(i-j)>3:
                pairing_list.append([i,j])
    return pairing_list


def possible_pairs(seq):
    pairs_dict = {("A","U"):1, ("U","A"):1, ("C","G"):1, ("G","C"):1, ("G","U"):1, ("U","G"):1}
    temp_map = np.zeros([len(seq),len(seq)])
    for i in range(len(seq)):
        for j in range(len(seq)):
            if ((seq[i],seq[j]) in pairs_dict) and np.abs(i-j)>3:
                temp_map[i][j] = pairs_dict[(seq[i],seq[j])]
    return temp_map.astype(int)

def conditional_argmax_pp(y_pred, threshold=0.5): # Input Shape: [1,L,L,1]
    map = np.zeros([1,y_pred.shape[1], y_pred.shape[1],1])
    idx_l = np.argmax(y_pred[0], axis=1)
    for i in range(y_pred.shape[1]):
        if y_pred[0, i, idx_l[i], 0] > threshold:
            #map[0, idx_l[i], i, 0] = y_pred[0, idx_l[i], i, 0]     
            map[0, i, idx_l[i], 0] = 1
    mapT = np.transpose(map, (0,2,1,3)) 
    result = (map+mapT)/2
    result[result < 1] = 0   
    return result # Output Shape: [1,L,L,1]


def map2idx(map):
    # Input shape: [L,L]
    pairing_list = map2pairs(map)
    idx = np.array(range(0,map.shape[1]))
    for pair in pairing_list:
        idx[pair[0]] = pair[1]
        idx[pair[1]] = pair[0]
    return idx

def pairing_idx_to_dot(pairing):
    mapping = ['.', ')', ']', '}', '{', '[', '(']
    n = len(pairing)
    #dot = ['.'] * n
    dot_bracket = np.array([0] * n)
    for i in range(n):
        if dot_bracket[i] != 0:
            continue
        j = pairing[i]
        if i==j:
            dot_bracket[i] = 0

        c = dot_bracket[min(i, j):max(i, j)+1]
        level = (np.sum(c)>0) * np.max(c) + int(i!=j)

        dot_bracket[min(i, j)] = -level
        dot_bracket[max(i, j)] = level

    return ''.join([mapping[c] for c in dot_bracket])

