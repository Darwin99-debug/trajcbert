import pickle
with open('/home/daril_kw/data/input_ids_small.pkl', 'rb') as f:
    input_ids = pickle.load(f)
with open('/home/daril_kw/data/attention_masks_small.pkl', 'rb') as f:
    attention_masks = pickle.load(f)
with open('/home/daril_kw/data/targets_matching_small.pkl', 'rb') as f:
    targets_matching = pickle.load(f)

with open('/home/daril_kw/data/targets_input_small.pkl', 'rb') as f:
    targets_inputs = pickle.load(f)




