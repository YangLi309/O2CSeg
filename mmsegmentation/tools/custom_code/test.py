import pickle

data = pickle.load(open('/home/aslab/code/yang_code/semantic_segmentation/mmsegmentation/checkpoints/mapillary_class_san_threshold_logits_0.1.pkl', 'rb'))
print(data)