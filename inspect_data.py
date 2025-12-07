import numpy as np

try:
    data = np.load('data/POP909_melody.npy', allow_pickle=True)
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    if len(data) > 0:
        first_elem = data[0]
        print(f"First element type: {type(first_elem)}")
        if hasattr(first_elem, 'shape'):
            print(f"First element shape: {first_elem.shape}")
        else:
            print(f"First element: {first_elem}")
except Exception as e:
    print(f"Error: {e}")
