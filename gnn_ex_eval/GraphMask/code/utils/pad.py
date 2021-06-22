import numpy as np

def pad_to_max(array, padding_variable):
    lengths = np.array([x.shape[0] for x in array]).astype(np.int32)
    max_length = max(lengths)

    result_shape = [len(array), max_length] + list(array[0].shape)[1:]

    output_tensor = np.full(result_shape, padding_variable)
    for i in range(len(lengths)):
        output_tensor[i][:lengths[i]] = array[i]

    return output_tensor, lengths