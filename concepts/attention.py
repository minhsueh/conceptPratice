import numpy as np
import math


"""
Input -> Embedding (positional and semantic) -> KVQ (attention layers) -> output
10 tokens -> 10 * word_embedding_size -> 10 * att_embedding_size -> 10 * word_embedding_size


projection matrix

"""

def positional_encoding(position, d_model):
    """
    # This is column-wise, which is not the normal pratical setting
    i = np.arange(d_model)
    angle_rads = position / (10000 ** (2 * (i // 2) / d_model))
    PE = np.where(i % 2 == 0, np.sin(angle_rads), np.cos(angle_rads))
    return(PE)
    """
    position_array = np.arange(position)[:, np.newaxis]
    dimension_array = np.arange(d_model)[np.newaxis, :]

    angle_rads = position_array / np.power(10000, (2 * (dimension_array // 2) / np.float32(d_model)))

    # even i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # odd i
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return(pos_encoding)


def embedding(input, word_embedding_size):

    """
    Params:
        input(str):
    Outputs:
        numpy.vector
    """
    pass
    input_list = input.split()
    input_list.append("<EOS>")

    # positional


    # semantic




class AttentionLayer:
    def __init__(self, att_embedding_size: int):

        pass

    def forward(self):
        pass


def main():
    # print(positional_encoding(0, 3))
    # print(positional_encoding(1, 3))
    print(positional_encoding(3, 5))

if __name__ == '__main__':
    main()