import numpy as np



def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    print(param_size + buffer_size)

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(size_all_mb)


def num_scale(x, div=None):
    x = np.array(x)
    assert div != 0
    if div is not None:
        x = x / div

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def zero_log(x):
        out = x.copy().astype(float)
        zero_mask = out == 0
        out[zero_mask] = 0
        out[~zero_mask] = np.log1p(out[~zero_mask])
        return out

    return (sigmoid(zero_log(x)) - 0.5) * 2
