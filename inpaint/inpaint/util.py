def crop_to_square(x):
    crop_width, remainder = divmod(abs(x.shape[0] - x.shape[1]), 2)
    if x.shape[0] < x.shape[1]:
        return x[:, crop_width : x.shape[1] - crop_width - remainder]
    else:
        return x[crop_width : x.shape[0] - crop_width - remainder]
