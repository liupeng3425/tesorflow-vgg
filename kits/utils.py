from kits.DataSet import DataSet


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        result = pickle.load(fo, encoding='bytes')
    return result


def read_data(path=''):
    import os
    train_files = []
    for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
        train_files.append(os.path.join(path, file))
    test_files = os.path.join(path, 'test_batch')
    return DataSet(train_files, test_files)


def get_category_name(index):
    return ['airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck'][index]
