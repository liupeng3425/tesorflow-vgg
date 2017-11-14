from kits import utils
import numpy


class DataSet(object):
    def __init__(self, data_files, test_files):
        self.__batch_index_data = 0
        self.__epochs_completed = 0
        assert len(data_files) > 0
        first_file = utils.unpickle(data_files[0])
        self.data_set = {'data': first_file[b'data'], 'labels': first_file[b'labels']}
        for file in data_files[1:]:
            data = utils.unpickle(file)
            self.data_set['data'] = numpy.append(self.data_set['data'], data[b'data'], axis=0)
            self.data_set['labels'] = numpy.append(self.data_set['labels'], data[b'labels'], axis=0)

        test_file = utils.unpickle(test_files)
        self.test_set = {'data': test_file[b'data'], 'labels': test_file[b'labels']}
        self.data_number = self.data_set['data'].shape[0]
        self.test_number = self.test_set['data'].shape[0]

        self.data_set['data'] = numpy.reshape(self.data_set['data'], [-1, 3, 32, 32])
        self.data_set['data'] = numpy.transpose(self.data_set['data'], [0, 2, 3, 1])
        self.test_set['data'] = numpy.reshape(self.test_set['data'], [-1, 3, 32, 32])
        self.test_set['data'] = numpy.transpose(self.test_set['data'], [0, 2, 3, 1])
        self.data_set['labels'] = numpy.reshape(self.data_set['labels'], [-1, 1])
        self.data_set['labels_one_hot'] = numpy.eye(10)[self.data_set['labels']]
        self.test_set['labels'] = numpy.reshape(self.test_set['labels'], [-1, 1])
        self.test_set['labels_one_hot'] = numpy.eye(10)[self.test_set['labels']]

    def next_batch_data(self, batch_size):
        # shuffle for the first time
        if self.__epochs_completed == 0:
            perm = numpy.arange(self.data_number)
            numpy.random.shuffle(perm)
            self.data_set['data'] = self.data_set['data'][perm]
            self.data_set['labels'] = self.data_set['labels'][perm]

        if batch_size + self.__batch_index_data > self.data_number:
            result = {'data': self.data_set['data'][self.__batch_index_data:self.data_number],
                      'labels': self.data_set['labels'][self.__batch_index_data:self.data_number]}

            self.__batch_index_data = 0

            # shuffle on next epoch
            perm = numpy.arange(self.data_number)
            numpy.random.shuffle(perm)
            self.data_set['data'] = self.data_set['data'][perm]
            self.data_set['labels'] = self.data_set['labels'][perm]

            left_number = batch_size - result['data'].shape[0]
            result['data'] = numpy.append(result['data'],
                                          self.data_set['data'][0:left_number],
                                          axis=0)
            result['labels'] = numpy.append(result['labels'],
                                            self.data_set['labels'][0:left_number],
                                            axis=0)
            self.__batch_index_data = left_number
        else:
            result = {'data': self.data_set['data'][self.__batch_index_data:self.__batch_index_data + batch_size],
                      'labels': self.data_set['labels'][self.__batch_index_data:self.__batch_index_data + batch_size]}
            self.__batch_index_data += batch_size

        self.__epochs_completed += 1
        return result
