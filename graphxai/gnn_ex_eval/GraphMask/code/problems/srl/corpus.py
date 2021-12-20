import itertools
import functools
import random


def simple_reader(data, batch_size):
    args = [iter(enumerate(data))] * batch_size
    for batch in itertools.zip_longest(*args, fillvalue=None):
        yield [s for s in batch if s is not None]


class Corpus(object):
    def __init__(self, parser, batch_size, path, reader=simple_reader):
        self.parser = parser
        self.path = path
        self.batch_size = batch_size
        self._size = None
        self.reader = reader

    def batches(self):
        with open(self.path, 'r') as data:
            for batch in self.reader(data, self.batch_size):
                batch = [(id, self.parser(record[:-1])) for (id, record) in
                         batch]
                yield batch

    def examples(self):
        with open(self.path, 'r') as data:
            for batch in self.reader(data, 1):
                batch = [(id, self.parser(record[:-1])) for (id, record) in
                         batch]
                yield batch[0]

    def get_batch_size(self):
        return self.batch_size

    def size(self):
        if self._size is None:
            for i, _ in enumerate(open(self.path, 'r')):
                pass
            self._size = i + 1
        return self._size
