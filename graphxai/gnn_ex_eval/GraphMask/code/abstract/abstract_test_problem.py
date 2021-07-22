import os
import pickle
import sys
import numpy as np
from tqdm import tqdm


class AbstractTestProblem:
    configuration = None

    def __init__(self, configuration):
        self.configuration = configuration

        if configuration["task"]["cache"] is not None:
            if not os.path.exists(configuration["task"]["cache"]):
                os.makedirs(configuration["task"]["cache"])

            if configuration["task"]["cache"] is not None:
                self.train_data = self.build_cached_dataset("train")
                self.dev_data = self.build_cached_dataset("dev")
                self.test_data = self.build_cached_dataset("test")
            else:
                self.train_data = [self.build_example(id, "train") for id in range(self.train_set_size)]
                self.dev_data = [self.build_example(id, "dev") for id in range(self.dev_set_size)]
                self.test_data = [self.build_example(id, "test") for id in range(self.test_set_size)]

            self.train_indexes = np.arange(len(self.train_data))

    def skip_example(self, d, split):
        pass

    def count_examples(self, split):
        if split == "train":
            return len(self.train_data)
        elif split == "test":
            return len(self.test_data)
        else:
            return len(self.dev_data)

    def build_cached_dataset(self, split):
        n_examples = self.count_raw_examples(split)

        data = []
        desc = "Building the " + split + " set" + (
            "" if self.configuration["task"]["clear_cache"] else " (or loading from cache if possible)")
        for example_id in tqdm(list(range(n_examples)), desc=desc):
            example_cache_location = self.configuration["task"]["cache"] + "/" + split + "." + str(
                example_id) + ".cache"

            broken_example = False

            if not self.configuration["task"]["clear_cache"] \
                    and os.path.exists(example_cache_location) \
                    and os.path.isfile(example_cache_location):
                example = pickle.load(open(example_cache_location, 'rb'))
            else:
                try:
                    example = self.build_example(example_id, split)
                except:
                    print("Warning: I threw an error trying to build this example. Skipping.", file=sys.stderr)
                    broken_example = True
                else:
                    pickle.dump(example, open(example_cache_location, 'wb'))

            skip = broken_example or self.skip_example(example, split)

            if not skip:
                data.append(example)

        return data

    def initialize_epoch(self):
        self.current_example = 0
        self.ready = True

        np.random.shuffle(self.train_indexes)

    def next_example(self, split):
        use_index = self.current_example

        if split == "train":
            example = self.train_data[self.train_indexes[use_index]]
        elif split == "test":
            example = self.test_data[use_index]
        else:
            example = self.dev_data[use_index]

        self.current_example += 1
        return example

    def has_next_example(self, split):
        return self.current_example < self.count_examples(split)

    def approximate_batch_count(self, batch_size, split):
        return int(self.count_examples(split) / batch_size)

    def next_batch(self, batch_size, split):
        if batch_size is None:
            batch_size = self.batch_size
        batch = []
        while len(batch) < batch_size and self.has_next_example(split):
            example = self.next_example(split)
            batch.append(example)

        return batch

    def iterate_batches(self, batch_size, split=None):
        while self.ready:
            batch = self.next_batch(batch_size, split)

            if len(batch) > 0:
                yield batch

            if not self.has_next_example(split):
                self.ready = False
