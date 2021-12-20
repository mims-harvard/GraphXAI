import numpy as np

class GloveEmbedder:

    extra_tokens = ["-lrb-", "-rrb-", "-start-", "-end-", "-unk-"]
    add_start_and_end = None
    embedding_dict = None
    index_dict = None

    def __init__(self, glove_embedding_path, add_start_and_end=False):
        self.glove_embedding_path = glove_embedding_path
        self.add_start_and_end = add_start_and_end

    def load_embeddings(self, path):
        embeddings_dict = {}
        index_dict = {}
        with open(path, 'r') as f:
            for line in f:
                values = line.split(" ")
                word = values[0]
                d = len(values[1:])
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

                if word not in index_dict:
                    index_dict[word] = len(index_dict)

        for token in self.extra_tokens:
            embeddings_dict[token] = np.random.uniform(low=-0.1, high=0.1, size=d)

            if token not in index_dict:
                index_dict[word] = len(index_dict)

        return embeddings_dict, index_dict

    def batch_to_embeddings(self, batch):
        if self.embedding_dict is None:
            self.embedding_dict, self.index_dict = self.load_embeddings(self.glove_embedding_path)

        embs = [None for _ in batch]

        for i, example in enumerate(batch):
            if self.add_start_and_end:
                text_example = ["-start-"] + example + ["-end-"]
            else:
                text_example = example
            emb = [self.embedding_dict[w] if w in self.embedding_dict else self.embedding_dict["-unk-"] for w in text_example]

            embs[i] = np.array(emb).astype(np.float32)

        longest = max([len(emb) for emb in embs])
        out_embs = np.zeros((len(batch), longest, embs[0].shape[-1]), dtype=np.float32)
        for i in range(len(embs)):
            out_embs[i, :len(embs[i])] = embs[i]

        return out_embs

    def batch_to_indexes_and_embeddings(self, batch):
        if self.embedding_dict is None:
            self.embedding_dict, self.index_dict = self.load_embeddings(self.glove_embedding_path)

        embs = [None for _ in batch]
        idxs = [None for _ in batch]

        for i, example in enumerate(batch):
            if self.add_start_and_end:
                text_example = ["-start-"] + example + ["-end-"]
            else:
                text_example = example
            emb = [self.embedding_dict[w] if w in self.embedding_dict else self.embedding_dict["-unk-"] for w in text_example]
            idx = [self.index_dict[w] if w in self.index_dict else self.index_dict["-unk-"] for w in text_example]

            embs[i] = np.array(emb).astype(np.float32)
            idxs[i] = np.array(idx).astype(np.int32)

        longest = max([len(emb) for emb in embs])
        out_embs = np.zeros((len(batch), longest, embs[0].shape[-1]), dtype=np.float32)
        out_indexes = np.ones((len(batch), longest), dtype=np.float32) * -1

        for i in range(len(embs)):
            out_embs[i, :len(embs[i])] = embs[i]
            out_indexes[i, :len(embs[i])] = idxs[i]

        return out_indexes, out_embs