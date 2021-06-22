
class Voc(object):
    def vocalize(self, seq):
        return [self.get_id(c) for c in list(seq)]

    def get_id(self):
        raise NotImplementedError()


class TwoWayVoc(Voc):
    def devocalize(self, seq):
        return [self.get_item(c) for c in list(seq)]

    def get_item(self):
        raise NotImplementedError()


class NullVoc(Voc):
    def __init__(self, size):
        self._size = size

    def get_id(self, entry):
        id = int(entry)
        if id > self._size - 1:
            raise Exception("bad null voc entry: %i" % id)
        return id

    def size(self):
        return self._size


class HashVoc(Voc):
    def __init__(self, size):
        self._size = size

    def get_id(self, entry):
        hash = 5381
        for c in entry:
            hash = (((hash << 5) + hash) + ord(c)) % 2147483647
        return hash % self._size

    def size(self):
        return self._size


class FileVoc(TwoWayVoc):
    def __init__(self, f):
        self._add_unks = False

        with open(f, 'r') as f:
            voc = [l[:-1] for l in f.readlines()]

            # unk, eos, bos = 'UNK', '$', '^'
            # voc += [unk, bos, eos]
            voc = ['_UNK'] + voc

            self.direct = [l.split('\t')[0] for l in voc]
            self.inverted = {id: token for token, id in enumerate(self.direct)}

    def add_unks(self):
        self._add_unks = True

    def get_item(self, id):
        return self.direct[id]

    def get_id(self, entry):
        if entry not in self.inverted:
            if self._add_unks:
                return self.unk()
            raise ValueError(
                "no such value in vocabulary and unks are disabled: %s" % entry)

        return self.inverted[entry]

    def unk(self):
        return 0

    # def bos(self):
    #     return len(self.direct) - 2
    #
    # def eos(self):
    #     return len(self.direct) - 1

    def size(self):
        return len(self.direct)

def frequency_voc(f):
    freq = dict()
    with open(f, 'r') as f:
        for line in f:
            line_split = line.split('\t')
            freq[line_split[0]] = int(line_split[1])
    return freq

def create_voc(name, *args, **kwargs):
    vocs = {
        'null': NullVoc,
        'hash': HashVoc,
        'file': FileVoc,
    }

    if name not in vocs:
        raise NotImplementedError(name)

    return vocs[name](*args, **kwargs)
