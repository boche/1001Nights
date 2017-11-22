import glob
import struct
import tensorflow as tf
import pickle
from tokens import *
from tensorflow.core.example import example_pb2

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

def example_generator(data_path, single_pass):
    """Generates tf.Examples from data files.

    Binary data format: <length><blob>. <length> represents the byte size
    of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
    the tokenized article text and summary.

  Args:
    data_path:
      Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
    single_pass:
      Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.

  Yields:
    Deserialized tf.Example.
  """
    while True:
        filelist = glob.glob(data_path) # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes: break # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                yield example_pb2.Example.FromString(example_str)
        if single_pass:
            print("example_generator completed reading all datafiles. No more data.")
            break

def text_generator(example_generator):
    """Generates article and abstract text from tf.Example.
    Args:
      example_generator: a generator of tf.Examples from file. See data.example_generator"""
    for e in example_generator:
        try:
            article_text = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
            abstract_text = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
        except ValueError:
            tf.logging.error('Failed to get article or abstract from example')
            continue
        if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
            tf.logging.warning('Found an example with empty article text. Skipping it.')
        else:
            yield (article_text, abstract_text)

def split_words(content, indexify):
    words = content.decode("utf-8").replace(SENTENCE_START, "").replace(
            SENTENCE_END, "").split(" ")
    res = []
    for w in words:
        if len(w) > 0:
            if indexify and w in word2idx:
                res.append(word2idx[w])
            else:
                res.append(w)
    return res

def read_file(bin_path, indexify = True):
    input_gen = text_generator(example_generator(bin_path, single_pass = True))
    res = []

    for article, abstract in input_gen:
        res.append(("fake_doc_id", split_words(abstract, indexify),
            split_words(article, indexify)))
    return res

def load_vocab(vocab_size):
    word2idx, idx2word = {}, []
    for word in [SOS, EOS, UNK]:
        idx2word.append(word)
        word2idx[word] = len(word2idx)

    cnt = 0
    with open(data_path + "vocab", 'r') as f:
        for line in f:
            word, freq = line.strip().split(" ")
            cnt += 1
            idx2word.append(word)
            word2idx[word] = len(word2idx)
            if cnt == vocab_size:
                break
    return word2idx, idx2word

if __name__ == "__main__":
    data_path = "/pylon5/ci560ip/bchen5/1001Nights/cnndaily/finished_files/"
    save_path = "/pylon5/ci560ip/bchen5/1001Nights/cnndaily/"
    word2idx, idx2word = load_vocab(50000)
    train_data = read_file(data_path + "train.bin")
    val_data = read_file(data_path + "val.bin")
    train_val_data = train_data + val_data
    test_data = read_file(data_path + "test.bin", indexify = False)

    train_data = {'text_vecs': train_data + val_data, 'word2idx': word2idx,
            'idx2word': idx2word}
    pickle.dump(train_data, open(save_path + "train/train_data.pkl", "wb"))
    pickle.dump(test_data, open(save_path + "test/test_data.pkl", "wb"))
