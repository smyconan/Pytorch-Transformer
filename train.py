import codecs
import regex
from collections import Counter
import numpy as np

from torch.autograd import Variable
import torch

from transformer import Transformer

PRE_TRAIN_SAVED = None

source_train = 'corpora/train.tags.de-en.de'
target_train = 'corpora/train.tags.de-en.en'

source_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
target_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'

EMBEDDING_DIM = 512
NUM_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1
FFN_DIM = 2048
HEAD_DIM = EMBEDDING_DIM / NUM_HEADS

BATCH_SIZE = 32
LR = 0.0001
NUM_EPOCHS = 20
EPS = 1e-8

MAX_SENT_LEN = 30
MIN_CUT = 20

SMOOTH_RATE = 0.1

def make_vocabulary(sents_file):
    text = codecs.open(sents_file, 'r', 'utf-8').read()
    text = regex.sub("[^\s\p{Latin}']", "", text)
    words = text.split()
    word2cnt = Counter(words)
    word2cnt = word2cnt.most_common(len(word2cnt))
    words = ["<PAD>", "<UNK>", "<S>", "</S>"] + [word for word, cnt in word2cnt if cnt >= MIN_CUT]
    word2idx = {words[i]:i for i in range(len(words))}
    idx2word = {i:words[i] for i in range(len(words))}
    return word2idx, idx2word

def indexlize(sents, word2idx):
    sents = [(sent + u" </S>").split() for sent in sents]
    sent_lengths = [len(sent) for sent in sents]
    max_len = min(max(sent_lengths), MAX_SENT_LEN)
    sents = [sent[:max_len] + max(0,max_len - len(sent))*["<PAD>"] for sent in sents]
    sents = [[word2idx.get(word, word2idx["<UNK>"]) for word in sent] for sent in sents]
    return sents, [min(length, max_len) for length in sent_lengths]

def load_training_data():
    # load sentences and vocabulary
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    de2idx, idx2de = make_vocabulary(source_train)
    en2idx, idx2en = make_vocabulary(target_train)
    # cut and indexlize sentences
    de_sents, de_sent_lens = indexlize(de_sents, de2idx)
    en_sents, en_sent_lens = indexlize(en_sents, en2idx)
    X, Y = np.matrix(de_sents, np.int32), np.matrix(en_sents, np.int32)
    return X, Y, de_sent_lens, en_sent_lens, de2idx, en2idx

def load_testing_data(de2idx, en2idx):
    # load sentences
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    de_sents = [_refine(line) for line in codecs.open(source_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(target_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    # cut and indexlize sentences
    de_sents, de_sent_lens = indexlize(de_sents, de2idx)
    en_sents, en_sent_lens = indexlize(en_sents, en2idx)
    Xt, Yt = np.matrix(de_sents, np.int32), np.matrix(en_sents, np.int32)
    return Xt, Yt, de_sent_lens, en_sent_lens

X, Y, lens_x, lens_y, vocab_x, vocab_y = load_training_data()
Xt, Yt, lens_xt, lens_yt = load_testing_data(vocab_x, vocab_y)
print(X.shape, Y.shape, len(lens_x), len(lens_y), len(vocab_x), len(vocab_y))
print(Xt.shape, Yt.shape, len(lens_xt), len(lens_yt))

NUM_BATCHS = len(X) // BATCH_SIZE

if PRE_TRAIN_SAVED != None:
    print("Use Pre-trained Parameters ...")
    model = torch.load(PRE_TRAIN_SAVED)
    START_EPOCH = int(PRE_TRAIN_SAVED.split(".")[1])
    START_BATCH = int(PRE_TRAIN_SAVED.split(".")[3]) + 1
else:
    print("Train from the very beginning ...")
    model = Transformer(src_vocab_size = len(vocab_x),
                        src_max_len = X.shape[1],
                        tgt_vocab_size = len(vocab_y),
                        tgt_max_len = Y.shape[1],
                        num_layers = NUM_LAYERS,
                        embedding_dim = EMBEDDING_DIM,
                        num_heads = NUM_HEADS,
                        FFN_dim = FFN_DIM,
                        dropout = DROPOUT)
    START_EPOCH = 0
    START_BATCH = 0

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=[0.9, 0.98], eps=EPS)

Xt = Variable(torch.LongTensor(Xt))
Yt = Variable(torch.LongTensor(Yt))
lenxt = Variable(torch.LongTensor(lens_xt))
lenyt = Variable(torch.LongTensor(lens_yt))
test_is_target = (1. - Yt.eq(vocab_x["<PAD>"]).float()).view(-1)

for epoch in range(START_EPOCH, NUM_EPOCHS):
    start_batch = START_BATCH if epoch == START_EPOCH else 0
    for batch in range(start_batch, NUM_BATCHS):
        # construct batch
        x_batch = Variable(torch.LongTensor(X[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE,:]))
        y_batch = Variable(torch.LongTensor(Y[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE,:]))
        lenx_batch = Variable(torch.LongTensor(lens_x[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]))
        leny_batch = Variable(torch.LongTensor(lens_y[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]))
        # build one-hot y and smooth it
        is_target = (1. - y_batch.eq(vocab_x["<PAD>"]).float()).view(-1)
        y_onehot = torch.zeros(BATCH_SIZE * y_batch.size(1), len(vocab_y))
        y_onehot = Variable(y_onehot.scatter_(1, y_batch.view(-1, 1).data, 1))
        y_smoothed = ((1 - SMOOTH_RATE) * y_onehot) + (SMOOTH_RATE / y_onehot.size()[-1])
        # train
        optimizer.zero_grad()
        softmax_pred = model(x_batch, lenx_batch, y_batch, leny_batch)
        _, predict = torch.max(softmax_pred, -1)
        acc = torch.sum(predict.eq(y_batch).float().view(-1) * is_target) / torch.sum(is_target)
        sum_loss = - torch.sum(y_smoothed * torch.log(softmax_pred.view(BATCH_SIZE * y_batch.size(1),-1)), dim=-1)
        loss = torch.sum(sum_loss * is_target) / torch.sum(is_target)
        loss.backward()
        optimizer.step()
        # display
        print("Epoch:", epoch, '\t', "Batch:", batch, '\t', "ACC:", acc.item(), '\t', "LOSS:", loss.item())
        if batch % 100 == 0:
            # test
            test_softmax = model(Xt, lenxt, Yt, lenyt)
            _, test_predict = torch.max(test_softmax, -1)
            acc = torch.sum(test_predict.eq(Yt).float().view(-1) * test_is_target) / torch.sum(test_is_target)
            print("TEST ACC:", acc.item())
        if batch % 1000 == 0:
            torch.save(model, "epoch."+str(epoch)+".batch."+str(batch)+".pkl")

