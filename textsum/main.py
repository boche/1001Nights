import sys
import argparse
import random
import string
import numpy as np
import time
import pickle
from tokens import *
from util import *
import torch
from model.Seq2Seq import Seq2Seq
from rouge import Rouge

def train(data):
    nbatch = len(data)
    nval = nbatch // 50
    identifier = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    random.seed(15213)
    random.shuffle(data)
    train_data, val_data = data[:-nval], data[-nval:]

    s2s = Seq2Seq(args)
    if args.use_cuda:
        s2s = s2s.cuda()
    s2s_opt = torch.optim.Adam(s2s.parameters(), lr = args.learning_rate,
            weight_decay = args.l2)
    print("Identifier:", identifier)
    print(s2s)

    for ep in range(args.nepochs):
        ts = time.time()
        random.shuffle(train_data)
        epoch_loss, epoch_p_gen, sum_len = 0, 0, 0
        s2s.train(True)

        for batch_idx, (inputs, targets, input_lens, target_lens) in enumerate(train_data[:5000]):
            # loc_word2idx, loc_idx2word: local oov indexing for a batch
            inputs, targets, loc_word2idx, loc_idx2word = index_oov(inputs,
                    targets, word2idx, args)
            oov_size = len(loc_word2idx)
            loss, p_gen = s2s(inputs, input_lens, targets, target_lens,
                    oov_size, is_volatile = False)
            sum_len += sum(target_lens) - len(target_lens)
            epoch_loss += loss.data[0]
            if args.use_copy:
                epoch_p_gen += mask_generation_prob(p_gen, target_lens)

            s2s_opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(s2s.parameters(), args.max_norm)
            s2s_opt.step()

            if batch_idx % 50 == 0:
                print("batch %d, time %.1f sec, loss %.2f" % (batch_idx,
                    time.time() - ts, loss.data[0]))
                s2s.train(False)
                summarize(s2s, inputs, input_lens, targets, target_lens,
                        loc_idx2word, oov_size, beam_search = False,
                        show_copy = True)
                sys.stdout.flush()
        train_loss, train_p_gen = epoch_loss / sum_len, epoch_p_gen / sum_len

        s2s.train(False)
        s2s.requires_grad = False
        epoch_loss, epoch_p_gen, sum_len = 0, 0, 0
        sum_len_fixed = 0

        for inputs, targets, input_lens, target_lens in val_data:
            inputs, targets, loc_word2idx, loc_idx2word = index_oov(inputs,
                    targets, word2idx, args)
            loss, p_gen = s2s(inputs, input_lens, targets, target_lens,
                    len(loc_word2idx), is_volatile = True)
            sum_len += sum(target_lens) - len(target_lens)
            epoch_loss += loss.data[0]
            if args.use_copy:
                epoch_p_gen += mask_generation_prob(p_gen, target_lens)

        val_loss, val_p_gen = epoch_loss / sum_len, epoch_p_gen / sum_len
        print("Epoch %d, train loss: %.2f, val loss: %.2f, train p_gen: %.2f, val p_gen: %.2f, #batch: %d, time %.2f sec"
                % (ep + 1, train_loss, val_loss, train_p_gen, val_p_gen,
                    batch_idx, time.time() - ts))
        model_fname = args.user_dir + args.model_fpat % (identifier, ep + 1)
        torch.save(s2s.state_dict(), model_fname)

def summarize(s2s, inputs, input_lens, targets, target_lens, loc_idx2word,
        oov_size, beam_search, show_copy):
    list_symbols, attns, p_gens = s2s.summarize(inputs, input_lens, oov_size,
            beam_search, is_volatile = True)
    symbols = torch.stack(list_symbols).transpose(0, 1) # b x s
    decode_approach = 'Beam' if beam_search else 'Greedy'

    use_visualization = beam_search is False and args.use_visualization and (
            args.attn_model != 'none')
    if use_visualization:   # only plot if it's not beam search
        attns = torch.stack(attns).transpose(0, 1)[0,:,:] # 1 x target_s x input_s
        p_gens = torch.stack(p_gens)[:,0,:] if args.use_copy else [] # max_seq_len x 1 x 1

    # In train, we only show the first instance. In test, batch is of size 1.
    srcText = idxes2sent(inputs[0].cpu(), idx2word, loc_idx2word, keepSrc=True)
    truth = idxes2sent(targets[0].cpu(), idx2word, loc_idx2word, keepSrc=True)
    prediction = idxes2sent(symbols[0].cpu().data, idx2word, loc_idx2word,
            keepSrc = not show_copy)

    if use_visualization:
        visualize(srcText, prediction, truth, attns, p_gens, args)

    print("<Source Text>: %s" % srcText)
    print("<Ground Truth>: %s" % truth)
    print("<%s>: %s\n%s" % (decode_approach, prediction, 80 * '-'))
    return prediction, truth

def test(model_path, testset):
    s2s = Seq2Seq(args)
    s2s.load_state_dict(torch.load(model_path))
    s2s = s2s.cpu()
    s2s.train(False)
    print(s2s)

    testset = vectorize(testset, word2idx, args)
    random.seed(15213)
    random.shuffle(testset)

    hyps, refs = {'Greedy':[], 'Beam':[]}, {'Greedy':[], 'Beam':[]}

    for _, headline, body in testset[:args.test_size]:
        inputs, targets, loc_word2idx, loc_idx2word = index_oov([body],
                [headline], word2idx, args)
        oov_size = len(loc_word2idx)
        input_lens, target_lens = [inputs.size(1)], [targets.size(1)]

        prediction, truth = summarize(s2s, inputs, input_lens, targets,
                target_lens, loc_idx2word, oov_size, beam_search = False,
                show_copy = True)
        hyps['Greedy'].append(prediction)
        refs['Greedy'].append(truth)

    rouge = Rouge()
    # for decode_approach in ["Greedy", "Beam"]:
    for decode_approach in ["Greedy"]:
        print("Decode Approach: {}".format(decode_approach))
        avg_score = rouge.get_scores(hyps[decode_approach],
                refs[decode_approach], avg=True)
        for metric, f1_prec_recl in avg_score.items():
            s = ', '.join(list(map(lambda x: '(%s, %.4f)' % (x[0], x[1]),
                f1_prec_recl.items())))
            print("%s: %s" % (metric, s))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--user_dir', type=str, default='/pylon5/ir3l68p/haomingc/1001Nights/')
    argparser.add_argument('--train_data_path', type=str, default="standard_giga/train/train_data_std_v50000")
    argparser.add_argument('--test_data_path', type=str, default="standard_giga/test/test_data.pkl")
    argparser.add_argument('--model_fpat', type=str, default="model/%s-%02d.model")
    argparser.add_argument('--model_path', type=str, default="model/5NSA-11.model")

    argparser.add_argument('--test_size', type=int, default=10000)
    argparser.add_argument('--batch_size', type=int, default=256)
    argparser.add_argument('--emb_size', type=int, default=64)
    argparser.add_argument('--hidden_size', type=int, default=256)
    argparser.add_argument('--nlayers', type=int, default=2)
    argparser.add_argument('--nepochs', type=int, default=30)
    argparser.add_argument('--max_title_len', type=int, default=20)
    argparser.add_argument('--max_text_len', type=int, default=32)
    argparser.add_argument('--learning_rate', type=float, default=0.001)
    argparser.add_argument('--teach_ratio', type=float, default=1)
    argparser.add_argument('--dropout', type=float, default=0.0)
    argparser.add_argument('--l2', type=float, default=0.001)
    # argparser.add_argument('--max_norm', type=float, default=100.0)

    argparser.add_argument('--test', action='store_true', default = False)
    argparser.add_argument('--data_src', type=str, choices=['xml', 'std'], default='std')
    argparser.add_argument('--rnn_model', type=str, choices=['gru', 'lstm'], default='lstm')
    argparser.add_argument('--attn_model', type=str, choices=['none', 'general', 'dot', 'concat'], default='none')
    argparser.add_argument('--use_cuda', action='store_true', default = False)
    argparser.add_argument('--use_bidir', action='store_true', default = False)
    argparser.add_argument('--use_copy', action='store_true', default = False)
    argparser.add_argument('--use_visualization', action='store_true', default = False)

    args = argparser.parse_args()
    # keepOOV keeps raw word for oovs, the other unk unk_idx
    args.train_data_path += "_keepOOV.pkl" if args.use_copy else ".pkl"
    check_args(args)

    for param in vars(args):
        if "path" in param:   # append user root dir as prefix to path-related parameters
            setattr(args, param, args.user_dir + getattr(args, param))
        print('- {} = {}'.format(param, getattr(args, param)))

    # only train_data has word dictionary
    train_data = pickle.load(open(args.train_data_path, "rb"))
    word2idx, idx2word = train_data["word2idx"], train_data["idx2word"]
    args.vocab_size = len(word2idx)

    if args.test:
        args.batch_size = 1
        args.use_cuda = False
        test_data = pickle.load(open(args.test_data_path, "rb"))
        test(args.model_path, test_data)
    else:
        # train_batch = group_data(train_data["text_vecs"][:20000], word2idx, args)
        train_batch = group_data(train_data["text_vecs"], word2idx, args)
        train(train_batch)
