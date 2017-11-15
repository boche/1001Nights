import torch
from torch.autograd import Variable

def logp_mask_loss(logp_list, target_lens, targets):
    """
    logp_list: list of torch tensors, (seq_len - 1) x batch x vocab_size
    target_lens: list of target lens
    targets: batch x seq
    """
    seq_len = targets.size(1)
    target_lens = torch.LongTensor(target_lens)
    target_lens = target_lens.cuda() if logp_list[0].is_cuda else target_lens
    loss = 0
    # offset 1 due to SOS
    for i in range(seq_len - 1):
        idx = Variable(targets[:, i + 1].contiguous().view(-1, 1)) # b x 1
        logp = torch.gather(logp_list[i], 1, idx).view(-1)
        loss += logp[target_lens > i + 1].sum()
    return -loss

def p_mask_loss(p_list, target_lens, targets):
    """
    p_list: list of torch tensors, (seq_len - 1) x batch x vocab_size
    target_lens: list of target lens
    targets: batch x seq
    """
    seq_len = targets.size(1)
    target_lens = torch.LongTensor(target_lens)
    target_lens = target_lens.cuda() if p_list[0].is_cuda else target_lens
    loss = 0
    # offset 1 due to SOS
    for i in range(seq_len - 1):
        idx = Variable(targets[:, i + 1].contiguous().view(-1, 1)) # b x 1
        logp = torch.log(torch.gather(p_list[i], 1, idx).view(-1))
        loss += logp[target_lens > i + 1].sum()
    return -loss

def mask_generation_prob(prob_list, target_lens):
    """
    prob_list: list of p_gens (not include prob for <SOS>); batch_size x 1
    target_lens: list; note that target_len includes <SOS> and <EOS> for each sentence
    """
    prob_sum = 0
    for i in range(len(prob_list)):
        p_gen = prob_list[i].data.cpu().numpy()
        for j in range(len(p_gen)):
            prob_sum += p_gen[j] if target_lens[j] > i + 1 else 0
    return prob_sum

def visualization(input_text, output_text, gold_text, attn):
    """
    attn: output_s x input_s
    """
    input_words = [''] + input_text.split(' ')
    output_words = [''] + output_text.split(' ') + [EOS]
    attn = attn.data.cpu().numpy()[:len(output_words) - 1, :]

    fig = plt.figure()
    fig.set_size_inches(8, 5)
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn, cmap='bone')
    fig.colorbar(cax, orientation='horizontal')

    # Set up axes
    ax.set_xticklabels(input_words, rotation=90)
    ax.set_yticklabels(output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("%sfigure/attn/%s.png" % (args.user_dir, gold_text.replace(" ", "_").replace('/', '_')), dpi = 200)
    plt.close()

def vec2text_from_full():
    idx2word_full = pickle.load(open(args.user_dir + 'nyt/idx2word_full.pkl', 'rb'))
    data = pickle.load(open(args.user_dir + 'nyt/nyt_eng_200912.pkl', 'rb'))
    data_text = []
    for docid, headline, body in data:
        if len(headline) > 0 and len(body) > 0:
            raw_headline = [idx2word_full[w] for w in headline]
            raw_body = [idx2word_full[w] for w in body]
            data_text.append((docid, raw_headline, raw_body))
    return data_text

