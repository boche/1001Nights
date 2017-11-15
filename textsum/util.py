import numpy as np
import torch
from torch.autograd import Variable
from tokens import *
import matplotlib as mpl
mpl.use('Agg') #adding this because otherwise plt will fail because of no display
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    probs = torch.stack(prob_list).transpose(0, 1).data.cpu().numpy() # b x s
    for i, prob in enumerate(probs):
        prob_sum += prob[:target_lens[i] - 1].sum()
    return prob_sum

def visualization(input_text, output_text, gold_text, attn, p_gen, args):
    """
    attn: output_s x input_s
    """
    input_words = [''] + input_text.split(' ')
    output_words = [''] + output_text.split(' ') + [EOS]
    attn = attn.data.cpu().numpy()[:len(output_words) - 1, :]
    p_gen = p_gen.data.cpu().numpy()[:len(output_words) - 1, :]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121 if args.use_pointer_net else 111)
    cax = ax.matshow(attn, cmap='bone')
    fig.colorbar(cax, orientation='horizontal')
    
    # Set up axes
    ax.set_xticklabels(input_words, rotation=90)
    ax.set_yticklabels(output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    if args.use_pointer_net:
        gs = GridSpec(5, 4)
        ax_attn = plt.subplot(gs[:, :-1])
        ax_prob = plt.subplot(gs[1:4, -1:])
       
        cax_attn = ax_attn.matshow(attn, cmap='bone') 
        # divider1 = make_axes_locatable(ax_attn)
        # cax1 = divider1.append_axes("right", size='5%', pad=0.05)
        # divider2 = make_axes_locatable(ax_prob)
        # cax2 = divider2.append_axes("right", size='50%', pad=0.05)
        # fig.colorbar(cax_attn, ax=ax_attn, cax=cax1, orientation='vertical')
        fig.colorbar(cax_attn, ax=ax_attn, orientation='horizontal')
        cax_prob = ax_prob.matshow(p_gen, cmap='bone')
        # fig.colorbar(cax_prob, ax=ax_prob, cax=cax2, orientation='vertical')    
        fig.colorbar(cax_prob, ax=ax_prob, orientation='vertical')    
        plt.tight_layout()
        
        ax_attn.set_title('attention scores') 
        ax_attn.set_xticklabels(input_words, rotation=90)
        ax_attn.set_yticklabels(output_words)
        
        ax_attn.xaxis.set_ticks_position('bottom')
        ax_attn.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax_attn.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        ax_prob.set_title('p_gen')
        ax_prob.set_xticks([])
        ax_prob.set_xticklabels([])
        ax_prob.set_yticklabels(output_words)
        ax_prob.yaxis.set_major_locator(ticker.MultipleLocator(1))

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
