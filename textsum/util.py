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
    probs = torch.stack(prob_list).transpose(0, 1).data.cpu().numpy() # b x s
    for i, prob in enumerate(probs):
        prob_sum += prob[:target_lens[i] - 1].sum()
    return prob_sum
