
# coding: utf-8

# In[11]:


from heapq import *
import numpy as np

UNK = 0
START = 1
END = 2

def get_hyp(beam_size, max_depth):
    def beam_search(beam_size, last_logp, last_word, prev_words, p):
        pruning = True
        if last_word == END:
            while len(final_candidates) >= beam_size and last_logp > final_candidates[0][0]:
                heappop(final_candidates)
            if len(final_candidates) < BEAM_SIZE:
                heappush(final_candidates, (last_logp, prev_words))
            return pruning
        if final_candidates and last_logp < final_candidates[0][0]:
            return pruning
        
        for word in np.argpartition(p, -beam_size)[-beam_size:]:
            current_logp = last_logp + np.log(p[word])

            while len(partial_candidates) + 1 > beam_size and current_logp > partial_candidates[0][0]:
                heappop(partial_candidates)

            if len(partial_candidates) + 1 <= beam_size:
                pruning = False
                heappush(partial_candidates, (current_logp, (word, prev_words + [word])))

        return pruning
        
    final_candidates = []
    last_candidates = [(0.0, (START, []))]
    current_depth = 0
    p = [0.4, 0.2, 0.1, 0.5, 0.6]
    while last_candidates and current_depth < max_depth:
        current_depth += 1
        partial_candidates = []
        for last_logp, (last_word, prev_words) in last_candidates:
            print(last_logp, prev_words)
            beam_search(beam_size, last_logp, last_word, prev_words, p)
        last_candidates = partial_candidates
        p = [0.5, 0.1, 0.1, 0.1, 0.1]
        
    if final_candidates:
        last_logp, result_sent = max(final_candidates)
    else:
        last_logp, (_, result_sent) = max(last_candidates)

    return result_sent[:-1] if result_sent[-1] == END else result_sent

print(get_hyp(3, 3))

