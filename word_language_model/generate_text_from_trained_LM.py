###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
import torch.nn.functional as F

from word_language_model.data_holder import Corpus


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data_dir',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=torch.device(device))
    # model = torch.load(f).to(device)
model.eval()

corpus = Corpus(args.data)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
# notice
# input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
input = torch.tensor([[corpus.dictionary.word2idx['<start>']]]).to(device)

# notice

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history

        #notice
        for i in range(10):
            word_idxxx = 0

            while word_idxxx != corpus.dictionary.word2idx['<end>']:

                if is_transformer_model:
                    output = model(input, False)
                    word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                    input = torch.cat([input, word_tensor], 0)
                else:
                    output, hidden = model(input, hidden)
                    word_weights = output.squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    input.fill_(word_idx)
                    word_idxxx = word_idx.item()

                word = corpus.dictionary.idx2word[word_idx]

                outf.write(word + ('\n' if word_idxxx ==  corpus.dictionary.word2idx['<end>'] else ' '))

#
# sentence = '<start> a person standing on a surfboard riding a wave <end>'
# pos = 'DET ADJ NOUN VERB ADP DET ADJ CONJ ADJ NOUN'.split()
# tokens = sentence.split()
# lm_prop = []
#
# for idx in range(len(tokens)-1):
#     #input for model
#     input = torch.tensor([[corpus.dictionary.word2idx[tokens[idx]]]]).to(device)
#     #get output
#     output, hidden = model(input, hidden)
# ######
#     word_weights = output.squeeze().div(args.temperature).exp().cpu()
#     word_idx = torch.multinomial(word_weights, 1)[0]
#     print(corpus.dictionary.idx2word[word_idx])
# #########
#     import numpy as np
#
#
#     log_likelihood = F.log_softmax(output[0][0]).detach()
#     prop = np.exp(log_likelihood)
#
#     lm_prop.append((tokens[idx+1], prop[corpus.dictionary.word2idx[tokens[idx+1]]].item()))
#
#
#
# print(lm_prop)
