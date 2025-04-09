import torch
import numpy as np
import transformers
from tqdm import tqdm
import linear_rep_geometry as lrg
from sklearn.linear_model import RidgeClassifier, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import json
from tqdm import tqdm




## get embeddings of each text
def get_embeddings(text_batch):
    tokenizer.pad_token = tokenizer.eos_token
    
    #7b and 13b
    #tokenized_output = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
    #70b
    #tokenized_output = tokenizer(text_batch, return_tensors="pt", padding=True, max_length=tokenizer.model_max_length, truncation=True).to(device)
    
    #pythia
    tokenized_output = tokenizer(text_batch, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)

    
    input_ids = tokenized_output["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states = True)
    hidden_states = outputs.hidden_states

    seq_lengths = tokenized_output.attention_mask.sum(dim=1).tolist()
    last_token_positions = [length - 1 for length in seq_lengths]
    text_embeddings = torch.stack([hidden_states[-1][i, pos, :] for i, pos in enumerate(last_token_positions)])

    return text_embeddings



###
def get_embedding_pairs(filename):
    fullpath = '/data/Yuhang/linear_rep-main/' + filename
    with open(fullpath, 'r') as f:
        lines = f.readlines()
    words_pairs = [line.strip().split('\t') for line in lines if line.strip()]

    lambdas_0 = []
    lambdas_1 = []

    for i in range(len(words_pairs)):
        first =  words_pairs[i][0] 
        second =  words_pairs[i][1] 
        lambdas_0.append(get_embeddings(first))
        lambdas_1.append(get_embeddings(second))
    

    return torch.cat(lambdas_0), torch.cat(lambdas_1)



device = torch.device("cuda:0")

model_name = "pythia-2.8b" #"DeepSeek-R1-Distill-Qwen-7B"  #"Meta-Llama-3-70B"

tokenizer = transformers.AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}", cache_dir = '/lts/yhliu/Pre-trained LLMs',load_in_8bit=True)
model = transformers.AutoModelForCausalLM.from_pretrained(f"EleutherAI/{model_name}",  low_cpu_mem_usage=True, trust_remote_code=True, cache_dir = '/lts/yhliu/Pre-trained LLMs')
model.to(device)


### load unembdding vectors ###
mm = model.name_or_path[-7:-1]

#gamma = model.lm_head.weight.detach().to(device) #Lamma

gamma = model.gpt_neox.embed_in.weight.detach().to(device) #pythia
W, d = gamma.shape



### concept data ###
filenames = ['word_pairs/[verb - 3pSg].txt',
             'word_pairs/[verb - Ving].txt',
             'word_pairs/[verb - Ved].txt',
             'word_pairs/[Ving - 3pSg].txt',
             'word_pairs/[Ving - Ved].txt',
             'word_pairs/[3pSg - Ved].txt',
             'word_pairs/[verb - V + able].txt',
             'word_pairs/[verb - V + er].txt',
             'word_pairs/[verb - V + tion].txt',
             'word_pairs/[verb - V + ment].txt',
             'word_pairs/[adj - un + adj].txt',
             'word_pairs/[adj - adj + ly].txt',
             'word_pairs/[small - big].txt',
             'word_pairs/[thing - color].txt',
             'word_pairs/[thing - part].txt',
             'word_pairs/[country - capital].txt',
             'word_pairs/[pronoun - possessive].txt',
             'word_pairs/[male - female].txt',
             'word_pairs/[lower - upper].txt',
             'word_pairs/[noun - plural].txt',
             'word_pairs/[adj - comparative].txt',
             'word_pairs/[adj - superlative].txt',
             'word_pairs/[frequent - infrequent].txt',
             'word_pairs/[English - French].txt',
             'word_pairs/[French - German].txt',
             'word_pairs/[French - Spanish].txt',
             'word_pairs/[German - Spanish].txt'
             ]

concept_names = []

for name in filenames: ##word_pairs/[verb - 3pSg].txt
    content = name.split("/")[1].split(".")[0][1:-1] # verb - 3pSg
    parts = content.split(" - ")  # 0: verb , 1: 3pSg
    concept_names.append(r'${} \Rightarrow {}$'.format(parts[0], parts[1])) #'$verb \\Rightarrow 3pSg$'


count = 0

# load embedding
embedding = []
for filename in filenames:
    embedding = []

    first, second = get_embedding_pairs(filename)
    embedding.append(first)
    embedding.append(second)
    
    first_second = filename.split("/")[1].split(".")[0][1:-1] # verb - 3pSg
    first_and_second = first_second.split(" - ")  # 0: verb , 1: 3pSg
    file_names=r'/data/Yuhang/linear_rep-main/data/yc_pythia{}_{}{}.pt'.format(mm, first_and_second[0], first_and_second[1])
    torch.save(embedding, file_names)

    count += 1
    
    
count = 0

#7b
np_embeddings = np.zeros((27, d))
np_weights = np.zeros((d, 27))


#13b model
# np_embeddings = np.zeros((27, 5120))
# np_weights = np.zeros((5120, 27))

#70b model
# np_embeddings = np.zeros((27, d))
# np_weights = np.zeros((d, 27))


# learning a linear classfication
for filename in filenames:
    
    first_second = filename.split("/")[1].split(".")[0][1:-1] # verb - 3pSg
    first_and_second = first_second.split(" - ")  # 0: verb , 1: 3pSg
    file_names=r'/data/Yuhang/linear_rep-main/data/yc_pythia{}_{}{}.pt'.format(mm, first_and_second[0], first_and_second[1])
    
    embedding = torch.load(file_names)
    
    
    label = np.ones(len(embedding[0]) + len(embedding[1]))
    label[len(embedding[0]):] = 0
    
    np_embedding = np.concatenate( (embedding[0].cpu().detach().numpy(), embedding[1].cpu().detach().numpy()), 0 )
    
    clf = LogisticRegression(max_iter=500,C=0.0001).fit(np_embedding, label) # for most case C=0.0001, C=0.01 for deepseek 7b 14b
    perf = clf.score(np_embedding, label)
    print('Loss: {:.3f}'.format(perf))
    
    # after training the LogisticRegression, print and save the log probability of each data for evaluation
    log_probs = clf.predict_log_proba(np_embedding)
    # torch.save(np_weights, '/data/Yuhang/linear_rep-main/data/deepseek-R1-7B_np_weights.pt')


    

