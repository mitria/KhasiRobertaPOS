import re
import torch


def prep_sent(tokenizer,tokens_a,max_seq_length,max_id_length=250):#prep_sent(sentence,max_seq_length):
    # tokens_a=sentence.split()
    print(tokens_a)
    
    if len(tokens_a) > max_seq_length-2:
        tokens_a = tokens_a[0 : (max_seq_length-2)]
    orig_to_tok_map = []              
    tokens = []

    tokens.append("<s>")

    orig_to_tok_map.append(len(tokens)-1)
    for token in tokens_a:
        orig_to_tok_map.append(len(tokens)) # keep first piece of tokenized term
        tokens.extend(tokenizer.tokenize(token))
        #orig_to_tok_map.append(len(tokens)-1) # # keep last piece of tokenized term -->> gives better results!
    tokens.append("</s>")
    print(tokens)
    orig_to_tok_map.append(len(tokens)-1)
    #print(orig_to_tok_map)
    input_ids = tokenizer.convert_tokens_to_ids([toks for toks in tokens])
    # orig_len=len(input_ids)-1
    # print('orig_len\t',orig_len)
    if len(input_ids)>max_id_length:
        input_ids=input_ids[0:max_id_length]
    while len(input_ids) < max_id_length:
        input_ids.append(0)
   # print(input_ids)
    return input_ids,orig_to_tok_map

def tag_sentence(tokenizer,sentence,model,id2tag,device):#, text_field, tag_field):   
    model.to(device)
    tokens_a=sentence.split()
    # orig_len=len(tokens_a)
    model.eval()
    token_tensor,orig_to_tok_map=prep_sent(tokenizer,tokens_a,max_seq_length=200)
    token_tensor=[token_tensor]
    # orig_len=len(token_tensor[0])
    # print('or len\t',orig_len)
    input_tensor=torch.tensor(token_tensor,dtype=torch.long).to(device)

    predicted_tags=[]
    predictions = model(input_tensor)
    #print('presha',predictions.shape)
    predictions=predictions.to('cpu')
    predictions = predictions.view(-1, predictions.shape[-1])
    # print(predictions[0][0])
    top_predictions = predictions.argmax(-1)
    #print(top_predictions)
    predictions=top_predictions.detach().numpy()
   
    
    for j  in orig_to_tok_map[1:len(orig_to_tok_map)-1]:#predictions:
        try:
            predicted_tags.append(id2tag[predictions[j]])#for id in j ]#.squeeze(1)]
        except:
            break
    return predicted_tags


def extract(line):
    raw_text=[]
    tags=[]
    for word in re.split('\s',line):
        wordtagsplit=word.split("/")
        if wordtagsplit[0]!='':
             raw_text.append(wordtagsplit[0])
             tags.append(wordtagsplit[1])
    
    if len(tags)!=len(raw_text):
        print(tags)
        print(raw_text)
    return(raw_text,tags)


def join(tags,sentence):
    tagged_line=''
    for i in range(len(tags)):
        print(sentence[i]+'/'+tags[i]+' ')
        tagged_line+=sentence[i]+'/'+tags[i]+' '
    return(tagged_line)