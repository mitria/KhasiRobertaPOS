
from src.metrics import categorical_accuracy
import torch
def train(batch_inputs,batch_tags,model, optimizer, criterion, tag_pad_idx,device):
   
    # (batch_inputs,batch_tags) = make_batches(train_input_ids,train_labels_ids, 2)
    epoch_loss = 0
    epoch_acc = 0
    #print(batch_inputs.shape)   
    model.train()
    
    #for batch in iterator:
   # for (inputs, labels,masks) in zip(batch_inputs, batch_labels,batch_masks):
    for step in range(0, len(batch_inputs)):   
        text = torch.tensor(batch_inputs[step],dtype=torch.int64).to(device)
        
        tags=torch.tensor(batch_tags[step],dtype=torch.int64).to(device)
        
        optimizer.zero_grad()
        
        #text = [sent len, batch size]
        
        predictions = model(text)
        
        #predictions = [sent len, batch size, output dim]
        #tags = [sent len, batch size]
        # print(predictions.shape)
        predictions = predictions.view(-1, predictions.shape[-1])
        
        
        tags = tags.view(-1)
       # print(tags.shape)
        
        #predictions = [sent len * batch size, output dim]
        #tags = [sent len * batch size]
        
        loss = criterion(predictions,tags)
                
        acc = categorical_accuracy(predictions, tags, tag_pad_idx,device)
        # all_loss.append(loss.item())
        # all_acc.append(acc.item())
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(batch_inputs), epoch_acc / len(batch_inputs)


def evaluate(batch_val_inputs,batch_val_tags,model,  criterion, tag_pad_idx,device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    # (batch_inputs,batch_tags) = make_batches(valid_input_ids,valid_labels_ids, 2)
    with torch.no_grad():
    
        #for batch in iterator:
         for step in range(0, len(batch_val_inputs)): 
            text = torch.tensor(batch_val_inputs[step],dtype=torch.int64).to(device)
            tags = torch.tensor(batch_val_tags[step],dtype=torch.int64).to(device)
            
            predictions = model(text)
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(predictions, tags)
            
            acc = categorical_accuracy(predictions, tags, tag_pad_idx,device)
            # all_val_acc.append(acc.item())
            # all_val_loss.append(loss.item())
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(batch_val_inputs), epoch_acc / len(batch_val_inputs)

def test(model,batch_test_inputs,batch_test_tags, criterion, tag_pad_idx,device):
    ypred=[]
    ytrue=[]
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    # (batch_inputs,batch_tags) = make_batches(valid_input_ids,valid_labels_ids, 2)
    with torch.no_grad():
    
        #for batch in iterator:
         for step in range(0,len(batch_test_inputs)): 
           # print(step)
            text = torch.tensor(batch_test_inputs[step],dtype=torch.int64).to(device)
            #print(text.shape)
            # text = torch.tensor(test_input_ids[step],dtype=torch.int64).to(device)
            tags = torch.tensor(batch_test_tags[step],dtype=torch.int64).to(device)
            # tags = torch.tensor(batch_test_tags[step],dtype=torch.int64).to(device)
            
            predictions = model(text)
            predictions1=predictions
            predictions = predictions.view(-1, predictions.shape[-1])
           # print(predictions.shape)
            tags = tags.view(-1)
           
            loss = criterion(predictions, tags)
            
            acc = categorical_accuracy(predictions, tags, tag_pad_idx,device)
            #tagret=tags.cpu().numpy()
            # predictions=predictions
            max_preds = predictions1.argmax(dim = 2)
           # print('max_preds:',max_preds)
            # ypred.append(max_preds.cpu().numpy())
            #ytrue.append(tagret)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(batch_test_inputs), epoch_acc / len(batch_test_inputs),max_preds.cpu().numpy()
    