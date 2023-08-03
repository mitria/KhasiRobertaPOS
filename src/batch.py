import random
from numpy.random import seed
seed(1)
def make_batches(ids,tags, batch_size):
 

    # samples=sorted(zip(ids,tags), key=lambda x: len(x[0]))
    batch_ordered_sentences = []
    batch_ordered_tags=[]
    ids=ids.tolist()
    tags=tags.tolist()
      
    while len(ids) > 0:
       
        
        to_take = min(batch_size, len(ids))

        # Pick a random index in the list of remaining samples to start
        # our batch at.
        select = random.randint(0, len(ids) - to_take)

        # Select a contiguous batch of samples starting at `select`.
        #print("Selecting batch from {:} to {:}".format(select, select+to_take))
        # batch = samples[select:(select + to_take)]
        id_bat=ids[select:(select + to_take)]
        tag_bat=tags[select:(select + to_take)]
        # batch=zip(ids[select:(select + to_take)],tags[select:(select + to_take)])
        # print("Batch length:", len(batch))

        #for s in batch: print(s[2])
        batch_ordered_sentences.append(id_bat)#([s[0] for s in batch])
        batch_ordered_tags.append(tag_bat)#([s[1] for s in batch])
        

        # Remove these samples from the list.
        del ids[select:select + to_take]
        del tags[select:select + to_take]
    print('\n  DONE - Selected {:,} batches.\n'.format(len(batch_ordered_sentences)))
  
    return( batch_ordered_sentences,batch_ordered_tags)


def make_test_batches(ids,tags, batch_size):
     # samples=sorted(zip(ids,tags), key=lambda x: len(x[0]))
    batch_ordered_sentences = []
    batch_ordered_tags=[]
    ids=ids.tolist()
    tags=tags.tolist()
    select=0  
    while len(ids) > 0:
       
        # `to_take` is our actual batch size. It will be `batch_size` until 
        # we get to the last batch, which may be smaller. 
        to_take = min(batch_size, len(ids))

       
        # batch = samples[select:(select + to_take)]
        id_bat=ids[select:(select + to_take)]
        tag_bat=tags[select:(select + to_take)]
        # batch=zip(ids[select:(select + to_take)],tags[select:(select + to_take)])
        # print("Batch length:", len(batch))

        
        #for s in batch: print(s[2])
        batch_ordered_sentences.append(id_bat)#([s[0] for s in batch])
        batch_ordered_tags.append(tag_bat)#([s[1] for s in batch])
        

        # Remove these samples from the list.
        del ids[select:select + to_take]
        del tags[select:select + to_take]
    print('\n  DONE - Selected {:,} batches.\n'.format(len(batch_ordered_sentences)))
  
    return( batch_ordered_sentences,batch_ordered_tags)