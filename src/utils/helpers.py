import lmdb
import tqdm
import pickle
from collections import namedtuple

CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])

def extract_latent(lmdb_env, net, dataloader):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        for batch in tqdm.tqdm(dataloader):
            imgs, labels, file_paths = batch 
            _,_,_, top_logits, bottom_logits = net.encode(imgs)

            for filepath, top, bottom in zip(file_paths, top_logits, bottom_logits):
                row = CodeRow(top=top, bottom=bottom, filename=filepath)
                txn.put(f"{index}".encode('utf-8'), pickle.dumps(row))
                index += 1
            txn.put('length'.encode('utf-8'), f"{index}".encode('utf-8'))
