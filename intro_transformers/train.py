import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm


from dataset import BilingualDataset, causal_mask
from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from config import get_config, get_weights_file_path

import warnings






def get_all_sentences(ds, lang):
    for item in ds: # a pair of sentences
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path): 
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2) #start of sentence, 
        #end of sentence, padding
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))


    return tokenizer



def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    #keep 90% of the data for training
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max length of source language: {max_len_src}")
    print(f"Max length of target language: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=True) #we wan to process one sentence at a time

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



def get_model(config, vocab_src_len, vocab_tgt_len, ):
   model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
   return model



def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model.to(device)

    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload'] is not None:
       model_filename = get_weights_file_path(config, config['preload'])
       print(f'Preloading model from {model_filename}')
       state = torch.load(model_filename)
       initial_epoch = state['epoch'] + 1
       optimizer.load_state_dict(state['optimizer_state_dict'])
       global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator: 
            encoder_input = batch["encoder_input"].to(device) #(B, Seq_Len)
            decoder_input = batch["decoder_input"].to(device) #(B, Seq_Len)
            encoder_mask = batch["encoder_mask"].to(device) #(B, 1, 1, Seq_Len)
            decoder_mask = batch["decoder_mask"].to(device) #(B, 1, Seq_Len, Seq_Len)

            encoder_output = model.encode(encoder_input, encoder_mask) #(B, Seq_Len, d_model)
            decoder_output = model.decode(decoder_input, decoder_mask, encoder_output, encoder_mask) #(B, Seq_Len, d_model)
            proj_output = model.project(decoder_output) # (B, Seq_Len, vocab_tgt_size)

            label = batch["labels"].to(device) #(B, Seq_Len)

            # (B, Seq_Len, vocab_tgt_size) -> (B*Seq_Len, vocab_tgt_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) 
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})  

            #log into tensorboard
            writer.add_scalar("train loss", loss.item(), global_step) 
            writer.flush()

            #backpropagate
            loss.backward()

            #update weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch, 
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

    








