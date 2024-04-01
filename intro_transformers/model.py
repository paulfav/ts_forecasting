import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model:int, vocab_size : int): #the dimension of the model and the vocab_size
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding= nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        #create a matrix of siez (seq_len, d_model) 
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #(1, seq_len, d_model) 

        self.register_buffer('pe', pe) #register the buffer to the model, this way we save the state of the 
        #tensor when we save the model

    def forward(self, x):
        x += (self.pe[:, :x.size(1), : ]).requires_grad_(False) #add the positional encoding to the input, and 
        #make it not trainable
        return self.dropout(x)
    

#normalization layer

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias



#feed forward layer

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and b1
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and b2
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


#multihead attention layer
    
class MultiHeadAttentionBlock(nn.Module):  
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.h = h
        assert d_model % h == 0, "d_model must be a multiple of h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model) #in the model, d_v = d_k
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout : nn.Dropout):
        d_k = query.size(-1)
        attention_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        #the (-2, -1) is to transpose the last two dimensions of the tensor
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) #(batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        output = attention_scores @ value
        return output, attention_scores

    def forward(self, q, k, v, mask = None):
        query = self.w_q(q) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        #(batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k)--> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k) #just d_model

        return self.w_o(x)
    


class ResidualConnection(nn.Module): 
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) #here we apply the sublayer on the normalized x,
    #in the original paper, the normalization is applied after the sublayer.




class EncoderBlock(nn.Module): 

    def __init__(self, self_attention_block : MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float): 
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x



class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #(batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    


class Transformer(nn.Module): 

    def __init__(
            self, encoder: Encoder, decoder: Decoder, 
            src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, 
            src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, 
            projection_layer: ProjectionLayer
            ):
        
        super().__init__()
        self.encoder = encoder  
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer


    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(
    src_vocab_size: int, tgt_vocab_size: int, 
    src_seq_len: int, tgt_seq_len: int,
    d_model: int = 512, N: int = 6, h: int = 8, d_ff: int = 2048, dropout: float = 0.1
    ) -> Transformer:
    #create embeddings
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    #create positional encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)


    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
  

    #initialize the parameters of the mod
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
        

    


    

    


    

    


    




