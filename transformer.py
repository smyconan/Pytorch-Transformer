import torch
import torch.nn as nn
import numpy as np

def get_padding_mask(key, query):
	return key.eq(0).unsqueeze(1).expand(-1, query.size(1), -1) # (BatchSize * QuerySeqLen * KeySeqLen)

def get_future_mask(seq):
	return torch.triu(torch.ones((seq.size(1), seq.size(1)), dtype=torch.uint8), diagonal = 1).unsqueeze(0).expand(seq.size(0), -1, -1)

class PositionalEncoder(nn.Module):
	def __init__(self, embedding_dim, max_seq_len):
		super(PositionalEncoder, self).__init__()
		weights = np.array([[position / np.power(10000, 2.0 * (idx // 2) / embedding_dim) 
								for idx in range(embedding_dim)] 
							for position in range(max_seq_len)])
		weights[:,0::2] = np.sin(weights[:,0::2])
		weights[:,1::2] = np.cos(weights[:,1::2])
		self.encoding = nn.Embedding(max_seq_len, embedding_dim)
		self.encoding.weight = nn.Parameter(torch.from_numpy(weights), requires_grad = False)
	def forward(self, batch_lengths, max_length):
		position_ids = torch.LongTensor([(list(range(length))+[0]*(max_length - length)) for length in batch_lengths])
		return self.encoding(position_ids)

class QKVAttentioner(nn.Module):
	def __init__(self, embedding_dim, head_dim, dropout):
		super(QKVAttentioner, self).__init__()
		self.dropout = nn.Dropout(dropout)
		self.softmax = nn.Softmax(dim=2)
		self.head_dim = head_dim
		self.Wq = nn.Linear(embedding_dim, head_dim)
		self.Wk = nn.Linear(embedding_dim, head_dim)
		self.Wv = nn.Linear(embedding_dim, head_dim)
	def forward(self, q, k, v, mask):
		QKSim = torch.bmm(self.Wq(q.float()), self.Wk(k.float()).transpose(1,2))/np.sqrt(self.head_dim)
		QKSim = QKSim.masked_fill_(mask, -np.inf)
		return torch.bmm(self.dropout(self.softmax(QKSim)), self.Wv(v.float())) # (BatchSize * SeqLen * HeadDim)

class MultiHeadAttentioner(nn.Module):
	def __init__(self, embedding_dim, num_heads, dropout):
		super(MultiHeadAttentioner, self).__init__()
		head_dim = embedding_dim // num_heads
		self.qkv_attentions = nn.ModuleList([QKVAttentioner(embedding_dim, head_dim, dropout) for i in range(num_heads)])
		self.linear = nn.Linear(embedding_dim, embedding_dim)
		self.dropout = nn.Dropout(dropout)
		self.normalize = nn.LayerNorm(embedding_dim)
	def forward(self, key, value, query, mask):
		residual = query
		attention = torch.cat([self.qkv_attentions[i](query, key, value, mask) for i in range(len(self.qkv_attentions))], 2)
		attention = self.dropout(self.linear(attention))
		return self.normalize(residual.float() + attention)

class FeedForwarder(nn.Module):
	def __init__(self,embedding_dim, FFN_dim, max_seq_len, dropout):
		super(FeedForwarder, self).__init__()
		self.layer1s = nn.ModuleList([nn.Linear(embedding_dim, FFN_dim) for i in range(max_seq_len)])
		self.layer2s = nn.ModuleList([nn.Linear(FFN_dim, embedding_dim) for i in range(max_seq_len)])
		self.dropout = nn.Dropout(dropout)
		self.normalize = nn.LayerNorm(embedding_dim)
	def forward(self, x):
		residual = x
		ffn_output = torch.cat([self.layer2s[i](self.layer1s[i](x[:,i,:])).unsqueeze(1) for i in range(len(self.layer1s))],1)
		return self.normalize(self.dropout(ffn_output) + residual)

class EncoderLayer(nn.Module):
	def __init__(self, embedding_dim, num_heads, FFN_dim, max_seq_len, dropout):
		super(EncoderLayer, self).__init__()
		self.attention = MultiHeadAttentioner(embedding_dim, num_heads, dropout)
		self.ffn = FeedForwarder(embedding_dim, FFN_dim, max_seq_len, dropout)
	def forward(self, seq, mask):
		return self.ffn(self.attention(seq, seq, seq, mask))

class Encoder(nn.Module):
	def __init__(self, vocab_size, max_seq_len, num_layers, embedding_dim, num_heads, FFN_dim, dropout):
		super(Encoder, self).__init__()
		self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
		self.position_encoder = PositionalEncoder(embedding_dim, max_seq_len)
		self.encoder_layers = nn.ModuleList([EncoderLayer(embedding_dim, num_heads, FFN_dim, max_seq_len, dropout) for i in range(num_layers)])
	def forward(self, seq, seq_length):
		emb_seq = self.embedding_layer(seq) + self.position_encoder(seq_length, seq.size(1))
		mask = get_padding_mask(seq, seq)
		for i in range(len(self.encoder_layers)):
			emb_seq = self.encoder_layers[i](emb_seq, mask)
		return emb_seq

class DecoderLayer(nn.Module):
	def __init__(self, embedding_dim, num_heads, FFN_dim, max_seq_len, dropout):
		super(DecoderLayer, self).__init__()
		self.attention = MultiHeadAttentioner(embedding_dim, num_heads, dropout)
		self.ffn = FeedForwarder(embedding_dim, FFN_dim, max_seq_len, dropout)
	def forward(self, encoder_outputs, decoder_seq, self_mask, context_mask):
		attention = self.attention(decoder_seq, decoder_seq, decoder_seq, self_mask)
		attention = self.attention(encoder_outputs, encoder_outputs, attention, context_mask)
		return self.ffn(attention)

class Decoder(nn.Module):
	def __init__(self, vocab_size, max_seq_len, num_layers, embedding_dim, num_heads, FFN_dim, dropout):
		super(Decoder, self).__init__()
		self.num_layers = num_layers
		self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
		self.position_encoder = PositionalEncoder(embedding_dim, max_seq_len)
		self.decoder_layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, FFN_dim, max_seq_len, dropout) for i in range(num_layers)])
	def forward(self, encoder_outputs, decoder_seq, decoder_seq_length, context_mask):
		emb_seq = self.embedding_layer(decoder_seq) + self.position_encoder(decoder_seq_length, decoder_seq.size(1))
		self_mask = get_padding_mask(decoder_seq, decoder_seq)
		future_mask = get_future_mask(decoder_seq)
		mask = torch.gt(self_mask + future_mask, 0) # great than
		for i in range(self.num_layers):
			emb_seq = self.decoder_layers[i](encoder_outputs, emb_seq, mask, context_mask)
		return emb_seq

class Transformer(nn.Module):
	def __init__(self, src_vocab_size, src_max_len, tgt_vocab_size, tgt_max_len, num_layers, embedding_dim, num_heads, FFN_dim, dropout):
		super(Transformer, self).__init__()
		self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, embedding_dim, num_heads, FFN_dim, dropout)
		self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, embedding_dim, num_heads, FFN_dim, dropout)
		self.linear = nn.Linear(embedding_dim, tgt_vocab_size, bias = False)
		self.softmax = nn.Softmax(dim = 2)
	def forward(self, src_seq, src_len, tgt_seq, tgt_len):
		encoder_outputs = self.encoder(src_seq, src_len)
		context_mask = get_padding_mask(src_seq, tgt_seq)
		output = self.decoder(encoder_outputs, tgt_seq, tgt_len, context_mask)
		return self.softmax(self.linear(output))
