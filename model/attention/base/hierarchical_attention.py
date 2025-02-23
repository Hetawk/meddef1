<<<<<<< HEAD
import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention


class HierarchicalAttention(nn.Module):
    def __init__(self, word_dim, sentence_dim, doc_dim, num_heads=8):
        super(HierarchicalAttention, self).__init__()
        self.word_attention = MultiHeadAttention(word_dim, word_dim, word_dim, num_heads)
        self.sentence_attention = MultiHeadAttention(sentence_dim, sentence_dim, sentence_dim, num_heads)
        self.doc_attention = MultiHeadAttention(doc_dim, doc_dim, doc_dim, num_heads)

    def forward(self, words, sentences, docs, word_mask=None, sentence_mask=None, doc_mask=None):
        word_output, word_attn = self.word_attention(words, words, words, word_mask)
        sentence_output, sentence_attn = self.sentence_attention(sentences, sentences, sentences, sentence_mask)
        doc_output, doc_attn = self.doc_attention(docs, docs, docs, doc_mask)

        return word_output, sentence_output, doc_output, word_attn, sentence_attn, doc_attn
=======
import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention


class HierarchicalAttention(nn.Module):
    def __init__(self, word_dim, sentence_dim, doc_dim, num_heads=8):
        super(HierarchicalAttention, self).__init__()
        self.word_attention = MultiHeadAttention(word_dim, word_dim, word_dim, num_heads)
        self.sentence_attention = MultiHeadAttention(sentence_dim, sentence_dim, sentence_dim, num_heads)
        self.doc_attention = MultiHeadAttention(doc_dim, doc_dim, doc_dim, num_heads)

    def forward(self, words, sentences, docs, word_mask=None, sentence_mask=None, doc_mask=None):
        word_output, word_attn = self.word_attention(words, words, words, word_mask)
        sentence_output, sentence_attn = self.sentence_attention(sentences, sentences, sentences, sentence_mask)
        doc_output, doc_attn = self.doc_attention(docs, docs, docs, doc_mask)

        return word_output, sentence_output, doc_output, word_attn, sentence_attn, doc_attn
>>>>>>> 16c5cfd9eac902321ee831908acfc69f3a52f936
