import numpy as np

"""
自建的数据, 结构为 Encoder_Input Decoder_Input Decoder_Output
这个训练的目的应该是让模型学会如何停止, 即如何预测下一个词为E的情况
"""
sentences = [
    ["我 是 学 生 P", "S I am a student", "I am a student E"],  # S: 开始符号
    ["我 喜 欢 学 习", "S I like learning P", "I like learning P E"],  # E: 结束符号
    ["我 是 男 生 P", "S I am a boy", "I am a boy E"],  # P: 占位符号, 句子长度不足的时候会使用P占位
]

src_vocab = {"P": 0, "我": 1, "是": 2, "学": 3, "生": 4, "喜": 5, "欢": 6, "习": 7, "男": 8}  # 词源字典
src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)

tgt_vocab = {"S": 0, "E": 1, "P": 2, "I": 3, "am": 4, "a": 5, "student": 6, "like": 7, "learning": 8, "boy": 9}  # 目标字典
tgt_idx2word = {tgt_vocab[key]: key for key in tgt_vocab}  # 索引到词的映射, 用于将one-hot编码转换为词
tgt_volab_size = len(tgt_vocab)

src_len = len(sentences[0][0].split())  # 输入序列长度, 若实际内容不足, 用P占位
tgt_len = len(sentences[0])  # 输出序列长度, 同理, 若实际内容不足, 用P占位

