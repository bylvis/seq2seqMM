
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense,TimeDistributed
import numpy as np
import keras.backend as K

def get_dataset(data_path, num_samples):
    input_texts = []
    target_texts = []

    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text, _ = line.split('\t')
        # 用tab作用序列的开始，用\n作为序列的结束
        target_text = '\t' + target_text + '\n'

        input_texts.append(input_text)
        target_texts.append(target_text)

        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    return input_texts,target_texts,input_characters,target_characters


#------------------------------------------#
#   init初始化部分
#------------------------------------------#
# 每一次输入64个batch
batch_size = 64
# 训练一百个世代
epochs = 100
# 256维神经元
latent_dim = 256
# 一共10000个样本
num_samples = 10000

# 读取数据集
data_path = 'fra.txt'

# 获取数据集
# 其中input_texts为输入的英文字符串
# target_texts为对应的法文字符串

# input_characters用到的所有输入字符,如a,b,c,d,e,……,.,!等
# target_characters用到的所有输出字符
input_texts,target_texts,input_characters,target_characters = get_dataset(data_path, num_samples)

# 对字符进行排序
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
# 计算共用到了什么字符
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
# 计算出最长的序列是多长
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('一共有多少训练样本：', len(input_texts))
print('多少个英文字母：', num_encoder_tokens)
print('多少个法文字母：', num_decoder_tokens)
print('最大英文序列:', max_encoder_seq_length)
print('最大法文序列:', max_decoder_seq_length)

# 建立字母到数字的映射
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

#---------------------------------------------------------------------------#

#--------------------------------------#
#   改变数据集的格式
#--------------------------------------#
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # 为末尾加上" "空格
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    # 相当于前一个内容的识别结果，作为输入，传入到解码网络中
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data不包括第一个tab
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.
#---------------------------------------------------------------------------#

# 第一个None代表可以输入一个不定长的英文序列 num_encoder_tokens代表多少个英文字母
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# return_state=True 获得单元状态和输出值
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_inputs)
# 单元状态和输出值变成一个列表
encoder_states = [state_h, state_c]
# 解码器的输入值 num_decoder_tokens代表多少个法文字母
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# 给解码器输入时需要给一个初始单元状态和输出值
decoder_outputs, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(decoder_inputs,initial_state=encoder_states)
# 将所有步进行全连接 维度是多少个法文字符 因为一个字母就是一个步 soft找出索引最大的法文字母
decoder_outputs = TimeDistributed(Dense(num_decoder_tokens, activation='softmax'))(decoder_outputs)
# 参数是编码输入(英文序列) 解码输出(以制表符开始的法文序列) 解码输入(以制表符结束的法文序列)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 开始训练
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=5,
          validation_split=0.2)
# 保存模型
model.save('out.h5')
K.clear_session()

# 定义一个编码输入 和前面也一样
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# 传入LSTM 获得单元状态和输出值 encoder_outputs就作为解码器输出了
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_inputs)
# 做成列表 作为decoder的初始状态
encoder_states = [state_h, state_c]
# 输入是以制表符开头的法文序列 输出是当前decoder的单元状态和输出值
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.load_weights("out.h5",by_name=True)
encoder_model.summary()

# decoder模型的法文序列输入
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# decoder模型的输入的值
decoder_state_input_h = Input(shape=(latent_dim,))
# decoder模型的输入的单元状态
decoder_state_input_c = Input(shape=(latent_dim,))
# 单元状态和值组成列表
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# 得到输出值 当前模型输出值 输出单元状态
decoder_outputs, state_h, state_c = LSTM(latent_dim, return_sequences=True,
                                        return_state=True)(decoder_inputs, initial_state=decoder_states_inputs)
# 当前新的单元状态和值组成列表
decoder_states = [state_h, state_c]

# 全连接层 对decoder_outputs每一个维度的内容进行全连接 获得最大概率的索引 获得法文字母的序列
decoder_outputs = TimeDistributed(Dense(num_decoder_tokens, activation='softmax'))(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
decoder_model.load_weights("out.h5",by_name=True)

# 建立序号到字母的映射 将预测结果转化为字符串
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

# 把英文序列输入进来 把单元状态和输出值转入到decoder当中 然后一步一步往下预测的过程
def decode_sequence(input_seq):
    # 将输入的英文序列输入LSTM输出单元状态和输出值
    states_value = encoder_model.predict(input_seq)
    #  建立了timeStep为1的阵列 模拟输入\t 获得decoder的第一个输出
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # 以\t为开头，一个一个向后预测
        # 传入到decoder的模型 制表符加初始状态 获取第一个输出和单元状态和值 output_tokens, h, c
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # 进行字符串操作 获得概率最大的序号 将序号中的字母取出来
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        # 如果达到结尾
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # 把新获得的单元状态 设置成新的单元状态 将新获得的法文字母作为下一轮的法文输入
        states_value = [h, c]
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.


    return decoded_sentence


# for seq_index in range(100):
#     input_seq = np.expand_dims(encoder_input_data[seq_index],axis=0)
#     decoded_sentence = decode_sequence(input_seq)
#     print('-')
#     print('Input sentence:', input_texts[seq_index])
#     print('Decoded sentence:', decoded_sentence)
input_seq = np.expand_dims(encoder_input_data[999],axis=0)
decoded_sentence = decode_sequence(input_seq)
print('-')
print('Input sentence:', input_texts[999])
print('Decoded sentence:', decoded_sentence)