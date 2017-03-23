#coding:utf-8
__author__ = 'root'

import collections
import numpy as np
import tensorflow as tf

poetry_file = 'poetry.txt'

#诗集
poetrys = []
with open(poetry_file,'r') as f:
    for line in f:
        try:
            title,content = line.strip().split(':')
            content = content.replace(" ",'')
            if '_' in content or '(' in content or '（' in content or '[' in content or '《' in content:
                continue
            if len(content) <5 or len(content) >79:
                continue
            content = '['+content+']'
            poetrys.append(content)
        except Exception as e :
            pass

#按诗的字数排序
poetrys =  sorted(poetrys,key=lambda line:len(line))
# print '唐诗总数',len(poetrys)


#统计每个字出现次数
all_words = []
for poetry in poetrys:
    all_words +=[word for word in poetry]
#all_words 为所有的字母的列表
counter = collections.Counter(all_words)
#counter   统计出每个字母的个数，｛‘a’:2,'b':1｝
count_pairs = sorted(counter.items(),key=lambda  x:-x[1])
#count_pairs　　　按照元祖索引１的数字大小从大到小排序的元祖，外面为列表的列表
#例如:[('a':6),('d':4),('g':2)]
words,_ = zip(*count_pairs)

#多少个常用字
words = words[:len(words)]+('',)

#每个字映射为一个数字ＩＤ
word_num_map = dict(zip(words,range(len(words))))
# print word_num_map
to_num = lambda word:word_num_map.get(word,len(words))
# print to_num
poetrys_vector = [list(map(to_num,poetry)) for poetry in poetrys]
#poetrys_vector 给每行的字母都转华为数字。
# print word_num_map


#每次取６４首诗进行训练

batch_size = 1
n_chunk = len(poetrys_vector)//batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
    start_index = i*batch_size
    end_index = start_index+ batch_size

    batches = poetrys_vector[start_index:end_index]
    length = max(map(len,batches))
    #length 找到所有中最长的那个长度
    xdata  = np.full((batch_size,length),word_num_map[''],np.int32)

    for row in range(batch_size):
        xdata[row,:len(batches[row])] = batches[row]

    ydata= np.copy(xdata)
    ydata[:,:-1] = xdata[:,1:]

    x_batches.append(xdata)
    y_batches.append(ydata)



#------------------------------------RNN----------


input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):

	if model == 'rnn':

		cell_fun = tf.nn.rnn_cell.BasicRNNCell
	elif model == 'gru':
		cell_fun = tf.nn.rnn_cell.GRUCell
	elif model == 'lstm':
		cell_fun = tf.nn.rnn_cell.BasicLSTMCell

	cell = cell_fun(rnn_size, state_is_tuple=True)
	cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

	initial_state = cell.zero_state(batch_size, tf.float32)

	with tf.variable_scope('rnnlm'):
		softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)+1])
		softmax_b = tf.get_variable("softmax_b", [len(words)+1])
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])
			inputs = tf.nn.embedding_lookup(embedding, input_data)

	outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
	output = tf.reshape(outputs,[-1, rnn_size])

	logits = tf.matmul(output, softmax_w) + softmax_b
	probs = tf.nn.softmax(logits)
	return logits, last_state, probs, cell, initial_state

#--------生成古诗-------------
#使用训练完成的模型

def gen_poetry():
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t,np.random.rand(1)*s))
        return words[sample]

    _,last_state,probs,cell,initial_state = neural_network()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_tables())
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess,'poetry.module-49')

        state_=sess.run(cell.zero_state(1,tf.float32))

        x = np.array([list(map(word_num_map.get,'['))])
        [probs_,state_] = sess.run([probs,last_state],feed_dict={input_data:x,initial_state:state_})
        word = to_word(probs_)
        poem = ''
        while word !=']':
            poem += word
            x = np.zeros((1,1))
            x[0,0] = word_num_map[word]
            [probs_,state_] = sess.run([probs,last_state],feed_dict={input_data:x,initial_state:state_})
            word = to_word(probs_)
        return poem

print gen_poetry()




















