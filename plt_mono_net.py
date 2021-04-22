import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpt

tf.enable_eager_execution()

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.W1 = tf.Variable(tf.random_normal([1, 10]))
        self.b1 = tf.Variable(tf.zeros([10]))
        self.W2 = tf.Variable(tf.random_normal([10, 1]))
        self.b2 = tf.Variable(tf.zeros([1]))
    def __call__(self, inputs):
        x = tf.nn.sigmoid(tf.multiply(inputs, self.W1) + self.b1)
        return tf.matmul(x, self.W2) + self.b2

model = MyModel()

def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

NUM_EXAMPLES = 1000

input = np.linspace(0, 1, num=1000)
input= np.reshape(input, (1000, 1))
input = tf.cast(input, tf.float32)
output = 2* input ** 2 + input + 3 * input **3


def train(model, input, output, learning_rate):
    with tf.GradientTape() as t:
        t.watch(model.W1)
        t.watch(model.W2)
        t.watch(model.b1)
        t.watch(model.b2)
        current_loss = loss(model(input), output)
    dW1, db1, dW2, db2 = t.gradient(current_loss, [model.W1, model.b1, model.W2, model.b2])
    model.W1 = (model.W1.assign_sub(learning_rate * dW1))
    model.W2 = (model.W2.assign_sub(learning_rate * dW2))
    model.b1 = (model.b1.assign_sub(learning_rate * db1))
    model.b2 = (model.b2.assign_sub(learning_rate * db2))
    print(current_loss)

ls_list = []
epochs = 5000
for epoch in range(epochs):
    print(epoch)
    current_loss = loss(model(input), output)
    ls_list.append(current_loss)
    train(model, input, output, learning_rate=0.1)

# index = [i for i in range(epochs)]
# plt.plot(index, ls_list, label='loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()


input_list = np.reshape(input, (1000))
output_list = np.reshape(output, (1000))
predict_list = np.reshape(model(input), (1000))
diff_list = []
for i in range(1000):
    diff_list.append(output_list[i] - predict_list[i])

fig, ax1 = plt.subplots(figsize = (6.5,4))
ax2 = ax1.twinx()

plt.title('Monotonic Network Approximation of a Cubic Function', fontsize = 16)
ax1.plot(input_list, output_list, linestyle='-',color = 'b', linewidth=1.0, label='target function',)
ax1.fill_between(input_list, output_list, color='b',alpha =0.25)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 6)
ax1.set_ylabel("target function & approximation",fontsize = 12)

ax1.plot(input, predict_list, linestyle='--',color = 'r',linewidth=4.0, label='approximation',)
ax2.plot(input_list, diff_list,color = 'g', label ='target function - approximation')
ax2.plot([0,0],[1,0])
ax2.set_ylim(-0.1, 0.2)
ax2.set_yticks([-0.1,0,0.1,0.2])
ax2.get_yaxis().set_major_formatter(mpt.ticker.FormatStrFormatter('%.1f'))
ax2.set_ylabel("target function - approximation", fontsize = 12)
line = [(0, 0), (1, 0)]
(line_xs, line_ys) = zip(*line)
ax2.add_line(
    mpt.lines.Line2D(line_xs, line_ys, linestyle = 'dotted', linewidth=1.0, color='gray')
    )
ax1.legend(loc='best')
ax2.legend(loc='best')
plt.fill_between([0.025,0.08],[0.19,0.19],[0.175,0.175], color='b',alpha =0.25, interpolate = True )
plt.show()
