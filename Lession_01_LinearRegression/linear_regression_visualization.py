import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

# Give example data
# x_data = np.random.rand(1000)
# y_label = 4 + 3 * x_data + .2*np.random.randn(1000) # noise added
# x = x_data.tolist()

# dataset = pd.DataFrame(data = {'X':x_data.tolist(), 'Y':y_label.tolist()})
# print(dataset)
# dataset.to_csv('dataset.csv')

dataset = pd.read_csv('dataset.csv')
x_data = dataset['X'].to_numpy()
y_label = dataset["Y"].to_numpy()

fig = plt.figure('dataset')
ax2 = fig.add_subplot(122)
ax1 = fig.add_subplot(121)

# refer dataset and what y will be if x is 0.456781

# 1. chose model as linear expression: y = w0 + w1*x
# the goal is find out w0 and w1, them is called weights of model
# 2. chose loss function as mean squared error
# 3. using gradient descent to optimize chose loss to find out the best w0 and w1
# other word that find w0 and w1 so that loss is get mininum

history_proc = []
history_loss = [] # To save loss value during training process
# Training process
lr = 0.001  # The learning Rate
epochs = 10000  # The number of iterations to perform gradient descent

# init w0 and w1
w0 = np.random.rand()
w1 = np.random.rand()
print('Init weights')
print('w0=',w0, ' w1=', w1)

print('Train model')
for epoch in range(epochs):
    y_pred = w0 + w1 * x_data       #The current predicted value of Y
    
    gradient_by_w0 = (-2/len(y_label)) * sum(y_label - y_pred)             # Derivative of loss function  respect to w0
    gradient_by_w1 = (-2/len(y_label)) * sum(x_data * (y_label - y_pred))  # Derivative of loss function  respect to w1

    w0 = w0 - lr*gradient_by_w0     #Update w0
    w1 = w1 - lr*gradient_by_w1     #Update w1

    loss_value = (1/len(y_label) * sum((y_label - y_pred)*(y_label - y_pred)))
    print('epoch={} loss={}'.format(epoch,loss_value))
    history_loss.append(loss_value)
    
    if epoch % 10 == 0:
        ax1.set_xlabel('x_value')
        ax1.set_ylabel('y_value')
        ax1.scatter(x_data, y_label,color='blue')

        x = np.linspace(0, 1, num=1000)
        img = ax1.plot(x, w0 + w1*x, animated = True)
        history_proc.append(img)

        ax2.set_xlabel('epoch ' + str(epoch))
        ax2.set_ylabel('loss')
        img = ax2.plot(history_loss, animated = True)
        history_proc.append(img)
           


ani = animation.ArtistAnimation(fig, history_proc, interval=50, blit=True, repeat_delay=1000)
plt.show()

print('trained weights value')
print('w0=',w0, ' w1=', w1)
print('Find out model with Y = {} + {} * X'.format(w0, w1))

