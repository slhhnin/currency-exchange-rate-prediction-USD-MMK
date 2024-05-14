import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import torch
import os
import torch.onnx
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable

df_final = pd.read_csv("USD_MMK_Historical_Data.csv",index_col='Date',parse_dates=True)
df_final= df_final[::-1]
df_final.index = pd.to_datetime(df_final.index)
price = df_final.Price



def scaling_window(data, seq_length, index):
    x = []
    y = []
    index_value = []
    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        index_value.append(index[i + seq_length])
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y), index_value

sc = MinMaxScaler()
training_data = sc.fit_transform(df_final[['Price']])

seq_length = 30
#x, y = scaling_window(training_data, seq_length)
x, y , new_index_value = scaling_window(training_data, seq_length, df_final.index)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size


dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))
print(dataX.shape)
trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
print('train_size', trainX.shape)
print('test_size', testX.shape)

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)

        return out


num_epochs = 2000
learning_rate = 0.001
input_size = 1
hidden_size = 60
linear_layer = 60
num_layers = 1
output_size = 1
seq_length = 30

lstm = LSTM(output_size, input_size, hidden_size, num_layers)
print(lstm)
criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

count=0
loss_list =[]
# Train the model
if(not os.path.exists('stock_price_test0.2_val0.1.pth')):
 for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()

        # obtain the loss function
    loss = criterion(outputs, trainY)

    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        loss_list.append(loss.item())
    torch.save(lstm.state_dict(), 'stock12.pth')


lstm.eval()

model = LSTM(output_size, input_size, hidden_size, num_layers)
model.load_state_dict(torch.load('stock_price_test0.2_val0.1.pth'))

root = tk.Tk()

test = model(testX)
test =test.detach().numpy()
testScore =mean_squared_error(testY, test, squared=False)
print('Test Score: %.2f RMSE' % (testScore))

canvas1 = tk.Canvas(root, width=1300, height=370)
filename = tk.PhotoImage(file ='trading_2.png')
background_label = tk.Label(root, image=filename)
background_label.place(x=1, y=1, relwidth=1, relheight=1)
canvas1.pack()

def pred_charts():
    global x2
    global x3
    global bar2
    x2 = dataY_plot
    x3 = data_predict

    fmt = mdates.DateFormatter('%Y/%m/%d')
    figure2 = Figure(figsize=(4,3), dpi=100)
    subplot2 = figure2.add_subplot(111)
    yAxis = x2
    zAxis = x3
    subplot2.plot(new_index_value,yAxis,label='Actual Price')
    subplot2.plot(new_index_value, zAxis,label='Predicted Price')
    subplot2.set_xlabel('Date')
    subplot2.set_ylabel('Currency Rate')

    subplot2.set_title('Real and Predicted Price of USD/MMK')
    subplot2.xaxis.set_major_formatter(fmt)
    figure2.autofmt_xdate()
    subplot2.legend()
    subplot2.grid(True)
    bar2 = FigureCanvasTkAgg(figure2, root)
    bar2.get_tk_widget().pack( side=tk.LEFT,fill=tk.BOTH, expand=0)


def forecast():
  all = data_X
  DAYS_TO_PREDICT = int(numdays.get())

  with torch.no_grad():
    test_seq = all[-1:]
    preds = []
    for _ in range(DAYS_TO_PREDICT):
      y_test_pred = model(test_seq)
      pred = torch.flatten(y_test_pred).item()
      preds.append(pred)
      new_seq = test_seq.numpy().flatten()
      new_seq = np.append(new_seq, [pred])
      new_seq = new_seq[1:]
      test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
  predicted_cases = sc.inverse_transform(
        np.expand_dims(preds, axis=0)
    ).flatten()
  print(test_seq)

  predicted_index = pd.date_range(
    start=df_final.index[-1],
    periods=DAYS_TO_PREDICT ,
    #closed='right'
   )

  predicted_cases = pd.Series(
    data=predicted_cases,
    index=predicted_index)
  print(predicted_cases)
  global bar3
  x4 = predicted_cases
  x5 = df_final[['Price']]

  fmt = mdates.DateFormatter('%Y/%m/%d')
  figure3 = Figure(figsize=(4,3), dpi=100)
  subplot3 = figure3.add_subplot(111)
  rAxis = x4
  tAxis = x5
  subplot3.plot(predicted_index, rAxis, label='Forecast Price')
  #subplot3.plot(new_index_value, tAxis,label='Actual Price')
  subplot3.set_xlabel('Date')
  subplot3.set_ylabel('Currency Rate')
  subplot3.set_title('Actual and Forecast Price of USD/MMK')
  subplot3.legend()
  subplot3.grid(True)
  subplot3.xaxis.set_major_formatter(fmt)
  figure3.autofmt_xdate()

  bar3 = FigureCanvasTkAgg(figure3, root)
  bar3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=0)


  # global label5
  # label5 = tk.Label(root, text=predicted_cases)
  # label5.config(fg='black', bg ='ghost white',font=('Arial', 11,'bold'))
  # canvas1.create_window(800, 486, window=label5)


def clear_charts():
    bar2.get_tk_widget().pack_forget()
    bar3.get_tk_widget().pack_forget()
    #label5.destroy()

def import_csv_data():
    global v, df
    csv_file_path = askopenfilename()
    print(csv_file_path)
    v.set(csv_file_path)
    df = pd.read_csv(csv_file_path,index_col='Date',parse_dates=True )
    return df, v


label1 = tk.Label(root, text='Currency Exchange Rate Prediction Using LSTM Network', font=('Times New Roman', 20, 'bold'))
label1.config(font=('Times New Roman', 20, 'bold'))
canvas1.create_window(700, 70, window=label1)

label2 = tk.Label(root, text='      File Path     ',  font=("Times New Roman", 13, 'bold'))#font=('Times New Roman', 20, 'bold')
canvas1.create_window(140, 180, window=label2)
v = tk.StringVar()

entry = tk.Entry(root, width=14, textvariable=v, font=("Times New Roman", 12))
canvas1.create_window(138, 225, window=entry)

button = tk.Button(root, text='Browse Data Set',  bg='#005700',  fg='white',command=import_csv_data, font=("Times New Roman", 12, 'bold'))# bg='medium sea green',
canvas1.create_window(270, 225, window=button)

df, v = import_csv_data()
df= df[::-1]
test_set = df[['Price']]

sc = MinMaxScaler()
testing_data = sc.fit_transform(test_set)

seq_length = 30
#x, y = scaling_window(testing_data, seq_length)
x, y , new_index_value = scaling_window(testing_data, seq_length, df.index)
data_X = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

train_predict = model(data_X)
data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)
testScore2 = mean_squared_error(dataY_plot, data_predict, squared=False)
print('Test Score: %.2f RMSE' % (testScore2))

button2 = tk.Button(root, text='     Predict      ', command=pred_charts,  bg='#005700',  fg='white', font=("Times New Roman", 12, 'bold'))
canvas1.create_window(200, 270, window=button2)


label3= tk.Label(root, text='  Enter Forecast Days  ',  font=("Times New Roman", 13, 'bold'))
canvas1.create_window(600, 180, window=label3)

numdays = tk.Entry(root,width=20, font=("Times New Roman", 12))
canvas1.create_window(600, 225, window=numdays)

button3 = tk.Button(root, text='    Forecast    ', command=forecast, bg='#005700',  fg='white', font=("Times New Roman", 12, 'bold'))
canvas1.create_window(600, 270, window=button3)



button4 = tk.Button (root, text='  Clear Charts  ', command=clear_charts, bg='#adadad',  font=("Times New Roman", 12, 'bold'))
canvas1.create_window(300, 320, window=button4)


label6 = tk.Label(root, text='      Currency Converter       ')
label6.config(font=("Times New Roman", 13, "bold"))
canvas1.create_window(1000, 180, window=label6)



OPTIONS = {
    "Australian Dollar": 49.10,
    "US Dollar": 0.00071,
    "Euro": 77.85,
    "myanmar": 1405.440
}

def ok():
    global y,x
    if(mmk.get()):
        USget = int(mmk.get())* OPTIONS["US Dollar"]
        USget = round(USget, 3)
        y.set(USget)
        #result.delete(1.0, tk.END)
        #result.insert(tk.INSERT, "Price In  ", tk.INSERT, int(mmk.get()), tk.INSERT,"   MMKs =  ", tk.INSERT, USget,tk.INSERT,"   USD")
        USdollar.insert(tk.INSERT, USget)
    else:
        answer =int(USdollar.get())
        myanmarget = answer * OPTIONS["myanmar"]
        myanmarget = round(myanmarget, 3)
        x.set(myanmarget)
        #result.delete(1.0, tk.END)
        #result.insert(tk.INSERT, "Price In  ", tk.INSERT, answer, tk.INSERT, "  USD = ", tk.INSERT, myanmarget,tk.INSERT,"  MMKs")
        mmk.insert(tk.INSERT, myanmarget)

x = tk.StringVar()
y = tk.StringVar()
# mmkyats = tk.Label(root, text="MMK", font=("Times New Roman", 11, 'bold'),bg="light steel blue",bd=5)
# canvas1.create_window(1100, 230, window=mmkyats)



# US = tk.Label(root, text="  USD  ", font=("Times New Roman", 11,"bold"),bg="light steel blue",bd=5)
# canvas1.create_window(1100, 270, window=US)




img = tk.PhotoImage(file="C:/Users/User/Downloads/12.png")
label7 = tk.Label(root, image=img)
canvas1.create_window(900, 270, window=label7)

USdollar = tk.Entry(root, font=("Times New Roman", 12))#,text=y)
canvas1.create_window(1010, 270, window=USdollar)

image = tk.PhotoImage(file="C:/Users/User/Downloads/13.png")
label8 = tk.Label(root, image=image)
canvas1.create_window(900, 230, window=label8)

mmk = tk.Entry(root, font=("Times New Roman", 12))#, text=x)
canvas1.create_window(1010, 230, window=mmk)


button = tk.Button(root, text=" Convert ",bg='#005700',  fg='white',font=("Times New Roman", 12, 'bold'), command=ok)
canvas1.create_window(950, 320, window=button)
def clear_chart():
    mmk.delete(0, 'end')
    USdollar.delete(0,'end')
    #result.delete(1.0,tk.END)

button7 = tk.Button(root, text='   Clear   ', command=clear_chart, bg='#adadad', font=("Times New Roman", 12, 'bold'))
canvas1.create_window(1040,320, window=button7)

button5 = tk.Button (root, text='  Exit Application ', command=root.destroy, bg='#adadad', font=("Times New Roman", 12, 'bold'))
canvas1.create_window(1030, 380, window=button5)

root.mainloop()