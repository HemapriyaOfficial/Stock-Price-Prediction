# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset


## Design Steps

### Step 1:
Write your own steps

### Step 2:

### Step 3:



## Program
#### Name:
#### Register Number:
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out
    
model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the Model

# Write your code here







```

## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/user-attachments/assets/2bbc1f55-d515-46bb-9235-de472b16b7c7)


### Predictions 

![image](https://github.com/user-attachments/assets/d5a5dad6-0f56-4717-b348-c32a767b800a)


## Result


