import numpy as np 
import tulipy 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("./DolarSarrafiMelliKharid1.csv")
data.columns = ['datetime', 'min','open','close','max','value']
print(data)

arr = data['close']
arr = np.float64(arr.to_numpy())
print(type(arr))

#14 means 2 weeks 
TWO_WEEKS = 14
out = tulipy.rsi(arr, TWO_WEEKS)
print(out)

############################
SHORT_P = 12
LONG_P = 26
SIGNAL = 9
out2 = tulipy.macd(real=arr, short_period=SHORT_P, long_period=LONG_P, signal_period=SIGNAL)
#3th element is difference between first and second 

plt.figure(1)
plt.plot(arr)

plt.figure(2)
plt.plot(out)

plt.figure(3)
plt.plot(out2[0], label = "macd")
plt.plot(out2[1], label = "signal")
plt.plot(out2[2], label = "histogram")
plt.legend()
plt.show()
