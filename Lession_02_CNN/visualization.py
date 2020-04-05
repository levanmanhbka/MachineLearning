import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_csv("log.csv")
print(data_frame.describe())

plt.plot(data_frame["loss"].to_numpy())
plt.legend("loss")
plt.show()