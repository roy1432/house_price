import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

#loading data set
df = pd.read_csv("data/houserates.csv")

#input data
X = df[["area","bed_rooms","age"]]

#output
y = df['price']
#create model
model = LinearRegression()
#trin the model with input and output features
model.fit(X,y)
#display learned parameters
print("Slope - m(weight) = ",model.coef_)
print("Intercept - c (constant) = ",model.intercept_)
#Test Prediction
sq_feet = [[1500,3,5]]
pred_price = model.predict(sq_feet)
print("Predicted Price = ",pred_price[0])
#save the model
joblib.dump(model,"model/house_price_model.pkl")
print("Model Saved Successfully")