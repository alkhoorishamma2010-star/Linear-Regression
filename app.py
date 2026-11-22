# ---------------------------------------------------------
# Simple Linear Regression App (for Grade 10)
# Predict Salary from Years of Experience
# ---------------------------------------------------------

# 1. IMPORT LIBRARIES
import streamlit as st                      # builds the web page (UI)
import pandas as pd                         # works with tables of data
import matplotlib.pyplot as plt             # draws graphs
from sklearn.model_selection import train_test_split   # splits data
from sklearn.linear_model import LinearRegression      # the ML model
from sklearn.metrics import mean_squared_error, r2_score  # checks accuracy

# 2. PAGE TITLE
st.title("📈 Simple Linear Regression")
st.write("Let's teach a computer to **predict Salary from Years of Experience**!")

# 3. THE DATA (built in, so no CSV file is needed)
data = {
    "YearsExperience": [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7,
                        3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0,
                        6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5],
    "Salary": [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189,
               63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940,
               91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872],
}
df = pd.DataFrame(data)

# Optional: students can upload their own CSV with 2 number columns
uploaded = st.file_uploader("Or upload your own CSV (2 columns: X, y)", type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)

x_col = df.columns[0]   # first column  = input (X)
y_col = df.columns[1]   # second column = output (y)

# 4. STEP 1 – LOOK AT THE DATA
st.header("Step 1: Look at the data")
st.dataframe(df.head(10))
st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")

fig1, ax1 = plt.subplots()
ax1.scatter(df[x_col], df[y_col], color="blue")
ax1.set_xlabel(x_col)
ax1.set_ylabel(y_col)
ax1.set_title("Do the dots follow a straight line?")
st.pyplot(fig1)

# 5. STEP 2 – SPLIT INTO TRAINING AND TESTING
st.header("Step 2: Split into training and testing")
test_percent = st.slider("How much data to keep for testing (%)", 10, 50, 20, step=5)

X = df[[x_col]]          # double brackets -> 2D table, which sklearn needs
y = df[y_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_percent / 100, random_state=0
)
st.write(f"Training rows: **{len(X_train)}** | Testing rows: **{len(X_test)}**")

# 6. STEP 3 – TRAIN THE MODEL
st.header("Step 3: Train the model")
model = LinearRegression()
model.fit(X_train, y_train)          # the computer learns m and c here

m = model.coef_[0]                   # slope
c = model.intercept_                 # intercept
st.latex(r"y = m \times x + c")
st.success(f"Our model learned:  {y_col} = {m:.2f} × {x_col} + {c:.2f}")

# 7. STEP 4 – TEST THE MODEL
st.header("Step 4: How good is our model?")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("R² score (closer to 1 is better)", f"{r2:.2f}")
col2.metric("Mean Squared Error (smaller is better)", f"{mse:,.0f}")

fig2, ax2 = plt.subplots()
ax2.scatter(X_train, y_train, color="blue", label="Training data")
ax2.scatter(X_test, y_test, color="green", label="Testing data")
ax2.plot(X, model.predict(X), color="red", linewidth=2, label="Best-fit line")
ax2.set_xlabel(x_col)
ax2.set_ylabel(y_col)
ax2.legend()
st.pyplot(fig2)

# 8. STEP 5 – MAKE YOUR OWN PREDICTION
st.header("Step 5: Try it yourself! 🎯")
user_x = st.number_input(f"Enter {x_col}:", value=5.0, step=0.5)
prediction = model.predict(pd.DataFrame({x_col: [user_x]}))[0]
st.info(f"Predicted {y_col}: **{prediction:,.2f}**")
