# python -m streamlit run "c:\Users\AMRINDER\Desktop\Project.py"
from tabulate import tabulate
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,GridSearchCV,RandomizedSearchCV,cross_val_score,ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA,KernelPCA
import streamlit as st
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler,MaxAbsScaler,OneHotEncoder,LabelEncoder
import io
from datetime import datetime
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
st.title('🏥 Healthcare Billing Prediction App')
st.write('''
This Streamlit-based machine learning app predicts hospital billing amounts using patient details like gender, blood type, medical condition, insurance provider, and medication. It helps healthcare administrators estimate costs in advance, improving budgeting, transparency, and trust with patients.

Key Features:

🎯 Accurate Cost Prediction using XGBoost Regressor

🛠️ Interactive Hyperparameter Tuning for model customization

🧠 Automated Preprocessing Pipeline with OneHotEncoding and RobustScaler

📊 Model Evaluation with R² Score and Mean Squared Error

🔁 Cross-validation to ensure robust performance

Real-World Challenges Solved:

💸 Unpredictable Medical Costs: Patients often face billing surprises. This app enables cost transparency by predicting likely expenses ahead of time.

🏥 Resource Planning: Hospitals can use this tool to forecast average billing trends, helping in insurance negotiations and internal budgeting.

🤝 Insurance Alignment: By analyzing patterns across different insurance providers, the model can support policy-making and cost optimization.''')
st.divider()
st.title("Heart Cleavland Dataset")
df=pd.read_csv("healthcare_dataset.csv")
st.header("Dataset")
st.code(df)
st.divider()
st.header("Data Insights")
cols=st.columns(3)
with cols[0]:
    st.subheader("Missing Values")
    st.code(df.isna().sum())
with cols[1]:
    st.subheader("Columns")
    for i in range(0, len(df.columns) - 1, 2):  
        col1, col2 = st.columns(2)
        with col1:
            st.code(df.columns[i])
        with col2:
            st.code(df.columns[i+1])
with cols[2]:
    st.subheader("Data Types")
    st.code(df.dtypes)
st.divider()
st.subheader("Data Description")
st.code(tabulate(df.describe().transpose(), headers='keys', tablefmt='fancy_grid'))
print(df.columns)
df.drop(['Name','Doctor','Hospital','Room Number'],axis=1,inplace=True)
days=[]
df['Date of Admission']=pd.to_datetime(df['Date of Admission'])
df['Discharge Date']=pd.to_datetime(df['Discharge Date'])
df['Hospitalized Days']=(df['Discharge Date']-df['Date of Admission']).dt.days
df['Cost_per_day'] = df['Billing Amount'] / df['Hospitalized Days']
df.drop(['Date of Admission', 'Discharge Date'], axis=1, inplace=True)
df['Admission Type'] = df['Admission Type'].map({'Elective': 1, 'Urgent': 2, 'Emergency': 3})
df['Risk Score'] = df['Age'] * df['Admission Type']
df['Test Results']=df['Test Results'].map({'Normal':0,"Inconclusive":1,"Abnormal":2})
st.markdown("""  
###  Feature Engineering  
```python  
df.drop(['Name','Doctor','Hospital','Room Number'],axis=1,inplace=True)  
days=[]  
df['Date of Admission']=pd.to_datetime(df['Date of Admission'])  
df['Discharge Date']=pd.to_datetime(df['Discharge Date'])  
df['Hospitalized Days']=(df['Discharge Date']-df['Date of Admission']).dt.days  
df['Cost_per_day'] = df['Billing Amount'] / df['Hospitalized Days']  
df.drop(['Date of Admission', 'Discharge Date'], axis=1, inplace=True)  
df['Admission Type'] = df['Admission Type'].map({'Elective': 1, 'Urgent': 2, 'Emergency': 3})  
df['Risk Score'] = df['Age'] * df['Admission Type']  
df['Test Results']=df['Test Results'].map({'Normal':0,"Inconclusive":1,"Abnormal":2})  
regressive_cols=['Age','Billing Amount','Hospitalized Days','Cost_per_day','Risk Score']
classifier_cols=['Admission Type','Test Results']
""")
regressive_cols=['Age','Billing Amount','Hospitalized Days','Cost_per_day','Risk Score']
classifier_cols=['Admission Type','Test Results']
st.divider()
st.header("Data Visualization")
fig,axs=plt.subplots(2,3,figsize=(20,10))
ax=axs.flatten()
for i in range(len(regressive_cols)):
    sns.boxplot(data=df,y=regressive_cols[i],ax=ax[i])
    ax[i].set_xlabel(f"{regressive_cols[i]}")
    
plt.suptitle("Boxplot of Different features")
st.pyplot(fig)

st.divider()
fig1=sns.pairplot(data=df[regressive_cols],kind='hist')
st.pyplot(fig1)
st.divider()
fig2,axs=plt.subplots(1,2,figsize=(20,8))
ax=axs.flatten()
for i in range(len(classifier_cols)):
    sns.countplot(x=df[classifier_cols[i]],ax=ax[i])
    ax[i].set_xlabel(f"{classifier_cols[i]}")
    ax[i].set_ylabel("Count")
plt.suptitle("Barplot of categorial features")
st.pyplot(fig2)
st.divider()



fig3, ax = plt.subplots(figsize=(10, 6))  
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig3)  

objects=df.select_dtypes(include='object').columns



st.divider()
st.title("Model Training")

x=df.drop("Billing Amount",axis=1)
y=df['Billing Amount']
st.header("XGBoost Regressor")
st.subheader("Hyperparameters")
eval_metric=st.pills("Evaluation Metric",['logloss', 'rmse', 'mse', 'error']) or 'mse'
cols=st.columns(4)


with cols[0]:
    n_estimators=st.number_input("Number of estimators",value=100,step=50)
    learning_rate=st.number_input("Learning Rate",value=0.05,step=0.01)
with cols[1]:
    max_depth=st.number_input("Max depth",value=3,step=1)
    gamma=st.number_input("Gamma",value=0,step=1)
    reg_alpha=st.slider("Reg_Alpha",min_value=0.0,max_value=10.0,value=0.0,step=0.1)
with cols[2]:
    subsample=st.number_input("Sub Sample",value=0.7,step=0.1)
    colsample_bytree=st.number_input("Max features per tree",value=1.0,step=0.1)
    reg_lambda=st.slider("Reg_lambda",min_value=0.0,max_value=10.0,value=1.0,step=0.1)

pipeline=Pipeline([
    ("scale",
    ColumnTransformer([
        ("Encode",OneHotEncoder(),objects),
        ("Robust",RobustScaler(),['Cost_per_day'])
    ],
    remainder='passthrough')),
    ("Train",XGBRegressor(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,gamma=gamma,subsample=subsample,colsample_bytree=colsample_bytree,reg_alpha=reg_alpha,reg_lambda=reg_lambda))
    
])


st.code(pipeline)
sss=ShuffleSplit(n_splits=5,test_size=0.2,random_state=42)
score=[]
for i,(train_index,test_index) in enumerate(sss.split(x,y)):
    x_train,y_train=x.iloc[train_index],y.iloc[train_index]
    x_test,y_test=x.iloc[test_index],y.iloc[test_index]
    model=pipeline.fit(x_train,y_train)
    y_pred=pipeline.predict(x_test)
    st.write(
            f"[`{i + 1:2}`] **LOSS**: `{mean_squared_error(y_test, y_pred):.3f}` **ACCURACY**: `{r2_score(y_test, y_pred):.3f}`"
    )
    accu=r2_score(y_test, y_pred)
    score.append(accu)
st.code(f"Mean of r2 score is {np.mean(score)}")
st.header("Classification Report")
st.code(classification_report(y_test,y_pred))
joblib.dump(pipeline,'ml_pipeline.pkl')

