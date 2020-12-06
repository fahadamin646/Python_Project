import pandas as pd
import pickle
def getPrediction(Matric,Inter,Degree,FO,Smoke):
    lst=[[Matric,Inter,Degree,FO,Smoke]]
    df=pd.DataFrame(lst,columns=['cancer','diabetes','heart_disease','belly','smoker'])
    with open('stand_scalar', 'rb') as f:
        sc=pickle.load(f)
    with open('model', 'rb') as f:
        ppn = pickle.load(f)
    dataf=sc.transform(df)
    pred=ppn.predict(dataf)
    return str(pred[0])

