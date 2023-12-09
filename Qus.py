import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #隨機森林
from sklearn.tree import DecisionTreeClassifier #決策樹模型
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from data import Data
class Qus(Data):
    
    def __init__(self, source="Customer Data/customer_data.csv") -> pd.DataFrame:
        super().__init__(source)
        plt.rcParams['font.sans-serif'] = ['Arial Unicode Ms']

    def Q2(self):
        """
        針對所有城市，計算總收入前三高城市．個別分析其欄位
        """
        df = self.read()
        df_total = df[['城市','總收入','總費用','總退款','額外數據費用',' 額外長途費用']]
        df_total = df_total.groupby('城市').sum().sort_values('總收入',ascending=False)
        # 輸出前三名的城市
        df_total = df_total.head(3)
        # Los Angeles  San Diego Sacramento
        top3_city = ['Los Angeles','San Diego','Sacramento']
        
        df = df[['城市'] in top3_city]
        # 網路、電話
        df_net = df[['網路服務','網路連線類型','平均下載量( GB)','線上安全服務','線上備份服務','設備保護計劃','技術支援計劃','電視節目','電影節目','音樂節目','無限資料下載','合約類型','支付帳單方式','每月費用','總費用','總退款','額外數據費用', '額外長途費用']]
        
        """畫圖"""
        df_total.value_counts()
        df_total = df_total.reset_index()
        bar_pic = df.plot.bar()
        plt.title('總收入前三名城市其數據')
        plt.ylabel('金額');plt.xlabel('城市')
        plt.grid(axis='y')
        plt.show()
        """
        過程：
        先將清洗完的資料以‘城市’為組別進行加總並透過排序取出總收入前三名的城市
        再將資料進行視覺化輸出．
        首先可從圖表中看出
        Los Angeles 為第一
        San Diego 為第二
        Sacramento 為第三
        其中可以看到Los Angeles 和 San Deigo的總費用及總收入的相差低
        (可考慮使用其他欄位進行變化)
        """
        
    def Q3(self,name):
        """
        根據顧客的狀態建立相關的決策數與規則
        \n並分析及評估決策樹的效能等相關指標
        """
        data = self.for_q2()
        data = data.dropna();print(data.shape)
        # 假設資料存在一個名為data的DataFrame中，並且目標是預測'客戶流失類別'
        # 請將特徵變數和目標變數分開，利用參數將特定資料進行篩選
        def connect_type():
            X = data.drop(['網路連線類型'], axis=1)
            y = data['網路連線類型']
            return X,y
        def page():
            X = data.drop(['合約類型'], axis=1)
            y = data['合約類型']
            return X,y
        def customer_status():
            X = data.drop(['客戶狀態'], axis=1)
            y = data['客戶狀態']
            
            return X,y
        if name == "connect_type":
            X,y = connect_type()
        elif name == "page":
            X,y = page()
        elif name =="c_status":
            X,y = customer_status()
        # 利用讀熱編碼，將類別資料轉換為數字
        X_encoded = pd.get_dummies(X)
        
        # 切割資料集為訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y)
        
        # 是用最佳參數進行訓練
        params = {'criterion': 'gini', 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
       
        # 建立決策樹模型
        model = RandomForestClassifier(**params)
        
        model.fit(X_train, y_train)
        self.__cross(model=model,X=X_encoded,y=y)
        # 預測測試集
        y_pred = model.predict(X_test)
        # 評估模型
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f'模型準確度：{accuracy}')
        


    def __get_best_params(self,X,y):
        """此方法用於找出隨機森林的最優參數利於模型訓練"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # 創建隨機森林分類器
        rf = RandomForestClassifier()

        # 使用網格搜索來尋找最佳參數組合
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
        grid_search.fit(X, y)

        # 最佳參數
        best_params = grid_search.best_params_
        print("最佳參數:", best_params)
    
    def __lookdatascale(self,X_test,X_train):
        """查看訓練資料及預測資料是否比例使用一致，避免模型過度擬合的狀況"""
        import pandas as pd 
        print(pd.Series(X_train).value_counts(normalize=True))
        print(pd.Series(X_test).value_counts(normalize=True))
    def __cross(self,model,X,y):
        """用於交叉驗證資料"""
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=5)
        print("交叉驗證準確度：", scores.mean())
    def __a(self,y_test,y_pred):
        """混肴矩陣"""
        from sklearn.metrics import confusion_matrix
        print('混淆矩陣')
        print(confusion_matrix(y_test,y_pred))
Qus().Q2()
# Qus().Q3(name='c_status')