import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split #分開訓練及預測

from sklearn.ensemble import RandomForestClassifier #隨機森林
from sklearn.tree import DecisionTreeClassifier #決策樹模型
# from sklearn.model_selection import GridSearchCVf

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN #分群
from sklearn.preprocessing import StandardScaler, OneHotEncoder , RobustScaler , OrdinalEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# 模型評分
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error,silhouette_score
# 畫畫
import matplotlib
from data import Data
class Qus(Data):

    def __init__(self, source="Customer Data/customer_data.csv") -> pd.DataFrame:
        super().__init__(source)
        # mac os 使用字體
        plt.rcParams['font.sans-serif'] = ['Arial Unicode Ms']
        # windows 使用字體
        # matplotlib.rc('font', family='Microsoft JhengHei')



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

    def Q5(self,way = 1):
        """
        \n題目:
        \n針對顧客的重要特徵分群，找出2~3群最有特色的顧客，並解釋其價值與意義。
        \n過程:
        \n
        """
        way_dict = {
            1: "kmeans" ,
            2: "DBSCAN" ,
        }
        data = self.read()

        # 針對資料挑出一些特徵
        data = data[(data['網路服務'] ==  "Yes")& (data['每月費用'] > 0)]
        feature = data[['年齡','額外數據費用','線上安全服務','線上備份服務',' 額外長途費用','電視節目','電影節目','音樂節目','總收入','平均下載量( GB)']]

        categorical_features = ['線上安全服務','線上備份服務','電視節目','電影節目','音樂節目']
        # categorical_features = ['合約類型','城市','線上安全服務','線上備份服務','設備保護計劃','技術支援計劃']
        numeric_features = ['額外數據費用', ' 額外長途費用','總收入','平均下載量( GB)','年齡']

        # 使用 ColumnTransformer 進行特徵轉換
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numeric_features),
                ('cat', OrdinalEncoder(), categorical_features)
            ])
        transformed_data = preprocessor.fit_transform(feature)

        # 使用PCA進行降維

        # 將資料轉換並標準化
        # imputer = SimpleImputer(strategy='mean')
        # transformed_data = imputer.fit_transform(transformed_data)

        # 設定分群數目，這裡假設分成3群
        if way_dict[way] == "kmeans":
            self.__kmeans_group_choice(transformed_data,30)
            self.__Q5_kmeans(data,transformed_data)

        elif way_dict[way] == "DBSCAN":
            # 看不出結果
            self.__Q5_DBSCAN(data,transformed_data)
        else:
            raise KeyError(f"輸入參數{way}未知，請依照{way_dict}進行輸入")

    def Q7(self):
        """第七題:以郵遞區號為主，顧客資料中使用電話服務的顧客，其特性可能為何"""
        data = self.read()

        # 針對資料挑出一些特徵
        data = data[(data['電話服務 '] == "Yes") & (data['每月費用'] > 0)]
        feature = data[['年齡', '電話服務 ', '平均長途話費',
                        '多線路服務', ' 額外長途費用', '總費用', '郵遞區號']]

        categorical_features = ['電話服務 ', '多線路服務', '郵遞區號']
        # categorical_features = ['合約類型','城市','線上安全服務','線上備份服務','設備保護計劃','技術支援計劃']
        numeric_features = ['年齡', '總費用', ' 額外長途費用', '平均長途話費']

        # 使用 ColumnTransformer 進行特徵轉換
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numeric_features),
                ('cat', OrdinalEncoder(), categorical_features)
            ])
        transformed_data = preprocessor.fit_transform(feature)
        # self.__kmeans_group_choice(transformed_data,100)
        self.__Q5_kmeans(data, transformed_data, "第七題")


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
    def __Q5_detail(self, data):
        pass

    def __Q7_detail(self, data: pd.DataFrame):
        data = data[data['電話服務 '] == "Yes"]
        data = data[[
            'Cluster', '年齡', '平均長途話費', '多線路服務', ' 額外長途費用', '總費用', '郵遞區號'
        ]]
        data_age = data[['Cluster', '年齡']].groupby('Cluster').mean()
        data_people = data[['Cluster', '年齡']].groupby('Cluster').count()
        print(data_age)
        print(data_people)
        data_area = data[['Cluster', '總費用']].groupby("Cluster").sum()
        print(data_area)
        data_service = data[['Cluster', '郵遞區號', '多線路服務']
                            ].groupby('Cluster').value_counts()
        plt.figure(figsize=(20, 8))

        # data_area.plot.bar()
        data_service.plot.bar()

        plt.show()

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
    def __kmeans(self,origin_data,transformed_data,x_label="年齡",y_label="總收入"):
        """
        處理Kmeans分群
        
        Params:
        origin_data:原始資料
        transformed_data:特徵化資料
        x_label:圖x資料
        y_label:圖y資料
        """
        kmeans = KMeans(n_clusters=20, random_state=0,n_init=100,max_iter=1000,init="k-means++")
        # 適應模型
        kmeans.fit(transformed_data)
        # 新增分群結果到原始資料中
        origin_data['Cluster'] = kmeans.labels_
        # 顯示分群結果
        # 視覺化散點圖
        plt.figure(figsize=(12, 8))
        # 繪製散點圖
        sns.scatterplot(x=x_label, y=y_label, hue='Cluster', data=origin_data, palette='viridis', s=100)
        plt.title('K-means 分群')
        plt.xlabel('age');plt.ylabel('total')
        plt.legend()
        plt.show()
        # 對於分群後的資料進行處理
        origin_data.to_csv('res_5.csv')
        print("輸出結束")

    def __DBSCAN(self,origin_data,transformed_data):
        dbscan = DBSCAN(eps=3 , min_samples=2)  # 調整 eps 和 min_samples 參數
        origin_data['Cluster'] = dbscan.fit_predict(transformed_data)
        # 顯示分群結果
        plt.figure(figsize=(12, 8))
        # 繪製散點圖
        sns.scatterplot(x='年齡', y='總費用', hue='Cluster', data=origin_data, palette='viridis', s=100)
        plt.title('DBSCAN 分群')
        plt.xlabel('年齡')
        plt.ylabel('總費用')
        plt.legend()
        plt.show()

    def __kmeans_group_choice(self,transformed_data,group_num):
        res = []
        for i in range(2,group_num):
            kmeans = KMeans(n_clusters=int(i), random_state=0,n_init=100,max_iter=1000,init="k-means++")
            # 適應模型
            kmeans.fit(transformed_data)
            res.append(silhouette_score(transformed_data,kmeans.labels_))
        plt.plot(range(2,group_num),res)
        plt.title('elbow');plt.xlabel('No. cluster')
        plt.show()

    def test(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        your_data = self.read()
        # 選擇特徵
        selected_features = ['平均下載量( GB)', '每月費用']

        # 提取選擇的特徵
        data_for_clustering = your_data[selected_features]
        data_for_clustering = data_for_clustering.fillna(0)
        # 特徵標準化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)

        # 應用K-means，假設分3群
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(scaled_data)

        # 新增分群結果到原始數據
        your_data['Cluster'] = kmeans.labels_

        # 繪製散點圖
        plt.figure(figsize=(12, 8))

        # 繪製散點圖
        sns.scatterplot(x='平均下載量( GB)', y='每月費用', hue='Cluster', data=your_data, palette='viridis', s=100)

        # 添加中心點
        centers = scaler.inverse_transform(kmeans.cluster_centers_)  # 反標準化得到原始數據中心點

        # 添加中心點到散點圖
        plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', label='Cluster Centers')

        # 添加標籤和標題
        plt.title('K-means 分群結果 with Cluster Centers')
        plt.xlabel('平均下載量( GB)')
        plt.ylabel('每月費用')
        plt.legend()
        plt.show()

Qus().Q7()
# Qus().test()