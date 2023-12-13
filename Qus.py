import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# 分開資料
from sklearn.model_selection import train_test_split
# 決策數、分類模型
from sklearn.ensemble import RandomForestClassifier #決策數-隨機森林
from sklearn.cluster import KMeans,DBSCAN #分群模型
from sklearn.tree import DecisionTreeClassifier #決策樹
from sklearn.model_selection import GridSearchCV
# Encoder 特徵轉換
from sklearn.preprocessing import StandardScaler, OneHotEncoder , RobustScaler , OrdinalEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# 計算模型的準確率或支持度
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.metrics import silhouette_score
# 內部類別
from data import Data
class Qus(Data):

    def __init__(self, source="Customer Data/customer_data.csv") -> pd.DataFrame:
        super().__init__(source)
        # 字體設定使用 
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
    
    def Q6(self):
            from sklearn.ensemble import RandomForestClassifier #隨機森林
            from sklearn.tree import DecisionTreeClassifier #決策樹模型
            from sklearn.model_selection import GridSearchCV
            from sklearn.preprocessing import LabelEncoder
            from mlxtend.frequent_patterns import fpgrowth
            from mlxtend.frequent_patterns import association_rules
            from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
            """根據顧客年齡差異（分成老中青)，比較其使用公司服務的關聯規則的異同"""
            df = self.read()
            df = df[['年齡','電話服務 ','多線路服務', '網路服務', '線上安全服務', '線上備份服務', '設備保護計劃', '技術支援計劃', '電視節目', '電影節目','音樂節目', '無限資料下載', '無紙化計費']]
            
            
            # 18 ~ 35 歲為青年，36 ~ 55 歲為中年，我和起來變青中年；56 歲以上為老年
            def age_class(age):
                if age <= 55:
                    return '青中年'
                else:
                    return '老年'
            
            # 創建col放年齡分類後的結果，刪除年齡蘭為
            df['年齡層分類'] = df['年齡'].apply(age_class)
            df = df.drop('年齡', axis=1)

            # encode所有欄位
            servers = ['電話服務 ', '多線路服務', '網路服務', '線上安全服務', '線上備份服務', '設備保護計劃', '技術支援計劃', '電視節目', '電影節目','音樂節目', '無限資料下載', '無紙化計費']
            # 創建 LabelEncoder 實例
            label_encoder = LabelEncoder()
            # 將 '年齡層分類' 欄位轉換為數值型別
            df['年齡層分類'] = label_encoder.fit_transform(df['年齡層分類'])
            # 進行獨熱編碼
            df_encoded = pd.get_dummies(df, columns=servers)
            # 年齡層分類放在第一列
            age_class_column = df_encoded.pop('年齡層分類')
            df_encoded.insert(0, '年齡層分類', age_class_column)


            # 關聯規則

            # FP-Growth效率佳，所以我就選這個：）
            # FP-Growth找出支持度超過 0.5 的頻繁項目集（frequent itemsets）
            frequent_itemsets = fpgrowth(df_encoded, min_support=0.5, use_colnames=True)
            print(frequent_itemsets)
            print('-'*50)

            # 使用信心度（confidence）作為評估指標，並設定閾值為 0.5。返回所有信心度大於等於 0.5 的關聯規則。
            association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
            # 使用提升度（lift）作為評估指標，並設定閾值為 1.1。返回所有提升度大於等於 1.1 的關聯規則。
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)

            """
            support用於衡量資料中包含特定項目集的頻率或出現次數。
                support很高 -> 表示該項目集是一個常見的模式或關聯規則。

            confidence用於評估一個規則的可靠程度或準確性。用來衡量「如果出現 A，則出現 B 的條件概率」。
                =1 -> 當 A 出現時，B 一定會出現。
                <1 -> 當 A 出現時，B 出現的機率較低。
            
            lift表示兩個事件（項目集）之間的相關性程度，特別是在資料挖掘和關聯規則分析中。
                >1 -> A 和 B 之間存在正相關性，它們的出現是非獨立的。
                =1 -> A 和 B 之間沒有相互關係，它們的出現是獨立的。
                <1 -> A 和 B 之間存在負相關性，它們的出現是相斥的。
            """


            #make it as a data frame
            df_rules = pd.DataFrame(rules)
            df_rules.to_excel('./association_rules/rule_retail.xlsx')
            print(df_rules)

            """
            分析結果：
            1. 顧客有使用'無限資料下載服務'時，也一定會使用'網路服務'。
            2. 顧客同時使用'無限資料下載服務'與'電話服務'時，也一定會使用'網路服務'。
            """
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
        feature = data[['年齡層','額外數據費用','線上安全服務','線上備份服務',' 額外長途費用','電視節目','電影節目','音樂節目','總收入','平均下載量( GB)']]

        categorical_features = ['年齡層','線上安全服務','線上備份服務']
        # categorical_features = ['合約類型','城市','線上安全服務','線上備份服務','設備保護計劃','技術支援計劃']
        numeric_features = ['額外數據費用', ' 額外長途費用','總收入','平均下載量( GB)']

        # 使用 ColumnTransformer 進行特徵轉換
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numeric_features),
                ('cat', OrdinalEncoder(), categorical_features)
            ])
        transformed_data = preprocessor.fit_transform(feature)

        # 將資料轉換並標準化
        # imputer = SimpleImputer(strategy='mean')
        # transformed_data = imputer.fit_transform(transformed_data)

        # 設定分群數目，這裡假設分成3群
        if way_dict[way] == "kmeans":
            self.__Q5_kmeans_group_choice(transformed_data)
            self.__Q5_kmeans(data,transformed_data)

        elif way_dict[way] == "DBSCAN":
            self.__Q5_DBSCAN(data,transformed_data)
        else:
            raise KeyError(f"輸入參數{way}未知，請依照{way_dict}進行輸入")    
        
    def Q7(self):
        """以郵遞區號為主，找到各電視、音樂、電影節目的主要特徵"""
        
        data = self.read()
        data = data[['郵遞區號','年齡層','電視節目','電影節目','音樂節目']]
        
        
       
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
    def sample_clusster(self):
        """用於瞭解正常分群圖"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        from sklearn.datasets import make_blobs

        # 生成模擬數據
        data, labels = make_blobs(n_samples=300, centers=4, random_state=42)

        # 初始化KMeans模型，設置群組數量（假設我們知道有4個群組）
        kmeans = KMeans(n_clusters=4)
        

        # 適應模型並進行預測
        predicted_labels = kmeans.fit_predict(data)

        # 可視化結果
        plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
        plt.title('K-Means Clustering')
        plt.show()
    def __Q5_kmeans(self,origin_data,transformed_data):
        kmeans = KMeans(n_clusters=3, random_state=0,n_init=100,max_iter=1000,init="k-means++")
        # 適應模型
        kmeans.fit(transformed_data)
        # 新增分群結果到原始資料中
        origin_data['Cluster'] = kmeans.labels_
        # 顯示分群結果
        # 視覺化散點圖
        plt.figure(figsize=(12, 8))
        # 繪製散點圖
        sns.scatterplot(x='年齡', y='總費用', hue='Cluster', data=origin_data, palette='viridis', s=100)
        # 繪製中心點
        # centers = MinMaxScaler.inverse_transform(kmeans.cluster_centers_)  # 反標準化得到原始數據中心點
        # plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', label='Cluster Centers')
        plt.title('K-means 分群')
        plt.xlabel('age')
        plt.ylabel('total')
        plt.legend()
        plt.show()
    def __Q5_DBSCAN(self,origin_data,transformed_data):
        dbscan = DBSCAN(eps=3, min_samples=2)  # 調整 eps 和 min_samples 參數
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
    def __Q5_kmeans_group_choice(self,transformed_data):
        # 使用輪廓分析法找到最佳的集群數
        silhouette_scores = []
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(transformed_data)
            silhouette_scores.append(silhouette_score(transformed_data, kmeans.labels_))

        # 繪製輪廓分析法圖
        plt.plot(range(2, 11), silhouette_scores)
        plt.title('Silhouette Analysis')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
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
        
# Qus().Q7()
Qus().Q5()
# Qus().Q3(name='c_status')
