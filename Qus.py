import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split #分開訓練及預測

from sklearn.ensemble import RandomForestClassifier #隨機森林
from sklearn.tree import DecisionTreeClassifier #決策樹模型
from sklearn.model_selection import GridSearchCV

from sklearn.impute import SimpleImputer
# 降維
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN #分群
from sklearn.preprocessing import StandardScaler, OneHotEncoder , RobustScaler , OrdinalEncoder,MinMaxScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 關聯
from mlxtend.frequent_patterns import apriori,association_rules,fpgrowth

# 模型評分
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error,silhouette_score
# 畫畫
import matplotlib
# inner
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

        df = df[df['城市'].isin(top3_city)]
        # 數值特徵資料
        df_num = df[['城市','每月費用','總費用','總退款','總收入']]
        df_num = df_num.groupby('城市').sum() ; print(df_num)
        # 類別特徵資料
        df_cat_net = df[['城市','網路連線類型','線上安全服務','線上備份服務','設備保護計劃','技術支援計劃']]
        df_cat_tv = df[['城市','電視節目','電影節目','音樂節目','無限資料下載']]
        df_cat_pho = df[['城市','電話服務 ','多線路服務']]
        """畫圖"""
        df_total.value_counts()
        df_total = df_total.reset_index()
        df_num.plot.bar()
        plt.title('總收入前三名城市其數據')
        plt.ylabel('金額');plt.xlabel('城市')
        plt.grid(axis='y')
        plt.show()
        # 各服務的人數
        df_people_use = df[['城市','電話服務 ','網路服務']]
        df_people_use = df_people_use.groupby('城市').value_counts()
        df_people_use = df_people_use.unstack(0).fillna(0)
        print(df_people_use)
        df_people_use.plot.bar()
        plt.title('前三名城市＿各服務使用比例');plt.ylabel('人數')
        plt.grid(axis='y')
        plt.show()
        # 電話服務
        df_cat_pho = df_cat_pho.groupby('城市').value_counts()
        df_cat_pho = df_cat_pho.unstack()
        df_cat_pho.plot.bar()
        plt.title('前三名城市＿電話使用分佈比例')
        plt.grid(axis='y')
        plt.ylabel('人數')
        plt.show()
        # 網路服務
        df_cat_net[['城市','網路連線類型']].groupby('城市').value_counts().unstack().plot.bar()
        plt.title("前三名城市_網路連線類型")
        plt.ylabel('人數')
        plt.show()
        df_cat_net = df_cat_net[['城市','線上安全服務','線上備份服務','設備保護計劃','技術支援計劃']].groupby('城市').value_counts()
        df_cat_net = df_cat_net.unstack(0)
        print(df_cat_net)
        df_cat_net.plot.bar()
        plt.title('前三名城市＿網路服務分佈比例')
        plt.ylabel('人數')
        plt.grid(axis='y')
        plt.show()
        # 娛樂服務
        df_cat_tv = df_cat_tv.groupby('城市').value_counts()
        df_cat_tv = df_cat_tv.unstack(0)
        df_cat_tv.plot.bar()
        plt.grid(axis='y')
        plt.title('前三名城市＿娛樂服務分佈比例')
        plt.ylabel('人數')
        plt.show()
        # 付款方式
        df_cat_payment = df[['城市','支付帳單方式']]
        df_cat_payment = df_cat_payment.groupby('城市').value_counts().unstack(0)
        df_cat_payment.plot.bar()
        plt.title('前三名城市＿支付方式分佈比例');plt.ylabel('人數');plt.grid(axis='y')
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
    def Q4(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        from mlxtend.frequent_patterns import apriori
        from mlxtend.frequent_patterns import association_rules
        from matplotlib.font_manager import FontProperties
        # 讀取資料
        df = pd.read_csv('./change_Final.csv', encoding='utf-8')

        feature = df[['經度','緯度 ']]

        # 使用K-means演算法分成兩群
        kmeans = KMeans(n_clusters=2, random_state=42)
        df['Cluster'] = kmeans.fit_predict(feature)

        # 顯示分群結果
        print(df[['客戶編號', '緯度 ', '經度', 'Cluster']])
        df[['客戶編號', '緯度 ', '經度', 'Cluster']].to_csv("5.csv")

        # 繪製分群結果的散點圖
        plt.scatter(df['緯度 '], df['經度'], c=df['Cluster'], cmap='viridis')
        plt.title('geographical grouping')
        plt.xlabel('latitude')
        plt.ylabel('longitude ')
        plt.show()
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        plt.rcParams['axes.unicode_minus'] = False

        # 透過群組分類取得兩個群的資料
        group_0_data = df[df['Cluster'] == 0]
        group_1_data = df[df['Cluster'] == 1]

        # 比較年齡的差異
        age_comparison = pd.DataFrame({
            'Group 0': [group_0_data['年齡'].mean()],
            'Group 1': [group_1_data['年齡'].mean()]
        }, index=['平均年齡'])

        print("\n平均年齡比較:")
        print(age_comparison)
        plt.bar(age_comparison.columns, age_comparison.iloc[0], color=['lightcoral', 'lightskyblue'])
        plt.xlabel('群組')
        plt.ylabel('平均年齡')
        plt.title('群組間年齡比較')
        plt.show()
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        plt.rcParams['axes.unicode_minus'] = False

        # 比較婚姻狀況的差異
        marital_status_comparison = pd.crosstab(df['婚姻'], df['Cluster'])
        print("\n婚姻狀況比較:")
        print(marital_status_comparison)
        marital_status_comparison.plot(kind='bar', stacked=True, color=['lightcoral', 'lightskyblue'])
        plt.xlabel('群組')
        plt.ylabel('人數')
        plt.title('群組間婚姻狀況比較')
        plt.show()
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        plt.rcParams['axes.unicode_minus'] = False

        # 比較扶養人數的差異
        dependents_comparison = pd.DataFrame({
            'Group 0': [group_0_data['扶養人數'].mean()],
            'Group 1': [group_1_data['扶養人數'].mean()]
        }, index=['平均扶養人數'])

        print("\n平均扶養人數比較:")
        print(dependents_comparison)
        # 绘制圆饼图
        plt.bar(dependents_comparison.columns, dependents_comparison.iloc[0], color=['lightcoral', 'lightskyblue'])
        plt.xlabel('群組')
        plt.ylabel('平均扶養人數')
        plt.title('群組間平均扶養人數比較')
        plt.show()
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        plt.rcParams['axes.unicode_minus'] = False

        # 比較電話服務的差異
        promotion_comparison = pd.crosstab(df['電話服務 '], df['Cluster'])
        print("\n電話服務比較:")
        print(promotion_comparison)
        promotion_comparison.plot(kind='bar', stacked=True, color=['lightcoral', 'lightskyblue'])
        plt.xlabel('群組')
        plt.ylabel('人數')
        plt.title('群組間電話服務比較')
        plt.show()
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        plt.rcParams['axes.unicode_minus'] = False

        # 選擇 Group 0 的數據
        group_0_data = df[df['Cluster'] == 0]

        # 選擇相關特徵
        selected_features = ['線上安全服務', '線上備份服務', '設備保護計劃', '技術支援計劃']
        selected_features1 = ['電視節目', '電影節目', '音樂節目', '無限資料下載']

        # 將特徵轉換成布林值類型
        group_0_data[selected_features] = group_0_data[selected_features].applymap(lambda x: True if x == 'Yes' else False)
        group_0_data[selected_features1] = group_0_data[selected_features1].applymap(lambda x: True if x == 'Yes' else False)

        # 使用 Apriori 算法找出頻繁項集
        frequent_itemsets = apriori(group_0_data[selected_features], min_support=0.2, use_colnames=True)
        frequent_itemsets1 = apriori(group_0_data[selected_features1], min_support=0.3 , use_colnames=True)

        # 根據頻繁項集生成關聯規則
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
        rules1 = association_rules(frequent_itemsets1, metric="confidence", min_threshold=0.3)

        # 顯示結果
        print("頻繁項集:")
        print(frequent_itemsets)
        print("\n關聯規則:")
        print(rules)
        print("頻繁項集:")
        print(frequent_itemsets1)
        print("\n關聯規則:")
        print(rules1)

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
        # (MinMaxTransformer:所有數據都會除以該列絕對值後的最大值。 數據會縮放到到[-1,1]之間)
        #（OrdinalEncoder:分類特徵 -> 整數特徵）
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
            # self.__kmeans_group_choice(transformed_data,30)
            self.__kmeans(data,transformed_data)

        elif way_dict[way] == "DBSCAN":
            # 看不出結果
            self.__Q5_DBSCAN(data,transformed_data)
        else:
            raise KeyError(f"輸入參數{way}未知，請依照{way_dict}進行輸入")
    def Q6(self):
        """根據顧客年齡差異（分成老中青)，比較其使用公司服務的關聯規則的異同"""

        df = self.read()
        df = df[['年齡', '電話服務 ', '多線路服務', '網路服務', '線上安全服務',
                '線上備份服務', '設備保護計劃', '技術支援計劃', '電視節目', '電影節目',
                '音樂節目', '無限資料下載', '無紙化計費']]
        
        print(df.isnull().sum())

        # 未啟用電話服務，多線路服務填補為‘No'；未啟用'網路服務'則後續服務填補為 "No"
        df.fillna('No', inplace=True)
        

        def age_class(age):
            if age <= 35:
                return '青年'
            elif age <= 55:
                return '中年'
            else:
                return '老年'

        # 創建col放年齡分類後的結果，刪除年齡欄
        df['年齡層分類'] = df['年齡'].apply(age_class)
        df = df.drop('年齡', axis=1)

        # 建立不同年齡層的資料框架
        df_young = df[df['年齡層分類'] == '青年'].copy()
        df_middle = df[df['年齡層分類'] == '中年'].copy()
        df_old = df[df['年齡層分類'] == '老年'].copy()

        # 創建 LabelEncoder 實例
        label_encoder = LabelEncoder()
        # 將 '年齡層分類' 欄位轉換為數值型別
        df_young['年齡層分類'] = label_encoder.fit_transform(df_young['年齡層分類'])
        df_middle['年齡層分類'] = label_encoder.fit_transform(df_middle['年齡層分類'])
        df_old['年齡層分類'] = label_encoder.fit_transform(df_old['年齡層分類'])


        # 進行獨熱編碼
        servers = ['電話服務 ', '多線路服務', '網路服務', '線上安全服務',
                '線上備份服務', '設備保護計劃', '技術支援計劃', '電視節目',
                '電影節目', '音樂節目', '無限資料下載', '無紙化計費']

        df_young_encoded = pd.get_dummies(df_young, columns=servers)
        df_middle_encoded = pd.get_dummies(df_middle, columns=servers)
        df_old_encoded = pd.get_dummies(df_old, columns=servers)

        # 進行關聯規則分析
        frequent_itemsets_young = fpgrowth(df_young_encoded, min_support=0.6, use_colnames=True)
        rules_young = association_rules(frequent_itemsets_young, metric="lift", min_threshold=1.1)
        print("青年關聯規則：")
        print(rules_young)
        # make it as a data frame
        df_rules = pd.DataFrame(rules_young)
        df_rules.to_excel('./association_rules/rules_young_retail.xlsx')
        print('-'*50)

        frequent_itemsets_middle = fpgrowth(df_middle_encoded, min_support=0.6, use_colnames=True)
        rules_middle = association_rules(frequent_itemsets_middle, metric="lift", min_threshold=1.1)
        print("中年關聯規則：")
        print(rules_middle)
        # make it as a data frame
        df_rules = pd.DataFrame(rules_middle)
        df_rules.to_excel('./association_rules/rules_middle_retail.xlsx')
        print('-'*50)

        frequent_itemsets_old = fpgrowth(df_old_encoded, min_support=0.6, use_colnames=True)
        rules_old = association_rules(frequent_itemsets_old, metric="lift", min_threshold=1.1)
        print("老年關聯規則：")
        print(rules_old)
        # make it as a data frame
        df_rules = pd.DataFrame(rules_old)
        df_rules.to_excel('./association_rules/rules_old_retail.xlsx')
        print('-'*50)
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


        """
        分析結果：
        1. 顧客有使用'無限資料下載服務'時，也一定會使用'網路服務'。
        2. 顧客同時使用'無限資料下載服務'與'電話服務'時，也一定會使用'網路服務'。
        """
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
        self.__kmeans(data, transformed_data, name="第七題")

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
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, s=3)
        grid_search.fit(X, y)

        # 最佳參數
        best_params = grid_search.best_params_
        print("最佳參數:", best_params)

    def __Q5_detail(self, data):
        """將Ｑ5的結果進行輸出後進行圖表分析"""
        # 資料分類
        data_age = data[['Cluster','年齡']]
        data_service = data[['Cluster','網路連線類型','線上安全服務','線上備份服務']]
        data_money = data[['Cluster','每月費用','總收入']]
        # 圖表解釋_組數、平均年齡
        data_people = data_service.groupby('Cluster').size().reset_index(name="count")
        data_age = data_age.groupby('Cluster').mean().sort_values('Cluster')
        data_people = data_people.sort_values(by='count',ascending=True)
        plt.figure(figsize=(20,10))
        plt.bar(
            data_people['Cluster'],
            data_people['count'],
        )
        plt.title("依照網路相關特徵，Kmeans分群分佈")
        plt.grid() ; plt.xticks(data_people['Cluster'])
        for a,b in zip(data_people['Cluster'],data_people['count']):
            plt.text(a, b+0.1, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)
        for a,b in zip(data_people['Cluster'],data_age['年齡']):
            plt.text(a, b+0.3, '%0.2f' % b, ha='center', va= 'bottom',fontsize=12)
        
        plt.xlabel("組別");plt.ylabel("人數")
        plt.show()
        # 圖表解釋_平均收入、每月支付
        data_money = data_money
        """由圖可知 0,3,4 為人數最多，以下為他們的比例"""
        gd = [0,3,4]
        data_service = data_service[data_service['Cluster'].isin(gd)]
        s_m = data_service.groupby('Cluster').value_counts().unstack(0)
        # s_md = data_service.groupby('Cluster').value_counts()
        s_m.plot.bar()
        plt.title("網路服務使用分佈圖")
        plt.show()

    def __Q7_detail(self, data: pd.DataFrame):
        data = data[(data['電話服務 '] == "Yes") & (data['網路服務'] == "No")]
        data = data[[
            'Cluster', '年齡', '平均長途話費', '多線路服務', ' 額外長途費用', '總費用', '郵遞區號','城市','客戶狀態','總收入','優惠方式'
        ]]
        data_age = data[['Cluster', '年齡']].groupby('Cluster').mean()
        data_people = data[['Cluster', '年齡']].groupby('Cluster').count()
        data_price = data[['Cluster', '平均長途話費',' 額外長途費用', '總費用','總收入']].groupby('Cluster').sum()
        data_service = data[['Cluster','多線路服務']].groupby('Cluster').value_counts().unstack(0)
        data_people_type = data[['Cluster','客戶狀態']].groupby('Cluster').value_counts().unstack(0)
        data_people_sales = data[['Cluster','優惠方式']].groupby("優惠方式").value_counts().unstack(0)
        plt.figure(figsize=(20,12))
        data_people_sales.plot.bar()
        plt.title('各群體的優惠方式使用狀況');plt.ylabel("人數")
        plt.grid(axis="y")
        plt.show()
        data_price.plot.bar()
        plt.title('各群體的費用和收入分佈');plt.ylabel("金額")
        plt.grid(axis="y")
        plt.show()
        data_service.plot.bar()
        plt.title('各群體是否使用多線路服務');plt.ylabel("人數")
        plt.grid(axis="y")
        plt.show()
        data_people_type.plot.bar()
        plt.title('各群體的客戶狀態分佈');plt.ylabel("人數")
        plt.grid(axis="y")
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
    def __kmeans(self,origin_data,transformed_data,name="第五題"):
        """
        處理Kmeans分群
        
        Params:
        origin_data:原始資料
        transformed_data:特徵化資料
        x_label:圖x資料
        y_label:圖y資料
        name:題目數
        """
        kmeans = KMeans(n_clusters=5, random_state=0,
                        n_init=100, max_iter=1000, init="k-means++")
        # 適應模型
        label = {"第五題":["年齡","總收入"],"第七題":['郵遞區號','總費用']}
        kmeans.fit(transformed_data)
        # 新增分群結果到原始資料中
        origin_data['Cluster'] = kmeans.labels_
        # 顯示分群結果
        # 視覺化散點圖
        plt.figure(figsize=(12, 8))
        # 繪製散點圖
        sns.scatterplot(x=label[name][0], y=label[name][1], hue='Cluster',
                        data=origin_data, palette='viridis', s=100)
        # 繪製中心點
        # centers = pca.inverse_transform(kmeans.cluster_centers_)
        # plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', label='Cluster Centers')
        plt.title('K-means 分群')
        plt.xlabel(label[name][0])
        plt.ylabel(label[name][1])
        plt.legend()
        plt.show()
        print('分群圖生成完畢')
        if name == "第五題":
            self.__Q5_detail(origin_data)
        elif name == "第七題":
            self.__Q7_detail(origin_data)

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

    def Q8(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import GridSearchCV
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori
        from mlxtend.frequent_patterns import association_rules

        # 讀取CSV檔案
        data = pd.read_csv('Customer Data/customer_data.csv', encoding='BIG5')

        # 選取所需的列進行分析
        data = data[['年齡', '支付帳單方式', '郵遞區號']]

        # 將年齡進行分層處理
        bins = [0, 18, 30, 45, 60, float('inf')]
        labels = ['少年', '青年', '中年', '壯年', '老年']
        data['年齡層'] = pd.cut(data['年齡'], bins=bins, labels=labels, right=False)

        # 對分類特徵進行獨熱編碼
        data = pd.get_dummies(data, columns=['年齡層'])

        # 分割資料集為訓練集和測試集
        X = data.drop(columns=['支付帳單方式'])  # 特徵
        y = data['支付帳單方式']  # 目標變數

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 建立決策樹模型
        clf = DecisionTreeClassifier(
            random_state=42,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=5
        )

        # 訓練模型、預測和評估
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'模型準確率：{accuracy}')

        print('*'*80)


        # 正規化處理後的結果
        scaler = MinMaxScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)

        # 使用正規化後的數據構建決策樹模型
        clf.fit(X_train_normalized, y_train)
        y_pred_normalized = clf.predict(X_test_normalized)

        accuracy_normalized = accuracy_score(y_test, y_pred_normalized)
        print(f'模型準確率（正規化後）：{accuracy_normalized}')

        print('*'*80)

        # 使用 GridSearchCV 尋找最佳參數
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        }

        clf = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print("最佳參數：", best_params)

        best_clf = DecisionTreeClassifier(**best_params, random_state=38)
        best_clf.fit(X_train, y_train)

        best_predictions = best_clf.predict(X_test)
        best_accuracy = accuracy_score(y_test, best_predictions)
        print(f"最佳模型準確度：{best_accuracy}")

        print('*'*80)

        # 讀取資料並初始化空列表
        records = []
        for i in range(len(data)):
            records.append([str(data.values[i, j]) for j in range(len(data.columns))])

        # 將資料進行轉換和編碼
        te = TransactionEncoder()
        te_ary = te.fit(records).transform(records)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # 應用 apriori 挖掘頻繁項集
        frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

        # 應用關聯規則分析並應用信心度過濾
        association_rules(frequent_itemsets, metric="confidence", min_threshold=0.15)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        # 將規則轉換為 DataFrame
        df_rules = pd.DataFrame(rules)
        print(df_rules)
# Qus().Q3()
# Qus().Q5(1)
Qus().Q7()