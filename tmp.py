import pandas as pd ; import matplotlib.pyplot as plt

def Q5_detail():
    """將Ｑ5的結果進行輸出後進行圖表分析"""
    data = pd.read_csv("./res_5.csv")
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
    s_m = data_service.groupby('Cluster').value_counts()
    print(s_m)
    # s_md = data_service.groupby('Cluster').value_counts()
    s_m.plot.bar()
    plt.title("")
    plt.show()
    
Q5_detail()