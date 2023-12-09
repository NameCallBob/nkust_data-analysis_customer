import matplotlib.pyplot as plt ; import pandas as pd
from data import Data
class A_Pic(Data):
    """
    主要為第一題，將欄位進行分析
    """
    def __init__(self) -> None:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode Ms']
        self.data = Data()
    def customer(self):
        """
        顧客特徵圖表
        """
        # self.__Age_Gender_Married()
        self.__customer_status()
        
    def __Age_Gender_Married(self):
        print(self.data.data_a_g_y())
        data = self.data.data_a_g_y()
        data.plot.bar(color=['pink','lightblue'])
        plt.title('性別的年齡層分佈圖')
        plt.legend(['女性','男性'])
        plt.show()
        data = self.data.read()
        data = data['婚姻'].value_counts()
        print(data)
        data.plot.pie(autopct = "%0.2f%%",labels=[
            '否','是'
        ])
        plt.title('是否婚姻')
        plt.ylabel('人數')
        plt.show()

    def __customer_status(self):
        sdata = self.data.data_customer_status()
        data = sdata['客戶狀態'].value_counts()
        print(data)
        data.plot.pie(autopct = "%0.2f%%",labels=[
            '原有','流失','加入'
        ])
        plt.title('客戶狀態');plt.ylabel('人數')
        plt.show()
        data = sdata[['客戶流失類別','客戶離開原因']]
        data1 = data[['客戶流失類別']].drop(index='NoData')
        data1.plot.pie(autopct = "%0.2f%%")
        
        print(data)
        
    def __phone(self):
        pass

    def __net(self):
        pass
    
    def __money(self):
        pass


def test():
    A_Pic().customer()
test()