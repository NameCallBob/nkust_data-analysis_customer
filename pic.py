import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib
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
        self.__Age_Gender_Married()
        self.__customer_status()
        self.__phone()
        self.__net()
        self.__money()

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
        data = sdata[sdata['客戶狀態'] != "Stayed"]
        data1 = data[['客戶流失類別','客戶離開原因']]
        data1 = data1.groupby("客戶流失類別").value_counts()
        data1.plot.pie(autopct = "%0.2f%%")
        plt.show()
        print(data)

    def __phone(self):
        df = self.data.data_phone()
        print(df.columns)

        # 優惠方式
        df_phone = df[['優惠方式']].value_counts()
        df_phone.plot.pie(
            autopct = "%0.2f%%",
            labels = ['None', 'Offer B', 'Offer E', 'Offer D', 'Offer A', 'Offer C']
        )
        plt.title("客戶_優惠方式比例圖")
        plt.xlabel("人數")
        plt.show()

        # 電話服務
        df_phone = df[['電話服務 ']].value_counts()
        df_phone.plot.pie(
            autopct = "%0.2f%%",
            labels = ['有', '無']
        )
        plt.title("客戶_電話服務比例圖")
        plt.xlabel("人數")
        plt.show()

        # 平均長途路費
        df_detail_1 = df[['平均長途話費']]
        df_detail_1.plot.box()
        plt.title('客戶_平均長途話費')
        plt.show()

        # 多線路服務
        df_detail = df[['多線路服務']].value_counts()
        df_detail.plot.pie(
            autopct = "%0.2f%%",
            labels = ["有","無"]
        )
        plt.title("客戶_多線路服務比例圖")
        plt.xlabel("人數")
        plt.show()
          
    def __net(self):
        df = self.data.data_net()
        print(df.columns)

        # 網路服務
        df_net_server = df[['網路服務']].value_counts()
        df_net_server.plot.pie(
            autopct = "%0.2f%%",
            labels = ['有', '無']
        )
        plt.title("客戶_網路服務比例圖")
        plt.xlabel("人數")
        plt.show()

        #  先取得有使用網路服務的使用者
        df_filtered = df[df['網路服務'] == 'Yes']

        #  網路連線類型
        df_connection_types = df_filtered['網路連線類型'].value_counts()
        df_connection_types.plot.pie(
            autopct = "%0.2f%%",
        )
        plt.title("客戶有使用網路服務_網路連線類型比例圖")
        plt.xlabel("人數")
        plt.show()

        # 平均下載量
        # #下載量分組
        bins = [0, 10, 20, 30, 40, 50, 60, float('inf')]  # 根據你的需求設定下載量的分組範圍
        labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61+']  # 每個分組的標籤
        df_filtered['下載量分組'] = pd.cut(df_filtered['平均下載量( GB)'], bins=bins, labels=labels, right=False)
        # 計算每個分組中的人數
        download_counts = df_filtered['下載量分組'].value_counts()
        download_counts.plot.pie(
            autopct = "%0.2f%%",
            labels=download_counts.index.tolist(),
        )
        plt.title("客戶有使用網路服務_平均下載量( GB)類型比例圖")
        plt.xlabel("人數")
        plt.show()

        # 線上安全服務
        df_safe_server = df[['線上安全服務']].value_counts()
        df_safe_server.plot.pie(
            autopct = "%0.2f%%",
            labels = ['無', '有']
        )
        plt.title("客戶_線上安全服務比例圖")
        plt.xlabel("人數")
        plt.show()

        # 線上備份服務
        df_safe_server = df[['線上備份服務']].value_counts()
        df_safe_server.plot.pie(
            autopct = "%0.2f%%",
            labels = ['無', '有']
        )
        plt.title("客戶_線上備份服務比例圖")
        plt.xlabel("人數")
        plt.show()

        # 設備保護計劃
        df_safe_server = df[['設備保護計劃']].value_counts()
        df_safe_server.plot.pie(
            autopct = "%0.2f%%",
            labels = ['無', '有']
        )
        plt.title("客戶_設備保護計劃比例圖")
        plt.xlabel("人數")
        plt.show()

        # 技術支援計劃
        df_tech_sup = df[['技術支援計劃']].value_counts()
        df_tech_sup.plot.pie(
            autopct = "%0.2f%%",
            labels = ['無', '有']
        )
        plt.title("客戶_技術支援計劃比例圖")
        plt.xlabel("人數")
        plt.show()

        # 電視節目
        df_tv = df[['電視節目']].value_counts()
        df_tv.plot.pie(
            autopct = "%0.2f%%",
            labels = ['無', '有']
        )
        plt.title("客戶_電視節目比例圖")
        plt.xlabel("人數")
        plt.show()

        # 電影節目
        df_movie = df[['電影節目']].value_counts()
        df_movie.plot.pie(
            autopct = "%0.2f%%",
            labels = ['無', '有']
        )
        plt.title("客戶_電影節目比例圖")
        plt.xlabel("人數")
        plt.show()

        # 音樂節目
        df_music = df[['音樂節目']].value_counts()
        df_music.plot.pie(
            autopct = "%0.2f%%",
            labels = ['無', '有']
        )
        plt.title("客戶_音樂節目比例圖")
        plt.xlabel("人數")
        plt.show()

        # 無限資料下載
        df_music = df[['無限資料下載']].value_counts()
        df_music.plot.pie(
            autopct = "%0.2f%%",
            labels = ['有', '無']
        )
        plt.title("客戶_無限資料下載比例圖")
        plt.xlabel("人數")
        plt.show()


    def __money(self):
        df = self.data.data_money()
        print(df.columns)
        print('-'*50)
        print(df.describe())
        # ['合約類型', '無紙化計費', '支付帳單方式', '每月費用', '總費用', '總退款', '額外數據費用', ' 額外長途費用', '總收入']
        # 合約類型
        df_net_server = df[['合約類型']].value_counts()
        df_net_server.plot.pie(
            autopct = "%0.2f%%",
            labels = ['Month-to-Month', 'Two Year', 'One Year']
        )
        plt.title("客戶_合約類型比例圖")
        plt.xlabel("人數")
        plt.show()

        # 無紙化計費
        df_no_paper = df[['無紙化計費']].value_counts()
        df_no_paper.plot.pie(
            autopct = "%0.2f%%",
            labels = ['有', '無']
        )
        plt.title("客戶_無紙化計費比例圖")
        plt.xlabel("人數")
        plt.show()

        # 支付帳單方式
        df_payment_method = df[['支付帳單方式']].value_counts()
        df_payment_method.plot.pie(
            autopct = "%0.2f%%",
            labels = ['Bank Withdrawal', 'Credit Card', 'Mailed Check']
        )
        plt.title("客戶_支付帳單方式比例圖")
        plt.xlabel("人數")
        plt.show()

        # 每月費用
        df_month_fee = df[df['每月費用'] >= 0]

        # #將費用分組
        bins = [0, 20, 40, 60, 80, 100, float('inf')]
        labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '100+']
        df_month_fee['費用分組'] = pd.cut(df_month_fee['每月費用'], bins=bins, labels=labels, right=False)

        # #計算每個分組中的人數
        fee_counts = df_month_fee['費用分組'].value_counts()
        fee_counts.plot.pie(
            autopct="%0.2f%%",
            labels=fee_counts.index.tolist(),
        )
        plt.title("客戶_每月費用比例圖")
        plt.xlabel("每月費用")
        plt.show()

        # 總費用
        df_total_fee = df[['總費用']]
        bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, float('inf')]
        labels = ['0-1000', '1001-2000', '2001-3000', '3001-4000', '4001-5000', '5001-6000', '6001-7000', '7001-8000', '8001+']
        df_total_fee['總費用分組'] = pd.cut(df['總費用'], bins=bins, labels=labels, right=False)
        # #計算每個分組中的人數
        total_fee_counts = df_total_fee['總費用分組'].value_counts()
        total_fee_counts.plot.pie(
            autopct="%0.2f%%",
            labels=total_fee_counts.index.tolist(),
        )
        plt.title("客戶_總費用比例圖")
        plt.xlabel("費用")
        plt.show()

        # 總退款
        df_refund = df[['總退款']]
        bins = [0, 10, float('inf')]
        labels = ['0-10', '11+']
        df_refund['總退款分組'] = pd.cut(df['總退款'], bins=bins, labels=labels, right=False)
        # #計算每個分組中的人數
        refund_counts = df_refund['總退款分組'].value_counts()
        refund_counts.plot.pie(
            autopct="%0.2f%%",
            labels=refund_counts.index.tolist(),
        )
        plt.title("客戶_總退款比例圖")
        plt.xlabel("費用")
        plt.show()

        # 額外數據費用
        df_extra_fee = df[['額外數據費用']]
        bins = [0, 10, float('inf')]
        labels = ['0-10', '11+']
        df_extra_fee['額外數據費用分組'] = pd.cut(df['額外數據費用'], bins=bins, labels=labels, right=False)
        # #計算每個分組中的人數
        extrafee_counts = df_extra_fee['額外數據費用分組'].value_counts()
        extrafee_counts.plot.pie(
            autopct="%0.2f%%",
            labels=extrafee_counts.index.tolist(),
        )
        plt.title("客戶_額外數據費用比例圖")
        plt.xlabel("費用")
        plt.show()

        # 額外長途費用
        df_long_fee = df[[' 額外長途費用']]
        bins = [0, 1000, 2000, 3000, float('inf')]
        labels = ['0-1000', '1001-2000', '2001-3000', '3001+']
        df_long_fee['額外長途費用分組'] = pd.cut(df[' 額外長途費用'], bins=bins, labels=labels, right=False)
        # #計算每個分組中的人數
        longfee_counts = df_long_fee['額外長途費用分組'].value_counts()
        longfee_counts.plot.pie(
            autopct="%0.2f%%",
            labels=longfee_counts.index.tolist(),
        )
        plt.title("客戶_額外長途費用比例圖")
        plt.xlabel("費用")
        plt.show()

        # 總收入
        df_t_income = df[['總收入']]
        bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, float('inf')]
        labels = ['0-1000', '1001-2000', '2001-3000', '3001-4000', '4001-5000', '5001-6000', '6001-7000', '7001-8000', '8001-9000', '9001-10000', '10000+']
        df_t_income['總收入分組'] = pd.cut(df['總收入'], bins=bins, labels=labels, right=False)
        # #計算每個分組中的人數
        tincome_counts = df_t_income['總收入分組'].value_counts()
        tincome_counts.plot.pie(
            autopct="%0.2f%%",
            labels=tincome_counts.index.tolist(),
        )
        plt.title("客戶_總收入比例圖")
        plt.xlabel("收入")
        plt.show()



        



def test():
    A_Pic().customer()


test()