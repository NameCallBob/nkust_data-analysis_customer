import pandas as pd

class Data():
    """
    用於處理資料處理
    """
    def __init__(self,source = "Customer Data/customer_data.csv",source_area = "Customer Data/customer_zip.csv") -> pd.DataFrame:
        self.source = source
        self.source_area = source_area
        self.data = pd.read_csv(self.source,encoding="big5")
        self.data_area_people = pd.read_csv(self.source,encoding="big5")
    def read(self) -> pd.DataFrame:
        """
        讀取資料
        \n其資料由於非utf8編碼所以使用big5進行分析
        \n並將不合適的資料進行清洗
        """
        try:
            df = self.data
            df = df.drop(columns=['客戶編號'])
            """
            註：
            原本使用dropna發現實際資料數量被刪除一大部分影響到資料分析結果
            暫時使用fillna(0)進行計算
            """
            # # 刪除缺失
            # df = df.dropna(axis=0)
            # # 移除重複 NOPE
            # df = df.drop_duplicates()
            # 先列出所有Nna的數量
            
            """先將欄位進行分類，針對NaN進行處理"""
            for i in df.columns:
                
                # 計算該欄位NaN的數量
                if (type(df[i].head(1)[0]) == "str" or type(df[i].head(1)[0]) == "object" or type(df[i].head(1)[0]) == "NaN") or type(df[i].head(1)[0]) == type(1.1):
                    if i == "客戶流失類別":
                        df[i] = df[i].fillna('stayed')
                    else:
                        df[i] = df[i].fillna('NoData')
                elif (type(df[i].head(1)[0]) == type(1)) :
                    df[i] = df[i].fillna(0)
            
            """
            '客戶編號', ' 性別', '年齡', '婚姻', '扶養人數', '城市', '郵遞區號', '緯度 ', '經度', '推薦次數',
            '加入期間 (月)', '優惠方式', '電話服務 ', '平均長途話費', '多線路服務', '網路服務', '網路連線類型',
            '平均下載量( GB)', '線上安全服務', '線上備份服務', '設備保護計劃', '技術支援計劃', '電視節目', '電影節目',
            '音樂節目', '無限資料下載', '合約類型', '無紙化計費', '支付帳單方式', '每月費用', '總費用', '總退款',
            '額外數據費用', ' 額外長途費用', '總收入', '客戶狀態', '客戶流失類別', '客戶離開原因'
            """

            """
            發現客戶流失類別,客戶離開原因沒資料原因與客戶狀態相關，因為他根本就沒離開 哈哈
            """
            # 年齡分層處理
            bins = [0,18,24,29,44,64,df['年齡'].max()]
            label = ['孩童或少年','青年','成年','壯年','中年','老年']
            df['年齡層']= pd.cut(df['年齡'],bins, labels = label)
            # 總體數量
            # print(df)
            return(df)
        except FileNotFoundError:
            raise FileNotFoundError('路徑為Customer Data/customer_data.csv，請注意是否依照程式碼放置正確地方')
        except UnicodeDecodeError:
            raise UnicodeDecodeError("檔案非utf8編碼")
        except Exception as e:
            raise (e)
        
        
    def __check_null(self):
        df = self.data
        for i in self.data.columns:
            print(df[i].isnull().value_counts())

    def data_a_g_y(self) -> pd.DataFrame:
        data = self.read()
        data = data[[' 性別','年齡層']]
        data = data.assign(年齡層 =data.年齡層.astype(object))
        data = data.groupby(' 性別').value_counts().unstack(level=0)
        return data
    
    def data_phone(self) -> pd.DataFrame:
        data = self.read()
        data = data[['電話服務 ','平均長途話費','多線路服務']]
        return data

    def data_net(self) -> pd.DataFrame :
        data = self.read()
        data = data[['網路服務','網路連線類型','平均下載量( GB)','線上安全服務','線上備份服務','設備保護計劃','技術支援計劃','電視節目','電影節目','音樂節目','無限資料下載']]
        print(data)
        return data

    def data_money(self) -> pd.DataFrame :
        data = self.read()
        data = data[['合約類型','優惠方式','無紙化計費','支付帳單方式','每月費用','總費用','總退款','額外數據費用',' 額外長途費用','總收入']]
        print(data)
        return data
    
    def for_q2(self) -> pd.DataFrame :
        data = self.read()
        return data[[
            '年齡層',' 性別','婚姻','城市','優惠方式','電話服務 ','網路服務','網路連線類型','合約類型','每月費用','客戶狀態','客戶流失類別','客戶離開原因'
        ]]

    def data_customer_status(self) -> pd.DataFrame :
        data = self.read()
        data = data[['客戶狀態','客戶流失類別','客戶離開原因']]
        print(data)
        return data