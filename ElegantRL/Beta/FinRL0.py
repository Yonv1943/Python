import os
import numpy as np
import numpy.random as rd
import pandas as pd
import yfinance as yf


class FinanceStockEnv:  # 2021-02-02
    """FinRL
    Paper: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
           https://arxiv.org/abs/2011.09607 NeurIPS 2020: Deep RL Workshop.
    Source: Github https://github.com/AI4Finance-LLC/FinRL-Library
    Modify: Github Yonv1943 ElegantRL
    """

    def __init__(self, tickers, initial_stocks, initial_capital=1e6, max_stock=1e2,
                 transaction_fee_percent=1e-3,
                 if_train=True,
                 train_beg=0, train_len=1024):
        tickers = ['AAPL', 'MSFT', 'JPM', 'V', 'RTX', 'PG', 'GS', 'NKE', 'DIS',
                   'AXP', 'HD', 'INTC', 'WMT', 'IBM', 'MRK', 'UNH', 'KO', 'CAT',
                   'TRV', 'JNJ', 'CVX', 'MCD', 'VZ', 'CSCO', 'XOM', 'BA', 'MMM',
                   'PFE', 'WBA', 'DD'] if ticker_list is None else ticker_list
        self.num_stocks = len(tickers)
        assert self.num_stocks == len(initial_stocks)
        self.initial_capital = initial_capital
        self.initial_stocks = initial_stocks
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = max_stock

        self.preprocess_data(tickers)
        ary = self.load_training_data_for_multi_stock(data_path='./FinanceStock.npy')
        assert ary.shape == (
        1699, 5 * self.num_stocks)  # ary: (date, item*stock_dim), item: (adjcp, macd, rsi, cci, adx)
        assert train_beg < train_len
        assert train_len < ary.shape[0]  # ary.shape[0] == 1699
        self.ary_train = ary[train_beg:train_len]
        self.ary_valid = ary[train_len:]
        self.ary = self.ary_train if if_train else self.ary_valid

        # reset
        self.day = 0
        self.initial_account__reset = self.initial_capital
        self.account = self.initial_account__reset
        self.day_npy = self.ary[self.day]
        # self.stocks = np.zeros(self.num_stocks, dtype=np.float32)  # multi-stack
        self.stocks = self.initial_stocks

        self.total_asset = self.account + (self.day_npy[:self.num_stocks] * self.stocks).sum()
        self.episode_return = 0.0  # Compatibility for ElegantRL 2020-12-21
        self.gamma_return = 0.0

        '''env information'''
        self.env_name = 'FinanceStock-v2'
        self.state_dim = 1 + (5 + 1) * self.num_stocks
        self.action_dim = self.num_stocks
        self.if_discrete = False
        self.target_reward = 1.25  # convergence 1.5
        self.max_step = self.ary.shape[0]

    def reset(self) -> np.ndarray:
        self.initial_account__reset = self.initial_capital * rd.uniform(0.9, 1.1)  # reset()
        self.account = self.initial_account__reset
        # self.stocks = np.zeros(self.num_stocks, dtype=np.float32)
        self.stocks = self.initial_stocks
        self.total_asset = self.account + (self.day_npy[:self.num_stocks] * self.stocks).sum()
        # total_asset = account + (adjcp * stocks).sum()

        self.day = 0
        self.day_npy = self.ary[self.day]
        self.day += 1

        state = np.hstack((self.account * 2 ** -16,
                           self.day_npy * 2 ** -8,
                           self.stocks * 2 ** -12,), ).astype(np.float32)
        return state

    def step(self, action) -> (np.ndarray, float, bool, None):
        action = action * self.max_stock

        """bug or sell stock"""
        for index in range(self.num_stocks):
            stock_action = action[index]
            adj = self.day_npy[index]
            if stock_action > 0:  # buy_stock
                available_amount = self.account // adj
                delta_stock = min(available_amount, stock_action)
                self.account -= adj * delta_stock * (1 + self.transaction_fee_percent)
                self.stocks[index] += delta_stock
            elif self.stocks[index] > 0:  # sell_stock
                delta_stock = min(-stock_action, self.stocks[index])
                self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
                self.stocks[index] -= delta_stock

        """update day"""
        self.day_npy = self.ary[self.day]
        self.day += 1
        done = self.day == self.max_step  # 2020-12-21

        state = np.hstack((self.account * 2 ** -16,
                           self.day_npy * 2 ** -8,
                           self.stocks * 2 ** -12,), ).astype(np.float32)

        next_total_asset = self.account + (self.day_npy[:self.num_stocks] * self.stocks).sum()
        reward = (next_total_asset - self.total_asset) * 2 ** -16  # notice scaling!
        self.total_asset = next_total_asset

        self.gamma_return = self.gamma_return * 0.99 + reward  # notice: gamma_r seems good? Yes
        if done:
            reward += self.gamma_return
            self.gamma_return = 0.0  # env.reset()

            # cumulative_return_rate
            self.episode_return = next_total_asset / self.initial_capital

        return state, reward, done, None

    @staticmethod
    def load_training_data_for_multi_stock(data_path='./FinanceStock.npy'):  # need more independent
        if os.path.exists(data_path):
            data_ary = np.load(data_path).astype(np.float32)
            assert data_ary.shape[1] == 5 * 30
            return data_ary
        else:
            raise RuntimeError(
                f'| Download and put it into: {data_path}\n for FinanceStockEnv()'
                f'| https://github.com/Yonv1943/ElegantRL/blob/master/FinanceMultiStock.npy'
                f'| Or you can use the following code to generate it from a csv file.')

    def preprocess_data(self, tickers):
        # the following is same as part of run_model()
        df = self.fecth_data(start_date='2009-01-01',
                             end_date='2021-01-01',
                             ticker_list=tickers).fetch_data()

        data = preprocess_data()
        data = add_turbulence(data)

        df = data
        rebalance_window = 63
        validation_window = 63
        i = rebalance_window + validation_window

        unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
        train__df = self.data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        # print(train__df) # df: DataFrame of Pandas

        train_ary = train__df.to_numpy().reshape((-1, 30, 12))
        '''state_dim = 1 + 6 * stock_dim, stock_dim=30
        n   item    index
        1   ACCOUNT -
        30  adjcp   2
        30  stock   -
        30  macd    7
        30  rsi     8
        30  cci     9
        30  adx     10
        '''
        data_ary = np.empty((train_ary.shape[0], 5, 30), dtype=np.float32)
        data_ary[:, 0] = train_ary[:, :, 2]  # adjcp
        data_ary[:, 1] = train_ary[:, :, 7]  # macd
        data_ary[:, 2] = train_ary[:, :, 8]  # rsi
        data_ary[:, 3] = train_ary[:, :, 9]  # cci
        data_ary[:, 4] = train_ary[:, :, 10]  # adx

        data_ary = data_ary.reshape((-1, 5 * 30))

        data_path = './FinanceStock.npy'
        os.makedirs(data_path[:data_path.rfind('/')])
        np.save(data_path, data_ary.astype(np.float16))  # save as float16 (0.5 MB), float32 (1.0 MB)
        print('| FinanceStockEnv(): save in:', data_path)
        return data_ary

    @staticmethod
    def data_split(df, start, end):
        data = df[(df.date >= start) & (df.date < end)]
        data = data.sort_values(["date", "tic"], ignore_index=True)
        data.index = data.date.factorize()[0]
        return data

    @staticmethod
    def fetch_data(start_date, end_date, ticker_list) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------
        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            temp_df = yf.download(tic, start=start_date, end=end_date)
            temp_df["tic"] = tic
            data_df = data_df.append(temp_df)
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop("adjcp", 1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        return data_df
