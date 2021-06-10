from elegantrl2.demo import *
from envs.FinRL.StockTrading import *

# from StockTrading import *
GAP = 4
MinActionRate = 0.25
Stock_Add = 32  # todo beta1


class StockTradingEnv:
    def __init__(self, cwd='./envs/FinRL', gamma=0.995,
                 max_stock=1e2, initial_capital=1e6, buy_cost_pct=1e-3, sell_cost_pct=1e-3,
                 start_date='2008-03-19', end_date='2016-01-01',
                 ticker_list=None, tech_indicator_list=None, initial_stocks=None, if_eval=False):

        price_ary, tech_ary = self.load_data(cwd, ticker_list, tech_indicator_list,
                                             start_date, end_date, )

        beg_i, mid_i, end_i = 0, int(2 ** 18), int(2 ** 19)  # int(528026)
        if if_eval:
            self.price_ary = price_ary[beg_i:mid_i:GAP]
            self.tech_ary = tech_ary[beg_i:mid_i:GAP]
        else:
            self.price_ary = price_ary[mid_i:end_i:GAP]
            self.tech_ary = tech_ary[mid_i:end_i:GAP]

        stock_dim = self.price_ary.shape[1]

        self.gamma = gamma
        self.max_stock = max_stock
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.initial_capital = initial_capital
        self.initial_stocks = np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.initial_total_asset = None
        self.gamma_reward = 0.0
        self.stock_cd = None  # stock_cd

        # environment information
        self.env_name = 'StockTradingEnv-v2'
        # self.state_dim = 1 + 2 * stock_dim + self.tech_ary.shape[1]
        self.state_dim = 1 + 3 * stock_dim + self.tech_ary.shape[1]  # stock_cd
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_discrete = False
        self.target_return = 5
        self.episode_return = 0.0

    def reset(self):
        self.day = 0
        price = self.price_ary[self.day]

        self.stocks = self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
        self.amount = self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum()

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0

        self.stock_cd = np.zeros_like(price)  # stock_cd

        state = np.hstack((max(self.amount, 1e4) * (2 ** -12),
                           price,
                           self.stock_cd,  # stock_cd
                           self.stocks,
                           self.tech_ary[self.day],
                           )).astype(np.float32) * (2 ** -5)
        return state

    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)

        self.day += 1
        price = self.price_ary[self.day]

        self.stock_cd += Stock_Add

        min_action = int(self.max_stock * MinActionRate)  # stock_cd
        for index in np.where(actions < -min_action)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.amount += price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                self.stock_cd[index] = 0  # stock_cd

        for index in np.where(actions > min_action)[0]:  # buy_index:
            if price[index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)
                buy_num_shares = min(self.amount // price[index], actions[index])
                self.stocks[index] += buy_num_shares
                self.amount -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                self.stock_cd[index] = 0  # stock_cd

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * 2 ** -10  # reward scaling
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, dict()

    def get_state(self, price):
        state = np.hstack((max(self.amount, 1e4) * (2 ** -12),
                           price,
                           self.stock_cd,  # stock_cd
                           self.stocks,
                           self.tech_ary[self.day],
                           )).astype(np.float32) * (2 ** -5)
        return state

    def load_data(self, cwd='./envs/FinRL', ticker_list=None, tech_indicator_list=None,
                  start_date='2016-01-03', end_date='2021-05-27'):
        raw_data1_path = f'{cwd}/StockTradingEnv_raw_data1.df'
        raw_data2_path = f'{cwd}/final1.df'
        data_path_array = f'{cwd}/StockTradingEnv_arrays_float16.npz'
        # start_date = '2008-03-19'
        # end_date = '2021-01-01'
        # raw_data1_path = f'{cwd}/StockTradingEnv_raw_data1.df'
        # raw_data2_path = f'{cwd}/StockTradingEnv_raw_data2.df'

        tech_indicator_list = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'
        ] if tech_indicator_list is None else tech_indicator_list
        # tech_indicator_list = [
        #     'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma'
        # ] if tech_indicator_list is None else tech_indicator_list

        # ticker_list = [
        #     'AAPL', 'MSFT', 'JPM', 'V', 'RTX', 'PG', 'GS', 'NKE', 'DIS', 'AXP', 'HD',
        #     'INTC', 'WMT', 'IBM', 'MRK', 'UNH', 'KO', 'CAT', 'TRV', 'JNJ', 'CVX', 'MCD',
        #     'VZ', 'CSCO', 'XOM', 'BA', 'MMM', 'PFE', 'WBA', 'DD'
        # ] if ticker_list is None else ticker_list  # finrl.config.DOW_30_TICKER
        ticker_list = [
            'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT', 'AMD', 'AMGN',
            'AMZN', 'ASML', 'ATVI', 'BIIB', 'BKNG', 'BMRN', 'CDNS', 'CERN', 'CHKP', 'CMCSA',
            'COST', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'CTXS', 'DLTR', 'EA', 'EBAY', 'FAST',
            'FISV', 'GILD', 'HAS', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG',
            'JBHT', 'KLAC', 'LRCX', 'MAR', 'MCHP', 'MDLZ', 'MNST', 'MSFT', 'MU', 'MXIM',
            'NLOK', 'NTAP', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PEP', 'QCOM', 'REGN',
            'ROST', 'SBUX', 'SIRI', 'SNPS', 'SWKS', 'TTWO', 'TXN', 'VRSN', 'VRTX', 'WBA',
            'WDC', 'WLTW', 'XEL', 'XLNX'
        ] if ticker_list is None else ticker_list  # finrl.config.NAS_74_TICKER
        # ticker_list = [
        #     'AMGN', 'AAPL', 'AMAT', 'INTC', 'PCAR', 'PAYX', 'MSFT', 'ADBE', 'CSCO', 'XLNX',
        #     'QCOM', 'COST', 'SBUX', 'FISV', 'CTXS', 'INTU', 'AMZN', 'EBAY', 'BIIB', 'CHKP',
        #     'GILD', 'NLOK', 'CMCSA', 'FAST', 'ADSK', 'CTSH', 'NVDA', 'GOOGL', 'ISRG', 'VRTX',
        #     'HSIC', 'BIDU', 'ATVI', 'ADP', 'ROST', 'ORLY', 'CERN', 'BKNG', 'MYL', 'MU',
        #     'DLTR', 'ALXN', 'SIRI', 'MNST', 'AVGO', 'TXN', 'MDLZ', 'FB', 'ADI', 'WDC',
        #     'REGN', 'LBTYK', 'VRSK', 'NFLX', 'TSLA', 'CHTR', 'MAR', 'ILMN', 'LRCX', 'EA',
        #     'AAL', 'WBA', 'KHC', 'BMRN', 'JD', 'SWKS', 'INCY', 'PYPL', 'CDW', 'FOXA', 'MXIM',
        #     'TMUS', 'EXPE', 'TCOM', 'ULTA', 'CSX', 'NTES', 'MCHP', 'CTAS', 'KLAC', 'HAS',
        #     'JBHT', 'IDXX', 'WYNN', 'MELI', 'ALGN', 'CDNS', 'WDAY', 'SNPS', 'ASML', 'TTWO',
        #     'PEP', 'NXPI', 'XEL', 'AMD', 'NTAP', 'VRSN', 'LULU', 'WLTW', 'UAL'
        # ] if ticker_list is None else ticker_list  # finrl.config.NAS_100_TICKER

        # print(raw_df.loc['2000-01-01'])
        # j = 40000
        # check_ticker_list = set(raw_df.loc.obj.tic[j:j + 200].tolist())
        # print(len(check_ticker_list), check_ticker_list)

        '''get: train_price_ary, train_tech_ary, eval_price_ary, eval_tech_ary'''
        if os.path.exists(data_path_array):
            load_dict = np.load(data_path_array)

            price_ary = load_dict['price_ary'].astype(np.float32)
            tech_ary = load_dict['tech_ary'].astype(np.float32)
        else:
            processed_df = self.processed_raw_data(raw_data1_path, raw_data2_path,
                                                   ticker_list, tech_indicator_list)

            def data_split(df, start, end):
                data = df[(df.date >= start) & (df.date < end)]
                data = data.sort_values(["date", "tic"], ignore_index=True)
                data.index = data.date.factorize()[0]
                return data

            train_df = data_split(processed_df, start_date, end_date)

            # print(processed_df.date) to show the start_date and end_date
            # print(processed_df.columns) to show the items name
            price_ary, tech_ary = self.convert_df_to_ary(train_df, tech_indicator_list)
            price_ary, tech_ary = self.deal_with_split_or_merge_shares(price_ary, tech_ary)

            np.savez_compressed(data_path_array,
                                price_ary=price_ary.astype(np.float16),
                                tech_ary=tech_ary.astype(np.float16), )
        return price_ary, tech_ary

    def processed_raw_data(self, raw_data_path, processed_data_path,
                           ticker_list, tech_indicator_list):
        if os.path.exists(processed_data_path):
            processed_df = pd.read_pickle(processed_data_path)  # DataFrame of Pandas
            # print('| processed_df.columns.values:', processed_df.columns.values)
            print(f"| load data: {processed_data_path}")
        else:
            print("| FeatureEngineer: start processing data (2 minutes)")
            fe = FeatureEngineer(use_turbulence=True,
                                 user_defined_feature=False,
                                 use_technical_indicator=True,
                                 tech_indicator_list=tech_indicator_list, )
            raw_df = self.get_raw_data(raw_data_path, ticker_list)

            processed_df = fe.preprocess_data(raw_df)
            processed_df.to_pickle(processed_data_path)
            print("| FeatureEngineer: finish processing data")

        '''you can also load from csv'''
        # processed_data_path = f'{cwd}/dow_30_2021_minute.csv'
        # processed_data_path = f'{cwd}/dow_30_daily_2000_2021.csv'
        # if os.path.exists(processed_data_path):
        #     processed_df = pd.read_csv(processed_data_path)
        return processed_df

    def draw_cumulative_return(self, args, _torch) -> list:
        state_dim = self.state_dim
        action_dim = self.action_dim

        agent = args.agent
        net_dim = args.net_dim
        cwd = args.cwd

        agent.init(net_dim, state_dim, action_dim)
        agent.save_load_model(cwd=cwd, if_save=False)
        act = agent.act
        device = agent.device

        state = self.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        with _torch.no_grad():
            for i in range(self.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = self.step(action)

                total_asset = self.amount + (self.price_ary[self.day] * self.stocks).sum()
                episode_return = total_asset / self.initial_total_asset
                episode_returns.append(episode_return)
                if done:
                    break

        import matplotlib.pyplot as plt
        plt.plot(episode_returns)
        plt.grid()
        plt.title('cumulative return')
        plt.xlabel('day')
        plt.xlabel('multiple of initial_account')
        plt.savefig(f'{cwd}/cumulative_return.jpg')
        print(f"| draw_cumulative_return: save in {cwd}/cumulative_return.jpg")
        return episode_returns

    def deal_with_split_or_merge_shares(self, price_ary, tech_ary):
        # print(price_ary.shape)  # (528026, 93)
        # print(tech_ary.shape)  # (528026, 93 * 7)
        tech_ary = tech_ary.reshape((tech_ary.shape[0], -1, 7))

        data_idx = list(range(price_ary.shape[1]))
        # for delete_idx in (77, 36, 31, 14):
        #     del data_idx[delete_idx]
        # data_idx = np.array(data_idx)

        price_ary = price_ary[:, data_idx]
        tech_ary = tech_ary[:, data_idx, :]
        for j in range(price_ary.shape[1]):
            x = price_ary[:, j]

            x = self.fill_nan_with_next_value(x)  # fill_nan_with_next_value

            x_offset0 = x[1:]
            x_offset1 = x[:-1]
            x_delta = np.abs(x_offset0 - x_offset1) / ((x_offset0 * x_offset1) ** 0.5)
            x_where = np.where(x_delta > 0.25)[0]
            # plt.plot(x)
            for i in x_where:
                # print(j, i, x[i] / x[i + 1])
                x[i + 1:] *= x[i] / x[i + 1]
            # plt.plot(x)
            # plt.show()
            price_ary[:, j] = x

        # print(price_ary.shape)  # (528026, 89)

        for tech_i1 in range(tech_ary.shape[1]):  # fill_nan_with_next_value
            for tech_i2 in range(tech_ary.shape[2]):
                tech_item = tech_ary[:, tech_i1, tech_i2]
                tech_item = self.fill_nan_with_next_value(tech_item)
                tech_ary[:, tech_i1, tech_i2] = tech_item

        tech_ary = tech_ary.reshape((528026, -1))
        # print(tech_ary.shape)  # (528026, 89 * 7)
        return price_ary, tech_ary

    @staticmethod
    def fill_nan_with_next_value(ary):
        x_isnan = np.isnan(ary)
        value = ary[np.where(~x_isnan)[0][0]]  # find the first value != nan
        for k in range(ary.shape[0]):
            if x_isnan[k]:
                ary[k] = value
            value = ary[k]
        return ary

    @staticmethod
    def get_raw_data(raw_data_path, ticker_list):
        if os.path.exists(raw_data_path):
            raw_df = pd.read_pickle(raw_data_path)  # DataFrame of Pandas
            # print('| raw_df.columns.values:', raw_df.columns.values)
            print(f"| load data: {raw_data_path}")
        else:
            print("| YahooDownloader: start downloading data (1 minute)")
            raw_df = YahooDownloader(start_date="2000-01-01",
                                     end_date="2021-01-01",
                                     ticker_list=ticker_list, ).fetch_data()
            raw_df.to_pickle(raw_data_path)
            print("| YahooDownloader: finish downloading data")
        return raw_df

    @staticmethod
    def convert_df_to_ary(df, tech_indicator_list):
        tech_ary = list()
        price_ary = list()
        for day in range(len(df.index.unique())):
            item = df.loc[day]

            tech_items = [item[tech].values.tolist() for tech in tech_indicator_list]
            tech_items_flatten = sum(tech_items, [])
            tech_ary.append(tech_items_flatten)
            price_ary.append(item.close)  # adjusted close price (adjcp)

        price_ary = np.array(price_ary)
        tech_ary = np.array(tech_ary)
        print(f'| price_ary.shape: {price_ary.shape}, tech_ary.shape: {tech_ary.shape}')
        return price_ary, tech_ary


class StockTradingVecEnv(StockTradingEnv):
    def __init__(self, env_num, **kwargs):
        super(StockTradingVecEnv, self).__init__(**kwargs)

        self.env_num = env_num
        self.initial_capital = np.ones(self.env_num, dtype=np.float32) * self.initial_capital
        self.initial_stocks = np.tile(self.initial_stocks[np.newaxis, :], (self.env_num, 1))
        self.price_ary = np.tile(self.price_ary[np.newaxis, :], (self.env_num, 1, 1))
        self.tech_ary = np.tile(self.tech_ary[np.newaxis, :], (self.env_num, 1, 1))

        self.device = torch.device('cuda')

    def reset_vec(self):
        with torch.no_grad():
            self.day = 0
            price = self.price_ary[:, self.day]

            self.stocks = self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)

            self.amount = self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum(axis=1)

            self.total_asset = self.initial_capital
            self.initial_total_asset = self.total_asset
            self.gamma_reward = np.zeros(self.env_num, dtype=np.float32)

            self.stock_cd = np.zeros_like(price)  # stock_cd
        return self.get_state(price)  # state

    def step_vec(self, actions):
        with torch.no_grad():
            actions = (actions * self.max_stock).long().detach().cpu().numpy()

            self.day += 1
            price = self.price_ary[:, self.day]

            self.stock_cd += Stock_Add

            min_action = int(self.max_stock * MinActionRate)  # stock_cd
            for j in range(self.env_num):
                for index in np.where(actions[j] < -min_action)[0]:  # sell_index:
                    if price[j, index] > 0:  # Sell only if current asset is > 0
                        sell_num_shares = min(self.stocks[j, index], -actions[j, index])
                        self.stocks[j, index] -= sell_num_shares
                        self.amount[j] += price[j, index] * sell_num_shares * (1 - self.sell_cost_pct)
                        self.stock_cd[j, index] = 0  # stock_cd

                for index in np.where(actions[j] > min_action)[0]:  # buy_index:
                    if price[j, index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)
                        buy_num_shares = min(self.amount[j] // price[j, index], actions[j, index])
                        self.stocks[j, index] += buy_num_shares
                        self.amount[j] -= price[j, index] * buy_num_shares * (1 + self.buy_cost_pct)
                        self.stock_cd[j, index] = 0  # stock_cd

            state = self.get_state(price)

            total_asset = self.amount + (self.stocks * price).sum(axis=1)
            reward = (total_asset - self.total_asset) * 2 ** -10  # reward scaling
            self.total_asset = total_asset

            self.gamma_reward = self.gamma_reward * self.gamma + reward
            done = self.day == self.max_step
            if done:
                reward = self.gamma_reward
                self.episode_return = total_asset / self.initial_total_asset

                state = self.reset_vec()
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            done = torch.as_tensor([done, ] * self.env_num, dtype=torch.float32, device=self.device)
        return state, reward, done, dict()

    def get_state(self, price):
        state = np.hstack((self.amount[:, np.newaxis].clip(None, 1e4) * (2 ** -12),
                           price,
                           self.stock_cd,  # stock_cd
                           self.stocks,
                           self.tech_ary[:, self.day],
                           ))
        return torch.as_tensor(state, dtype=torch.float32, device=self.device) * (2 ** -5)

    def check_vec_env(self):
        # env = StockTradingVecEnv()
        self.reset_vec()

        a = torch.zeros((self.env_num, self.action_dim), dtype=torch.float32, device=self.device)
        s, r, d, _ = self.step_vec(a)
        print(s.shape)
        print(r.shape)
        print(d.shape)


def demo_custom_env_finance_rl():
    from elegantrl2.agent import AgentPPO

    '''choose an DRL algorithm'''
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()
    args.agent.lambda_entropy = 0.01  # todo ceta3
    args.gpu_id = sys.argv[-1][-4]
    args.random_seed = 1943210

    "TotalStep: 10e4, TargetReturn: 3.0, UsedTime:  200s, FinanceStock-v1"
    "TotalStep: 20e4, TargetReturn: 4.0, UsedTime:  400s, FinanceStock-v1"
    "TotalStep: 30e4, TargetReturn: 4.2, UsedTime:  600s, FinanceStock-v1"
    # from envs.FinRL.StockTrading import StockTradingEnv
    args.gamma = 0.999
    # args.env = StockTradingEnv(if_eval=False, gamma=gamma)
    args.env = StockTradingVecEnv(if_eval=False, gamma=args.gamma, env_num=2)
    args.env_eval = StockTradingEnv(if_eval=True, gamma=args.gamma)

    args.net_dim = 2 ** 9
    args.batch_size = args.net_dim * 4
    args.target_step = args.env.max_step
    args.repeat_times = 2 ** 4

    args.eval_gap = 2 ** 8
    args.eval_times1 = 2 ** 0
    args.eval_times2 = 2 ** 1
    args.break_step = int(8e6)

    '''train and evaluate'''
    # train_and_evaluate(args)
    args.worker_num = 2
    train_and_evaluate_mp(args)


demo_custom_env_finance_rl()
# stable
