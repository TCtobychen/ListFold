import numpy as np
import copy
import os 
import argparse

parser = argparse.ArgumentParser(description='Rolling Training')
parser.add_argument(
    '--trnln',
    type=int,
    default=300,
    help='rolling length of training')
parser.add_argument(
    '--tstln',
    type=int,
    default=16,
    help='rolling length of test')


to_use = [81, 95, 101, 155, 157, 166, 177, 198, 200, 231, 336, 373, 405, 417, 442, 481, 496, 516, 576, 642, 653, 750, 766, 2206, 2208, 2211, 2223, 2224, 2228, 2244, 2252, 2267, 2286, 2296, 2303, 2309, 2352, 2354, 2359, 2368, 2389, 2396, 2419, 2436, 2443, 2466, 2468, 2484, 2486, 2488, 2507, 2523, 2582, 2585, 2624, 2626, 2627, 2651, 2669, 2672, 2675, 2684, 2695, 2710, 2726, 2740, 2772, 2778, 2802, 2827, 2830, 2852, 2886, 2889, 2922, 2933, 2941, 2947, 2951, 2983, 3004, 3036, 3051, 3063, 3199]
good_stocks = []
for i, ind in enumerate(to_use):
    if i >= 50 and i < 55:
        continue
    good_stocks.append(ind)
print(good_stocks)

feature_cols = ['alpha_100w', 'amount_21', 'amount_5',
       'amount_63', 'amount_div', 'avg_volume_21', 'avg_volume_5',
       'avg_volume_63', 'beta_100w', 'close_low_high', 'close_s_vwap5',
       'close_vwap5', 'c_l2_ibm', 'dlt_miclo', 'highlow_1', 'highlow_12',
       'highlow_3', 'highlow_6', 'ibm_close', 'ibm_svlo', 'IR_netasset_252',
       'IR_roe_252', 'l2_ibm_ewma', 'l2_lbm_ewma', 'magm_yop',
       'ma_crossover_15_36', 'net_assets', 'n_buy_value_small_order', 'pb',
       'pcf_gm', 'period_return', 'q_s_fa_yoyocf', 'rank_amount_div',
       'rank_close_low_high', 'rt_10', 'rt_126', 'rt_12_1', 'rt_15', 'rt_21',
       'rt_252', 'rt_5', 'rt_5_Skewness_10', 'rt_5_Skewness_15',
       'rt_5_Skewness_20', 'rt_5_Skewness_5', 'rt_63', 'std_deviation_100w',
       'sw_first_industry', 's_dq_mv', 's_price_div_dps', 's_val_mv',
       'trade_status', 'trk_rk_pe_re', 'ttm_pcf', 'ttm_pe', 'ttm_ps',
       'ttm_roa', 'ttm_roe', 'turnover_21', 'turnover_5', 'turnover_63',
       'vol_1', 'vol_12', 'vol_3', 'vol_6', 'yeildvol_1m', 'yeildvol_3m',
       'yeildvol_6m', 'yop_pcf', 'yop_pe', 'z_rank_pe', 'z_sde_pe']

no_cheating_feature = []
for i, col in enumerate(feature_cols):
    if col == 'period_return' or col == 'sw_first_industry' or col == 'trade_status' or col == 's_price_div_dps':
        continue
    no_cheating_feature.append(i)

print('Reading data')
m = np.load('features_processed.npz', allow_pickle = True)
m = m['arr_0']
print(m.shape)
m = m[:,good_stocks,:]
m = m[:,:,no_cheating_feature]
print(m.shape)

price = np.load('Y_cl.npy')
print(price.shape)
price_next_point = price[1:]
price_today = price[:-1]
price_ratio = np.divide(price_next_point, price_today)

returns = np.log(price_ratio)
print(returns.shape)
cnt = 0
for i in range(len(returns)):
    for j in range(len(returns[i])):
        if np.isnan(returns[i][j]):
            cnt += 1
            returns[i][j] = 0.0

D = {}

def get_preprocess_stock(data):
    "data is M * F"
    data = np.array(data, dtype = np.float32)
    a = np.zeros((3, data.shape[-1]))
    t = np.nan_to_num(data, nan = np.nan, neginf = 1e9)
    a[0, :] = np.nanmin(t, axis = 0)
    t = np.nan_to_num(data, nan = np.nan, posinf = -1e9)
    a[2, :] = np.nanmax(t, axis = 0)
    for i in range(data.shape[-1]):
        data[:,i] = np.nan_to_num(data[:,i], nan = np.nan, posinf = a[2,i], neginf = a[0,i])
        try:
            data[:,i] = (data[:,i] - a[0,i]) / (a[2,i] - a[0,i])
        except:
            if i not in D.keys():
                D[i] = 0
            D[i] += 1
            print(i)
            print(data[:,i])
    for i in range(data.shape[-1]):
        nan_value = 0.0 if np.nanmean(data[:,i]) == np.nan else np.nanmean(data[:,i])
        data[:,i] = np.nan_to_num(data[:,i], nan = nan_value)
        a[1, i] = nan_value
    return data, a

def get_preprocess(data):
    A = []
    for i in range(data.shape[1]):
        data[:,i,:], a = get_preprocess_stock(data[:,i,:])
        A.append(a)
    return data, A

def preprocess_stock(data, a):
    for i in range(data.shape[-1]):
        data[:,i] = np.nan_to_num(data[:,i], nan = a[1,i], posinf = a[2,i], neginf = a[0,i])
    for i in range(data.shape[0]):
        a[0,:] = np.minimum(a[0,:], data[i,:])
        a[2,:] = np.maximum(a[2,:], data[i,:])
        for j in range(data.shape[-1]):
            try:
                data[i,j] = (data[i,j] - a[0,j]) / (a[2,j] - a[0,j])
            except:
                print("!!!!!!\n\n")
                print(i,j)
    return data

def preprocess(data, A):
    for i in range(data.shape[1]):
        data[:,i,:] = preprocess_stock(data[:,i,:], A[i])
    return data

args = parser.parse_args()
rolling_train_length = args.trnln
rolling_test_length = args.tstln
for ind, i in enumerate(range(rolling_train_length, len(m), rolling_test_length)):
    train, test = copy.deepcopy(m[i-rolling_train_length:i,:,:]), copy.deepcopy(m[i:i+rolling_test_length,:,:])
    train, a = get_preprocess(train)
    test = preprocess(test, a)
    if not os.path.exists('./rolling_' + str(rolling_train_length) + '_' + str(rolling_test_length)):
        os.makedirs('./rolling_' + str(rolling_train_length) + '_' + str(rolling_test_length))
    np.save('./rolling_' + str(rolling_train_length) + '_' + str(rolling_test_length) + '/features_train_' + str(ind) + '.npy', train)
    np.save('./rolling_' + str(rolling_train_length) + '_' + str(rolling_test_length) + '/features_test_' + str(ind) + '.npy', test)
    np.save('./rolling_' + str(rolling_train_length) + '_' + str(rolling_test_length) + '/ranks_train_' + str(ind) + '.npy', returns[i-rolling_train_length:i, good_stocks])
    np.save('./rolling_' + str(rolling_train_length) + '_' + str(rolling_test_length) + '/ranks_test_' + str(ind) + '.npy', returns[i:i + rolling_test_length, good_stocks])
    print(np.max(train, axis = (0,1)))
    print(np.max(test, axis = (0,1)))
