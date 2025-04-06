import torch, pickle, pdb 
from datetime import datetime, timedelta 
from data_proc import DataProcConfig, get_single_action_model_train_test_data_from_config
    
date_format = "%Y-%m-%d"
datetime_format = "%Y-%m-%d %H:%M:%S"

test_mode = False

abs_start_date = '2023-12-25'
abs_stop_date = '2025-04-01'
tr_days = 360 
bs_days = 25 
get_news_features = True

abs_stop_date_T = datetime.strptime(abs_stop_date, date_format)

interval_days = 30
monthly_interval = timedelta(days=interval_days)
yearly_interval = timedelta(days=360)
buy_sell_nonoverlap_interval = timedelta(days=30)
training_data_start_date = abs_start_date

data_proc_config = DataProcConfig(
    training_time_length_days = tr_days,
    buy_sell_time_length_days = bs_days,
    training_data_start_date = training_data_start_date,
    test_data_start_date = (datetime.strptime(training_data_start_date, date_format) + buy_sell_nonoverlap_interval).strftime(datetime_format)[:10].strip(),
    data_list_file = f"/home/ubuntu/code/HCL/invest/data/data_list_{abs_start_date}_{abs_stop_date}_tr{tr_days}d_bs{bs_days}d_{interval_days}dinterval_newsFeature{get_news_features}_testmode{test_mode}.txt",
    is_prod = False,
    get_news_features=get_news_features,
)

while datetime.strptime(data_proc_config.test_data_start_date.strip(), date_format) + timedelta(days=data_proc_config.training_time_length_days) + buy_sell_nonoverlap_interval < abs_stop_date_T:
    print("--------- processing dates: ---------")
    print("training_data_start_date: "+data_proc_config.training_data_start_date)
    print("test_data_start_date: "+data_proc_config.test_data_start_date)
    get_single_action_model_train_test_data_from_config(
        data_proc_config, 
    )
    data_proc_config.training_data_start_date = (datetime.strptime(data_proc_config.training_data_start_date.strip(), date_format) + monthly_interval).strftime(datetime_format)[:10].strip()
    data_proc_config.test_data_start_date = (datetime.strptime(data_proc_config.training_data_start_date.strip(), date_format) + buy_sell_nonoverlap_interval).strftime(datetime_format)[:10].strip()

print('processing finished, data_list_file:')
print(data_proc_config.data_list_file)
