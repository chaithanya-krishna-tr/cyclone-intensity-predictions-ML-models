import numpy as np
#np.random.seed(11)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import h5py
import os.path as osp
import os
from scipy import ndimage
from glob import glob
from tqdm import tqdm
import sys
import warnings 
warnings.filterwarnings('ignore')
'''
Functions used throughout the project.

Set data_root to where your data is saved.
'''

# data_root = '/raid/data/hurricane/'
data_root = ''

hand_features = ['vs0', 'PSLV_v2', 'PSLV_v3', 'PSLV_v4', 'PSLV_v5', 'PSLV_v6', 'PSLV_v7',
                 'PSLV_v8', 'PSLV_v9', 'PSLV_v10', 'PSLV_v11', 'PSLV_v12', 'PSLV_v13',
                 'PSLV_v14', 'PSLV_v15', 'PSLV_v16', 'PSLV_v17', 'PSLV_v18', 'PSLV_v19',
                 'MTPW_v2', 'MTPW_v3', 'MTPW_v4', 'MTPW_v5', 'MTPW_v6', 'MTPW_v7',
                 'MTPW_v8', 'MTPW_v9', 'MTPW_v10', 'MTPW_v11', 'MTPW_v12', 'MTPW_v13',
                 'MTPW_v14', 'MTPW_v15', 'MTPW_v16', 'MTPW_v17', 'MTPW_v18', 'MTPW_v19',
                 'MTPW_v20', 'MTPW_v21', 'MTPW_v22', 'IR00_v2', 'IR00_v3', 'IR00_v4',
                 'IR00_v5', 'IR00_v6', 'IR00_v7', 'IR00_v8', 'IR00_v9', 'IR00_v10',
                 'IR00_v11', 'IR00_v12', 'IR00_v13', 'IR00_v14', 'IR00_v15', 'IR00_v16',
                 'IR00_v17', 'IR00_v18', 'IR00_v19', 'IR00_v20', 'IR00_v21', 'CSST_t24',
                 'CD20_t24', 'CD26_t24', 'COHC_t24', 'DTL_t24', 'RSST_t24', 'U200_t24',
                 'U20C_t24', 'V20C_t24', 'E000_t24', 'EPOS_t24', 'ENEG_t24', 'EPSS_t24',
                 'ENSS_t24', 'RHLO_t24', 'RHMD_t24', 'RHHI_t24', 'Z850_t24', 'D200_t24',
                 'REFC_t24', 'PEFC_t24', 'T000_t24', 'R000_t24', 'Z000_t24', 'TLAT_t24',
                 'TLON_t24', 'TWAC_t24', 'TWXC_t24', 'G150_t24', 'G200_t24', 'G250_t24',
                 'V000_t24', 'V850_t24', 'V500_t24', 'V300_t24', 'TGRD_t24', 'TADV_t24',
                 'PENC_t24', 'SHDC_t24', 'SDDC_t24', 'SHGC_t24', 'DIVC_t24', 'T150_t24',
                 'T200_t24', 'T250_t24', 'SHRD_t24', 'SHTD_t24', 'SHRS_t24', 'SHTS_t24',
                 'SHRG_t24', 'PENV_t24', 'VMPI_t24', 'VVAV_t24', 'VMFX_t24', 'VVAC_t24',
                 'HE07_t24', 'HE05_t24', 'O500_t24', 'O700_t24', 'CFLX_t24', 'DELV-12']




def compute_metrics(y_true, y_predict, print_them=False):
    me=[]
    for i in range(len(y_true)):
        me.append(y_true[i]-y_predict[i])
    mean_error=np.mean(me)
    standard_deviation=np.std(me)
    metrics = {'MAE': mean_absolute_error(y_true, y_predict),
               'RMSE': np.sqrt(mean_squared_error(y_true, y_predict)),
               'R^2': r2_score(y_true, y_predict),
               'ME': mean_error,
               'STD': standard_deviation
              }
    if print_them:
        for k, v in metrics.items():
            print(f'{k}: {v:.2f}')
        print()
    return metrics


def load_loyo_data(leave_out_year, get_hand=False, get_images=False, scale=False, remove_oprreadup=False, remove_oprfortraining=False, data_root=data_root):
    df = pd.read_csv(osp.join(data_root, 'train_global_fill_REA_na_wo_img_scaled.csv')) #58995 rows
    #df = pd.read_csv(osp.join(data_root, 'train_global_fill_na_w_img_scaled.csv')) # 38k data
    # train
    train_df = df.loc[~((df.basin=='AL') & (df.year==leave_out_year))]
    # if remove duplicated opr and rea training events (the rea part)for AL 2010-2018:
    if remove_oprreadup:
        train_df = train_df.loc[~((train_df.type=='rea') & (train_df.basin=='AL') & (train_df.year>=2010))]
    # remove all opr data points for training:
    if remove_oprfortraining:
        train_df = train_df.loc[~(train_df.type=='opr')]
    ids = train_df['name'].values
    y_train = train_df[['dvs24']].values
    # test
    test_df = df.loc[((df.year==leave_out_year) & (df.type=='opr'))]
    y_test = test_df[['dvs24']].values

    # hand features
    if get_hand:
        x_train_hand = train_df[hand_features].values
        x_test_hand  = test_df[hand_features].values

    return x_train_hand, x_test_hand, y_train, y_test, ids

