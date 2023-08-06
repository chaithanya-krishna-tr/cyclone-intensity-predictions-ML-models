#Importing Dependencies

import sys
import numpy as np
np.random.seed(int(1))
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(int(1))
import utils
import models
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
import os
import datetime
import seaborn as sns

if __name__ == '__main__':
    
    # input architecture 
    architecture = "random_forest"

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, restore_best_weights=True)]

    fit_kwargs = {'epochs': 1,
                  'verbose': 2,
                  'callbacks': callbacks}

    # define plot lists
    x_plot = []
    y_plot = []
    y_error = []

    # write the csv header
    colNames = ['Leave Out Year', 'MAE', 'RMSE', 'R^2','ME','STD']
    # create directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'LOYO_results', r'seeds')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    if architecture == "mlp":
        filename = 'LOYO_results/seeds/LOYO_np' +str(1)  +'_tf' +str(1) +'_mlp'+ '.csv'
    elif architecture == "lasso":
        filename = 'LOYO_results/seeds/LOYO_np' +str(1)  +'_tf' +str(1) +'_lasso'+ '.csv'
    elif architecture == "linear":
        filename = 'LOYO_results/seeds/LOYO_np' +str(1)  +'_tf' +str(1) +'_linear'+ '.csv'
    elif architecture == "random_forest":
        filename = 'LOYO_results/seeds/LOYO_np' +str(1)  +'_tf' +str(1) +'_rf'+ '.csv'
    with open(filename, 'a+') as f:
        line = ','.join(colNames) + ','
        f.write(line + '\n')
    flag = True
    for leave_out_year in range(2010, 2019): #(2020, 2021) <<<< update years here
        # Load data
        print(f'Loading data for year: {leave_out_year}...')
        if architecture == 'mlp':
            x_train, x_test, y_train, y_test, ids= utils.load_loyo_data(leave_out_year, scale=True, get_hand=True, remove_oprreadup=False,remove_oprfortraining=True)
        elif architecture == "linear":
            x_train, x_test, y_train, y_test, ids = utils.load_loyo_data(leave_out_year, scale=True, get_hand=True, remove_oprreadup=False,remove_oprfortraining=True)
        elif architecture == "lasso":
            x_train, x_test, y_train, y_test, ids = utils.load_loyo_data(leave_out_year, scale=True, get_hand=True, remove_oprreadup=False,remove_oprfortraining=True)
        elif architecture == "random_forest":
            x_train, x_test, y_train, y_test, ids = utils.load_loyo_data(leave_out_year, scale=True, get_hand=True, remove_oprreadup=False,remove_oprfortraining=True)
        else:
            raise Exception(f'Invalid architecture name: {architecture}')
        train_hurricane_names = ids

        # Init CV
        n_splits = 1
        ss = ShuffleSplit(n_splits=n_splits, test_size=0.1)
        metrics = {'MAE': [],
                   'RMSE': [],
                   'R^2': [],
                   'ME': [],
                   'STD': []}
        
        
        # Cross validation loop
        for i, (train_idxs, val_idxs) in enumerate(ss.split(train_hurricane_names)):
            print(f'\n--- Fold {i+1} of {n_splits} ---')

            if architecture == 'mlp':
                # Extract CV fold
                x_train_cv = x_train[train_idxs]
                y_train_cv = y_train[train_idxs]
                x_val = x_train[val_idxs]
                y_val = y_train[val_idxs]
                # Train
                model = models.mlp(input_shape=x_train_cv[0].shape)
                model.fit(x_train_cv, y_train_cv, batch_size=32,
                          validation_data=(x_val, y_val),
                          **fit_kwargs)
                y_predict = model.predict(x_test)
            if architecture == "linear":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                print('Fitting model...')
                model.fit(x_train, y_train)
                y_predict=model.predict(x_test)
                
            if architecture == "lasso":
                from sklearn.linear_model import LassoCV
                print('Fitting model...')
                model = LassoCV(cv=5, random_state=0, max_iter=2000)
                model.fit(x_train, y_train)
                y_predict = model.predict(x_test)
                
            if architecture == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                print('Fitting model...')
                model = RandomForestRegressor(max_depth=200, random_state=0, n_estimators=100, n_jobs=3, verbose=1)
                model.fit(x_train, y_train)
                y_predict = model.predict(x_test)
                
                if flag == True:
                    feature_importance = model.feature_importances_
                    sorted_idx = feature_importance.argsort()[::-1]
                    sorted_features = np.array(utils.hand_features)[sorted_idx]
                    sorted_importance = feature_importance[sorted_idx]
                    fig_width = len(sorted_features)*0.5
                    fig_height = fig_width*1.5
                    fig,ax = plt.subplots(figsize=(fig_width, fig_height))
                    sns.barplot(x=sorted_importance, y=sorted_features, palette="Blues_d")
                    ax.set_title("Feature Importance (Random Forest)")
                    ax.tick_params(axis='both', which='major', labelsize=15)
                    ax.set_xlabel("Importance",fontsize = 15)
                    ax.set_ylabel("Feature",fontsize = 15)
                    plt.tight_layout()
                    plt.savefig(f"LOYO_results/seeds/LOYO_np{1}_tf{1}_rf_feature_importance.png")
                    plt.show()
                    flag = False
               


            tmp_metrics = utils.compute_metrics(y_test, y_predict, print_them=True)
            metrics = {k: v+[tmp_metrics[k]] for k, v in metrics.items()}

            # output predicted labels along with true labels for the RI analysis
            df_pred = pd.DataFrame({'y_test':list(y_test.reshape([-1,])), 'y_predict':list(y_predict.reshape([-1,]))})
            if architecture == "mlp":
                pred_filename = 'LOYO_results/LOYO_pred_np' +str(1)  +'_tf' +str(1) + str(leave_out_year) +"_mlp"+ '.csv'
            elif architecture == "lasso":
                pred_filename = 'LOYO_results/LOYO_pred_np' +str(1)  +'_tf' +str(1) + str(leave_out_year) +"_lasso"+ '.csv'
            elif architecture == "random_forest":
                pred_filename = 'LOYO_results/LOYO_pred_np' +str(1)  +'_tf' +str(1) + str(leave_out_year) +"_rf"+ '.csv'
            elif architecture == "linear":
                pred_filename = 'LOYO_results/LOYO_pred_np' +str(1)  +'_tf' +str(1) + str(leave_out_year) +"_linear"+ '.csv'
            df_pred.to_csv(pred_filename, index=False)

        # Print metrics
        print(f'\n--- Cross Validation Test Metrics for year: {leave_out_year} ---')
        for k, v in metrics.items():
            print(f'{k}: {np.mean(v):.2f} +/- {2 * np.std(v):.2f}')

        # write results to csv
        with open(filename, 'a+') as f:
            for name in colNames:
                if name == 'Leave Out Year':
                    line = leave_out_year
                else:
                    line = f'{np.mean(metrics[name]):.2f} +/- {2 * np.std(metrics[name]):.2f}'
                    if name == "MAE":
                        x_plot.append(leave_out_year)
                        y_plot.append(np.mean(metrics[name]))
                        y_error.append(2 * np.std(metrics[name]))
                f.write(str(line) + ',')
            f.write('\n')

    # plot values
    x_labels = list(map(lambda x: str(x), x_plot))
    plt.figure()
    plt.errorbar(x_plot, y_plot, yerr=y_error)
    plt.title("MAE vs Year Left Out")
    plt.xlabel("Year Left Out")
    plt.xticks(rotation=45)
    plt.ylabel("Mean Absolute Error")
    plt.xticks(x_plot, x_labels)
    plt.savefig(filename[:-4] + ".png")
