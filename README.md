# stock_directional-prediction_powered_by_gcn
This project is built on GCN for machine learning, which helps predict one day or 5 days ahead of stock direction. The steps to achieve such a goal involve contextual data gathering, hyperparameters tuning and feature engineering.
******MAKE SURE YOU HAVE PLACED THE MODEL FOLDERS AND STOCK DATA FILES NAMED AS "stock_data_10years_yyyymmdd.csv" UNDER THE SAME DIRECTORY WHEN YOU TUNE AND DEPLOY THIS MODEL ON YOUR COMUPTER******

HERE ARE THE FILES FOR YOUTO CUSTOMIZE YOUR OWN STOCK PREDICTOR:

./fetch_data_improved.py==>scrape 10 years of trading data and contextual data of stocks included in the script, you can scrape the stock you want by changing the 'Ticker' list at the beigning of the script

./tuned_gnn_models_5D.zip==>unzip this file to access the GCN models that predict 5 days ahead

./tuned_gnn_models.zip==>unzip this file to access the GCN models that predict 1 day ahead

./predict_all_tickers.py==>load the models in ./tuned_gnn_models/ to run prediction 1 day ahead

./tune_and_save_gnn.py==>my version of script to hyperparameters tune and feature engineer a new gcn model

