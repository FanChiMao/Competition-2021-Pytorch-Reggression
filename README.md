# Pytorch-2021-IMBD-Reggression
- [2021全國智慧製造大數據分析競賽](https://imbd2021.thu.edu.tw/)  

```
└── README.md 

主要訓練程式碼
├── train.py                執行訓練檔
├── training.yaml           調整訓練參數
├── dataset.py              讀取訓練驗證資料
├── model.py                網路架構
└── checkpoint              訓練完成存模型及log的資料夾

主要測試程式碼   
├── demo.py                 執行預測並匯出結果csv
└── test.py                 執行預測並計算成績(有答案)

其他程式碼
├── utils
|    ├── csv_utils          csv檔相關函式
|    ├── dir_utils          路徑相關函式
|    ├── model_utils        網路模型相關函式
|    └── score_utils        計算分數相關函式
├── csv_data
|    ├── testing            測試csv資料夾    
|    └── training           訓練csv資料夾
└── colab ver_              Colab版本
```  
# 0. Competition result  
- Preliminary  
    - [Explain File](https://drive.google.com/file/d/1Gpo4ZX1VsPFwPaziXvJywOSG_OVPekNr/view?usp=sharing)  
    - [Test Result](https://drive.google.com/file/d/1rZZ0k7-cOi32kJDaIHenxfYZJqKRJcFo/view?usp=sharing)  
    - [Test Report](https://drive.google.com/file/d/1vYPjHpsvyp_bYV4o27Jkvc-ftF0_sLav/view?usp=sharing)  

# 1. Training   

## 1.1 Prepair training data  
- Official [training data](https://drive.google.com/file/d/1xj7Wpev5k48hP6nBoEFJURd-hoPy4Bzv/view?usp=sharing): 98072  
- Data format (Our objective is entering the input features `F1 ~ F13` and predict the final `Output`)  

    | Data number | F1  | F2  | F3  | F4  | F5  | F6  | F7  | F8  | F9  | F10  | F11  | F12  | F13  |Output|  
    | ----------- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|:----:|:----:|:----:|  
    | 1           |0	|23.5 |23.6 |23.6 |23.6	|23.8 |24.3 |23.6 |23.5 |22.6  |23.3  |	23.1 |22.3  |0     |  
    | 2           |0	|23.5 |23.6 |23.6 |23.6	|23.8 |24.3 |23.6 |23.5 |22.6  |23.3  |	23.1 |22.3  |-0.6  |  
    | 3           |0	|23.5 |23.5 |23.6 |23.6	|23.8 |24.3 |23.6 |23.5 |22.6  |23.3  |	23.1 |22.3  |0.6   |  
    | 4           |0	|23.5 |23.5 |23.6 |23.6	|23.8 |24.3 |23.6 |23.5 |22.6  |23.3  |	23.1 |22.3  |-0.6  |  
    | 5           |0	|23.5 |23.6 |23.6 |23.6	|23.8 |24.3 |23.6 |23.5 |22.6  |23.3  |	23.1 |22.3  |-0.3  |  
    | ......      |...	|...  |...  |...  |...  |...  |...  |...  |...  |...   |...   |...   |...   |...   |  
    
- CSV to independent data  
  Because the training data has some deviations which the same input feature values get different output results as showed below:  
  
    | Data number | F1  | F2  | F3  | F4  | F5  | F6  | F7  | F8  | F9  | F10  | F11  | F12  | F13  |Output|  
    | ----------- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|:----:|:----:|:----:|  
    | 1           |0	|23.5 |23.6 |23.6 |23.6	|23.8 |24.3 |23.6 |23.5 |22.6  |23.3  |	23.1 |22.3  |0     |  
    | 2           |0	|23.5 |23.6 |23.6 |23.6	|23.8 |24.3 |23.6 |23.5 |22.6  |23.3  |	23.1 |22.3  |-0.6  |  
    | ......      |...	|...  |...  |...  |...  |...  |...  |...  |...  |...   |...   |...   |...   |...   |  
    
  You could run `csv_utils.py` to let all training data are independent with mean or mediam value of output.  
    
## 1.2 Set hyperparameters and train  
- Configuration file: `training.yaml`  

  ```
    TRAINING:
      Network: 'MLP'
      EPOCH: 1000
      LR: 0.01
      LR_MIN: 0.0001
      GPU: true
      BATCH: 1000
      VAL_RATE: 0.8  # split validation set from training set
      VAL_AFTER_EVERY: 1  # save the model per ? epoch
      TRAIN_DIR: './csv_data/training/independent_mean.csv'  # path to training data
      SAVE_DIR: './checkpoints'  # path to save models and images

  ```
  
- Start training: `train.py`  

    ```
    python train.py
    ```  
## 1.3 Training and validation loss curve  
- log file direction: `checkpoints -> log` folder  

    ```
    tensorboard --logdir [log path]
    ```

# 2. Testing
## 2.1 Prepair preliminary testing data  
- Official [preliminary testing data](https://drive.google.com/file/d/17b03rxEfXTGlcSLJCv-W-ctTWsYwhA3c/view?usp=sharing): 7222  


## 2.2 Load the model and test  
- Pretrained model:  
  - [MLP 13-128-256-128-1 mean 0.8](https://drive.google.com/file/d/1vpIN1N8Xfaxp7h4iz8obXfX7Oy1u_6Um/view?usp=sharing)  
  - [MLP 13-128-256-128-1 mean 0.9](https://drive.google.com/file/d/17m4Hl1eZGfNbYU0eFcxVHqWXHlkVBTpc/view?usp=sharing)  
  - [MLP 13-128-256-128-1 median 0.8](https://drive.google.com/file/d/128De9g4p_i0reXA7YYXJaRxXq6GyYya9/view?usp=sharing)  
  - [MLP 13-128-256-256-256-128-1 mean 0.8](https://drive.google.com/file/d/1K6RQYvPYYKmWcTYGP6Y7ZVRK2Os61_iz/view?usp=sharing)  


- Model weight file direction: `checkpoints -> model` folder  
- Start testing: `demo.py`  
  
    ```
    python demo.py  
    ```
    
## 2.3 Score  
- Official score calculate rule:  
  <img src="https://github.com/FanChiMao/Pytorch-2021-IMBD-Reggression/blob/main/figures/score_rule.JPG" alt="arch" width="600" style="zoom:100%;" />  

# 3. Reference  
- https://lulaoshi.info/machine-learning/neural-network/pytorch-kaggle-house-prices.html  
- https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1  
- https://blog.csdn.net/just_sort/article/details/103110806  



