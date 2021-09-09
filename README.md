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
|    ├── csv_utils           csv檔相關函式
|    ├── dir_utils           路徑相關函式
|    ├── model_utils         網路模型相關函式
|    └── score_utils         計算分數相關函式
├── csv_data
|    ├── testing             測試csv資料夾    
|    └── training            訓練csv資料夾
└── colab ver_               Colab版本
```  

# 1. Training   

## 1.1 Prepair training data  
- Official [training data](https://drive.google.com/file/d/1xj7Wpev5k48hP6nBoEFJURd-hoPy4Bzv/view?usp=sharing): 98072  

## 1.2 Set hyperparameters and train  
- Configuration file: `training.yaml`  
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
- Model weight file direction: `checkpoints -> model` folder  
- Start testing: `demo.py`  
    ```
    python test.py
    ```
    
## 2.3 Score  
- Official score calculate rule:  

  <img src="figures/score_rule.JPG" alt="arch" width="500" style="zoom:100%;" />  

# 3. Reference  
- https://lulaoshi.info/machine-learning/neural-network/pytorch-kaggle-house-prices.html  
- https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1  
- https://blog.csdn.net/just_sort/article/details/103110806  

  
# 4. Contact us:  
- Chi-Mao Fan: qaz5517359@gmail.com  
- Yu-Fang Huang: lin12099@yahoo.com.tw  
- Kai-Hua Yeh: kateyehyeh@gmail.com  
- 


