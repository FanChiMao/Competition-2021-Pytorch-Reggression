# Pytorch-2021-IMBD-Reggression
- [2021全國智慧製造大數據分析競賽](https://imbd2021.thu.edu.tw/)  

```
├── README.md    

主要訓練程式碼
├── train.py                執行訓練檔
├── training.yaml           調整訓練參數
├── dataset.py              讀取訓練驗證資料
├── model.py                網路架構

主要測試程式碼   
├── test.py                 還沒寫      

其他程式碼
├── utils                   相關函式
|   ├── csv_utils           
|   ├── dir_utils
|   ├── model_utils
├── csv_data                csv檔案資料夾
|   ├── result              預計放預測結果的csv檔      
|   |   ├── result.csv      結果檔    
|   ├── teesting
|   |   ├── test.csv        
|   ├── training
|   |   ├── val.csv    
|   |   ├── train.csv    
├── checkpoint              儲存模型的權重檔資料夾
|   ...  
......

```

# 1. Training  

## 1.1 Prepair training data  
- Official [training data](https://drive.google.com/file/d/1xj7Wpev5k48hP6nBoEFJURd-hoPy4Bzv/view?usp=sharing): 98072  

- Split training data to train, val and test part:  
  - [train.csv](https://drive.google.com/file/d/1L389britWH1_e1Xb_3XACHeV0Yz2RwqV/view?usp=sharing): 97000  
  - [val.csv](https://drive.google.com/file/d/1dZtR1xRfyLnoGqfuenvAWMCxprxZ8D3K/view?usp=sharing): 1000  
  - [test.csv](https://drive.google.com/file/d/1AShQtKNL_d_ePbihX2n2lEyrsGCP5fJs/view?usp=sharing): 72  

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
- example:  



# 2. Testing

## 2.1 Load the model and test  
- Model weight file direction: `checkpoints -> model` folder  
- Start testing: `test.py`  
    ```
    python test.py
    ```
    
## 2.2 Score  
- Official score calculate rule:  
- example:  







