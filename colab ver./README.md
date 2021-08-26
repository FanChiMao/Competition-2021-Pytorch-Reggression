# Colab-2021-IMBD-Reggression

# 1. Training  

## 1.1 Prepair training data  
- Official [training data](https://drive.google.com/file/d/1xj7Wpev5k48hP6nBoEFJURd-hoPy4Bzv/view?usp=sharing): 98072  

- Split training data to train, val and test part:  
  - [train.csv](https://drive.google.com/file/d/1L389britWH1_e1Xb_3XACHeV0Yz2RwqV/view?usp=sharing): 97000  
  - [val.csv](https://drive.google.com/file/d/1dZtR1xRfyLnoGqfuenvAWMCxprxZ8D3K/view?usp=sharing): 1000  
  - [test.csv](https://drive.google.com/file/d/1AShQtKNL_d_ePbihX2n2lEyrsGCP5fJs/view?usp=sharing): 72  

## 1.2 Upload relevant csv data to colab  
- Pretrained model: [13-128-256-128-1](https://drive.google.com/file/d/1iimuaBDnGSLTyGZLR-vB-O9tGtkMIeo1/view?usp=sharing)  
- Example:  
<img src="https://i.ibb.co/jw8VjJ5/image.jpg" alt="image" border="0"></a>  


## 1.3 Training and validation loss curve  



# 2. Testing

## 2.1 Load the model and test  
- Model weight file direction: `checkpoints -> model` folder  
- Start testing: `test.py`  
    ```
    python test.py
    ```
    
## 2.2 Score  
- Official score calculate rule:  

  <img src="figures/score_rule.JPG" alt="arch" width="500" style="zoom:100%;" />  
  
- Final score example:  

  <img src="figures/score.jpg" alt="arch" style="zoom:100%;" />  


# 3. Reference  
- https://lulaoshi.info/machine-learning/neural-network/pytorch-kaggle-house-prices.html  
- https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1  
- https://blog.csdn.net/just_sort/article/details/103110806  

  
# 4. Contact us:  
- Chi-Mao Fan: qaz5517359@gmail.com  
- 
