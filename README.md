## ListFold

Files needed to run the code: `factor_data.npy`, `Y_cl.npy`. 

`factor_data.npy` contains the features needed for A share stocks and can be downloaded from 

`Y_cl.npy` contains the corresponding weekly price and can be downloaded from 

### Flowchart

1. We first run `first_process.py`, which would turn `factor_data.npy` to `features_processed.npz`. It will change all `True, False` to `1, 0` and compress the factor data. 

   Estimated time of running: 5 minutes. 

2. Then we run `process_rolling.py`. Now the default training and testing length is 300 16. This code will rolling split the data (features and prices) and put them in a folder.  

   Estimated time of running: 5 minutes, also depends on training and testing length. 

3. Then we train the model using `train_20200111_rolling_torch.py`. It will save all the models to a folder. An example run is `python train_20200111_rolling_torch.py --pp 21`.Here `21` denotes the total number of train-test pairs.  

   Estimated time of running: 1 hour, also depends on training and testing length. 
   
4. For back test, an example run is `python back_test.py`, which will generate a csv file, recording all positive and negative positions and combined return each week. 

### Arguments

In `train.py`, we can control training and testing length, batch size, training epochs. Use `python train.py -h` to see details. 

In `back_test.py`, we can pick different saved models to load. Details see `python back_test.py -h`. 

### MSE

We don't have `relu` for MSE loss, so we use `train_mlp.py` and `back_test_mlp.py`, who have slightly different network structures. 

