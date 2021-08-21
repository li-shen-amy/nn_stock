# nn_stock
Implement Nearest Neighbor method to predict stock

Creates forecasts of a time series on t+1 using nearest neighbour algorithm.

## Usage:
```.py
[OutSample_For,InSample_For,InSample_Res]=nearest_neighbour.nn(data, train_len, embed_dim, k_nearest, method='correlation', n_out_sample=0)
```
### INPUT:  

data - The time series to be forecasted (the function was originally made for stock price, but it accepts any kind of time series).

train_len - the observation where the InSample forecasts will start. It also defines the training period of the algorithm. 
            For example, if len(data)=500 and train_len=400, the values of 1:400 will be the training period for the forecasted value
            of 401. For the forecast of 402, the training period is 1:401, meaning that each time a new observation is available, the
            algorithm adds it to the training period. Please notes that the parameter train_len doesn't have any effect on the out of 
            sample forecasts.
              
embed_dim - Embedding dimension (size of the histories)
              
k_nearest - The number of nearest neighbours to be used in the construction of the forecasts

n_out_sample - Number of outsample Forecasts. The default value is 0.

### OUTPUT:
OutSample_For - A vector with the out of sample forecasted values of the time series. The length of [OutSample_For] is n_out_sample. 
                Notes that the out sample forecasts are build with the whole modeled series data. 
               
InSample_For  - A vector with the in sample forecasted values of the time series. The length of [InSample_For] is length(data)-embed_dim    

InSample_Res  - A vector with the out-of-sample residues from the in-sample forecasts.
              
## Example:
```.py
      train_len = 50
      embed_dim = 3
      k_nearest = 20
      method_1 = 'correlation'
      method_2 = 'absolute_distance'
      [OutSample_For_Corr, InSample_For_Corr, InSample_Res_Corr] = nearest_neighbour.nn(data, train_len, embed_dim, k_nearest, method_1)
      [OutSample_For_Abs, InSample_For_Abs, InSample_Res_Abs] = nearest_neighbour.nn(data, train_len, embed_dim, k_nearest, method_2)
```
Predicitions:
<p align="center">
  <img src="https://github.com/li-shen-amy/profile/blob/master/images/projects/nn_stock1.jpg" />
</p>
<p align="center">
  <img src="https://github.com/li-shen-amy/profile/blob/master/images/projects/nn_stock2.jpg" />
</p>
