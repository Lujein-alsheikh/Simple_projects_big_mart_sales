In this read_me file, you can find: explanation of the problem and features, key moments in preprocessing the data, and designing the neural net. 

<details>
<summary>Problem statement</summary>
The goal is to predict the sales of different products across different big mart outlets using a neural network. This problem is found on the Analytics Vidhya
hackathon website so training set and test set are already split.
</details>

<details>
<summary>The features are</summary>

| Column | Description | 
| ---------|----------|
| Item_Identifier | Unique product ID | 
| Item_Weight | Weight of product | 
| Item_Fat_Content | Whether the product is low fat or not |	
| Item_Visibility | The % of total display area of all products in a store allocated to the particular product |
| Item_Type | The category to which the product belongs |
| Item_MRP | Maximum Retail Price (list price) of the product |
| Outlet_Identifier | Unique store ID |
| Outlet_Establishment_Year | The year in which store was established |
| Outlet_Size |	The size of the store in terms of ground area covered |
| Outlet_Location_Type | The type of city in which the store is located|
| Outlet_Type |	Whether the outlet is just a grocery store or some sort of supermarket |
| Item_Outlet_Sales | Sales of the product in the particular store. This is the outcome variable to be predicted|
</details>


<details>
<summary>Key moments or notes about preprocessing </summary>
Look for cells that start with "idea" or "Thinking moment" or "question". <br/>

+ <strong>Important observations about the data:</strong> The same outlet always has the same establishment year, size, location type, and type. Those columns are redundant. <br/>
An item always has the same weight, fat content, and type. But, two different items can still have the same weight. The weight, fat content, and item type can (almost) uniquely identify an item (there are ~200 items that can't be identified using those 3 pieces of information). <br/>
The problem really boils down to predicting sales for a specific product in a specific outlet. <br/>
I tried using only the item ids and outlet id's and got good results. <br/>

+ <strong>Should we keep the Item_Identifier column? Does it not cause leakage?</strong> Some items have similar sales across different outliers and some don't. There is no (immediate) leakage. I kept this column. Some contestants removed it and with different preprocessing methods than mine still got good results though. <br/>

+ <strong>How to encode the Item_Identifier column that has 1559 different categories?</strong> The library category_encoders comes in handy. I used binary encoding and was able to encode 1559 values using 11 columns only. <br/>

+ <strong>Creating a model to predict missing values of the "Outlet_Size" column: </strong> (I didn't try it): If we want to create a model to predict the missing values of "Outlet_Size", we can not use the original target variable as a feature because it causes data leakage. Another important question would be how to assess the accuracy of the predictions if there are no ground truth values to compare with?  <br/>

+ <strong>Weird performance from Python: df.groupby('Item_Identifier')['Item_Weight'].mean() gives strange results.</strong> For example, instead of returning 4.59 it returns 4.58999999. Another example, instead of returning 6.52 it returns 6.5200000005. I need that df.groupby('Item_Identifier')['Item_Weight'].mean() returns exactly the same values as in the dataset. I rounded the output. <br/>

+ <strong>Note about encoding:</strong> When applying binary encoding on item_ids, we have to .fit_transform on the whole dataset because we need the coding to be consistent. <br/>

+ <strong>Downside of one-hot-encoding:</strong> that the data becomes sparse. Some models might not work well with sparse data. I apply one-hot-encoding on two columns Item_Type and Outlet_Identifier and end up with 21 columns which is acceptable in my case. <br/>

+ <strong>Other ideas for encoding:</strong> frequency encoding or target encoding. Caution that target encoding leads to leakage.
</details>

<details>
<summary>Summary of how I preprocessed each column</summary>

| Column | info about this column | How to preprocess it? |
| ---------|----------|----------|
| Item_Identifier | 1559 unique values | binary encoding |
| Item_Weight | float | filled in missing values easily since each item has the same weight and the same item is repeated many times |
| Item_Fat_Content | - | label encoding |
| Item_Visibility | float | - |
| Item_Type | 16 categories | grouping sparse categories and then one-hot-encoding. We end up with 11 categories |
| Item_MRP | float | - |
| Outlet_Identifier | 10 categories | one-hot-encoding |
| Outlet_Establishment_Year | int | I treated it as numerical value even though it is discrete |
| Outlet_Size | 3 categories: small, medium, and high | missing values for outlets 10, 45, and 17. I fill them in with the mode (Medium) |
| Oulet_Location_Type | 3 categories: tiers 1,2, and 3 | label encoding |
| Outlet_Type | 4 categories: Sypermarket type 1,2, and 3 and grocery store | one-hot-encoding |
</details>

<details>
<summary>Outlets information</summary>

| Outlet_Identifier | Establishment_Year | Outlet_Size | Location_Type | Outlet_Type |
| ---------|----------|----------|----------|----------|
| 049 | 1999 | Medium | tier 1 | Supermarket Type 1 |
| 018 | 2009 | Medium | tier 3 | Supermarket Type 2 |
| 010 | 1998 |  | tier 3 | Grocery Store |
| 013 | 1987 | High | tier 3 | Supermarket Type 1 |
| 027 | 1985 | Medium | tier 3 | Supermarket Type 3 |
| 045 | 2002 | | tier 2 | Supermarket Type 1 |
| 017 | 2007 | | tier 2 | Supermarket Type 1|
| 046 | 1997 | Small | tier 1 | Supermarket Type 1 |
| 035 | 2004 | Small | tier 2 | Supermarket Type 1 |
| 019 | 1985 | Small | tier 1 | Grocery Store | 
</details>


<details>
<summary>Hypotheses testing</>

+ $H_{0}$: the average of meat sales equals the average of seafood sales. Test used: anova. Result: Fail to reject the null hypothesis. However, the result of this test is not reliable because not all requirements are met. More precisely, one requirement for anova is that the population standard deviations of all groups should all be equal. In reality, we don't know if this is true or not. Another requirement is that the distributions should be Gaussian which is clearly not the case. Another factor is the sizes of the groups. The seafood group is very small. </br>

+ Pair-wise t-tests of sales with respect to item types. Example from the results: there is a difference between the dairy and softdrinks groups. There is no difference between the dairy and seafood groups and there is no difference between the softdrinks and seafood groups. This inconsistency is due to the fact that the seafood group is very small.<br/>

+ For each item type whether there is a significant difference between the averages of its sales across the three different outlet locations (tiers) using anova. </br>

</details>

<details>
<summary>Building the NN and judging the performance and results</summary>

- I tried first a NN with one hidden layer only (of 500 neurons). I tried different activation functions and got negative prediction which is why I applied a custom loss function. The latter lowered the number of test data points whose predictions were negative when the activation function was leaky relu. When the activation function was relu, the custom function actually increased the number of negatively predicted values. Anyhow, adding a custom loss function didn't solve the problem. Conclusion: the baseline model does not work. <br/>

- I tried another NN with two hidden layers (500 and 100) which gave acceptable results namely, a loss of ~1300 on the public part of the test_data (on the analytics vidhya hackathon platform). <br/>

- I tried the latter NN using only the Item_Id and Outlet_Identifier which gave much better results: ~ 1158 loss on the public part of the test_data. <br/>

- No overfitting observed in neither neural nets I tried.
</details>

<details>
<summary>Explaining the content of the .csv files</summary>

- "submission_data.csv" has the output of the first NN. <br/>
- "submission_data_3.csv" has the output of the same NN using Item_Identier, Outlet_Identifier, visibility, and mrp as features only. <br/>
- "output_file.csv" has the result of df.groupby('Item_Identifier')['Item_Weight']
</details>

<details>
<summary>Points to investigate further</summary>

* Why did the baseline NN including varying the activation function give negative predictions? Where do they come from? <br/>
* In the baseline NN, why does using sigmoid and tanh predict the same value for all test points? <br/>
</details>

<details>
<summary>How can this code be improved?</summary>
Keeping track of different experiments namely, applying the same baseline model but varying the activation function and loss function. 
</details>









