In this read_me file, you can find: explanation of the problem and features and key moments in preprocessing the data and designing the neural net. 

<details>
<summary>Problem statement</summary>
The goal is to predict the sales of different products across different big mart outlets using a neural network. This problem is found on the Analytics Vidhya
hackathon website so training set and test set are already split.
</details>

<details>
<summary>The features are:</summary>

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
<summary>Key moments or notes about preprocessing: </summary>
Look for Mardown cells that start with "idea" or "Thinking moment" or "question". <br/>
1. **Important observations about the data:** The same outlet always has the same establishment year, size, location type, and type. Those columns are redundant. <br/>
An item always has the same weight, fat content, and type. But, two different items can still have the same weight. The weight, fat content, and item type can (almost) uniquely identify an item (there are ~200 items that can't be identified using those 3 pieces of information). <br/>
The problem really boils down to predicting sales for a specific product in a specific outlet. <br/>
I tried using only the item ids and outlet id's and got good results. 
1. **Should we keep the Item_Identifier column? Does it not cause leakage?** Some items have similar sales across different outliers and some don't. There is no (immediate) leakage. I kept this column. Some contestants removed it and with different preprocessing methods than mine still got good results though.
3. **How to encode the Item_Identifier column that has 1559 different categories?** The library category_encoders comes in handy. I used binary encoding and was able to encode 1559 values using 11 columns only.
4.**Creating a model to predict missing values of the "Outlet_Size" column:** (I didn't try it): If we want to create a model to predict the missing values of "Outlet_Size", we can not use the original target variable as a feature because it causes data leakage. Another important question would be how to assess the accuracy of the predictions if there are no ground truth values to compare with? 
5. **Weird performance from Python: df.groupby('Item_Identifier')['Item_Weight'].mean() gives strange results.** For example, instead of returning 4.59 it returns 4.58999999. Another example, instead of returning 6.52 it returns 6.5200000005. I need that df.groupby('Item_Identifier')['Item_Weight'].mean() returns exactly the same values as in the dataset. I rounded the output.
6. **Note about encoding:** When applying binary encoding on item_ids, we have to .fit_transform on the whole dataset because we need the coding to be consistent.
7. **Downside of one-hot-encoding:** that the data becomes sparse. Some models might not work well with sparse data. I apply one-hot-encoding on two columns Item_Type and Outlet_Identifier and end up with 21 columns which is acceptable in my case.
8. **Other ideas for encoding:** frequency encoding or target encoding. Caution that target encoding leads to leakage.
</details>

<details>
<summary>Summary of how I preprocessed each column</summary>
| Column | info about this column | How to preprocess it? |
| ---------|----------|----------|
| Item_Identifier | 1559 unique values | binary encoding |
| Item_Weight | float | filled in missing values easily since each item has the same weight and the same item is repeated many times |
| Item_Fat_Content | | label encoding |
| Item_Visibility | float | |
| Item_Type | 16 categories | grouping sparse categories and then one-hot-encoding. We end up with 11 categories |
| Item_MRP | float | |
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
<summary>Building the NN and judging the performance and results</summary>
6. I tried first a NN with one hidden layer only (of 500 neurons). I tried different activation functions and got negative prediction which is why I applied a custom loss function. The latter lowered the number of test data points whose predictions were negative, but didn't solve the whole problem.
7. I tried another NN with two hidden layers (500 and 100) which gave acceptable results namely, a loss of ~1300 on the public part of the test_data.
8. I tried the latter NN using only the Item_Id and Outlet_Identifier which gave much better results: ~ 1158 loss on the public part of the test_data.
9. No overfitting observed in neither neural nets I tried.
</details>
