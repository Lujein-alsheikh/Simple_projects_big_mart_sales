In this read_me file, you can find: explanation of the problem and features and key moments in preprocessing the data and designing the neural net. 

The goal is to predict the sales of different products across different big mart outlets using a neural network. This problem is found on the Analytics Vidhya
hackathon website so training set and test set are already split.

The features are:
Item_Identifier:	Unique product ID
Item_Weight:	Weight of product
Item_Fat_Content:	Whether the product is low fat or not
Item_Visibility:	The % of total display area of all products in a store allocated to the particular product
Item_Type:	The category to which the product belongs
Item_MRP:	Maximum Retail Price (list price) of the product
Outlet_Identifier:	Unique store ID
Outlet_Establishment_Year:	The year in which store was established
Outlet_Size:	The size of the store in terms of ground area covered
Outlet_Location_Type:	The type of city in which the store is located
Outlet_Type:	Whether the outlet is just a grocery store or some sort of supermarket
Item_Outlet_Sales:	Sales of the product in the particular store. This is the outcome variable to be predicted.

Key moments: (Look for Mardown cells that start with "idea" or "Thinking moment" or "question")
1. Should we keep the Item_Identifier column? Does it not cause leakage? Some items have similar sales across different outliers and some don't. There is no (immediate) leakage. I kept this column. Some contestants removed it and with different preprocessing methods than mine still got good results though.
2. Important observations about the data: The same outlet always has the same establishment year, size, location type, and type. Those columns are redundant.
An item has the same weight, fat content, and type. But, two different items can still have the same weight. The weight, fat content, and item type can (almost) uniquely identify an item (there are ~200 items that can't be identified using those 3 pieces of information).
3. How to encode the Item_Identifier column that has 1559 different categories? The library category_encoders comes in handy. I used binary encoding and was able to encode 1559 values using 11 columns only.
4. Idea (I didn't try it): If I want to create a model to predict the missing values of "Outlet_Size", do I use the original target ("Item_Outlet_Sales") as a feature? Does it cause data leakage? In this case, I have to use the training set only. The size of each outlet doesn't change. This allows me to fill in the missing values in the test set as well. Another important question would be how to assess the accuracy of the predictions if there are no ground truth values to compare with?
5. Weird performance from Python: df.groupby('Item_Identifier')['Item_Weight'].mean() gives strange results. For example, instead of returning 4.59 it returns 4.58999999. Another example, instead of returning 6.52 it returns 6.5200000005. I had to investigate where the weird results were coming from. The solution is round the results.
6. I tried first a NN with one hidden layer only (of 500 neurons). I tried different activation functions and got negative prediction which is why I applied a custom loss function. The latter lowered the number of test data points whose predictions were negative, but didn't solve the whole problem.
7. I tried another NN with two hidden layers (500 and 100) which gave acceptable results namely, a loss of ~1300 on the public part of the test_data.
8. I tried the latter NN using only the Item_Id and Outlet_Identifier which gave much better results: ~ 1158 loss on the public part of the test_data.
9. No overfitting observed in both neural nets I tried.
