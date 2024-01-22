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
1. Should we keep the Item_Identifier column? Does it not cause leakage?
2. 
