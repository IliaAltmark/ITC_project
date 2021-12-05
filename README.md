# Amazon Reviews Classification
Ilia Altmark, Alejandro Vidaurrazaga Iturmendi and Bazham Khanatayev


For our project we chose Amazon reviews data where different users rate the products that they have ordered. The idea is to do sentiment analysis based on the text provided in the reviews. We believe that understanding the user’s sentiment towards the product has many uses. Online reviews are an integral part of a brand’s reputation and anything related to better understanding consumer’s attitudes towards a product can be used to improve the product and increase revenue. One might also wonder why sentiment analysis is useful when most review platforms also have an option to give a star rating. The ratings may actually not be the best representation of the sentiment of the review. This was demonstrated in a paper presented at an NLP conference in Europe. 

For our models we used the reviews as the unstructured data which we classified to 3 different classes (initially started as 5 classes which are the rating scores), ‘Positive’, ‘Neutral’ and ‘Negative’.

# EDA
We’ve combined about 10 categories together into one dataset (arts crafts and sewing, automotive, CDs and Vinyl, cell phones and accessories, grocery and gourmet food, kindle store, musical iInstruments, office products, patio lawn and garden, pet supplies) and used under sampling in order to make sure the data is balanced. Overall 250,000 reviews. All the reviews are in English and from Amazon. In fact about 90% of the reviews have less than 200 words.

# Baseline
For our baseline we chose a simple logistic regression model. One of the challenges we had to face was dealing with an unbalanced dataset. We soon realized that and created a balanced dataset consisting of 10 different categories. The preprocessing we performed was removing punctuation and creating a bag of words. 

After seeing the results we realized that dealing with 5 different classes might be more challenging than we realized. We decided to combine the rating the following way:
1 and 2 became ‘Negative’, 3 was ‘Neutral’, 4 and 5 became ‘positive’. 

# RNN with LSTM
For our second model we decided to try an RNN model with one LSTM layer. For this model the preprocessing was removing various unrelated characters (like new lines and email and web addresses), balancing the data with undersampling, translating to tokens and padding.

# Transformers with BERT
We assumed we could get better results with BERT. The preprocessing involved using a dedicated BERT tokenizer which involved tokenizing and padding as well as adding mask values (tell which is a value and which is a padding).
