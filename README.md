# COMP-472-A3

## nb_bow model object
**To train a model and use it to predict:**
- create a model object 

  `model = nb_bow()`


- train the model by calling the predict method on the object and passing the training features and labels arrays.  
  This sets the internal attributes the object needs to make a prediction
  
  `model.train(features, labels)`


- use the trained model to make predictions by passing an array of tweets organized into an array of how many of each 
  word in the vocabulary occurs in the tweet to the predict method. The method returns an array of yes/no values the 
  same length as the testing set.  (one prediction pre tweet input)
  
  `model.predict(test_features)`
  


**Libraries used:**
- csv
