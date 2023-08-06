Deep learning and Machine Learning can be used to diagnose various diseases like liver diseases. India has more than a million people getting diagnosed with liver diseases each year hence it's essential to detect them early. These are caused due to consumption of alcohol, contaminated food, and obesity, thus we need a system that can predict the symptoms of liver diseases. We can predict these diseases using patient data and deep learning algorithms. The performance of this system can be measured in terms of accuracy, recall f-measure, etc. In this project, I have tried to use deep learning and data mining techniques to help this noble cause of detecting liver diseases at an early stage.  I have used three hybrid algorithms: CNN combined with LSTM(99.02%), CNN combined with GRU(98.38%), and CNN combined with RNN(99.48%).

We are going to use the following deep learning algorithms for our project and compare their accuracy – 

**1.	 LSTM-CNN:**
There are several ways to enhance model performance, such as changing batch size and number of epochs, dataset curating, adjusting the ratio of the training, validation, and test datasets, changing loss functions and model architectures, and so on. In this project, we will improve model performance by changing the model architecture. More specifically, we will see if the CNN-LSTM model can predict liver disease cases better than the LSTM model.
The CNN layers extract the feature from input data and LSTM layers to provide sequence prediction.

**2.	CNN + GRU:**
To combine the advantages of the GRU module which can well process time sequence data and the advantages of the CNN module which is ideal for handling high-dimensional data, the GRU-CNN hybrid neural networks were proposed
The proposed GRU-CNN hybrid neural network framework consists of GRU and CNN modules. The inputs are time series data collected from the energy system and information from the spatiotemporal matrix. The output is a prediction of future load values. As for the CNN module, it is good at processing two-dimensional data such as B. Spatio-temporal matrices and images. The CNN engine uses local connections and shared weights to directly extract local features from spatiotemporal matrix data and obtain efficient representations through convolution and pooling layers. The structure of the CNN module contains two layers of convolutions and one flattening operation, and each layer of convolutions contains one convolution operation and one pooling operation. After the second pooling operation, the high-dimensional data is flattened to 1-dimensional data, and the output of the CNN module is combined into a fully connected layer. On the other hand, the purpose of the GRU module is to grasp long-term dependencies, and the GRU module can learn useful information from historical data through memory cells over a long period, and unneeded information can be learned over a long period. be forgotten. Gate of Oblivion. The input to the GRU module is time series data. The GRU module contains many gate recursion units, and the outputs of all these gate recursion units are connected to fully connected layers. Finally, the load prediction result can be obtained by averaging all neurons in the fully connected layer.

**3.	CNN + RNN:**
The proposed model makes use of the ability of the CNN to extract local features and of the LSTM to learn long-term dependencies. First, a CNN layer of Conv1D is used for processing the input vectors and extracting the local features that reside at the text level. The output of the CNN layer (i.e. the feature maps) are the input for the RNN layer of LSTM units/cells that follows. The RNN layer uses the local features extracted by the CNN and learns the long-term dependencies of the local features The proposed model makes use of the ability of the CNN to extract local features and of the LSTM to learn long-term dependencies. First, a CNN layer of Conv1D is used for processing the input vectors and extracting the local features that reside at the text level. The output of the CNN layer (i.e. the feature maps) are the input for the RNN layer of LSTM units/cells that follows. The RNN layer uses the local features extracted by the CNN and learns the long-term dependencies of the local features.

![image](https://github.com/sompuradhruv/liver/assets/78086198/bd3f033c-0e54-42c1-8b38-5e61c59708ae)

![image](https://github.com/sompuradhruv/liver/assets/78086198/daebc555-3f5f-498a-8a20-e1a573c4ab22)

**Data Set Information:**
The data was received from UCI Machine Learning Repository. The information about the dataset is below. (UCI Machine Learning Repository, 2013). The data set contains 416 liver patient records and 167 nonliver patient records collected from North East of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into the liver patient (liver disease) or not (no dis-ease). This data set contains 441 male patient records and 142 female patient records.

**Attribute Information:**
•	Age of the patient
•	Gender of the patient
•	Total Bilirubin
•	Direct Bilirubin
•	Alkaline Phosphatase
•	Alamine Aminotransferase
•	Aspartate Aminotransferase
•	Total Proteins
•	Albumin
•	Albumin and Globulin Ratio
•	Class: field used to split the data into two sets (patient with liver disease, or no disease)
Before loading the dataset, we should import all the required libraries such as pandas, tokenizer, numpy, seaborn, and label encoder to perform operations of implementing deep-learning models as well as to perform steps of data pre- pro-cessing. Here, we have downloaded the dataset from the UCI repository and saved it as indian_liver_patient.csv which is now loaded and can be read as a data frame which is now named as data.
5.2	Data Pre-processing & Visualization
While creating our project, the dataset which we imported from the repository was not clean and formatted, and before employing the deep learning models on the data, it is very necessary to clean and put formatted data, hence data pre-processing is required and is basically the process of preparing the raw data and making it ready for the deep learning model. The following graphs show a number of liver and non-liver diseases along with males and females in the dataset.

**Observations**
By using the command data.describe, we can figure out some of the observations of the dataset such as:
•	Gender is a non-numerical variable and other all are numeric values.
•	There are 10 features and 1 output which is the dataset.
•	In the Albumin and Globulin ratio we can see that there are four missing values.
•	Values of Alkaline_Phosphatase, Alamine_Aminotransferase, 
Aspartate_Aminotransferase which is int should be converted for float values for better accuracy.

**Filling of Missing Values**
It is the process of identifying the missing variables and adding the mean values. For our dataset, the Albumin and Globulin ratio had four missing values which are replaced by considering the mean of that column which is 94.7. These values are filled in the second fig which shows that the column A/G ratio has no more null values.

**Identifying Duplicate Values**
Duplicate values were identified and by the observations, we can see around 13 duplicate values but for a medical dataset duplicate values can exist and thus we are not dropping any of the duplicate values.

**Resampling**
Because of the imbalance in the dataset where we can observe a majority in liver disease patients and a minority in non-liver disease patients, smote is a synthesized minority oversampling technique which generates new values for the minority data and then synthesizes new samples for minorities. This will help in obtaining a better accuracy for the model during the implementation of machine learning models to the dataset in the Weka Tool. Also, we have applied PCA to achieve better results and then lastly made combinations using smote and PCA to compare the accuracy among various ML algorithms.

**Feature Selection**
Feature Selection is a process of figuring out which inputs are the best for the model and checking if there is a possibility of eliminating certain inputs. Considering the Dataset, we can see a very high linear relationship between Total and Direct Bilirubin and by considering this linear relationship, Direct Bilirubin can opt to be dropped, but as per medical analysis Direct Bilirubin constitutes almost 10% of the Total Bilirubin and this 10% may prove crucial in obtaining higher accuracy for the model, thus none of the features are removed.

**Train-Test Split**
We can use the train-test split technique. It is a technique for evaluating the performance of a deep-learning algorithm. The procedure involves taking a dataset and dividing it into two subsets. It is a fast and easy procedure to perform, the results of which allow us to compare the performance of deep learning algorithms for our predictive modeling problem. For the liver disease prediction model, we have considered 80 % of training data and 20 % of data for testing. 

**Result Analysis**
We have used three hybrid algorithms: CNN+LSTM(99.02%), CNN+GRU(98.38%), and CNN+RNN(99.48%), and have achieved accuracy as high as 99.48% using filters like upscaling and PCA. We also got used various algorithms and got the following accuracies- naïve bayes: 76%, random forest: 80.26%, logistic: 72%, SVM: 76.93%, knn: 76.67%. We used PCA and SMOTE to increase the number of cases in the dataset in a balanced way. This gave us better accuracies.



