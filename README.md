Classifier Algorithm Analysis for Liver Disease Prediction
 Dhruv Umesh Sompura1, *B.K. Tripathy[0000-0003-3455-4549], Ishan Rajesh Kasat2 
School of Information Technology and Engineering, VIT, Vellore – 632014, Tam-il Nadu, India

dhruvumesh.sompura2020@vitstudent.ac.in, * tripathybk@vit.ac.in,
ishanrajesh.kasat2020@vitstudent.ac.in
Abstract. Deep learning and Machine Learning can be used in automated di-agnosis of various diseases like liver diseases. India has more than a million people getting diagnosed with liver diseases each year hence it's important to detect them at an early stage. These are caused due to consumption of alco-hol, contaminated food, obesity, thus we need a system that can predict the symptoms of liver diseases. We can predict these diseases using patient data and deep learning algorithms. Performance of this system can be measured in terms of accuracy, recall f-measure, etc. In this project we have tried to use deep learning and data mining techniques to help this noble cause of de-tecting liver diseases at an early stage.  In the existing method[ref. 10] the re-searchers have used MLCNN-LDPS which provided an accuracy of 90.75%. We have used three hybrid algorithms: CNN combined with LSTM(99.02%), CNN combined with GRU(98.38%) and CNN combined with RNN(99.48%) 
Keywords: Liver Disease, SVM , Machine learning, Naïve Bayes, CNN, LSTM, GRU, RNN
1	Introduction
Deep learning is a subtopic of machine learning that consists of three or more neural networks layers. Deep learning mimics the human brain to process data by integrating data to create accurate predictions. Deep learning algorithms rely on large amounts of data and work multiple times, always tweaking to improve re-sults. 
We all know that the liver is the largest internal organ of the body, which per-forms very important bodily functions including the formation of blood clotting factors and proteins, the production of triglycerides and cholesterol, the synthe-sis of glycogen and the production of bile. Usually, more than 75% of the liver tissue needs to be affected by the decline in function. It is therefore important to detect the disease at an early stage so that the disease can be treated before it becomes serious. Deep learning techniques have become very important in healthcare nowadays for disease prediction from medical database. Many re-searchers and companies are using deep learning to improve medical diagnostics. Among various deep learning techniques, classification algorithms are widely used in disease prediction. Some of the popular algorithms include KNN, SVM, RF, NBC etc. We have studied the same and made some points on it.
2	Background
The main objective is to analyze the parameters of various classification algo-rithms and compare their predictive accuracies so as to find out the best classifier for determining the liver disease. Also, the other objectives could be listed as –
1.	Obtaining a suitable dataset to implement the prediction model
The first and foremost objective of implementing a deep learning model is to find a suitable dataset which contains liver disease patients records which will include the values of several attributes. 
2.	Training the model to obtain a very high accuracy
Training the model plays an important role, the more data collected, the more the model is trained. The process of training a DL model involves providing an DL algorithm (that is, the learning algorithm) with the training dataset. Ob-taining high accuracy is important.
3.	Implementing performance metrics on the model
Performance metrics play an important role in deep learning models and only with these performance measures one can identify how good the model is and how effectively it is performing.
4.	Develop a prototype web-based interface for prediction
The development of a prototype web-based interface is necessary and it is made available to all users. The user is required to have a necessary blood test report by which the user can enter the values and thus the system predicts the results based on the values.
5.	Encouraging diagnostic centers to implement high accuracy models
The ultimate objective of this whole project is to encourage diagnostics to implement deep learning models for fast and accurate diagnostics of liver disease.

3	Literature Review
[1] uses ml algorithms like LR, KNN, SVM, DT, NB and RF and their performance was calculated and compared using various metrics like f-1 score, recall, preci-sion, accuracy. The goal was to reduce the cost of liver disease diagnosis. [2] focuses on using ml algorithms to classify whether patient is liver patient or not. It uses logistic regression to compare the results with that of other researchers. Logistic regression was found to have better accuracy than Naïve Bayes Classifi-er, Decision tree, support vector machine, artificial neural network and k nearest neighbors. [3] proposes the research work of Naïve Bayes and Support Vector Machine (SVM) classifier algorithms used for liver disease prediction and analy-sis. It mentions that there are two major parameters that are involved in under-standing the suitability of the respective methodologies: time taken to execute the prediction process and accuracy of the predictive results. It is clear that SVM classifier is the best algorithm owing to high accuracy rates. But when it comes to the time taken to execute the predictive process, the Naive Bayes classifier re-flects higher suitability since it takes the least possible time to execute the pro-cess. [4] uses decision tree algorithms and compares them with respect to seven performance metrics (ACC%, MAE, PRE, REC, FME, Kappa Statistics and runtime). The algorithms used were J48, LMT, Random Tree, Random Forest, REPTree, Decision Stump and Hoeffding Tree. The analysis proves that Decision Stump provides the highest accuracy than other techniques. [5] uses a software engineering approach using classification and feature selection technique. Six classification algorithms J-48, Random Forest, Logistic Regression, SMO (Sup-port Vector Machine), IBk (k nearest Neighbor), Logistic Regression, Naïve Bayes have been considered for implementation and comparing their results based on the ILPD (Indian Liver Patient Dataset). The development of intelligent liver disease prediction software (ILDPS) is done by using feature selection and classification prediction techniques based on software engineering model. The proposed work focuses on the development of the software that will help in the prediction of the level diseases based upon the various symptoms. 
[6] builds classification models to predict liver disease outcome. It uses four phases to build a model. First min max normalization algorithm is applied on the original liver patient datasets collected from UCI repository. In second phase, by the use of PSO feature selection, subset (data) of liver patient dataset from whole normalized liver patient datasets are obtained which comprises only significant attributes. Third phase, classification algorithms are applied on the data set. In the fourth phase, the accuracy will be calculated using root mean square value, root mean error value. J48 algorithm is considered as the better performance al-gorithm after applying PSO feature selection. Finally, the evaluation is done based on accuracy values. Thus outputs show from proposed classification im-plementations indicate that J48 algorithm performances all other classification algorithm. [7] uses the genetic algorithm to optimize boosted C5.0 algorithm which is a data mining algorithm and it has used to find rules for liver disease by considering a dataset and its results are compared with other proposed approach-es. Then, finally, we will have optimal and accurate rules for the liver disease diagnosis. After the genetic algorithm was implemented, totally 24 rules were generated instead of 92 rules and the comparable statistical parameters show that the proposed method has better performance than boosted C5.0. Also, having lesser rules will help to reduce the time of diagnosis. So instead of using an evo-lutionary algorithm for producing rules, the genetic algorithm is used for improv-ing and reducing rules of another algorithm. In [8] various classification algo-rithms were investigated and analyzed such as Naïve Bayes, Decision Tree, Mul-ti-Layer Perceptron and k-NN which were used in a previous study, and helped to develop the dataset. They were compared and it was seen that in view of preci-sion, Naïve Bayes is preferable than others, but in other criteria such as Recall and Sensitivity, Logistic and Random Forest took precedence over other algo-rithms in the performance of prediction test. [9] surveyed some data mining tech-niques to predict liver disease at an earlier stage. The study analyzed algorithms such as C4.5, Naive Bayes, Decision Tree, Support Vector Machine, Back Propa-gation Neural Network and Classification and Regression Tree Algorithms. It is seen that C4.5 gives better results compared to other algorithms. In future an im-proved C4.5 could be derived with various parameters. In [10] a liver disease prediction system based on a modified convolutional neural network (MCNN-LDPS) was presented for accurate liver disease prediction results.  This research method was analyzed on a dataset of Indian liver patients.  The analysis of the researched work proves that the proposed MCNN-LDPS method achieves better results in terms of increased accuracy and precision. This research method was compared with the existing multi-layer perceptron neural network (MLPNN) for performance analysis.  The main limitation of CNN was its inability to encode orientation and relative spatial relations, and viewpoint.  CNN did not encode the position and orientation of the data.  Lack of ability has been spatially invariant to the input data sample.  This was solved in this researched worked by combin-ing the genetic algorithm with CNN method.
3.1	Gaps In Literature
When used to a big dataset, it has been found that some algorithms do not pro-duce effective results, but rather classify a sample of datasets more accurately than the entire dataset. Additionally, it has been observed that certain techniques may not exhibit the same degree of accuracy when used with real-time data. The majority of research has divided their training and testing datasets using the ran-dom-sampling method. However, there isn't much research comparing various sampling methods' outcomes across the dataset. Additionally, different sampling techniques are likely appropriate for certain contexts and datasets. Relying solely on one way restricts the scope for future improvements. While there is a lot of study on predicting liver illnesses, there are very few studies that concentrate on the disease's progression. A lot of focus should be placed on determining the illness' stage so that doctors may immediately provide patients treatment instruc-tions without wasting time on expensive tests.
4	Proposed Algorithm 
Existing methods-
•	Classified liver disease using artificial neural network (ANN) classification algorithm resulted in low accuracy.
•	Testing accuracy of MLP was found to be 77.54 %, logistic regression method gave 74.36% and for SMO it gave 71.36 %.
We are going to use the following deep learning algorithms for our project and compare their accuracy – 
4.1	 LSTM-CNN:	
There are several ways to enhance model performance, such as changing batch size and number of epochs, dataset curating, adjusting the ratio of the training, validation, and test datasets, changing loss functions and model architectures, and so on. In this project, we will improve model performance by changing the model architecture. More specifically, we will see if the CNN-LSTM model can predict liver disease cases better than the LSTM model.
The CNN layers that extract the feature from input data and LSTMs layers to provide sequence prediction
4.2	CNN + GRU:
To combine the advantages of the GRU module which can well process time se-quence data and the advantages of the CNN module which is ideal for handling high-dimensional data, the GRU-CNN hybrid neural networks was proposed
The proposed GRU-CNN hybrid neural network framework consists of GRU and CNN modules. The inputs are time series data collected from the energy sys-tem and information from the spatiotemporal matrix. The output is a prediction of future load values. As for the CNN module, it is good at processing two-dimensional data such as: B. Spatio-temporal matrices and images. The CNN en-gine uses local connections and shared weights to directly extract local features from spatio-temporal matrix data and obtain efficient representations through convolution and pooling layers. The structure of the CNN module contains two layers of convolutions and one flattening operation, and each layer of convolu-tions contains one convolution operation and one pooling operation. After the second pooling operation, the high-dimensional data is flattened to 1-dimensional data and the output of the CNN module is combined into a fully connected layer. On the other hand, the purpose of the GRU module is to grasp long-term dependencies, and the GRU module can learn useful information from historical data through memory cells over a long period of time, and unneeded information can be learned over a long period of time. be forgotten. Gate of Oblivion. The input to the GRU module is time series data. The GRU module con-tains many gate recursion units, and the outputs of all these gate recursion units are connected to fully connected layers. Finally, the load prediction result can be obtained by averaging over all neurons in the fully connected layer. 
4.3	CNN + RNN:
The proposed model makes use of the ability of the CNN to extract local features and of the LSTM to learn long-term dependencies. First, a CNN layer of Conv1D is used for processing the input vectors and extracting the local features that re-side at the text-level. The output of the CNN layer (i.e. the feature maps) are the input for the RNN layer of LSTM units/cells that follows. The RNN layer uses the local features extracted by the CNN and learns the long-term dependencies of the local features The proposed model makes use of the ability of the CNN to extract local features and of the LSTM to learn long-term dependencies. First, a CNN layer of Conv1D is used for processing the input vectors and extracting the local features that reside at the text-level. The output of the CNN layer (i.e. the feature maps) are the input for the RNN layer of LSTM units/cells that follows. The RNN layer uses the local features extracted by the CNN and learns the long-term de-pendencies of the local features
4.4	Algorithm:


 
Fig. 1. Flowchart of all the deep learning algorithms used in the prediction of result
4.5	Architecture:
                                                     
Fig. 2. Flowchart of the system architecture that our experimental code is based on
5	Experimentation
5.1	Datasets Description & Sample Data
Data Set Information:
 The data was received from UCI Machine Learning Repository. The infor-mation about the dataset is below. (UCI Machine Learning Repository, 2013). The data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into liver patient (liver disease) or not (no dis-ease). This data set contains 441 male patient records and 142 female patient records.
Attribute Information:
•	Age of the patient
•	Gender of the patient
•	Total Bilirubin
•	Direct Bilirubin
•	Alkaline Phosphotase
•	Alamine Aminotransferase
•	Aspartate Aminotransferase
•	Total Protiens
•	Albumin
•	Albumin and Globulin Ratio
•	Class: field used to split the data into two sets (patient with liver disease, or no disease)
Before loading the dataset, we should import all the required libraries such as pandas, tokenizer, numpy, seaborn, label encoder to perform operations of im-plementing deep-learning models as well to perform steps of data pre- pro-cessing. Here, we have downloaded the dataset from the UCI repository and saved it as indian_liver_patient.csv which is now loaded and can be read as a data frame which is now named as data.
5.2	Data Pre-processing & Visualization
While creating our project, the dataset which we imported from the repository was not clean and formatted and before employing the deep learning models on the data, it is very necessary to clean and put formatted data, hence data pre-processing is required and it is basically the process of preparing the raw data and making it ready for the deep learning model. The following graphs show number of liver and non-liver disease along with male and females in the dataset.

 
Fig. 3. Shows visual representation of the dataset and its columns
Observations	
By using the command data.describe, we can figure out some of the observa-tions of the dataset such as:
•	Gender is a non- numerical variable and other all are numeric values.
•	There are 10 features and 1 output which is the dataset.
•	In Albumin and Globulin ratio we can see that there are four missing values.
•	Values of Alkaline_Phosphatase, Alamine_Aminotransferase, 
Aspartate_Aminotransferase which are int should be converted for float values for better accuracy.

Variable	Data Type
Age	int64
Total_Bilirubin	float64
Direct_Bilirubin	float64
Alkaline_Phosphotase	int64
Alamine_Aminotransferase	int64
Aspartate_Aminotransferase	int64
Total_Protiens	float64
Albumin	float64
Albumin_and_Globulin_Ratio	float64
Dataset	int64
dtype	object
Table. 1. Dataset values used for the calculation along with their data types
Filling of Missing Values
 It is the process of identifying the missing variables and adding the mean val-ues. For our dataset, the Albumin and Globulin ratio had four missing values which are replaced by considering the mean of that column which is 94.7. These values are filled in the second fig which shows that the column A/G ratio has no more null values.
Identifying Duplicate Values
 Duplicate values were identified and by the observations we can see around 13 duplicate values but for a medical dataset duplicate value can exist and thus we are not dropping any of the duplicate values.
Resampling
 Because of the imbalance in the dataset where we can observe a majority in liver disease patients and a minority in non-liver disease patients, smote which is synthesize minority oversampling technique which generates new values for the minority data and then synthesizes new samples for minorities. This will help in obtaining a better accuracy for the model during the implementation of machine learning models to the dataset in Weka Tool. Also, we have applied PCA to achieve better results and then lastly made combinations using smote and PCA to compare the accuracy among various ML algorithms.
5.3	Feature Selection
Feature Selection is a process of figuring out which inputs are the best for the model and checking if there is a possibility of eliminating certain inputs. Consid-ering the Dataset, we can see a very high linear relationship between Total and Direct Bilirubin and by considering this linear relationship, Direct Bilirubin can be opted to be dropped, but as per medical analysis Direct Bilirubin constitutes to almost 10% of the Total Bilirubin and this 10% may prove crucial in obtaining higher accuracy for the model, thus none of the features are removed.
5.4	Train-Test Split
We can use the train-test split technique. It is a technique for evaluating the per-formance of a deep learning algorithm. The procedure involves taking a dataset and dividing it into two subsets. It is a fast and easy procedure to perform, the results of which allows us to compare the performance of deep learning algo-rithms for our predictive modelling problem. For the liver disease prediction model, we have considered 80 % of training data and 20 % of data for testing. 
5.5	Code Implementation
 
Fig. 4. Shows the code execution for Simple RNN+CNN algorithm

 
Fig. 5. Shows the code execution for LSTM+CNN algorithm

 
Fig. 6. Shows the code execution for GRU+CNN algorithm
6	Result Analysis
In the existing method[10] the researchers have used MLCNN-LDPS which pro-vided an accuracy of 90.75%. We have used three hybrid algorithms: CNN+LSTM(99.02%), CNN+GRU(98.38%), CNN+RNN(99.48%) and have achieved an accuracy as high as 99.48% using filters like upscaling and PCA. We also got used various algorithms and got the following accuracies- naïve bayes: 76%, random forest: 80.26%, logistic: 72%, svm: 76.93%, knn: 76.67%. We used PCA and SMOTE to increase the number of cases in the dataset in a balanced way. This gave us better accuracies.
Algorithms	With SMOTE	Without SMOTE	PCA With SMOTE	PCA Without SMOTE
Naïve Bayes	76 %	62.6072%	47.4667%	31.38%
Random Forest	80.2667 %	72.0412%	84.2667 %	72.3842 %
Logistic	72%	60.2058%	88.5333 %	72.3842 %
SVM	76.9333 %	64.494%	78.5333 %	72.3842 %
KNN	76.6667 %	66.7238%	88.5333 %	72.3842 %
Table. 2. The table contains the accuracy of all the algorithms run on weka before and after applying pca and smote

 
Fig. 7. Graphical Representation of the table above to give a clear understanding of the accuracies
7	Conclusion 
With the help of performance measure and analysis the performance of various deep algorithms are evaluated. The dataset was obtained from UCI repository on which data pre-processing techniques such as filling missing values, replacing duplicate values was performed. Oversampling was performed using SMOTE and with the help of data visualization the model was trained to understand the dupli-cate values present. Feature selection showed a linear relationship on certain attributes of the dataset. The highest accuracy was obtained by using CNN+RNN model and thus the performance was measured based on a classification report and performance measures such as accuracy and precision. This result which we obtained relies upon various deep learning algorithms which provides high accu-racy and consumes very less time for the entire processing. We can conclude that CNN+RNN model proved its worthiness in prediction of liver patients by achiev-ing high accuracy amongst the other hybrid algorithms.
8	Future Work
Future work can incorporate the use of fast dataset techniques. The dataset may contain several additional instances for better prediction. Along with increasing incidences, various attributes important for predicting liver disease such as tri-glycerides, urinary copper, serum cholesterol, and serum glutamate-oxaloacetic transaminase (SGOT) could be added to improve chances of predicting liver dis-ease. Data mining techniques can be used to provide hidden aspects in blood test results, genetic profiles, heart rate, bone density, etc. This can help achieve effi-cient diagnostic system.
References
1.	 Rahman, A. S., Shamrat, F. J. M., Tasnim, Z., Roy, J., & Hossain, S. A. (2019). A comparative study on liver disease prediction using supervised machine learning algo-rithms. International Journal of Scientific & Technology Research, 8(11), 419-422.
2.	 Adil, S. H., Ebrahim, M., Raza, K., Ali, S. S. A., & Hashmani, M. A. (2018, August). Liver patient classification using logistic regression. In 2018 4th International Confer-ence on Computer and Information Sciences (ICCOINS) (pp. 1-5). IEEE.
3.	 Vijayarani, S., & Dhayanand, S. (2015). Liver disease prediction using SVM and Naïve Bayes algorithms. International Journal of Science, Engineering and Technology Re-search (IJSETR), 4(4), 816-820.Kefelegn, S., & Kamat, P. (2018). Prediction and anal-ysis of liver disorder diseases by using data mining technique: survey. International Journal of pure and applied mathematics, 118(9), 765-770.
4.	 Nahar, N., Ara, F., Neloy, M. A. I., Barua, V., Hossain, M. S., & Andersson, K. (2019, December). A comparative analysis of the ensemble method for liver disease predic-tion. In 2019 2nd International Conference on Innovation in Engineering and Technol-ogy (ICIET) (pp. 1-6). IEEE.
5.	 Singh, J., Bagga, S., & Kaur, R. (2020). Software-based prediction of liver disease with feature selection and classification techniques. Procedia Computer Science, 167, 1970-1980.
6.	 Priya, M. B., Juliet, P. L., & Tamilselvi, P. R. (2018). Performance analysis of liver disease prediction using machine learning algorithms. Int. Res. J. Eng. Technol, 5(1), 206-211.
7.	 Hassoon, M., Kouhi, M. S., Zomorodi-Moghadam, M., & Abdar, M. (2017, Septem-ber). Rule optimization of boosted c5. 0 classification using genetic algorithm for liver disease prediction. In 2017 international conference on computer and applications (ic-ca) (pp. 299-305). IEEE.
8.	 Jin, H., Kim, S., & Kim, J. (2014). Decision factors on effective liver patient data pre-diction. International journal of Bio-science and Bio-Technology, 6(4), 167-178.
9.	 Kefelegn, S., & Kamat, P. (2018). Prediction and analysis of liver disorder diseases by using data mining technique: survey. International Journal of pure and applied mathe-matics, 118(9), 765-770.
10.	Jeyalakshmi, K., & Rangaraj, R. (2021). Accurate liver disease prediction system using convolutional neural network. Indian Journal of Science and Technology, 14(17), 1406-1421.
