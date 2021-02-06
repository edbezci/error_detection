import time
import os
import pandas as pd
import random
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer




class error_detect:
    '''
    This object conducts NLP based error checking to detect erroneous/fraudulent labels in binary data.
    It is based on Logistic Regression Classification for NLP purposes. 
    It requires the path/name of the data folder holding the data as CSV, and returns a CSV report for the error detection.
    '''

    def __init__(self):
        self.folder = input("What is the name of the folder?....  ")
        self.file = []
        self.file_names = []

    def detect(self):
        self.file_find()
        for d in self.file:
            self.file_names.append(error_detect.classification(str(d)))
        print ("File(s) " +str(self.file_names) + " created." )
    def file_find(self):
        files=[]   
        for f in os.listdir(self.folder):
            files.append(str(self.folder) + '/' + str(f)) 
        print ("Number of files in the folder...:  " + str(len(files)))   
        return self.file.extend(files)

    @staticmethod
    def classification(file):


        start = time.time()
        df = pd.read_csv(file, encoding =  "ISO-8859-1")

    # it is pythonic in order to prevent ambuguity, class is not used as a variable name so we change class to label
    
        #df.columns = ["text","label"]
    
    # Doing this to increase speed and prevent any possible issues per unbalanced datasets. 
        no_lower_class = min(df['label'].value_counts())  # whatever is the lower class
    
    #Unique labels
        col_labels= df.label.unique()
    # we are sure that our sampled dataset has all of the lower-class; and the same amount of the other class
        label1 = df[df.label == col_labels[0]]
        label2 = df[df.label == col_labels[1]]
    
        smpldt = pd.concat([label1.sample(n=no_lower_class, replace=False), label2.sample(n=no_lower_class, replace=False)])
        
    # make one random error here to test later on:
    
        error_df = smpldt.sample(n=1, replace=False)
    
    # Changing the label value of the random row to create a random error
    
        if error_df["label"].item() == col_labels[0]:
            error_df["label"] = col_labels[1]
        else:
            error_df["label"] = col_labels[0]
    
    #removing chosen row from the other dataframe
    
        new_smpldt = pd.concat([smpldt,error_df]).drop_duplicates(subset = ["text"], keep=False)
    
    #splitting the data for test and training 
        training_data, testing_data = train_test_split(new_smpldt,random_state = 2000)

    # GET LABELS
        Y_train=training_data['label'].values
        # Y_test=testing_data['label'].values # Comment out/in if necessary
    
    # Feature extraction and count vectorizing for logistic classification. Features are the Labels, i.e. 1 or 0, Include/Exclude,Approve/Deny,Dangerous/Safe etc.
        cv = CountVectorizer(stop_words="english", binary=True, max_df=0.95)
        cv.fit_transform(training_data["text"].values)
        
        train_feature_set=cv.transform(training_data["text"].values)
        test_feature_set=cv.transform(testing_data["text"].values)
        test_labels = testing_data["label"].values
    
    # assinging test and trains.
        X_train,X_test = train_feature_set,test_feature_set
    
    
    # Gearing the Classifier model 
        classifier = LogisticRegressionCV(verbose=0, solver='liblinear',random_state=0, cv=10, penalty='l2',max_iter=100).fit(X_train,Y_train)
    
    # Getting diagnostic results
      # predicted_label = classifier.predict(X_test)  ## comment out/in if necessary
        diagnostic_score = classifier.score(X_test,test_labels)
    
    # Conducting error-cheking
        possible_error = cv.transform(error_df["text"].values)
        suggested_label = classifier.predict(possible_error)
        error_label = error_df["label"].values
        end = time.time()
    
        speed = round(end-start,3)

        text = "file: "+str(file)+"; Speed (seconds):"+str(speed)+";Original_Error_Label:"+str(error_label)+";Predicted_Label:"+str(suggested_label)+";Diagnostic Score:"+str(diagnostic_score)

        file_name = str(random.randint(1, 1000))+"_error_check.csv" 

        print("Speed (seconds)..: " + str(speed) + " ;Original_Error_Label: " +str(error_label) + " ;Predicted_Label: " + str(suggested_label) + " ; Diagnostic Score: " + str(diagnostic_score))

        with open(file_name,'w') as file:
            for line in text:
                file.write(line)

        return file_name