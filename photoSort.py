#Written By:
#Andrew Williams
#(except where specifically noted)
#andrewwi@uw.edu
#
#Developed and tested using Python 3.6.3
#
#The following code is designed to run a machine learning algorithm which attempts to 
#separate in-focus photos from out-of-photos as shot by a professional photographer in 
#low-light conditions.
#
#It contains the following classes:
#-A "brisqueCalculator" class. The code in this class is taken from the OpenCV guide to BRISQUE
# image quality assessment, which extracts a variety of spatial and distortion features from the 
# photograph. The source is available at https://www.learnopencv.com/image-quality-assessment-brisque/
# and https://github.com/spmallick. Credit to Satya Mallick.
#
#-A "photograph" class which contains the features and feature extraction functions for each photo.
#
#-A "machineLearningTests" class which iterates through a variety of tests and outputs statistics and results
#
#-A "main" function which controls the overall flow of the program. At the start of the main function is a variety 
#of boolean operators which control program flow.

import numpy as np
import cv2
import exifread
import time
import heapq
from imutils import paths
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from fractions import Fraction
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import math as m
import sys
from scipy.special import gamma as tgamma
import os

#The Brisque Calculator class is the only class I did not write myself. The writer is Satya Mallick.
#The source code is available at https://github.com/spmallick/learnopencv/tree/master/ImageMetrics
#I adapted his code to work as a class rather than a standalone command line application.
#
#The constructor is empty, it is used by passing the grayscale image to the compute_features() function.
#It returns an array of spatial and distortion features (see paper for more details.)
class brisqueCalculator:

    def __init__(self):

        pass

    # AGGD fit model, takes input as the MSCN Image / Pair-wise Product
    def AGGDfit(self, structdis):
        # variables to count positive pixels / negative pixels and their squared sum
        poscount = 0
        negcount = 0
        possqsum = 0
        negsqsum = 0
        abssum   = 0

        poscount = len(structdis[structdis > 0]) # number of positive pixels
        negcount = len(structdis[structdis < 0]) # number of negative pixels
    
        # calculate squared sum of positive pixels and negative pixels
        possqsum = np.sum(np.power(structdis[structdis > 0], 2))
        negsqsum = np.sum(np.power(structdis[structdis < 0], 2))
    
        # absolute squared sum
        abssum = np.sum(structdis[structdis > 0]) + np.sum(-1 * structdis[structdis < 0])

        # calculate left sigma variance and right sigma variance
        lsigma_best = np.sqrt((negsqsum/negcount))
        rsigma_best = np.sqrt((possqsum/poscount))

        gammahat = lsigma_best/rsigma_best
    
        # total number of pixels - totalcount
        totalcount = structdis.shape[1] * structdis.shape[0]

        rhat = m.pow(abssum/totalcount, 2)/((negsqsum + possqsum)/totalcount)
        rhatnorm = rhat * (m.pow(gammahat, 3) + 1) * (gammahat + 1)/(m.pow(m.pow(gammahat, 2) + 1, 2))
    
        prevgamma = 0
        prevdiff  = 1e10
        sampling  = 0.001
        gam = 0.2

        # vectorized function call for best fitting parameters
        vectfunc = np.vectorize(self.func, otypes = [np.float], cache = False)
    
        # calculate best fit params
        gamma_best = vectfunc(gam, prevgamma, prevdiff, sampling, rhatnorm)

        return [lsigma_best, rsigma_best, gamma_best] 

    def func(self, gam, prevgamma, prevdiff, sampling, rhatnorm):
        while(gam < 10):
            r_gam = tgamma(2/gam) * tgamma(2/gam) / (tgamma(1/gam) * tgamma(3/gam))
            diff = abs(r_gam - rhatnorm)
            if(diff > prevdiff): break
            prevdiff = diff
            prevgamma = gam
            gam += sampling
        gamma_best = prevgamma
        return gamma_best

    def compute_features(self, grayscale):

        scalenum = 2
        feat = []
        # make a copy of the image 
        im_original = grayscale.copy()

        # scale the images twice 
        for itr_scale in range(scalenum):
            im = im_original.copy()
            # normalize the image
            im = im / 255.0

            # calculating MSCN coefficients
            mu = cv2.GaussianBlur(im, (7, 7), 1.166)
            mu_sq = mu * mu
            sigma = cv2.GaussianBlur(im*im, (7, 7), 1.166)
            sigma = (sigma - mu_sq)**0.5
        
            # structdis is the MSCN image
            structdis = im - mu
            structdis /= (sigma + 1.0/255)
        
            # calculate best fitted parameters from MSCN image
            best_fit_params = self.AGGDfit(structdis)
            # unwrap the best fit parameters 
            lsigma_best = best_fit_params[0]
            rsigma_best = best_fit_params[1]
            gamma_best  = best_fit_params[2]
        
            # append the best fit parameters for MSCN image
            feat.append(gamma_best)
            feat.append((lsigma_best*lsigma_best + rsigma_best*rsigma_best)/2)

            # shifting indices for creating pair-wise products
            shifts = [[0,1], [1,0], [1,1], [-1,1]] # H V D1 D2

            for itr_shift in range(1, len(shifts) + 1):
                OrigArr = structdis
                reqshift = shifts[itr_shift-1] # shifting index

                # create transformation matrix for warpAffine function
                M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
                ShiftArr = cv2.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))
            
                Shifted_new_structdis = ShiftArr
                Shifted_new_structdis = Shifted_new_structdis * structdis
                # shifted_new_structdis is the pairwise product 
                # best fit the pairwise product 
                best_fit_params = self.AGGDfit(Shifted_new_structdis)
                lsigma_best = best_fit_params[0]
                rsigma_best = best_fit_params[1]
                gamma_best  = best_fit_params[2]

                constant = m.pow(tgamma(1/gamma_best), 0.5)/m.pow(tgamma(3/gamma_best), 0.5)
                meanparam = (rsigma_best - lsigma_best) * (tgamma(2/gamma_best)/tgamma(1/gamma_best)) * constant

                # append the best fit calculated parameters            
                feat.append(gamma_best) # gamma best
                feat.append(meanparam) # mean shape
                feat.append(m.pow(lsigma_best, 2)) # left variance square
                feat.append(m.pow(rsigma_best, 2)) # right variance square
        
            # resize the image on next iteration
            im_original = cv2.resize(im_original, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        return feat

#This class contains an array of machine learning tests and functions. It runs tests with 
#the following algorithms: Random Forest, KNN, SVM, Logistic Regression, and Neural Network.
#
#datasetFeatures: A 2D array that includes the filename of the photo and all its features that we want to use for testing
#datasetLabels: The correct label of each photo
#featNames: The name of each feature
#split: What % of the dataset we want to dedicate to training (the rest will be dedicated to testing/validation)   
class machineLearningTests:

    def __init__(self, datasetFeatures, datasetLabels, featNames, split):

        self.features = datasetFeatures
        self.labels = datasetLabels
        self.trainingSize = split
        self.featureNames = featNames
        self.featureTracker = {}
        self.iterations = 0
        self.totalIterations = 1
        self.metricsTracker = {}

    #Function in the machineLearningTests class
    #If we're running a new round of tests, reset our metrics tracker and iteration count
    def startNewRoundOfTests(self):

        self.metricsTracker = {
            "accuracyScore" : 0,
            "precisionScore" : 0,
            "recallScore" : 0,
            "trainingTime" : 0,
            "trueZero" : 0,
            "falseZero" : 0,
            "falseOne" : 0,
            "trueOne" : 0
        }

        self.iterations = 0
    
    #Function in the machineLearningTests class
    #
    #Runs random forest feature selection and if runFullTest is true, runs machine learning 
    #tests as well. Repeats for numberRuns times.
    def runRandomForest(self, runFullTest, numberRuns):

        #Set total iterations to number of runs (if zero, only this line is executed)
        self.totalIterations = numberRuns
        if numberRuns > 0:

            print("Running Random Forest...")

            #Initialize metrics tracker and update run goal
            self.startNewRoundOfTests()
            
        #Repeat for given number of runs
        for self.iterations in range(0,self.totalIterations):

            #If we're doing machine learning, split data. If just feature selection, train on everything
            if runFullTest:
                trainData, testData, trainLabels, testLabels = train_test_split(self.features, self.labels, train_size=self.trainingSize)
            else:
                trainData = self.features
                trainLabels = self.labels

            model = RandomForestClassifier()
            print("Fitting Random Forest Model...")
            #Get training time.
            startTime = time.time()
            #Remove file name from training data, then send it to fit function.
            model.fit(self.stripFileName(trainData),trainLabels)
            endTime = time.time()
            elapsedTime = endTime - startTime
            print("Random Forest Fitted.")
    
            print("Random Forest Feature Importances:")
            #Combine the list of feature names and feature importances
            featureList = dict(zip(self.featureNames, model.feature_importances_))
            #Sort features by importances
            sorted_features = sorted(featureList.items(), key=lambda x: x[1])
            #Output list of features to screen
            for key, value in sorted_features:
                print("%s: %s" % (key,value))

            #Track feature importances over multiple runs
            self.trackFeatureList(featureList)

            #If we're doing the machine learning as well, run our predictions
            if runFullTest:
                predictions = []
                predictions = model.predict(self.stripFileName(testData))
                self.printStatsAndResults(testData, testLabels, predictions, elapsedTime, 'RandomForest')

    #Function in the machineLearningTests class
    #
    #Runs logistic regression RFE and if runFullTest is true, runs prediction 
    #tests as well. Repeats for numberRuns times.
    def runLogisticRegression(self, runFullTest, numberRuns):

        #Set total iterations to number of runs (if zero, only this line is executed)
        self.totalIterations = numberRuns
        if numberRuns > 0:

            self.startNewRoundOfTests()
            
            print("Running Logistic Regression...")

            #If just running RFE feature selection, train on all data, otherwise, split it.
            if runFullTest == False:

                trainData = self.features
                trainLabels = self.labels
        
            else:

                trainData, testData, trainLabels, testLabels = train_test_split(self.features, self.labels, train_size=self.trainingSize) 

            # create a base classifier used to evaluate a subset of attributes
            clf = LogisticRegression()
            # create the RFE model and select n attributes
            rfe = RFE(clf, 40)
            print("Fitting Logistic Regression RFE...")
            startTime = time.time()
            rfe = rfe.fit(self.stripFileName(trainData), trainLabels)
            endTime = time.time()
            print("Logistic Regression RFE Fitted.")
            elapsedTime = endTime - startTime
            print("Training Time: %s" % elapsedTime)
            #Use dictionary objects to store and track the support and ranking output from RFE
            supportDict = dict(zip(self.featureNames,rfe.support_))
            rankingDict = dict(zip(self.featureNames,rfe.ranking_))
            sortedRanking = sorted(rankingDict.items(), key=lambda x: x[1])
            #Send RFE output to file
            rfeOutput = "RFE Test Results:\n"
            for key,value in sortedRanking:
                print("%s %s %s" % (key,supportDict.get(key), value) )
                rfeOutput += str(key) + " " + str(supportDict.get(key)) + " " + str(value) + "\n"
            
            ostream = open("Results-LogisticRegressionRFESearch.txt",'w')
            ostream.write(rfeOutput)
            ostream.close()

            #If running full tests, split data and do predictions
            if runFullTest:

                for self.iterations in range(0,self.totalIterations):

                    trainData, testData, trainLabels, testLabels = train_test_split(self.features, self.labels, train_size=self.trainingSize) 

                    predictions = []
                    predictions = rfe.predict(self.stripFileName(testData))
                    self.printStatsAndResults(testData, testLabels, predictions, elapsedTime, 'LogisticRegression')
    
    #Function in the machineLearningTests class
    #
    #Runs the k-Nearest Neighbors algorithm and repeats numberRuns times.
    def runKNNTest(self, numberRuns):

        self.totalIterations = numberRuns
        if numberRuns > 0:

            print("Running KNN Test...")

            self.startNewRoundOfTests()            

        for self.iterations in range(0,self.totalIterations):
        
            trainData, testData, trainLabels, testLabels = train_test_split(self.features, self.labels, train_size=self.trainingSize) 

            clf = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')

            #Fit model to data and time how long it takes to train
            print("Fitting KNN Model...")
            startTime = time.time()
            clf = clf.fit(self.stripFileName(trainData), trainLabels)
            endTime = time.time()
            print("KNN Model Fitted.")
            elapsedTime = endTime - startTime

            #Predict from test data using fitted decision tree
            predictions=[]
            predictions = clf.predict(self.stripFileName(testData))

            #Track and output stats
            self.printStatsAndResults(testData, testLabels, predictions, elapsedTime, 'KNN')
    

    #Function in the machineLearningTests class
    #
    #Runs the SVM Neighbors algorithm and repeats numberRuns times.
    def runSVMTest(self, numberRuns):

        self.totalIterations = numberRuns
        if numberRuns > 0:

            print("Running SVM Test...")
            self.startNewRoundOfTests()  

        for self.iterations in range(0, self.totalIterations):

            trainData, testData, trainLabels, testLabels = train_test_split(self.features, self.labels, train_size=self.trainingSize)

            #SVM Parameters. If degree = 0, linear, otherwise poly
            degreePoly = 3

            clf = svm.SVC(kernel='poly',decision_function_shape='ovr',degree=degreePoly)

            startTime = time.time()
            clf = clf.fit(self.stripFileName(trainData), trainLabels)
            endTime = time.time()
            elapsedTime = endTime - startTime
        
            predictions=[]
            predictions = clf.predict(self.stripFileName(testData))
        
            #Track and output stats
            self.printStatsAndResults(testData, testLabels, predictions, elapsedTime, 'SVM')

    #Function in the machineLearningTests class
    #
    #
    def runNeuralNetwork(self, runGridSearch, runFullNNTest, numberRuns):

        self.totalIterations = numberRuns
        #If we're running the grid search (Note: We only run the grid search once, regardless of numberRuns)
        if runGridSearch and self.totalIterations > 0:
    
            print("Running Neural Network Grid Search...")
    
            #Here's our grid of possible parameters (does not reflect all tested values, just latest test)
            param_grid = {'solver': ['adam','lbfgs',],
                        'activation':['relu'],
                        'max_iter':[500,],
                        'hidden_layer_sizes':[(500,),(250,250),(150,150,150,150)],
                        'learning_rate':['adaptive'],
                        'batch_size':[25,50,100],
                        'alpha':[0.1,0.0001],
                        'early_stopping':[False],
                        'tol':[0.001,0.0001]
                    }

            #For the grid search we use all data for training
            trainData = self.features
            trainLabels = self.labels

            #Run the grid search using all available processors on the machine
            clf_grid = GridSearchCV(MLPClassifier(), param_grid, n_jobs=-1, verbose=2)
            clf_grid.fit(self.stripFileName(trainData),trainLabels)

            #Print out the best results to the screen
            searchOutput = "-----------------Original Features--------------------\n"
            searchOutput += "Best score: " + str(clf_grid.best_score_)
            searchOutput = ("\nUsing the following parameters:\n")
            searchOutput += str(clf_grid.best_params_)
            print(searchOutput)

            #And output the results to file as well
            ostream = open("Results-NNGridSearchResults.txt",'w')
            ostream.write(searchOutput)
            ostream.close

        #If we're doing the full testing...
        if runFullNNTest and self.totalIterations > 0:

            #Initialize metrics and run goal
            self.startNewRoundOfTests()
            
            #Iterate the given number of times
            for self.iterations in range(0,self.totalIterations):

                #Split our data
                trainData, testData, trainLabels, testLabels = train_test_split(self.features, self.labels, train_size=self.trainingSize)

                #I didn't implement a way to automatically transfer the results from the grid search to 
                #the NN hyperparameters. These need to be updated manually.
                #
                #MLP Parameters:
                #Regularization Parameter
                alphaParam = 0.0001
                #Activation function is 'identity','logistic','tanh', or 'relu'
                activationFunc = 'relu'
                #Solver for weight optimization: 'lbfgs','sgd', or 'adam'
                solverFunc = 'adam'
                #Batch size for stochastic optimizers. Default to 200 with 'auto'
                sBatchSize = 25
                #Maximum number of iterations, or epochs for stochastic solvers
                maxEpochs = 500
                #How the Learning Rate Changes: 'constant','invscaling','adaptive'
                learningRateType = 'adaptive'
                tolerance = 0.0001
                earlyStopping = False

                #Set up the classifier with the given parameters
                clf = MLPClassifier(hidden_layer_sizes=(500), activation=activationFunc, solver=solverFunc,
                    alpha=alphaParam, batch_size=sBatchSize, learning_rate=learningRateType, 
                    early_stopping=earlyStopping, max_iter=maxEpochs,tol=tolerance)

                #Fit model to data and time how long it takes to train
                print("Fitting Neural Network Model...")
                startTime = time.time()
                clf = clf.fit(self.stripFileName(trainData), trainLabels)
                endTime = time.time()
                print("Fitted Neural Network Model.")
                elapsedTime = endTime - startTime

                #Predict from test data using fitted decision tree
                predictions=[]
                predictions = clf.predict(self.stripFileName(testData))
                endTime = time.time()

                #Track and output stats
                self.printStatsAndResults(testData, testLabels, predictions, elapsedTime, 'NeuralNetwork')

    #Function in the machineLearningTests class
    #
    #Strips the file name from the feature data so it can be submitted to the fit() function.
    def stripFileName(self, origData):

        correctData=[]

        for i in range(0,len(origData)):
            correctData.append(origData[i][1:])

        return correctData

    #Function in the machineLearningTests class
    #
    #Tracks feature importances across multiple iterations when running random forest feature selection.
    def trackFeatureList(self, latestFeatureList):

        #If it's the first iteration, then just load the given list into the feature tracker
        if self.iterations == 0:

            self.featureTracker = latestFeatureList

        else:
            #Otherwise add the values from the given list into the feature tracker
            for key,val in latestFeatureList.items():

                self.featureTracker[key] = self.featureTracker.get(key) + val

        #If this is the last iteration, convert each value in the tracker from the sum to the mean
        if self.iterations == (self.totalIterations - 1):
            
            for key,val in self.featureTracker.items():

                self.featureTracker[key] = self.featureTracker.get(key) / (self.iterations + 1)
                
            #Sort our list by mean importance and output to file
            all_sorted_features  = sorted(self.featureTracker.items(), key=lambda x: x[1])
            featureFile = "results-featureSelection.txt"
            outstream = open(featureFile,'w')
            for key,val in all_sorted_features:
                outstream.write(str(key) + "," + str(val) + "\n")

            outstream.close

    #Function in the machineLearningTests class
    #
    #This function tracks statistics between iterations, outputs results to screen, and 
    #also outputs all the results to text files at the last iteration. For each test run,
    #it outputs two files: a list of individual results (a label for each tested file), 
    #and a summary containing the statistics for each run. The final line in the summary 
    #file will be a TOTAL line containing the average statistics from each run.
    def printStatsAndResults(self, testData, trueLabels, predictedLabels, trainingTime, algString):

        #Get statistics
        accuracyScore = accuracy_score(trueLabels, predictedLabels)
        precisionScore = precision_score(trueLabels, predictedLabels)
        recallScore = recall_score(trueLabels, predictedLabels)
        #Need filenames in order to output to individual results file
        nameArray = np.array(testData)
        fileNames = nameArray[:,0]
        
        cnf_matrix = confusion_matrix(trueLabels, predictedLabels)

        print("\n\n*************************** %s Results *****************************\n\n" % algString)

        #Print statistics to screen
        print("Accuracy: " + str(accuracyScore))
        self.metricsTracker["accuracyScore"] += accuracyScore
        print("Precision: " + str(precisionScore))
        self.metricsTracker["precisionScore"] += precisionScore
        print("Recall: " + str(recallScore))
        self.metricsTracker["recallScore"] += recallScore
        print("Training Time: " + str(trainingTime) + " seconds")
        self.metricsTracker["trainingTime"] += trainingTime

        #Print Confusion Matrix
        cnf_matrix = confusion_matrix(trueLabels, predictedLabels)
        self.metricsTracker["trueZero"] += cnf_matrix[0][0]
        self.metricsTracker["falseZero"] += cnf_matrix[0][1]
        self.metricsTracker["falseOne"] += cnf_matrix[1][0]
        self.metricsTracker["trueOne"] += cnf_matrix[1][1]
        print("\nConfusion Matrix:")
        print(cnf_matrix)
        print("\n\n\n")

        #outputSummary is a string to output to the summary file
        outputSummary = algString + "," + str(self.iterations + 1) + ","
        outputSummary += str(accuracyScore) + "," + str(precisionScore) + "," + str(recallScore) + "," + str(trainingTime) + ","
        outputSummary += str(cnf_matrix[0][0]) + "," + str(cnf_matrix[0][1]) + ","
        outputSummary += str(cnf_matrix[1][0]) + "," + str(cnf_matrix[1][1]) + "\n"

        #Stream for the individual results file
        resultsFile = "results-" + algString + "Individual.txt"
        #Stream for the summary results file
        summaryFile = "results-" + algString + "Summary.txt"

        #If this is the first iteration, create a new file, otherwise append existing file
        if self.iterations == 0:
            outResultStream = open(resultsFile,'w')
            outSummaryStream = open(summaryFile, 'w')
        else:
            outResultStream = open(resultsFile,'a')
            outSummaryStream = open(summaryFile, 'a')

        #Find the proper label for each individual photo in the test dat.
        for i in range(0,len(testData)):

            outputLabel = "Error"
            if trueLabels[i] == 0:
                if predictedLabels[i] == 0:
                    outputLabel = "True Zero"
                elif predictedLabels[i] == 1:
                    outputLabel = "False One"
            elif trueLabels[i] == 1:
                if predictedLabels[i] == 0:
                    outputLabel = "False Zero"
                elif predictedLabels[i] == 1:
                    outputLabel = "True One"
            nextLine = str(self.iterations) + "," + str(fileNames[i]) + "," + outputLabel + "\n"
            #Output the file  name, the iteration count (which test run this is) and the label.
            outResultStream.write(nextLine)

        #If this is the final iteration, output a TOTAL line containing averages to the summary file
        if (self.iterations + 1) == self.totalIterations:

            for key in self.metricsTracker.keys():
                self.metricsTracker[key] = self.metricsTracker.get(key) / (self.iterations + 1)

            outputSummary += algString + ",TOTAL,"
            outputSummary += str(self.metricsTracker["accuracyScore"]) + ","
            outputSummary += str(self.metricsTracker["precisionScore"]) + ","
            outputSummary += str(self.metricsTracker["recallScore"]) + ","
            outputSummary += str(self.metricsTracker["trainingTime"]) + ","
            outputSummary += str(self.metricsTracker["trueZero"]) + ","
            outputSummary += str(self.metricsTracker["falseZero"]) + ","
            outputSummary += str(self.metricsTracker["falseOne"]) + ","
            outputSummary += str(self.metricsTracker["trueOne"]) + "\n"            

        outSummaryStream.write(outputSummary)

        outSummaryStream.close
        outResultStream.close  

#The Photograph class is an object that represents
#each individual photograph and all its features.
class photograph:

    #It is initialized by feeding in a file location and a rating.
    #During intialization, it will read in or calculate all features.
    #
    #idNum: the unique ID for each photo. Set by a simple counter in the main function.
    #fileLocation: The location of the photo file on disk
    #rating: The rating of the photo (1-4 stars) that determine which binary class it falls into.
    #1-2 is class 0 (bad), 3-4 is class 1 (good). This will set the label further down in the constructor.
    #readFromTxt: A boolean, if true, we are reading in from a text file, if false, the constructor 
    #will extract all features from the photo file at the fileLocation.
    def __init__(self, idNum, fileLocation, rating, readFromTxt):

        self.fileLocation = fileLocation
        self.fileName = fileLocation[-12:]
        self.rating = rating
        self.id = idNum
        self.label = 0

        #Dictionary object that represent all the features we extract and store for each photograph.
        self.features = {
            "laplacianGray" : 0,
            "laplacianGrayMax" : 0,
            "laplacianGraySum" : 0,
            "laplacianMod" : 0,
            "laplacianColor" : 0,
            "laplacianColorMax" : 0,
            "laplacianColorSum" : 0,
            "faceLaplacianExists" : False,
            "faceLaplacian" : 0,
            "faceLaplacianExists2" : False,
            "faceLaplacian2" : 0,
            "bodyLaplacianExists" : False,
            "bodyLaplacian" : 0,
            "profileLaplacianExists" : False,
            "profileLaplacian" : 0,
            "centralLaplacianGray" : 0,
            "centralLaplacianGraySum" : 0,
            "centralLaplacianGrayMax" : 0,
            "centralLaplacianMod" : 0,
            "centralLaplacianColor" : 0,
            "centralLaplacianColorSum" : 0,
            "centralLaplacianColorMax" : 0,
            "contourLaplacianGray" : 0,
            "contourLaplacianGrayMax" : 0,
            "contourLaplacianGraySum" : 0,
            "contourLaplacianMod" : 0,
            "contourLaplacianColor" : 0,
            "contourLaplacianColorMax" : 0,
            "contourLaplacianColorSum" : 0,
            "sobelVar" : 0,
            "sobelMean" : 0,
            "sobelContourVar" : 0,
            "sobelContourMean" : 0,
            "blurredLaplacianGray" : 0,
            "blurredLaplacianGrayMax" : 0,
            "blurredLaplacianGraySum" : 0,
            "blurredLaplacianColor" : 0,
            "blurredLaplacianColorMax" : 0,
            "blurredLaplacianColorSum" : 0,
            "tenengradContour" : 0,
            "tenengradVarianceContour" : 0,
            "tenengrad" : 0,
            "tenengradVariance" : 0,
            "faceCount" : 0,
            "eyeCount" : 0,
            "faceCount2" : 0,
            "eyeCount2" : 0,
            "fNumber" : 0,
            "iso" : 0,
            "ggdshape": 0,
            "ggdvariance": 0,
            "aggdhshape": 0,
            "aggdhmean": 0,
            "aggdhlvar": 0,
            "aggdhrvar": 0,
            "aggdvshape": 0,
            "aggdvmean": 0,
            "aggdvlvar": 0,
            "aggdvrvar": 0,
            "aggddlshape": 0,
            "aggddlmean": 0,
            "aggddllvar": 0,
            "aggddlrvar": 0,
            "aggddrshape": 0,
            "aggddrmean": 0,
            "aggddrlvar": 0,
            "aggddrrvar": 0,
            "smallggdshape": 0,
            "smallggdvariance": 0,
            "smallaggdhshape": 0,
            "smallaggdhmean": 0,
            "smallaggdhlvar": 0,
            "smallaggdhrvar": 0,
            "smallaggdvshape": 0,
            "smallaggdvmean": 0,
            "smallaggdvlvar": 0,
            "smallaggdvrvar": 0,
            "smallaggddlshape": 0,
            "smallaggddlmean": 0,
            "smallaggddllvar": 0,
            "smallaggddlrvar": 0,
            "smallaggddrshape": 0,
            "smallaggddrmean": 0,
            "smallaggddrlvar": 0,
            "smallaggddrrvar": 0,
            "cannyPixels": 0,
            "cannySharpness": 0,
        }

        #This is a list of all the features the machine learning algorithm will actually use.
        #To utilize all 87 features, either (1)set the useAllFeatures boolean to True in the main function, or 
        #(2)make sure all lines are uncommented in the list below.
        #
        #To restrict which features are used, you must both (1)set the useAllFeatures boolean to False in the 
        #main function and (2)comment out the lines below representing the features you do not wish to use.
        self.featuresToUse=[#"laplacianGray",
                            #"laplacianGrayMax",
                            #"laplacianGraySum",
                            #"laplacianMod",
                            #"laplacianColor",
                            #"laplacianColorMax",
                            #"laplacianColorSum",
                            #"faceLaplacianExists",
                            #"faceLaplacian",
                            #"faceLaplacianExists2",
                            #"faceLaplacian2",
                            #"bodyLaplacianExists",
                            #"bodyLaplacian",
                            #"profileLaplacianExists",
                            #"profileLaplacian",
                            #"centralLaplacianGray",
                            #"centralLaplacianGraySum",
                            "centralLaplacianGrayMax",
                            #"centralLaplacianMod",
                            #"centralLaplacianColor",
                            #"centralLaplacianColorSum",
                            #"centralLaplacianColorMax",
                            #"contourLaplacianGray",
                            #"contourLaplacianGrayMax",
                            #"contourLaplacianGraySum",
                            #"contourLaplacianMod",
                            #"contourLaplacianColor",
                            #"contourLaplacianColorMax",
                            #"contourLaplacianColorSum",
                            #"sobelVar",
                            #"sobelMean",
                            #"sobelContourVar",
                            #"sobelContourMean",
                            #"blurredLaplacianColor",
                            #"blurredLaplacianColorSum",
                            #"blurredLaplacianColorMax",
                            #"blurredLaplacianGray",
                            #"blurredLaplacianGrayMax",
                            #"blurredLaplacianGraySum",
                            #"tenengradContour",
                            "tenengradVarianceContour",
                            #"tenengrad",
                            #"tenengradVariance",
                            #"faceCount",
                            #"eyeCount",
                            #"faceCount2",
                            #"eyeCount2",
                            "fNumber",
                            #"iso",
                            #"ggdshape",
                            #"ggdvariance",
                            #"aggdhshape",
                            "aggdhmean",
                            "aggdhlvar",
                            #"aggdhrvar",
                            #"aggdvshape",
                            "aggdvmean",
                            "aggdvlvar",
                            #"aggdvrvar",
                            #"aggddlshape",
                            #"aggddlmean",
                            #"aggddllvar",
                            #"aggddlrvar",
                            #"aggddrshape",
                            #"aggddrmean",
                            #"aggddrlvar",
                            #"aggddrrvar",
                            #"smallggdshape",
                            #"smallggdvariance",
                            #"smallaggdhshape",
                            #"smallaggdhmean",
                            #"smallaggdhlvar",
                            #"smallaggdhrvar",
                            #"smallaggdvshape",
                            "smallaggdvmean",
                            #"smallaggdvlvar",
                            #"smallaggdvrvar",
                            #"smallaggddlshape",
                            #"smallaggddlmean",
                            #"smallaggddllvar",
                            #"smallaggddlrvar",
                            #"smallaggddrshape",
                            #"smallaggddrmean",
                            #"smallaggddrlvar",
                            #"smallaggddrrvar",
                            "cannyPixels",
                            "cannySharpness"
                        ]

        #Files rated 3 or higher are given a label of 1, or "good."
        #Files with a lower rating are marked as 0, or "bad."
        if (rating >= 3):
            self.label = 1

        #If we're not reading from the text file, we have to extract all info from the photo file
        if readFromTxt == False:

            #Get certain EXIF tags as features
            exifreader = open(fileLocation, 'rb')
            tags = exifread.process_file(exifreader)
            self.features["fNumber"] = float(Fraction(str(tags["EXIF FNumber"])))
            self.features["iso"] = int(str(tags["EXIF ISOSpeedRatings"]))

            #Read in the photo
            photo = cv2.imread(fileLocation)

            #Apply a bilateral filter to reduce noise but preserve edges
            photo = cv2.bilateralFilter(photo,7,25,25,borderType=cv2.BORDER_DEFAULT)

            #Create a grayscale version of the image
            gray = cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)

            #Get the height and width of the photo
            h, w = photo.shape[:2]
        
            #Get these values to divide the photo into a 3x3 grid. We'll use this to find the center region
            gw = w // 3
            gh = h // 3

            #These will be used to find the contour region. We want to crop away the edges to get rid of things 
            #like curtains and stage lights. (See paper for more info)
            minwidth = w // 8
            minheight = h // 8
            maxwidth = minwidth * 7
            maxheight = minheight * 7

            #These two photos are where we'll be looking for contours
            contourRegion = photo[minheight:maxheight, minwidth:maxwidth]
            contourRegionGray = gray[minheight:maxheight, minwidth:maxwidth]
            #Zero in on the ROI and set the contour regions properly
            crxmin, crxmax, crymin, crymax = self.findContours(contourRegion, contourRegionGray, False, False)
            contourRegion = contourRegion[crymin:crymax, crxmin:crxmax]
            contourRegionGray = contourRegionGray[crymin:crymax, crxmin:crxmax]
            
            #Get laplacian statistics on our contour region (ROI)
            lvar,lsum,lmax = self.getLaplacian(contourRegionGray, False)
            self.features["contourLaplacianGray"] = lvar
            self.features["contourLaplacianGraySum"] = lsum
            self.features["contourLaplacianGrayMax"] = lmax
            self.features["contourLaplacianMod"] = self.getModifiedLaplacian(contourRegionGray)
            lvar,lsum,lmax = self.getLaplacian(contourRegion, False)
            self.features["contourLaplacianColor"] = lvar
            self.features["contourLaplacianColorSum"] = lsum
            self.features["contourLaplacianColorMax"] = lmax

            #Get canny edge data
            self.getCannySharpness(contourRegionGray, False)
            #I was not able to extract useful stats from the Scharr filter so it is disabled
            #self.getScharr(gray, True)
            #Get sobel variance and mean on both the entire photo and the contour region
            sobelvar, sobelmean = self.getSobel(gray, False)
            self.features["sobelGray"] = sobelvar
            self.features["sobelMean"] = sobelmean
            sobelvar, sobelmean = self.getSobel(contourRegionGray, False)
            self.features["sobelContourGray"] = sobelvar
            self.features["sobelContourMean"] = sobelmean

            #The central square represents the middle part of the 3x3 grid. An alternate way of finding the ROI
            centralSquare = photo[gh:gh+gh, gw:gw+gw]
            centralSquareGray = gray[gh:gh+gh, gw:gw+gw]
            #Get the laplacian statistics and modified laplacian of the central square only
            lvar,lsum,lmax = self.getLaplacian(centralSquareGray, False)
            self.features["centralLaplacianGray"] = lvar
            self.features["centralLaplacianGraySum"] = lsum
            self.features["centralLaplacianGrayMax"] = lmax
            #Only need to get the modified laplacian of the grayscale image
            self.features["centralLaplacianMod"] = self.getModifiedLaplacian(centralSquareGray)
            #Get the laplacian statistics and modified laplacian of the color central square
            lvar,lsum,lmax = self.getLaplacian(centralSquare, False)
            self.features["centralLaplacianColor"] = lvar
            self.features["centralLaplacianColorSum"] = lsum
            self.features["centralLaplacianColorMax"] = lmax
            
            #Get laplacian statistics of the overall grayscale image
            lvar,lsum,lmax = self.getLaplacian(gray, False)
            self.features["laplacianGray"] = lvar
            self.features["laplacianGraySum"] = lsum
            self.features["laplacianGrayMax"] = lmax
            self.features["laplacianMod"] = self.getModifiedLaplacian(gray)
            #Get laplacian variance of the overall color image
            lvar,lsum,lmax = self.getLaplacian(photo, False)
            self.features["laplacianColor"] = lvar
            self.features["laplacianColorSum"] = lsum
            self.features["laplacianColorMax"] = lmax

            #The following set of functions are designed to detect faces, bodies, and contours in the image,
            #in order to extract the laplacian variance around those features. The two boolean values in each 
            #call determine whether the functions draws and outputs what it finds (can be enabled for testing purposes).
            self.faceDetect(photo, gray, False, False)
            self.faceDetectAlt(photo, gray, False, False)
            self.bodyDetect(photo, gray, False, False)
            self.profileDetect(photo, gray, False, False)
            
            #This is where we call the BrisqueCalc (Blind/Referenceless Image Spatial Quality Evaluator)
            #class to get Brisque-related statistics.
            brisqueCalc = brisqueCalculator()
            featuresReturned = brisqueCalc.compute_features(contourRegionGray)
            self.readInBrisqueFeatureArray(featuresReturned)

            #This is put in as a test to try an additional Gaussian blur to reduce noise, then 
            #get the laplacian statistics on the resulting blurred image. 
            blurredPhoto = cv2.GaussianBlur(photo, (3,3), 0)
            grayBlurred = cv2.cvtColor(blurredPhoto,cv2.COLOR_BGR2GRAY)  
            lvar,lsum,lmax = self.getLaplacian(grayBlurred, False)
            self.features["blurredLaplacianGray"] = lvar
            self.features["blurredLaplacianGraySum"] = lsum
            self.features["blurredLaplacianGrayMax"] = lmax
            lvar,lsum,lmax = self.getLaplacian(blurredPhoto, False) 
            #Get laplacian statistics of the overall color image
            self.features["blurredLaplacianColor"] = lvar
            self.features["blurredLaplacianColorSum"] = lsum
            self.features["blurredLaplacianColorMax"] = lmax

            #Tenengrad is another measure of edge. Get the tenengrad of the 
            #whole grayscale image and the grayscale of our contour region.
            ten, tenVar = self.getTenengrad(gray)
            self.features["tenengrad"] = ten
            self.features["tenengradVariance"] = tenVar
            ten, tenVar = self.getTenengrad(contourRegionGray)
            self.features["tenengradContour"] = ten
            self.features["tenengradVarianceContour"] = tenVar

    #Function in photograph class
    #
    #Applies a tenengrad filter and gets the mean and focus measure.
    def getTenengrad(self, img):

        scale = 5
        gx = cv2.Sobel(img, cv2.CV_16S, 1, 0, scale)
        gy = cv2.Sobel(img, cv2.CV_16S, 0, 1, scale)

        FM = gx**2 + gy**2
        meanVal = cv2.mean(FM)
        focusMeasure = meanVal[0]

        tvar = FM.var()

        print("Tenengrad: " + str(focusMeasure) + " " + str(tvar))

        return focusMeasure, tvar

    #Function in photograph class
    #
    #Applies a modified laplacian and gets the mean.
    def getModifiedLaplacian(self, img):

        M = cv2.getGaussianKernel(3,-1,cv2.CV_64F)
        M = cv2.transpose(M)
        M[0][0] = -1
        M[0][1] = 2
        M[0][2] = -1
        G = cv2.getGaussianKernel(3,-1,cv2.CV_64F)

        LX = cv2.sepFilter2D(img, cv2.CV_64F, M, G)
        LY = cv2.sepFilter2D(img, cv2.CV_64F, G, M)
        FM = cv2.convertScaleAbs(LX) + cv2.convertScaleAbs(LY)
        meanVal = cv2.mean(FM)
        print("Modified Laplacian: " + str(meanVal[0]))
        return meanVal[0]

    #Function in photograph class
    #
    #Applies a Laplacian filter and gets the variance, sum, and max value, in an attempt 
    #to find the sharpest edge.
    def getLaplacian(self, img, output):

        ll = cv2.Laplacian(img, cv2.CV_64F)
        testll = np.ravel(ll)
        sumLaplacian = testll.sum()
        maxLaplacian = testll.max()
        lapVariance = ll.var()
        if output:
            cv2.imwrite("zll_" + str(self.fileName),ll)
        return lapVariance, sumLaplacian, maxLaplacian 

    #Function in photograph class
    #
    #Applies a Scharr filter and gets the variance. On further research I decided the Scharr filter wasn't 
    #useful so this is not actually called, but remains for possible further testing.
    def getScharr(self, gray, output):

        scale = 5
        delta = 0
        ddepth = cv2.CV_16S

        scharr_x = cv2.Scharr(gray, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        scharr_y = cv2.Scharr(gray, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
        abs_scharr_x = cv2.convertScaleAbs(scharr_x)
        abs_scharr_y = cv2.convertScaleAbs(scharr_y)
    
        scharrImg = cv2.addWeighted(abs_scharr_x, 0.5, abs_scharr_y, 0.5, 0)
        if output:
            cv2.imwrite("zscharr_" + str(self.fileName),scharrImg)
        return scharrImg.var()
    
    #Function in photograph class
    #
    #Applies a second-derivative sobel filter and gets the mean and variance
    def getSobel(self, gray, output):

        scale = 1
        delta = 0
        ddepth = cv2.CV_16S

        sobel_x = cv2.Sobel(gray, ddepth, 2, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        sobel_y = cv2.Sobel(gray, ddepth, 0, 2, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    
        sobelImg = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        if output:
            cv2.imwrite("zsobel_" + str(self.fileName),sobelImg)
        meanVal = cv2.mean(sobelImg)
        return sobelImg.var(), meanVal[0]

    #Function in photograph class
    #
    #Takes the featureArray returned from the compute_features
    #function in the brisqueCalculator class and reads the 
    #items in the array into the appropriate features in the 
    #photo object.
    def readInBrisqueFeatureArray(self, featureArray):

        self.features["ggdshape"] = featureArray[0]
        self.features["ggdvariance"] = featureArray[1]
        self.features["aggdhshape"] = featureArray[2]
        self.features["aggdhmean"] = featureArray[3]
        self.features["aggdhlvar"] = featureArray[4]
        self.features["aggdhrvar"] = featureArray[5]
        self.features["aggdvshape"] = featureArray[6]
        self.features["aggdvmean"] = featureArray[7]
        self.features["aggdvlvar"] = featureArray[8]
        self.features["aggdvrvar"] = featureArray[9]
        self.features["aggddlshape"] = featureArray[10]
        self.features["aggddlmean"] = featureArray[11]
        self.features["aggddllvar"] = featureArray[12]
        self.features["aggddlrvar"] = featureArray[13]
        self.features["aggddrshape"] = featureArray[14]
        self.features["aggddrmean"] = featureArray[15]
        self.features["aggddrlvar"] = featureArray[16]
        self.features["aggddrrvar"] = featureArray[17]
        self.features["smallggdshape"] = featureArray[18]
        self.features["smallggdvariance"] = featureArray[19]
        self.features["smallaggdhshape"] = featureArray[20]
        self.features["smallaggdhmean"] = featureArray[21]
        self.features["smallaggdhlvar"] = featureArray[22]
        self.features["smallaggdhrvar"] = featureArray[23]
        self.features["smallaggdvshape"] = featureArray[24]
        self.features["smallaggdvmean"] = featureArray[25]
        self.features["smallaggdvlvar"] = featureArray[26]
        self.features["smallaggdvrvar"] = featureArray[27]
        self.features["smallaggddlshape"] = featureArray[28]
        self.features["smallaggddlmean"] = featureArray[29]
        self.features["smallaggddllvar"] = featureArray[30]
        self.features["smallaggddlrvar"] = featureArray[31]
        self.features["smallaggddrshape"] = featureArray[32]
        self.features["smallaggddrmean"] = featureArray[33]
        self.features["smallaggddrlvar"] = featureArray[34]
        self.features["smallaggddrrvar"] = featureArray[35]
      
    #Function in photograph class
    #
    #Find contours in the image. If draw/output are set to true, 
    #then draw/output contours on image.
    def findContours(self, photo, gray, draw, output):

        #Get the mean brightness of the image
        mean = cv2.mean(gray)
        print("Mean Brightness: " + str(mean[0]))

        contourThresh = round(mean[0])
        height, width = gray.shape[:2]
        #Set our minimum and maximum x and y (which will eventually bound our ROI)
        #to starting values.
        min_x, min_y = width, height
        max_x = max_y = 0
        #This boolean tells us if we've succeeded
        areaFound = False
        #This tells us the total area bounded by our contours so far
        totalArea = 0
        #The following two values are configured. If we can't find a contour that covers more 
        #than 1% of the image, we're too dim, if a contour exceeds more than 50%, we're too bright.
        minArea = (width * height) * 0.01
        maxArea = (width * height) * 0.5

        #Keeping looping until we've found our target ROI
        while areaFound == False:

            try:
                #Our contour threshold is the mean brightness.
                print("Trying contour detection threshold value = " + str(contourThresh))
                #Get a black/white binary threshold version of our image
                ___, thresh = cv2.threshold(gray, contourThresh, 255, cv2.THRESH_BINARY)
                #And find the contours
                ___, contours, ____ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                #Loop through each contour we found
                for contour in contours:
                    #Find the enclosed area
                    area = cv2.contourArea(contour)
                    #If area greater than min area, add it to total area found
                    if area > minArea:
                        totalArea += area
                        #If area of contour is more than half the image, we're too bright. Up our threshold and try again.
                        if area > maxArea:
                            contourThresh += 20
                            totalArea = 0
                        #Otherwise update the bounding rectangle (min and max x,y)
                        else:
                            (x,y,w,h) = cv2.boundingRect(contour)
                            min_x, max_x = min(x, min_x), max(x+w, max_x)
                            min_y, max_y = min(y, min_y), max(y+h, max_y)
                    
                    #If our total area is more than 5% of the image we've found our ROI
                    if totalArea > minArea * 5:
                        areaFound = True
            
            #If we get an exception when finding contours, something has gone wrong. Reset our bounding rectangle and 
            #we'll loop through the while loop again
            except:
                areaFound = False
                min_x, min_y = width, height
                max_x = max_y = 0

            #If our contour threshold is too bright or too dark, something's wrong with 
            #the image and we'll default our ROI to the central region
            if contourThresh < 10 or contourThresh > 256:
                print("Unable to find contours, defaulting to middle third")
                min_x = width // 3
                max_x = (2 * width) // 3
                min_y = height // 3
                max_y = (2 * height) // 3
                areaFound = True

            #For testing purposes, we can draw the ROI on the photo
            if areaFound==True and draw==True:
                cv2.rectangle(photo, (min_x, min_y), (max_x, max_y), (255, 255, 0), 2)

            #If we've fallen this far the contour threshold is too dim and we need to reduce it (unless we 
            # hit the "contour too bright" section earlier, where we upped the threshold by 20. In that case 
            # 20 - 10 will still increase the threshold by 10)
            contourThresh = contourThresh - 10

        #If output is true, output a photo with the ROI drawn on it
        if output == True:
            outputfile = "zcontours_" + str(self.fileName)
            cv2.imwrite(outputfile, photo)

        #Return ROI coordinates
        return min_x, max_x, min_y, max_y

    def getCannySharpness(self, gray, output):

        cannyImg = cv2.Canny(gray, 200, 250)
        self.features["cannyPixels"] = cv2.countNonZero(cannyImg)
        self.features["cannySharpness"] = (self.features["cannyPixels"] * 1000) / (gray.size)
        print("Canny Pixels: " + str(self.features["cannyPixels"]))
        print("Canny Sharpness: " + str(self.features["cannySharpness"]))

        if output == True:
            outputfile = "zcanny_" + str(self.fileName)
            cv2.imwrite(outputfile, cannyImg)

    #Function in photograph class
    #
    #Find profiles of faces in the image. If draw/output are set to true, 
    #then draw/output rectangular boundaries of face profiles on image in white
    def profileDetect(self, photo, gray, draw, output):

        profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

        profiles = profile_cascade.detectMultiScale(gray, 1.3, 5)
        print("Profiles detected: " + str(len(profiles)))
        for (x,y,w,h) in profiles:
            self.features["profileLaplacianExists"] = True
            if draw == True:
                cv2.rectangle(photo,(x,y),(x+w,y+h),(255,255,255),2)
            roi_gray = gray[y:y+h, x:x+w]

            self.features["profileLaplacian"] = cv2.Laplacian(roi_gray, cv2.CV_64F).var()

        if output == True:
            outputfile = "zprofile_" + str(self.fileName)
            cv2.imwrite(outputfile, photo)     

    #Function in photograph class
    #
    #Find bodies in the image. If draw/output are set to true, 
    #then draw/output rectangular boundaries of bodies on image in yellow
    def bodyDetect(self, photo, gray, draw, output):

        body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

        bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
        print("Bodies detected: " + str(len(bodies)))
        for (x,y,w,h) in bodies:
            self.features["bodyLaplacianExists"] = True
            if draw == True:
                cv2.rectangle(photo,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h, x:x+w]

            self.features["bodyLaplacian"] = cv2.Laplacian(roi_gray, cv2.CV_64F).var()

        if output == True:
            outputfile = "zbody_" + str(self.fileName)

            cv2.imwrite(outputfile, photo)        
       
    #Function in photograph class
    #
    #Find faces in the image. If draw/output are set to true, 
    #then draw/output rectangular boundaries of faces on image
    #in blue and boundaries of eyes in green
    def faceDetect(self, photo, gray, draw, output):

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print("Faces detected: " + str(len(faces)))
        for (x,y,w,h) in faces:
            if draw == True:
                cv2.rectangle(photo,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = photo[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)

            print("Eyes detected: " + str(len(eyes)))

            if len(eyes) > 0:
                self.features["faceLaplacianExists"] = True
                self.features["faceLaplacian"] += cv2.Laplacian(roi_gray, cv2.CV_64F).var()
                self.features["faceCount"] += 1
                self.features["eyeCount"] += len(eyes)

            if draw == True:
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

        if self.features["faceCount"] > 0:
            self.features["faceLaplacian"] = self.features["faceLaplacian"] / self.features["faceCount"]

        if output == True:
            outputfile = "wface_" + str(self.fileName)
            cv2.imwrite(outputfile, photo)

    #Function in photograph class
    #
    #Use alternative classifiers to find faces in the image. If 
    #draw/output are set to true, then draw/output rectangular 
    #boundaries of faces on image in aqua and boundaries of eyes in purple
    def faceDetectAlt(self, photo, gray, draw, output):

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print("Alt faces detected: " + str(len(faces)))
        for (x,y,w,h) in faces:
            if draw == True:
                cv2.rectangle(photo,(x,y),(x+w,y+h),(255,255,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = photo[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)

            print("Alt eyes detected: " + str(len(eyes)))

            if len(eyes) > 0:
                self.features["faceLaplacianExists2"] = True
                self.features["faceLaplacian2"] += cv2.Laplacian(roi_gray, cv2.CV_64F).var()
                self.features["faceCount2"] += 1
                self.features["eyeCount2"] += len(eyes)

            if draw == True:
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)

        if self.features["faceCount2"] > 0:
            self.features["faceLaplacian2"] = self.features["faceLaplacian2"] / self.features["faceCount2"]

        if output == True:
            outputfile = "zfacealt_" + str(self.fileName)
            cv2.imwrite(outputfile, photo)

    #Function in photograph class
    #
    #Return a string representing the filename and features of each photograph. Used
    #when outputting dataset info to a file or the console.
    def getFeatureString(self):

        outputLine = str(self.fileName) + " " + str(self.rating)
        for nextVal in self.features.values():
                outputLine += " " + str(nextVal)

        return outputLine

    #Function in photograph class
    #
    #Return a list of features to be used in machine learning testing. If useAll is true, 
    #then return all features, otherwise return only the features listed in  
    #featuresToUse.
    def getFeatureListForML(self, useAll):

        featureList=[]
        #print(str(self.fileName))
        featureList.append(str(self.fileName))
        if useAll == False:
            for nextFeature in self.featuresToUse:
                featureList.append(self.features[nextFeature])
        else:
            for nextFeature in self.features.values():
                featureList.append(nextFeature)

        return featureList

#Main function
if __name__=='__main__':

    #*************START CONFIGURABLE VALUES*******************

    #All the following booleans determine the behavior of the operation.
    #If true, read in features from datafilepath, otherwise, read in features from photos
    readFromTextFile = False
    #If true, output data to datafilepath
    outputDataToTextFile = False
    
    #If true, use all features, otherwise, only use features listed in featuresToUse list in photo class
    useAllFeatures = True
    #How much of the total set to dedicate to training (the rest will be used for testing)
    trainingRatio = 0.8    

    #If true, run machine learning tests as well as feature selection.
    #This affects: Random Forest (if false, selector only), Neural 
    #Network (if false, grid search only), Logistic Regression (if false, RFE only)
    runMachineLearningTests = False
    #If true, run the neural network grid search (only applicable if neuralNetworkCount > 0)
    runNNGridSearch = False
    #The number of runs to do for each machine learning algorithm.
    logisticRegressionCount = 0
    randomForestCount = 0
    knnCount = 0
    neuralNetworkCount = 0
    svmCount = 0

    #File paths for operation of the program
    #File to read in and/or output files, labels, and features
    datafilepath = 'dataset.txt'

    #*************END CONFIGURABLE VALUES*******************

    #List to store all photographs
    photos=[]

    #If this is true, then output photograph information and features to text file.
    #This is so we don't have to re-read all 2,000 photographs every time and can 
    #just read in information from text file next time.
    if outputDataToTextFile == True and readFromTextFile == False:

        outstream = open(datafilepath,'a')
    
    else:
        #If we're reading in from the text file, we shouldn't be outputting right 
        #back out. This is to prevent output overwriting data that has not yet been 
        #read in
        outputDataToTextFile = False

    #This is where we read in from the text file
    if readFromTextFile == True:

        with open(datafilepath) as fp:
            thisID = 0
            for line in fp:
                #File is a space-separated data file
                readFeatures = line.strip().split(' ')
                #Use thisID as unique identifier
                thisID += 1
                
                #First item is the filename
                thisFile = readFeatures[0]
                #Second item is the rating (1-4)
                thisRating = int(readFeatures[1])
                #Create a photograph object with this data
                thisPhoto = photograph(thisID, thisFile, thisRating, True)

                j = 2
                
                #Now go through the rest of the items read in from this line and build the feature array
                for key, val in thisPhoto.features.items():
                 
                    #Most values are floats but a few are not, need to be read in correctly accounting for data type
                    if key == "fNumber":
                        thisPhoto.features[key] = float(Fraction(str(readFeatures[j])))                        
                    elif key == "iso":
                        thisPhoto.features[key] = int(readFeatures[j])
                    elif "Exists" not in key:
                        thisPhoto.features[key] = float(readFeatures[j])
                    else:
                        thisPhoto.features[key] = bool(readFeatures[j])

                    j += 1
                                                        
                #After feature list is populated, append this photograph object to master list of photo
                photos.append(thisPhoto)

    else:

        #If we're not reading in from a text file, then we're reading in from actual photos
        count = 1
        for i in range(1,5,3):
        
            #This is where files have been stored for testing
            imageLocation = 'C:\\Users\\Andrew\\Dropbox\\School\\581\\Project\\Photos\\sampleData\\' + str(i) + 'Star\\'
            for nextPhoto in paths.list_images(imageLocation):

                #If we feed a photo location into the constructor then the constructor will populate all feature data
                print("Reading in photo # " + str(count) + ": " + nextPhoto)
                photoObj = photograph(count,nextPhoto,i, False)
                #Once photo object is created, append it to our master list
                photos.append(photoObj)
                count += 1

                #If we're outputting data to text file, output photo data to text file now
                if outputDataToTextFile:

                    nextLine = photoObj.getFeatureString() + "\n"
                    outstream.write(nextLine)

    #Close our stream if we've been outputting our data to text file
    if outputDataToTextFile:

        outstream.close

    #We're going to populate features and labels (data and targets) into these arrays to feed into the machine learning class
    labels=[]
    features=[]
    featureNames=[]
    
    #If we want to use all features, get all features, otherwise only get features in the featuresToUse array in the photo class
    #Every photo will have the same features in the featuresToUse array, they cannot be set differently
    if useAllFeatures:
        featureNames = list(photos[0].features.keys())
    else:
        featureNames = photos[0].featuresToUse

    #Convert the photographs into a list of labels and a list of features to be 
    #fed into the machine learning algorithm
    for nextPhoto in photos:

        labels.append(nextPhoto.label)
        features.append(nextPhoto.getFeatureListForML(useAllFeatures))

    #This is our machine class, which contains all functions for machine learning tests. We feed it in:
    #Features: Includes the filename of the photo and all its features that we want to use for testing
    #Labels: The correct label of each photo
    #FeatureNames: The name of each feature
    #Ratio: What % of the dataset we want to dedicate to training (the rest will be dedicated to testing/validation)    
    tester = machineLearningTests(features, labels, featureNames, trainingRatio)

    #Run logistic regression test
    tester.runLogisticRegression(runMachineLearningTests, logisticRegressionCount)

    #Run neural network tests
    tester.runNeuralNetwork(runNNGridSearch, runMachineLearningTests, neuralNetworkCount)

    #Run random forest tests
    tester.runRandomForest(runMachineLearningTests, randomForestCount)

    #Run KNN tests
    tester.runKNNTest(knnCount)
    
    #Run SVM tests
    tester.runSVMTest(svmCount)