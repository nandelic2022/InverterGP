import inspect
import random 
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import gplearn 
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_random_state
import gplearn.genetic
from gplearn.functions import make_function
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import scipy.special as sp 
from sklearn.model_selection import KFold 

###############################################################################
# Load dataset 
###############################################################################
df = pd.read_csv('Inverter Data Set.csv')
dfx1 = df[['u_a_k-1',
           'u_b_k-1',
           'u_c_k-1',
           #'d_a_k-3',
           #'d_b_k-3',
           #'d_c_k-3',
           'i_a_k-3',
           'i_b_k-3',
           'i_c_k-3',
           'i_a_k-2',
           'i_b_k-2',
           'i_c_k-2',
           'u_dc_k-3',
           'u_dc_k-2']]

dfy1 = df.pop("d_a_k-2")
X_train, X_test, y_train, y_test = train_test_split(dfx1,dfy1, test_size = 0.3)
ColumnNames = ["X{}".format(i) for i in range(len(list(X_test.columns)))]
X_train.columns = ColumnNames
X_test.columns = ColumnNames
def generateGPParameters(): 
    parameters = [] 
    PopSize = random.randint(100, 500)
    noGen = random.randint(100,200)
    while True: 
        tourSize = random.randint(10,50)
        if tourSize < PopSize:
            break 
        else: 
            pass 
    treeDepth = (random.randint(3,7), random.randint(8,15))
    while True: 
        x = 0
        crosCoeff = random.uniform(0.01,1)
        pSubMute = random.uniform(0.001,1)
        pHoistMute = random.uniform(0.001,1)
        pPointMute = random.uniform(0.001,1)
        x = crosCoeff + pSubMute + pHoistMute + pPointMute 
        if x <= 1: 
            print("Crossover = {}".format(crosCoeff))
            print("SubtreeMute = {}".format(pSubMute))
            print("HoistMute = {}".format(pHoistMute))
            print("PointMute = {}".format(pPointMute))
            break
        else: 
            pass
    stoppingCrit = random.uniform(0,1)/1000.0
    maxSamples = random.uniform(0.9,1)
    constRange = (-random.uniform(0.1,1)*10000, random.uniform(0.1,1)*10000)
    parsimony = random.uniform(0,1)/1000000000.0
    parameters = [PopSize, \
                  noGen,\
                  tourSize,\
                  treeDepth,\
                  crosCoeff,\
                  pSubMute,\
                  pHoistMute,\
                  pPointMute,\
                  stoppingCrit,\
                  maxSamples,\
                  constRange,\
                  parsimony]
    print("Chosen Parameters = {}".format(parameters))
    file0.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(PopSize, \
                                                                           noGen,\
                                                                           tourSize,\
                                                                           treeDepth,\
                                                                           crosCoeff,\
                                                                           pSubMute,\
                                                                           pHoistMute,\
                                                                           pPointMute,\
                                                                           stoppingCrit,\
                                                                           maxSamples,\
                                                                           constRange,\
                                                                           parsimony))
    file1.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(PopSize, \
                                                                           noGen,\
                                                                           tourSize,\
                                                                           treeDepth,\
                                                                           crosCoeff,\
                                                                           pSubMute,\
                                                                           pHoistMute,\
                                                                           pPointMute,\
                                                                           stoppingCrit,\
                                                                           maxSamples,\
                                                                           constRange,\
                                                                           parsimony))
    file0.flush()
    file1.flush()
    return parameters 

def GP(genes, X_train, X_test, y_train, y_test, ColumnNames):
    R2Train = []; R2Valid = []; R2Test = []
    MAETrain = []; MAEValid = []; MAETest = []
    RMSETrain = []; RMSEValid = []; RMSETest = []
    CleanFormulas = [] 
    NumpyFormulas = [] 
    function_set = ['add','sub', 'mul','div', 'sqrt','abs',\
                      'log', 'sin', 'cos', 'tan', 'min', 'max']
    est_gp = gplearn.genetic.SymbolicRegressor(population_size = genes[0], 
                                                generations = genes[1], 
                                                tournament_size = genes[2],
                                                init_depth = genes[3],
                                                p_crossover = genes[4],
                                                p_subtree_mutation= genes[5],
                                                p_hoist_mutation= genes[6], 
                                                p_point_mutation= genes[7],
                                                stopping_criteria = genes[8],
                                                max_samples = genes[9], 
                                                const_range=genes[10],
                                                parsimony_coefficient = genes[11],
                                                n_jobs = -1,
                                                verbose = True,
                                                warm_start= False,
                                                function_set = function_set)
    
    k = 5 
    kf = KFold(n_splits = k, random_state = None)
    for train_index, test_index in kf.split(X_train):
        X_Train, X_Valid = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
        y_Train, y_Valid = y_train.iloc[train_index], y_train.iloc[test_index]
        est_gp.fit(X_Train, y_Train)
        print("Formula = {}".format(est_gp._program))
        print("Formula Depth = {}".format(est_gp._program.depth_))
        print("Formula Length = {}".format(est_gp._program.length_))
        # Write formula into files 
        file0.write("Formula = {}, depth = {}, length = {}".format(est_gp._program, est_gp._program.depth_, est_gp._program.length_))
        file2.write("Formula = {}, depth = {}, length = {}\n".format(est_gp._program, est_gp._program.depth_, est_gp._program.length_))
        CleanFormulas.append([str(est_gp._program)]) 
        print("Current Clean Formula status = {}".format(len(CleanFormulas)))
        NumpyFormulas.append(processingFormulas([str(est_gp._program)]))
        print("Current Numpy Formula status = {}".format(len(NumpyFormulas)))
        print("Numpy formula list = {}".format(NumpyFormulas))
        #######################################################################
        # Obtain Scores on Train part of each CV
        #######################################################################
        R2_train = est_gp.score(X_Train, y_Train)
        MAE_train = mean_absolute_error(y_Train, est_gp.predict(X_Train))
        RMSE_train = np.sqrt(mean_squared_error(y_Train, est_gp.predict(X_Train)))
        R2Train.append(R2_train)
        MAETrain.append(MAE_train)
        RMSETrain.append(RMSE_train)
        #######################################################################
        # Obtain Scores on Validation Fold in each CV 
        #######################################################################
        R2_valid = est_gp.score(X_Valid, y_Valid)
        MAE_valid = mean_absolute_error(y_Valid, est_gp.predict(X_Valid))
        RMSE_valid = np.sqrt(mean_squared_error(y_Valid, est_gp.predict(X_Valid)))
        R2Valid.append(R2_valid)
        MAEValid.append(MAE_valid)
        RMSEValid.append(RMSE_valid)
    ###########################################################################
    # Calculate average R2, MAE and RMSE
    ###########################################################################
    file4.write("R2TrainCV = {}\n R2ValidCV = {}\n".format(R2Train, R2Valid))
    file4.write("MAETrainCV = {}\n MAEValidCV = {}\n".format(MAETrain, MAEValid))
    file4.write("RMSETrainCV = {}\n RMSEValidCV = {}\n".format(RMSETrain, RMSEValid))
    R2TrainVal = np.mean([np.mean(R2Train),np.mean(R2Valid)])
    MAETrainVal = np.mean([np.mean(MAETrain),np.mean(MAEValid)])
    RMSETrainVal = np.mean([np.mean(RMSETrain) + np.mean(RMSEValid)])
    R2TrainVal_STD = np.std([np.mean(R2Train),np.mean(R2Valid)])
    MAETrainVal_STD = np.std([np.mean(MAETrain),np.mean(MAEValid)])
    RMSETrainVal_STD = np.std([np.mean(RMSETrain) + np.mean(RMSEValid)])
    print("####################################################################")
    print("R2 CV = {}".format(R2TrainVal))
    print("MAE CV = {}".format(MAETrainVal))
    print("RMSE CV = {}".format(RMSETrainVal))
    print("R2 CV STD = {}".format(R2TrainVal_STD))
    print("MAE CV STD = {}".format(MAETrainVal_STD))
    print("RMSE CV STD = {}".format(RMSETrainVal_STD))
    print("####################################################################")
    for i in range(len(NumpyFormulas)):
        file3.write("Clean Fomula Fold {} = {}\n".format(i, NumpyFormulas[i]))
    file4.write("R2_CV_mean = {}\n R2_CV_STD = {}\n".format(R2TrainVal, R2TrainVal_STD))
    file4.write("MAE_CV_mean = {}\n MAE_CV_STD ={}\n".format(MAETrainVal, MAETrainVal_STD))
    file4.write("RMSE_CV_mean = {}\n RMSE_CV_STD = {}\n".format(RMSETrainVal, RMSETrainVal_STD))
    if R2TrainVal > 0.99:
        #############################################
        #Calculate the output of each formula 
        #############################################
        est_gp=gplearn.genetic.SymbolicRegressor(population_size = genes[0], 
                                                    generations = genes[1], 
                                                    tournament_size = genes[2],
                                                    init_depth = genes[3],
                                                    p_crossover = genes[4],
                                                    p_subtree_mutation= genes[5],
                                                    p_hoist_mutation= genes[6], 
                                                    p_point_mutation= genes[7],
                                                    stopping_criteria = genes[8],
                                                    max_samples = genes[9], 
                                                    const_range=genes[10],
                                                    parsimony_coefficient = genes[11],
                                                    n_jobs = -1,
                                                    verbose = True,
                                                    warm_start= False,
                                                    function_set = function_set).fit(X_train, y_train)
        print("Final formula = {}".format(str(est_gp._program)))
        print("Final formula length = {}".format(str(est_gp._program.length_)))
        print("Final formula depth = {}".format(str(est_gp._program.depth_)))
        file2.write("Final formula = {}\n".format(str(est_gp._program)))
        file2.write("Final formula length = {}\n".format(str(est_gp._program.length_)))
        file2.write("Final formula depth = {}\n".format(str(est_gp._program.depth_)))
        file3.write("clean final formula = {}\n".format(processingFormulas([str(est_gp._program)])))
        R2FinalTrain = est_gp.score(X_train, y_train)
        R2FinalTest = est_gp.score(X_test, y_test)
        MAEFinalTrain = mean_absolute_error(y_train, est_gp.predict(X_train))
        MAEFinalTest = mean_absolute_error(y_test, est_gp.predict(X_test))
        RMSEFinalTrain = np.sqrt(mean_squared_error(y_train, est_gp.predict(X_train)))
        RMSEFinalTest = np.sqrt(mean_squared_error(y_test, est_gp.predict(X_test)))
        R2FinalMean = np.mean([R2FinalTrain, R2FinalTest])
        MAEFinalMean = np.mean([MAEFinalTrain, MAEFinalTest])
        RMSEFinalMean = np.mean([RMSEFinalTrain, RMSEFinalTest])
        R2FinalSTD = np.std([R2FinalTrain, R2FinalTest])
        MAEFinalSTD = np.std([MAEFinalTrain, MAEFinalTest])
        RMSEFinalSTD = np.std([RMSEFinalTrain, RMSEFinalTest])
        file4.write("R2FinalTrain = {}\nMAEFinalTrain = {}\nRMSEFinalTrain = {}\n".format(R2FinalTrain, MAEFinalTrain, RMSEFinalTrain))
        file4.write("R2FinalTest = {}\nMAEFinalTest = {}\nRMSEFinalTest = {}\n".format(R2FinalTest, MAEFinalTest, RMSEFinalTest))
        file4.write("R2FinalMean = {}\nMAEFinalMean ={}\nRMSEFinalMean = {}\n".format(R2FinalMean, MAEFinalMean, RMSEFinalMean))
        file4.write("R2FinalStd = {}\nMAEFinalStd ={}\nRMSEFinalStd = {}\n".format(R2FinalSTD, MAEFinalSTD, RMSEFinalSTD))
        file0.flush(); file1.flush(); file2.flush(); file3.flush(); file4.flush()
        return R2FinalMean
    else:
        file0.flush(); file1.flush(); file2.flush(); file3.flush(); file4.flush()
        return R2TrainVal
def processingFormulas(formulaList): 
    for i in range(len(formulaList)):
        if "add" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("add", "np.add")
        if "sub" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("sub", "np.subtract")
        if "mul" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("mul", "np.multiply")
        if "neg" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("neg", "np.negative")
        if "abs" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("abs", "np.abs")
        if "sin" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("sin", "np.sin")
        if "cos" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("cos", "np.cos")
        if "tan" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("tan", "np.tan")
    procFormulaList = formulaList
    return procFormulaList
def log(x1):
      with np.errstate(divide = "ignore", invalid = "ignore"):
          return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)),0.)
def sqrt(x1):
    return np.sqrt(np.abs(x1))
def div(x1,x2):
    with np.errstate(divide = "ignore", invalid = "ignore"):
        return np.where(np.abs(x2) > 0.001, np.divide(x1,x2), 1.)    

def Calculation(FormulaList, X_test, y_test,ColumnNames): 
    #Cahnge the name of the columns in X_test 
    X_test.columns=ColumnNames
    CalculatedValues =[[[] for j in range(len(FormulaList[i]))] for i in range(len(FormulaList)) ]
    print(CalculatedValues)
    #Detect variables in the formula 
    FilterdVariables = [[[] for j in range(len(FormulaList[i]))] for i in range(len(FormulaList)) ]
    for i in range(len(FormulaList)):
        for j in range(len(FormulaList[i])):
            for k in range(len(ColumnNames)):
                if ColumnNames[k] in FormulaList[i][j]:
                    FilterdVariables[i][j].append(ColumnNames[k])
    print(FilterdVariables)
    #Create Simplifed Datasets 
    for i in range(len(FormulaList)):
        for j in range(len(FormulaList[i])): 
            X_test2 = X_test.filter(items = FilterdVariables[i][j]).reset_index(drop=True)
            print(X_test2)
            for k in range(len(X_test2)):
                for l in range(len(FilterdVariables[i][j])):
                    exec("%s= X_test2.loc[%d, '%s']"%(FilterdVariables[i][j][l],k,FilterdVariables[i][j][l]))
                #print(FormulaList[i][j])
                y_true = eval(FormulaList[i][j])
                CalculatedValues[i][j].append(y_true)
                print(y_true)
    # Compare Results 
    R2TestFinal = [[] for i in range(len(FormulaList))]
    MAETestFinal = [[] for i in range(len(FormulaList))]
    RMSETestFinal = [[] for i in range(len(FormulaList))]
    for i in range(len(CalculatedValues)):
        for j in range(len(CalculatedValues[i])):
            R2Score = r2_score(y_test, CalculatedValues[i][j])
            R2TestFinal[i].append(R2Score)
            MAEScore = mean_absolute_error(y_test, CalculatedValues[i][j])
            MAETestFinal[i].append(MAEScore)
            RMSEScore = np.sqrt(mean_squared_error(y_test, CalculatedValues[i][j]))
            RMSETestFinal[i].append(RMSEScore)
    return R2TestFinal, MAETestFinal, RMSETestFinal
name="Dak-2"
file0 = open("{}_GP_History_log.data".format(name),"w")
file1 = open("{}_GP_Parameters.data".format(name),"w")
file2 = open("{}_GP_Raw_Formulas.data".format(name),"w")
file3 = open("{}_GP_Clean_Formulas.data".format(name),"w")
file4 = open("{}_GP_Scores.data".format(name),"w")
k = 0 
while True:
    file0.write("k = {}\n".format(k))
    file1.write("k = {}\n".format(k))
    file2.write("k = {}\n".format(k))
    file3.write("k = {}\n".format(k))
    file4.write("k = {}\n".format(k))
    Parameters = generateGPParameters()
    R2MeanScore = GP(Parameters, X_train, X_test, y_train, y_test, ColumnNames)
    if R2MeanScore > 0.999:
        print("Solution is Found!")
        file0.write("Solution is Found!!\n")
        file1.write("Solution is Found!!\n")
        file2.write("Solution is Found!!\n")
        file3.write("Solution is Found!!\n")
        file4.write("Solution is Found!!\n")
    else:
        k += 1
        pass
file0.close()
file1.close()
file2.close()
file3.close()
file4.close()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        