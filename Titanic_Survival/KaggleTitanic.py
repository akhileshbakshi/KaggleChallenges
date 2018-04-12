import warnings
warnings.filterwarnings('ignore')

# import packages -------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.cross_validation import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# functions for visualization, data extraction --------------------------------
def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew()),))) 
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = titanic.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def plot_box_plot (X, xvar, yvar):
    data = pd.concat([X[xvar], X[yvar]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=xvar, y=yvar, data=data)
    #fig.axis(ymin=0, ymax=800000)    

def plot_pairscatter (X, cols):
    X = X.dropna()
    sns.set()
    sns.pairplot(X[cols], size = 2.5)
    plt.show()
    

# extracts prefix of ticket, returns 'XXX' if no prefix
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

# runs pca, returns plots for explained variance, heatmap, transformed data 
def runPCA (trainData, cutoffVariance, testData): 
    pca = PCA()
    pca.fit(trainData)
    
    # plot explained variance 
    fig = plt.figure( figsize = (4, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylabel('Cumsum Explained Variance')
    plt.xlabel('# components')
    plt.xticks(np.arange(0,np.shape(trainData)[1],2))
    plt.show() 
    
    # plot heatmap 
    fig = plt.figure( figsize = (8, 8))
    sns.heatmap(np.abs(np.log(pca.inverse_transform(np.eye(trainData.shape[1])))), cmap="hot", cbar=False)
    plt.xticks(np.arange(trainData.shape[1]),list(trainData),rotation =90)
    plt.xticks(np.arange(1,1+trainData.shape[1]),list(trainData))
    plt.ylabel('Principal Component')
    plt.xlabel('Input Feature')
    
    nComponentsReturn = sum(np.cumsum(pca.explained_variance_ratio_)<cutoffVariance)
    
    return pd.DataFrame(pca.transform(trainData)[:,:nComponentsReturn]), \
            pd.DataFrame(pca.transform(testData)[:,:nComponentsReturn]) 
    
    
    
    
 # import data ----------------------------------------------------------------
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

full = train.append( test , ignore_index = True )
titanic = full[ :891 ]
del train , test




# # descriptive statistics ----------------------------------------------------
#titanic.describe()
## see correlation map 
#plot_correlation_map( titanic )
## Plot distributions for continuous variables
#plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )
## Plot bars for categorical variables 
#plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )





# data preparation & feature selection ----------------------------------------

# convert male-female data into 0s and 1s 
sex = pd.Series(np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )

# create new variables for unique values of Embarked, pclass
embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

# fill missing values in variables with median  
ageFare = pd.DataFrame()         
ageFare[ 'Age' ] = full.Age.fillna( full.Age.median() )
ageFare[ 'Fare' ] = full.Fare.fillna( full.Fare.median() )

# extract titles from passenger names 
title = pd.DataFrame()
title[ 'Title' ] = full['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )


cabin = pd.DataFrame()
# replacing missing cabins with U (for Unknown)
cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )
# mapping each Cabin value with the cabin (first) letter
cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )
cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )

# extract critical alphabets from tickets
ticket = pd.DataFrame()
ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )


family = pd.DataFrame()
# introducing a new feature : size of families (including the passenger)
family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )


# embarked.C has significantly higher survival rate than other two (which are similar) 
# 957 unknown tickets (Ticket_XXX), and rest can be clubbed e.g. SOTO** ... but ignoring
# remove Master, Miss, Mr and Mrs because age, gender have been captured separately 
full_X = pd.concat([embarked.Embarked_C, pclass, sex, ageFare, title, family.FamilySize],axis=1)


# model selection -------------------------------------------------------------

trainValid_X = full_X[ 0:891 ]
trainValid_y = titanic.Survived
finalTest_X = full_X[ 891: ]

# scale data 
scaler = StandardScaler()
scaler.fit(trainValid_X)
columns = list(trainValid_X) 
trainValid_X = pd.DataFrame(scaler.transform(trainValid_X), columns = columns)
finalTest_X = pd.DataFrame(scaler.transform(finalTest_X), columns = columns)  

# pca analysis to reduce dimensions, from 14 -> 10 
trainValid_X, finalTest_X = runPCA(trainValid_X, 0.98, finalTest_X)

# split train, valid, test data 
train_X, valid_X, train_y, valid_y = train_test_split(trainValid_X, trainValid_y, train_size=.8)
cv_X, test_X, cv_y, test_y = train_test_split(valid_X, valid_y, train_size=.5 )


# For each model, optimize parameters and compare against one another 
# NOTE: this can be done faster using gridsearchCV 
#def switchmodel(model, ctr):
#    if model==0: 
#       parameters = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
#       return LogisticRegression(C = parameters[ctr]) 
#    elif model==1: 
#        parameters = np.arange(1,9,1)
#        return GradientBoostingClassifier(max_depth = parameters[ctr])
#    elif model==2:
#        parameters = np.arange(1,81,10)
#        return RandomForestClassifier(n_estimators=1000, min_samples_leaf = parameters[ctr])  
#    elif model==3: 
#        parameters = np.arange(3,11,1)
#        return SVC(C = parameters[ctr])          
#
#modelnum = 2     
#trainscore = np.zeros((8,1))
#CVscore = np.zeros((8,1))
#for counter in range(8):
#    model = switchmodel(modelnum, counter)
#    model.fit(train_X, train_y)
#    trainscore[counter] = model.score(train_X, train_y)
#    CVscore[counter] = model.score(cv_X, cv_y)
#    
#print(trainscore)
#print(CVscore) 


# pick model 
#model = LogisticRegression(C = 0.3) 
#model = GradientBoostingClassifier(max_depth = 7)
#model = RandomForestClassifier(n_estimators=1000, min_samples_leaf = 1) 
#model = SVC(C = 5) 
#model.fit(train_X.append(cv_X , ignore_index = True ), train_y.append(cv_y , ignore_index = True ))
#testscore = model.score(test_X, test_y)   
#print(testscore) 



# model training --------------------------------------------------------------

alldata_X = train_X.append(cv_X , ignore_index = True ).append(test_X , ignore_index = True )
alldata_y = train_y.append(cv_y , ignore_index = True ).append(test_y , ignore_index = True )
# SVC - optimize over C 
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
Cs = np.arange(0.05,10,0.05)
param_grid = {'C': Cs}
grid = GridSearchCV(SVC(), param_grid, cv=kfold, verbose = 10)
grid_result = grid.fit(alldata_X, alldata_y)


# predict yesy daya 
model = SVC(C = 0.25)
model.fit(alldata_X, alldata_y)
model.predict(finalTest_X)



