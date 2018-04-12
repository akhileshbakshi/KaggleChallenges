# Potential improvements: 
# 1. account for inflation in house sale prices 
# 2. remove outliers before regression 

import warnings
warnings.filterwarnings('ignore')

# import packages -------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
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
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
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
    corr = np.abs(df.corr())
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(corr, cmap = cmap, square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, vmax=.8, vmin=.1, #annot = True, annot_kws = { 'fontsize' : 12 }
    )

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

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
                        
# runs pca, returns plots for explained variance, heatmap, transformed data 
def addDumies (data, targetColumn): 
    return pd.get_dummies( data[targetColumn] , prefix=targetColumn)
    
  
    

# import data ----------------------------------------------------------------

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
full = train.append( test , ignore_index = True )
del test



# # descriptive statistics ----------------------------------------------------

## box plot 
#plot_box_plot(train, 'OverallQual', 'SalePrice')
## pair scatter plots 
#cols = ['LotFrontage', 'LotArea', 'OverallQual', 'MasVnrArea', 'BsmtFinSF1', 'GrLivArea',  'GarageArea', 'TotalBath', 'totalPorch']
#plot_pairscatter(train, cols) 
## correlation map 
#plot_correlation_map(train)
## box plot of categorical variables 
#sns.boxplot(x="Neighborhood", y="SalePrice", data=full)



# data preparation & feature selection ----------------------------------------

LotFrontage, LotArea, OverallQual, MasVnrArea, BsmtFinSF1, GrLivArea, GarageArea = \
        full.LotFrontage, full.LotArea, full.OverallQual, full.MasVnrArea, \
        full.BsmtFinSF1, full.GrLivArea, full.GarageArea 
totalPorch = pd.Series(full.WoodDeckSF+full.OpenPorchSF, name = 'totalPorch') 
TimeFromRemodel = pd.Series(full.YrSold - full.YearRemodAdd, name = 'TimeFromRemodel') 
TotalBath = pd.Series(full.BsmtFullBath+0.5*full.BsmtHalfBath+full.FullBath+0.5*full.HalfBath, name='TotalBath') 
Fireplaces = pd.Series(full.Fireplaces.map( lambda s : 0 if s == 0 else 1 ), name = 'Fireplaces')

# convert to binary data 
Street = pd.Series(addDumies(full,'Street').Street_Grvl, name = 'Street')   
Lotshape = pd.Series(addDumies(full,'LotShape').LotShape_Reg, name = 'Lotshape')
LandContour = pd.Series(addDumies(full,'LandContour').LandContour_Bnk, name = 'LandContour')
LotConfig = pd.Series(addDumies(full,'LotConfig').LotConfig_CulDSac + \
                      addDumies(full,'LotConfig').LotConfig_FR3, name = 'LotConfig')

# categorize neighborhood as cheap based on median prices of houses within neighborhood  
variable = addDumies(train,'Neighborhood')
NeighborhoodMedian = np.zeros((np.shape(variable)[1],1))  
for icolumn, columnName in enumerate(list(variable)):
    NeighborhoodMedian[icolumn] = np.median(np.unique(variable.ix[:,icolumn] * train.SalePrice)[1:])
cutoff1 = np.sort(NeighborhoodMedian,axis=0)[11]
CheapNeighborhood = (NeighborhoodMedian <= cutoff1).astype(int) 

HouseinCheapNeighborhood = np.zeros((np.shape(full)[0],1))
for ineighborhood in range(np.shape(variable)[1]):
    if CheapNeighborhood[ineighborhood] == 1: 
        indices = np.where(variable.ix[:,ineighborhood]==1)
        HouseinCheapNeighborhood[indices] = 1
HouseinCheapNeighborhood = pd.Series(np.concatenate(HouseinCheapNeighborhood, axis = 0), name = 'HouseinCheapNeighborhood') 


ExternalQlty = pd.Series(addDumies(full,'ExterQual').ExterQual_Gd + \
                         addDumies(full,'ExterQual').ExterQual_Ex, name = 'ExternalQlty') 
CentralAirCon = pd.Series(np.where(full.CentralAir == 'Y', 1, 0) , name = 'CentralAir')
KitchenQlty = pd.Series(np.where(full.KitchenQual == 'Ex', 1, 0), name = 'KitchenQual')
SaleType = pd.Series(np.where(full.SaleType == 'New', 1, 0) , name = 'SaleType')
SaleCondition = pd.Series(np.where(full.SaleCondition == 'Partial', 1, 0) , name = 'SaleCondition')

# full data set 
full_X = pd.concat([LotFrontage, LotArea, OverallQual, MasVnrArea, BsmtFinSF1, \
                   GrLivArea, GarageArea, totalPorch, TimeFromRemodel, TotalBath, \
                   Fireplaces, Street, Lotshape, LandContour, LotConfig, HouseinCheapNeighborhood, \
                   ExternalQlty, CentralAirCon, KitchenQlty, SaleCondition, SaleType], axis = 1)
full_X.fillna(full_X.median(), inplace=True) # NOTE: potential tampering of train data with test data 

train_Xy = full_X[:1460]
train_Xy['SalePrice'] = train.SalePrice
# remove features based on very poor corr w/ SalePrice, or high corr. w/ OveralQual 
full_X = full_X.drop(['Street', 'SaleType', 'LandContour', 'LotConfig', 'ExternalQlty'], axis = 1) 

# model selection -------------------------------------------------------------

trainValid_X = full_X[ 0:1460 ]
trainValid_y = train.SalePrice 
finalTest_X = full_X[ 1460: ]

# scale data 
scaler = StandardScaler()
scaler.fit(trainValid_X)
columns = list(trainValid_X) 
trainValid_X = pd.DataFrame(scaler.transform(trainValid_X), columns = columns)
finalTest_X = pd.DataFrame(scaler.transform(finalTest_X), columns = columns)  

# pca analysis to reduce dimensions, from 18 -> 14 
# trainValid_X, finalTest_X = runPCA(trainValid_X, 0.98, finalTest_X)

# split train, valid, test data 
train_X, valid_X, train_y, valid_y = train_test_split(trainValid_X, trainValid_y, train_size=.8)
cv_X, test_X, cv_y, test_y = train_test_split(valid_X, valid_y, train_size=.5 )


# regression methods from here 
# NOTE: haven't tried adaboost regressor 

## linear regression 
#model = LinearRegression()
#model.fit(train_X, train_y)
#print(model.score(cv_X, cv_y))
#
## decision tree regression 
#trainscore, CVscore = np.zeros((11,8)),  np.zeros((11,8))
#for i, max_depth in enumerate(np.arange(1,12)): 
#    for j, min_samples_leaf in enumerate(np.arange(1,9)):
#            model = DecisionTreeRegressor(max_depth = max_depth, min_samples_leaf = min_samples_leaf)  
#            model.fit(train_X, train_y)
#            trainscore[i,j] = model.score(train_X, train_y)
#            CVscore[i,j] = model.score(cv_X, cv_y)
## optimal parameter for decision tree:  
#iopt = np.unravel_index(CVscore.argmax(), CVscore.shape)[0]
#jopt = np.unravel_index(CVscore.argmax(), CVscore.shape)[1]
#DT_max_depth = np.arange(1,12)[iopt]
#DT_min_samples_leaf = np.arange(1,8)[jopt]
#print(DT_max_depth, DT_min_samples_leaf, trainscore[iopt, jopt], CVscore[iopt, jopt])
#
## random forest regression     
#trainscore, CVscore = np.zeros((11,8)),  np.zeros((11,8))
#for i, max_depth in enumerate(np.arange(4,15)): 
#    for j, min_samples_leaf in enumerate(np.arange(1,8)):
#            model = RandomForestRegressor(n_estimators = 1000, max_depth = max_depth, min_samples_leaf = min_samples_leaf)  
#            model.fit(train_X, train_y)
#            trainscore[i,j] = model.score(train_X, train_y)
#            CVscore[i,j] = model.score(cv_X, cv_y)
## optimal parameter for decision tree:  
#iopt = np.unravel_index(CVscore.argmax(), CVscore.shape)[0]
#jopt = np.unravel_index(CVscore.argmax(), CVscore.shape)[1]
#RF_max_depth = np.arange(4,15)[iopt]
#RF_min_samples_leaf = np.arange(1,8)[jopt]
#print(RF_max_depth, RF_min_samples_leaf, trainscore[iopt, jopt], CVscore[iopt, jopt])


# fine tuning parameters for RandomForestRegressor 
alldata_X = train_X.append(cv_X , ignore_index = True ).append(test_X , ignore_index = True )
alldata_y = train_y.append(cv_y , ignore_index = True ).append(test_y , ignore_index = True )
kfold = KFold(n_splits=10, random_state=7)
maxDepths = np.arange(8,24,2)
minLeafs = np.arange(1,5)
param_grid = {'max_depth': maxDepths, 'min_samples_leaf': minLeafs}
model = RandomForestRegressor(n_estimators = 1000)
grid = GridSearchCV(model, param_grid, cv=kfold, verbose = 10, n_jobs = 3)
grid_result = grid.fit(alldata_X, alldata_y)


model = RandomForestRegressor(n_estimators = 1000, max_depth = 16, min_samples_leaf = 2)
#scores = cross_val_score(model, alldata_X, alldata_y, cv = kfold)
model.fit(alldata_X, alldata_y)
model.predict(finalTest_X)
        
        
            








    
    
    
  






