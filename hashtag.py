# -*- coding: utf-8 -*-
"""
Created on Fri Jan 03 11:30:33 2014

@author: NANDAN
"""

import numpy as np
import pandas as pn
import matplotlib.pylab as mp
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import sklearn.linear_model as lm
import sklearn.preprocessing as prep
from sklearn.ensemble import RandomForestRegressor 
from sklearn.naive_bayes import BaseNB,GaussianNB,MultinomialNB,BernoulliNB
from sklearn.decomposition import TruncatedSVD
from nltk.stem import LancasterStemmer,SnowballStemmer 
from nltk.stem.snowball import EnglishStemmer 
from nltk import word_tokenize,wordpunct_tokenize
import re
import string
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn import svm

class LancasterTokenizer(object):
        def __init__(self):
            self.wnl = LancasterStemmer()
        def __call__(self, doc):
            return [self.wnl.stem(t) for t in wordpunct_tokenize(doc)]

def main():
    traindata = np.array(pn.read_csv("C:\Users\DAN\Documents\Python Scripts\Hashtags\\train.csv",encoding='utf-8'))[:,:]
    testdata = np.array(pn.read_csv("C:\Users\DAN\Documents\Python Scripts\Hashtags\\test.csv",encoding='utf-8'))[:,1]
    ys = traindata[:,4:9]
    yw = traindata[:,9:13]
    yk = traindata[:,13:28]
    #y = y.astype(int)
  
    #mp.plot(traindata[:,0],yk, 'rx')
    traindata = list(traindata[:,1])
    testdata = list(testdata)
    X_all = traindata + testdata
    lentrain = len(traindata)
    print ys.shape, len(X_all)
    X_sum = X_all
    
#Well, well we thought removing the unecessary stuff would help but no (almost same result)
#Nevertheless, My Regex Expertise would come into play someday!
    for index in range(0,len(X_sum)):
        X_sum[index] = string.lower(X_sum[index])
        X_sum[index] = re.sub(r'RT|\@mention',"",X_sum[index])
        X_sum[index] = re.sub(r'cloud\w+(\s|\W)',"cloud ",X_sum[index])
        X_sum[index] = re.sub(r'rain\w+(\s|\W)',"rain ",X_sum[index])
        X_sum[index] = re.sub(r'hot\w+(\s|\W)',"hot ",X_sum[index])
        X_sum[index] = re.sub(r'thunder\w+(\s|\W)',"thunder ",X_sum[index])
        X_sum[index] = re.sub(r'freeze\w+(\s|\W)',"freeze ",X_sum[index])
        X_sum[index] = re.sub(r'rain\w+(\s|\W)',"rain ",X_sum[index])
        X_sum[index] = re.sub(r' sun\w+(\s|\W)'," sun ",X_sum[index])
        X_sum[index] = re.sub(r' wind\w+(\s|\W)',"wind ",X_sum[index])
    
    X_1 = TFID(X_all,1)
 #Okay, so using ensemble of models with word and char vectors increases the performance    
    X_2 = TFID(X_all,2)
    X_4 = TFID(X_all, 4)
    X_5 = TFID(X_sum,1)
    
    
#    tsvd = TruncatedSVD(n_components=100)
#    print 'Fitting the SVD'
#    tsvd.fit(X_all)
#    X_svd = tsvd.transform(X_all)
#    X = X_svd[:lentrain]
#    X_test = X_svd[lentrain:]
#    print X_svd.shape
    
    os1,ow1,ok1 = modelsCombo(X_1,ys,yw,yk,lentrain)
    os2,ow2,ok2 = modelsCombo(X_2,ys,yw,yk,lentrain)
    os4,ow4,ok4 = modelsCombo(X_4,ys,yw,yk,lentrain)
    os5,ow5,ok5 = modelsCombo(X_5,ys,yw,yk,lentrain)  
    
   
    out = np.hstack(((os1+os2+os4+os5)/4,(ow1+ow2+ow4+ow5)/4,(ok1+ok2+ok4+ok5)/4))
    #out = np.hstack(((outs+outsc)/2,(outw+outwc)/2,(outk+outkc)/2))
    
    #out = np.hstack((os4,ow4,ok4))
    #print out[1,:]
    
    
    #SGD(X,yk,X_test)
    #gridLog(X,yw)
    
    write(list(out))
    
    #Clean up
    del (traindata,testdata,X_sum)
     
     
def TFID(data , choice):
    #Again removing stop words increased the efficiency, same for Snowball
    
    if(choice==1):    
        tfv = TfidfVectorizer(min_df=3,   max_features=None, strip_accents='unicode',  analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1)
        #, tokenizer=Snowball()    
        print "fitting pipeline and transforming for ", len(data), ' entries'
        tfv.fit(data)
        vect = tfv.transform(data)
        print vect.shape
        return vect
    elif(choice==2):
        print 'Fitting char pipeline'
        tfvc = TfidfVectorizer(norm='l2',min_df=3,max_df=1.0,strip_accents='unicode',analyzer='char',ngram_range=(2,7),use_idf=1,smooth_idf=1,sublinear_tf=1)  
        tfvc.fit(data)    
        vectc = tfvc.transform(data)
        print 'vectc',vectc.shape 
        return vectc
    elif(choice==3):    
        tfv = TfidfVectorizer(min_df=3,   max_features=None, strip_accents='unicode',  analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1, tokenizer=LancasterTokenizer())
        #, tokenizer=Snowball()    
        print "fitting pipeline and transforming for ", len(data), ' entries'
        tfv.fit(data)
        vect = tfv.transform(data)
        print vect.shape
        return vect
    elif(choice==4):    
        tfv = CountVectorizer(min_df=3,   max_features=None, strip_accents='unicode',  analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1, 3), binary=True)  
        print "fitting count pipeline and transforming for ", len(data), ' entries'
        tfv.fit(data)
        vect = tfv.transform(data)
        print vect.shape
        return vect
    else:
        return []
    

def modelsCombo(X_all,ys,yw,yk,lentrain):
   
    X = X_all[:lentrain]
    X_test = X_all[lentrain:]  
   
    print 'S'
    outs = ridge(X,ys,X_test)
    #print outs[1,:]
    outs = np.clip(outs,0,1)
    outs = prep.normalize(outs,norm='l1')   
    #print outs[1,:]
    print np.sum(outs,axis=1)
    
    print 'W'
    outw = ridge(X,yw,X_test)
    outw = prep.normalize(outw,norm='l1') 
    outw = np.clip(outw,0,1)    
    
    outk = ridge(X,yk,X_test)
    outk = np.clip(outk,0,1)
    print 'All'
    
    return outs,outw,outk

def ridge(X,y,X_test):
    ri = lm.Ridge(alpha=1,tol=0.001,solver='auto',fit_intercept=True) 
        
#    c = cross_validation.cross_val_score(ri, X, y, cv=5, scoring='mean_squared_error')
#    print "5 Fold CV Score: via mse scoring ", np.mean(c) , "+/-" , np.std(c)*2
#    print c
    
    ri.fit(X,y)
    print 'Ridge Hashtagged'
    out = ri.predict(X_test)
    #print 'root mean sq error %f', mean_squared_error(ri.predict(X),y)
    
#Strange but ensemble of ridge classifiers yields no change
#    kf = StratifiedKFold(y,n_folds=5)
#    i=0
#    for train_index, test_index in kf:
#        ri.fit(X[train_index],y[train_index])
#        if(i==0):        
#            oute = ri.predict(X_test)
#            print len(oute)
#        else:
#            print oute
#            print i
#            oute = oute + ri.predict(X_test)
#        
#        i = i+1    
#    print 'i',len(oute/5)
#    return np.array(oute/5)
    return np.array(out)


def rfor(X,y,X_test):
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features=100)
    rf.fit(X,y)
    
    out = rf.predict(X_test)
    
    return np.array(out)
        
        
def write(pred):
    testfile = pn.read_csv('C:\Users\DAN\Documents\Python Scripts\Hashtags\\test.csv', na_values=['?'], index_col=0)
    pred_df = pn.DataFrame(pred, index=testfile.index, columns=['s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15'])
    pred_df.to_csv('C:\Users\DAN\Documents\Python Scripts\Hashtags\\hash.csv')
    

     
def gridLog(X,y):
    # Demonstraing the use of Grid Search to effectively search for parameters best suited
    ri = lm.Ridge(solver='auto',fit_intercept=True,tol=0.001)    
    param_grid = {'alpha': [0.1,0.5,1]}    
    #param_grid = {'alpha': [0.1,0.5,1],'tol':[0.001,0.01]}
    g = GridSearchCV(ri,param_grid,cv=10)
 
    g.fit(X,y)
    
    print("Best score: %0.3f" % g.best_score_)
 
    print("Best parameters set:")
    best_parameters = g.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
if __name__ == '__main__': 
    main()