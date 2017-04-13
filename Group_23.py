import pandas
from sklearn import preprocessing
from sklearn import linear_model
import numpy as np
#importing training data
train_data= pandas.read_csv('train.csv')

#combination of features
train_data['weekday-hour'] = train_data.apply(lambda x:'%s-%s' % (x['weekday'],x['hour']),axis=1)
train_data['slotwidth-slotheight'] = train_data.apply(lambda x:'%s-%s' % (x['slotwidth'],x['slotheight']),axis=1)
train_data['useragent-slotvis'] = train_data.apply(lambda x:'%s-%s' % (x['useragent'],x['slotvisibility']),axis=1)

#splitting user tags
unique_usertags = []
def func(group):
    unique_groups_tags = set(group.usertag.str.split(',').values[0])
    for i in unique_groups_tags:
        if (len(unique_usertags) > 0):
            unique_groups_set = unique_usertags[0]
            if i not in unique_usertags:
                unique_usertags.append(i)
        else:
            unique_usertags.append(i)
            
        
    return pandas.Series(group.usertag.str.split(',').values[0], name='usertag')

#hot encode the data
def one_hot_encode(df):
    ser = df.groupby(level=0).apply(func)
    pandas.get_dummies(ser)
    df_dummies_utag = pandas.get_dummies(ser).groupby(level=0).apply(lambda group: group.max())
    del df['usertag']
    df = df.join(df_dummies_utag)
    return df

le = preprocessing.LabelEncoder()

#dropping extra columns that we do not need for the model training
train_data = train_data.drop(['IP','logtype','adexchange','url','urlid','slotid','keypage'], axis=1)
train_data = one_hot_encode(train_data)
train_data = train_data.apply(le.fit_transform)

#loading test set
test_data= pandas.read_csv('test.csv')

bidlist = []
bidlist1 = []
bidlist2 = []
bidlist3 = []

for i in test_data['bidid']:
    bidlist.append(i)
    bidlist1.append(i)
    bidlist2.append(i)
    bidlist3.append(i)

# feature combine for test data
# note, might need to combine '2'+'SecondView' / '1'+'FirstView' / '(0'+Other?)
test_data['weekday-hour'] = test_data.apply(lambda x:'%s-%s' % (x['weekday'],x['hour']),axis=1)
test_data['slotwidth-slotheight'] = test_data.apply(lambda x:'%s-%s' % (x['slotwidth'],x['slotheight']),axis=1)
test_data['useragent-slotvis'] = test_data.apply(lambda x:'%s-%s' % (x['useragent'],x['slotvisibility']),axis=1)
test_data = test_data.drop(['IP','logtype','adexchange','url','urlid','slotid','keypage'], axis=1)
test_data = one_hot_encode(test_data)
test_data = test_data.apply(le.fit_transform)

#defining targets for the model
training_data_targets = np.array([train_data.click]).T
train_data.convert_objects(convert_numeric=True)

#defining features for the model
training_data_many_features = train_data[['weekday','city','region','hour','useragent','domain','slotprice','advertiser','creative','weekday-hour','useragent-slotvis','slotwidth-slotheight']+unique_usertags].values
test_data_many_features = test_data[['weekday','city','region','hour','useragent','domain','slotprice','advertiser','creative','weekday-hour','useragent-slotvis','slotwidth-slotheight']+unique_usertags].values


#model no. 1 logistic regression

from sklearn.linear_model import LogisticRegression
log_reg_model = LogisticRegression()
log_reg_model.fit(training_data_many_features, training_data_targets)

click_pred=[]
click_pred_prob_zero=[]
click_pred_prob_one=[]

for a in range(0,len(test_data)):
    click=test_data._slice(slice(a,a+1))
    click_df = click.iloc[0]
    click_many_features=click[['weekday','city','region','hour','useragent','domain','slotprice','advertiser','creative','weekday-hour','useragent-slotvis','slotwidth-slotheight']+unique_usertags].values
    predicted_click = log_reg_model.predict(click_many_features)
    predicted_click_probability = log_reg_model.predict_proba(click_many_features)
    click_pred.append(predicted_click)
    #probability of preditcted click being a zero
    click_pred_prob_zero.append(predicted_click_probability[0][0])
    #probability of preditcted click being a one
    click_pred_prob_one.append(predicted_click_probability[0][1])
    
    
#model no. 2 Gradient boosting classifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
params = {'n_estimators': 100, 'max_depth': 6,
        'learning_rate': 0.1, 'alpha':0.98}
clf_init = LogisticRegression();
clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                             n_estimators=100, subsample=1.0,
                             min_samples_split=2,
                             min_samples_leaf=5,
                             max_depth=6,
                             random_state=None,
                             max_features=None,
                             verbose=2)

clf.fit(training_data_many_features, training_data_targets)

click_p=[]
click_p_prob_zero=[]
click_p_prob_one=[]
for a in range(0,len(test_data)):
    click1=test_data._slice(slice(a,a+1))
    click_df1 = click.iloc[0]
    click_many_features1=click1[['weekday','city','region','hour','useragent','domain','slotprice','advertiser','creative','weekday-hour','useragent-slotvis','slotwidth-slotheight']+unique_usertags].values
    predicted_click1 = clf.predict(click_many_features1)
    predicted_click_probability1 = clf.predict_proba(click_many_features1)
    click_p.append(predicted_click1)
    #probability of preditcted click being a zero
    click_p_prob_zero.append(predicted_click_probability1[0][0])
    #probability of preditcted click being a one
    click_p_prob_one.append(predicted_click_probability1[0][1])
    
#model no. 3 Gradient boosting regressor to predicit values    
    
from sklearn.ensemble import GradientBoostingRegressor
training_data_targets_payprice = np.array([train_data.payprice]).T
params = {'n_estimators': 100, 'max_depth': 6,
        'learning_rate': 0.1, 'alpha':0.98}
gbr = GradientBoostingRegressor(**params).fit(training_data_many_features, training_data_targets_payprice)
predict_payprice=[]
for a in range(0,len(test_data)):
    bid_=test_data._slice(slice(a,a+1))
    bid_features=bid_[['weekday','city','region','hour','useragent','domain','slotprice','advertiser','creative','weekday-hour','useragent-slotvis','slotwidth-slotheight']+unique_usertags].values
    predicted_price = gbr.predict(bid_features)
    predict_payprice.append(predicted_price)

#combination of all models

#getting the predict payprice form boosting regressor model
predict_payprice_final=predict_payprice

for i in range (0,len(predict_payprice)):
    #probability of being one from gradient classifier
    if (click_p_prob_one[i]>0.0001):
        predict_payprice_final[i]=300
        #probability of being zero from gradient classifier
        if (click_p_prob_zero[i]>0.8):
            predict_payprice_final[i]=predict_payprice_final[i]/2
        #probability of being zero from log regression
        if (click_pred_prob_zero[i]>0.97):
            predict_payprice_final[i]=predict_payprice_final[i]/2

            df_pred_pp = pandas.DataFrame(predict_payprice_final)
df_bid = pandas.DataFrame(bidlist)
    
d_f1 = pandas.concat([df_bid,df_pred_pp], axis=1)
d_f1.to_csv('Group_23.csv', index=False)