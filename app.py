import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('random_forest_picklefile.pkl', 'rb'))
#Ipkl = pickle.load(open('leI.pkl'))
#ICpkl=  pickle.load(open('leI.pkl'))
items = pd.read_csv('items.csv')
items_cat = pd.read_csv('item_categories.csv')
sim = pd.read_csv('shop_item_sale.csv')
colm = ['item_cnt_month_mean', 'date_block_num', 'year','item_id',
                      'item_name', 'item_name_length','item_name_word_count', 'item_category_id' ,
                      'item_category_name','item_categories_name_length','shop_id']

def  inputdata(s,itd,m,y):
    out = []
    for i in range(11):
        out.append(0)
    icm = sim[(sim['item_id']==itd) & (sim['shop_id']==s)]['item_cnt_month_mean'].values
    if(icm.size):
        out[0]=icm[0]
    out[1]=((y-2013)*12 + m)-1
    out[2]=y
    out[3]=itd
    iname = items[items['item_id']==itd]['item_name'].values
    #inval = Ipkl.transform(iname.astype(str))
    out[4] =len(iname[0])
    out[5]=len(iname[0])
    out[6]=len(iname[0].split(' '))
    icid = items[items['item_id']==itd]['item_category_id'].values
    out[7] = icid[0]
    icname =items_cat[items_cat['item_category_id']==icid[0]]['item_category_name'].values
    #icnval = ICpkl.transform(icname.astype(str))
    out[8]=len(icname[0])
    out[9]=len(icname[0])
    out[10]=s
    return out
     
    
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #print("refdsd")
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    temp=pd.DataFrame(features,columns=['shop_id','item_id','month','year'])
    shopid = temp['shop_id']
    shopid = int(shopid[0])
    itemid = temp['item_id']
    itemid= int(itemid[0])
    
    month = temp['month']
    month = int(month[0])
    
    year = temp['year']
    year = int(year[0])
    
    final_features = inputdata(shopid,itemid,month,year)
    
    input_variables = pd.DataFrame([final_features],columns=colm,dtype ='int',index =['input'])
    print(input_variables)
                                     
    prediction = model.predict(input_variables).clip(0., 20.)
    output=[]
    output.append(prediction[0].round(4))
    output.append(final_features)

    return render_template('index.html', prediction_text='future sales count is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    features = [np.array(data)]
    temp=pd.DataFrame(features,columns=['shop_id','item_id','month','year'])
    shopid = temp['shop_id']
    shopid = int(shopid[0])
    itemid = temp['item_id']
    itemid= int(itemid[0])
    
    month = temp['month']
    month = int(month[0])
    
    year = temp['year']
    year = int(year[0])
    
    final_features = inputdata(shopid,itemid,month,year)
    
    input_variables = pd.DataFrame([final_features],columns=colm,dtype ='int',index =['input']) 
    
    prediction = model.predict(input_variables).clip(0.,20)
    output = prediction[0].round(4)
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)