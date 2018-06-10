import sys
sys.path.append("//home/ubuntu/fastai/")
from fastai.imports import *
from fastai.torch_imports import *
from fastai.dataset import *
from fastai.plots import *
from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from IPython.display import HTML

def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())

def train_model(lrnRate, trainDataFile, testDataFile, cat_vars, contin_vars):

    PATH="//home/ubuntu/fastai/courses/data/EnergyPriceDLModel/"
    data_Train = pd.read_csv(f'{PATH}'+trainDataFile, parse_dates=['Date'])    
    data_Test = pd.read_csv(f'{PATH}'+testDataFile, parse_dates=['Date'])
    add_datepart(data_Train, "Date", drop=False)    
    add_datepart(data_Test, "Date", drop=False)
    
    columns = ['Is_month_end', 'Is_month_start', 'Is_quarter_end','Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Elapsed']
    data_Train.drop(columns,1,inplace=True)
    data_Test.drop(columns,1,inplace=True)
    
    #Convert to feather
    data_Train.reset_index(inplace=True)    
    data_Test.reset_index(inplace=True)    
    data_Train.to_feather(f'{PATH}df')
    data_Test.to_feather(f'{PATH}df_test')
    data_Train = pd.read_feather(f'{PATH}df')    
    data_Test= pd.read_feather(f'{PATH}df_test')
    data_Train["Date"] = pd.to_datetime(data_Train.Date)
    data_Test["Date"] = pd.to_datetime(data_Test.Date)

    #Catorgorise Variables and apply to data
    ##cat_vars = ['Date', 'Time','TimeOfDay','Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear']
    ##contin_vars = ['Demand']
    dep = 'Price'
    data_Train = data_Train[cat_vars+contin_vars+[dep, 'Index']].copy()   
    data_Test[dep] = 0
    data_Test = data_Test[cat_vars+contin_vars+[dep, 'Index']].copy() 
    for v in cat_vars: data_Train[v] = data_Train[v].astype('category').cat.as_ordered()

    apply_cats(data_Test, data_Train)

    for v in contin_vars:
        data_Train[v] = data_Train[v].astype('float32')  
        data_Test[v] = data_Test[v].astype('float32')      
    
    data_Train = data_Train.set_index("Index")
    data_Test = data_Test.set_index("Index")

    #Start to build model
    df, y, nas, mapper = proc_df(data_Train, 'Price', do_scale=True)
    yl = np.log(y)
    df_test, _, nas, mapper = proc_df(data_Test, 'Price', do_scale=True, mapper=mapper, na_dict=nas)

    samp_size = len(df)
    train_ratio = 0.75
    train_size = int(samp_size * train_ratio); train_size
    val_idx = list(range(train_size, len(df)))

    max_log_y = np.max(yl)
    y_range = (0, max_log_y*1.2)

    md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128, test_df=df_test)
    cat_sz = [(c, len(data_Train[c].cat.categories)+1) for c in cat_vars]
    emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
    m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars), 0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
    m.fit(lrnRate, 5, metrics=[exp_rmspe], cycle_len=1)
    m.fit(lrnRate, 3, metrics=[exp_rmspe], cycle_len=3)

    return m

def predict_single_dataPoint(m, data, catVars):
    
    cat = data[catVars].values.astype(np.int64)[None]
    contin = data.drop(catVars).values.astype(np.float32)[None]

    model = m.model
    model.eval()
    prediction = to_np(model(V(cat), V(contin)))
    price = np.exp(prediction)

    return price


