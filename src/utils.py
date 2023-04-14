import plotfa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report ,multilabel_confusion_matrix ,ConfusionMatrixDisplay
sns.set_style('darkgrid')

from hazm import Normalizer , word_tokenize
from cleantext import clean
import regex as re
import json


stop_words = set(open('/content/drive/MyDrive/Roshan_Internship1401/nlp/dataset/stop-words.txt', encoding='utf8').read().splitlines())

def clean_html(s):
    return re.sub(re.compile('<.*?>'), '', s)

def text_preprocessing(s):
  s = s.strip()
  
  s = clean(s,
    fix_unicode=True,               
    to_ascii=False,                  
    lower=True,                    
    no_line_breaks=True,          
    no_urls=True,                 
    no_emails=True,                
    no_phone_numbers=True,         
    no_digits=False,               
    no_currency_symbols=False,      
    no_punct=False,                
    replace_with_punct="",         
    replace_with_url="",
    replace_with_email="",
    replace_with_phone_number="",
    replace_with_number="",
    replace_with_digit="0",
    replace_with_currency_symbol="",
  )
  
  s = re.sub(re.compile('<.*?>'), '', s)    #clean html tags

  normalizer = Normalizer()
  s = normalizer.normalize(s)
  s = normalizer.character_refinement(s)

  s = re.sub("\s+", " ", s)         # trailing whitespace
  s = re.sub(r'(@.*?)[\s]', ' ', s) # @ mentions
  s = re.sub("#", "", s)            # hashtags

  tokens = word_tokenize(s)
  s = ' '.join(tokens)
  return s


#----------------------------------------------------------------------------------------------------

def get_classes(labels) :
  topics = list()
  for row in labels:
    for x in row :
      if x[0] not in topics :
        topics.append(x[0])
  return topics


# string lists to list       '[['اشکال فنی', False]]'          --->         [['اشکال فنی', False]]
def labels_to_list( labels ):
  for i ,lbl in enumerate(labels) :
    if isinstance(lbl ,str) :
      try :
        labels[i] = json.loads(str(lbl))
      except :
        print(i)
        break
  return labels



def df_with_hot_label(df) :
  df_copy = df.copy()
  list_labels = labels_to_list(df.labels.tolist())
  classes = get_classes(list_labels)
  for lbl in classes :
    df_copy[lbl] = np.zeros((df_copy.shape[0] ,1))
    for j in range(df.shape[0]) :
      for ls in list_labels[j] :
        if ls[0]==lbl and ls[1]==True :
          df_copy[lbl][j] = 1
  return df_copy



def reduce_data(df) : 
  zeros = df[(df[df.columns[3:]] == 0).all(axis=1)]
  num = int(zeros.shape[0]*0.15)
  zeros = zeros.head(num)
  non_zeros = df[(df[df.columns[3:]] != 0).any(axis=1)]
  result = pd.concat([zeros, non_zeros], axis=0)
  df = result.sample(frac=1, random_state=42).reset_index(drop=True)

  return df










class EvaluateModel():
  def __init__(self ,y_true ,y_preds ,id_label,model_name):

    self.y_true = y_true
    self.y_preds = y_preds
    self.id_label = id_label
    self.model_name = model_name
    self.save_dir = os.path.join(os.getcwd(),'drive',"MyDrive", 'Roshan_Internship1401','nlp' ,'models', model_name,'logs')
    os.makedirs( os.path.join(os.getcwd(),'drive',"MyDrive", 'Roshan_Internship1401','nlp' ,'models', model_name,'logs')  ,exist_ok=True)



  def find_best_threshold(self ,step = 0.001  ,confidence = 1):
      best_threshold = 0
      start ,end ,step = (0 ,1 ,step)
      f1_scores = []
      for i in range(confidence):
        thresholds = np.arange(start, end, step)
        for threshold in thresholds:
          y_pred = (self.y_preds > threshold).astype(int)
          f1 = f1_score(self.y_true, y_pred ,average='micro')
          f1_scores.append(f1)

        best_threshold = thresholds[np.argmax(f1_scores)]
        start = thresholds[np.argmax(f1_scores)] - step
        end = thresholds[np.argmax(f1_scores)] + step
        step*=0.1

      return best_threshold

  def report(self,):
    labels ,preds = self.y_true ,(self.y_preds > self.find_best_threshold()).astype(int)
    result_dic = classification_report( labels, preds, target_names=list(self.id_label['label2id'].keys()) ,output_dict=True)
    df = pd.DataFrame.from_dict(result_dic).T
    df.to_csv(os.path.join(self.save_dir+'/report.csv'))
    print("Report CSV file added to logs .")
    return df

    return result_dic

  def get_confusion(self,):
    labels ,preds= self.y_true.astype(int) , (self.y_preds > self.find_best_threshold()).astype(int)
    return multilabel_confusion_matrix( labels, preds )   

  def plot_confusion(self,):
    cm = self.get_confusion()  

    f, axes = plt.subplots(3, 3, figsize=(9, 9) ,sharey=True ,sharex=True)
    f.suptitle(self.model_name)
    axes = axes.ravel()
    for i in range(9):
      disp = ConfusionMatrixDisplay(cm[i])
      axes[i].grid(False)
      axes[i].set_xticks([1, 0])
      axes[i].set_yticks([1, 0])
      disp.plot(ax=axes[i], values_format='.4g')
      disp.ax_.set_title(plotfa.fa(self.id_label['id2label'][i]) ,color = 'black').set_fontsize(10)
      disp.ax_.set_xlabel('')
      disp.ax_.set_ylabel('')
      disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    dest = os.path.join(self.save_dir+'/confusion_matrix.png')
    plt.savefig( dest )
    plt.show(block=False)
