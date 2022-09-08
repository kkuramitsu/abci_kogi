import csv
import sys
import Levenshtein
import nltk
#nltk.download('punkt')
from nltk import bleu_score
from nltk.translate.bleu_score import corpus_bleu
import re
from sumeval.metrics.rouge import RougeCalculator
from janome.tokenizer import Tokenizer
import mojimoji
import gs

import warnings
warnings.filterwarnings('ignore')

def read_tsv(filename):
    ss = []
    index_id, pred_id=2,1
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
          row_index=mojimoji.zen_to_han(row[index_id])
          row_pred=mojimoji.zen_to_han(row[pred_id])
          ss.append((row_index,row_pred))
    return ss


def Exact_Match(ss,textlist):
  
  #正答数
  correct=0

  for line in ss:
    index=line[0]
    pred=line[1]

    if index == pred:
      correct+=1

  #誤答数
  no_correct=len(ss)-correct

  #正答率
  correct_answer_rate=correct/len(ss)

  textlist.append(f'全体件数：{len(ss)}')
  textlist.append(f'正答数：{correct}')
  textlist.append(f'誤答数：{no_correct}')
  textlist.append(f'正答率：{round(correct_answer_rate,5)}')

def Levenstein(ss,textlist):

  #合計
  sum_Levenstein=0

  for line in ss:
    index=line[0]
    pred=line[1]
    sum_Levenstein += Levenshtein.ratio(index,pred)

  #平均値
  leven=sum_Levenstein/len(ss)
  
  textlist.append(f'leven：{round(leven,5)}')

def BLEU(ss,textlist):
  
  pattern = re.compile(r'[\(, .\+\-\)]')

  def tokenize_japanese(japanese):
    try:
      t=Tokenizer()
      tokens=t.tokenize(japanese)
      ss=[token.surface for token in tokens]
      return ss
    except:
      return pattern.split(japanese)

  references=[]
  predictions=[]

  for line in ss:
    index=line[0]
    pred=line[1]

    references.append([tokenize_japanese(index)])
    predictions.append(tokenize_japanese(pred))
    
  bleu=corpus_bleu(references,predictions)
  textlist.append(f'BLEU：{round(bleu,5)}')


def ROUGE_L(ss,textlist):

  rouge = RougeCalculator(lang='ja')
  sum_ROUGE_score=0

  for line in ss:
    index=line[0]       
    pred=line[1]

    ROUGE_score = rouge.rouge_l(
            summary=pred,
            references=index)
    
    sum_ROUGE_score+=ROUGE_score
  #平均
  ROUGE_score=sum_ROUGE_score/len(ss)

  textlist.append(f'ROUGE-L：{round(ROUGE_score,5)}')


def main():

  textlist=[]

  ss = read_tsv(sys.argv[1])
  
  #resultのファイル名記録
  try:
    textlist.append(sys.argv[2])
  except:
    textlist.append(sys.argv[1])

  Exact_Match(ss,textlist)
  BLEU(ss,textlist)
  ROUGE_L(ss,textlist)
  Levenstein(ss,textlist)

  text1=""
  for textlist_line in textlist:
    text1+=textlist_line+'\n'
  
  gs.send_gs(text1)

main()