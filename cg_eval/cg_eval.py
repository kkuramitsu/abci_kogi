import csv
import sys
import black
import Levenshtein
import nltk
#nltk.download('punkt')
from nltk import bleu_score
from nltk.translate.bleu_score import corpus_bleu
from io import BytesIO
from tokenize import tokenize, open
import re
from sumeval.metrics.rouge import RougeCalculator
import gs

import warnings
warnings.filterwarnings('ignore')

def read_tsv(filename):
    ss = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
          ss.append((row[index_id], row[pred_id]))
    return ss


def Exact_Match(ss,textlist):

  #print("<BLACK_NG>")
  
  #正答数
  correct=0

  #blackが使用できない数
  black_NG=0

  for line in ss:
    index=line[0]
    pred=line[1]

    try:
      index_black=black.format_str(index,mode=black.Mode())[:-1]
      pred_black=black.format_str(pred,mode=black.Mode())[:-1]

      if index_black==pred_black:
        correct+=1

    except:
      black_NG+=1

  #誤答数
  no_correct=len(ss)-correct

  #正答率
  correct_answer_rate=correct/len(ss)

  textlist.append(f'全体件数：{len(ss)}')
  textlist.append(f'BLACK_NG：{black_NG}')
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

  def tokenize_pycode(code):
      try:
          ss=[]
          tokens = tokenize(BytesIO(code.encode('utf-8')).readline)
          for toknum, tokval, _, _, _ in tokens:
              if toknum != 62 and tokval != '' and tokval != 'utf-8':
                  ss.append(tokval)
          return ss
      except:
          return pattern.split(code)

  references=[]
  predictions=[]

  for line in ss:
    index=line[0]
    pred=line[1]

    references.append([tokenize_pycode(index)])
    predictions.append(tokenize_pycode(pred))
    
  bleu=corpus_bleu(references,predictions)
  textlist.append(f'BLEU：{round(bleu,5)}')


def ROUGE_L(ss,textlist):

  rouge = RougeCalculator()
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


def arg(textlist):
  try:
    #textlist.append(f"index = {sys.argv[2]}, pred = {sys.argv[3]}")
    return int(sys.argv[2]),int(sys.argv[3])
  except:
    # textlist.append("index = 2, pred = 1")
    return 2, 1
    

def main():
  global index_id
  global pred_id

  textlist=[]

  index_id, pred_id = arg(textlist)

  ss = read_tsv(sys.argv[1])
  try:
    textlist.append(sys.argv[2])
  except:
    textlist.append(sys.argv[1])

  Exact_Match(ss,textlist)
  BLEU(ss,textlist)
  ROUGE_L(ss,textlist)
  Levenstein(ss,textlist)

  text1=''
  for textlist_line in textlist:
    text1+=textlist_line+'\n'
  
  gs.send_gs(text1)
  
main()
