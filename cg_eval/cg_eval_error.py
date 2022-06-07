import csv
import sys
import black
import Levenshtein
import nltk
#nltk.download('punkt')
from nltk import bleu_score
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
  textlist.append('\n')
  textlist.append(f'正答数：{correct}')
  textlist.append('\n')
  textlist.append(f'誤答数：{no_correct}')
  textlist.append('\n')
  textlist.append(f'正答率：{round(correct_answer_rate,5)}')
  textlist.append('\n')

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
  textlist.append('\n')

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

  #合計
  sum_bleu = 0

  for line in ss:
    index=line[0]
    pred=line[1]
    sum_bleu += bleu_score.sentence_bleu([tokenize_pycode(index)],tokenize_pycode(pred))

     #平均値
  bleu = sum_bleu / len(ss)

  textlist.append(f'BLEU：{round(bleu,5)}')
  textlist.append('\n')

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
  textlist.append('\n')

def arg(textlist):
  try:
    textlist.append(f"index = {sys.argv[2]}, pred = {sys.argv[3]}")
    textlist.append('\n')
    return int(sys.argv[2]),int(sys.argv[3])
  except:
    textlist.append("index = 2, pred = 1")
    textlist.append('\n')
    return 2, 1
    

def main():
  global index_id
  global pred_id

  textlist=[]

  index_id, pred_id = arg(textlist)

  ss = read_tsv(sys.argv[1])
  textlist.append(sys.argv[1])
  textlist.append('\n')

  Exact_Match(ss,textlist)
  BLEU(ss,textlist)
  ROUGE_L(ss,textlist)
  Levenstein(ss,textlist)

  text1=""
  for textlist_line in textlist:
    text1+=textlist_line
  
  gs.send_gs(text1)

main()