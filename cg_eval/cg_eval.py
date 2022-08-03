import csv
import sys
import black
import Levenshtein
import nltk
#nltk.download('punkt')
from nltk import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu
from io import BytesIO
from tokenize import tokenize, open
import re
from sumeval.metrics.rouge import RougeCalculator
import gs

import warnings
warnings.filterwarnings('ignore')

def read_tsv(filename,textlist):
    ss = []
    row_count=0
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")

        #IndexError専用
        index_error_list=[]

        for row in reader:
            row_count+=1
            try:
                ss.append((row[index_id], row[pred_id]))
        
            except IndexError:
                index_error_list.append(row_count)

    if len(index_error_list)==0:
        pass
    else:
        textlist.append(f'Indexerrorが発生した件数：{len(index_error_list)}')
        textlist.append(f'Indexerrorが発生した行数：{index_error_list}')

    return ss


def Exact_Match(ss,textlist):
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
    textlist.append(f'正答率：{round(correct_answer_rate,6)}')


def Levenstein(ss,textlist):
    #合計
    sum_Levenstein=0
    
    for line in ss:
        index=line[0]
        pred=line[1]
        sum_Levenstein += Levenshtein.ratio(index,pred)

    #平均値
    leven=sum_Levenstein/len(ss)
  
    textlist.append(f'Leven：{round(leven,6)}')


def CORPUS_BLEU(ss,textlist):
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
    textlist.append(f'corpus_BLEU：{round(bleu,6)}')


def CONALA_BLEU(ss,textlist):
    smoother = SmoothingFunction()

    sum_b1 = 0
    sum_b2 = 0
    sum_b3 = 0
    sum_b4 = 0
    sum_b5 = 0

    pattern = re.compile(r'[\(, .\+\-\)]')

    def tokenize_for_bleu_eval(code):
        code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
        code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = code.replace('"', '`')
        code = code.replace('\'', '`')
        tokens = [t for t in code.split(' ') if t]
        return tokens

    for line in ss:
        py=line[0].strip()
        pred=line[1].strip()

        py = [tokenize_for_bleu_eval(py)]
        pred = tokenize_for_bleu_eval(pred)

        sum_b1 += bleu_score.sentence_bleu(py, pred, smoothing_function=smoother.method1)
        sum_b2 += bleu_score.sentence_bleu(py, pred, smoothing_function=smoother.method2)
        sum_b3 += bleu_score.sentence_bleu(py, pred, smoothing_function=smoother.method3)
        sum_b4 += bleu_score.sentence_bleu(py, pred, smoothing_function=smoother.method4)
        sum_b5 += bleu_score.sentence_bleu(py, pred, smoothing_function=smoother.method5)

    bleu1 = sum_b1 / len(ss)
    bleu2 = sum_b2 / len(ss)
    bleu3 = sum_b3 / len(ss)
    bleu4 = sum_b4 / len(ss)
    bleu5 = sum_b5 / len(ss)

    textlist.append(f'-conala_bleu_smooth1 : {round(bleu1,6)}')
    textlist.append(f'-conala_bleu_smooth2 : {round(bleu2,6)}')
    textlist.append(f'-conala_bleu_smooth3 : {round(bleu3,6)}')
    textlist.append(f'-conala_bleu_smooth4 : {round(bleu4,6)}')
    textlist.append(f'-conala_bleu_smooth5 : {round(bleu5,6)}')
 

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
    
    textlist.append(f'ROUGE-L：{round(ROUGE_score,6)}')


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
    
    ss = read_tsv(sys.argv[1],textlist)

    try:
        textlist.append(sys.argv[2])
    except:
        textlist.append(sys.argv[1])

    Exact_Match(ss,textlist)
    CORPUS_BLEU(ss,textlist)
    CONALA_BLEU(ss,textlist)
    ROUGE_L(ss,textlist)
    Levenstein(ss,textlist)

    text1=''
    for textlist_line in textlist:
        text1+=textlist_line+'\n'
    
    gs.send_gs(text1)
  
main()