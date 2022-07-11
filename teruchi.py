import random
import re
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
from io import BytesIO

def get_token(text):
    text = re.sub(r'([^A-Za-z0-9_])', r' \1 ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '`')
    text = text.replace('\'', '`')
    tokens = [t for t in text.split(' ') if t]
    return tokens

def py_token(text):
    lst = []
    tokens = tokenize(BytesIO(text.encode('utf-8')).readline)
    for token in tokens:
      lst.append(token.string)
    return lst[1:-2] #文頭に文字コード、文末に改行が入ってしまうため

def mask(s:str, ratio=0.4):
  token = py_token(s)
  token_idx = [idx for idx in range(len(token))] 
  buffer = []
  num = 0

  for index in token_idx:
    if random.random() < ratio:       # Mask rate
      if random.random() < 0.8:     # 40% of the time, replace with [MASK]
        masked_token = f'<extra_id_{num}>'
        if index == 0:
          buffer.append(masked_token)
          num += 1
        else:
          if masked_token != buffer[-1]: #ひとつ前がMASKではなかったらMASKを追加
            buffer.append(masked_token)
            num += 1
          else:
            pass
      elif random.random() > 0.8 and random.random() < 0.9: # 10% of the time, replace with random word
        masked_token = token[random.randint(0, len(token_idx) - 1)]
        buffer.append(masked_token)
        num += 1
      else:      # 10% of the time, keep original
        buffer.append(token[index])
    else:
     buffer.append(token[index])
  buffer = "".join(buffer)
  return buffer



def suff_3(shuffle_lst):
  tmp = random.sample(shuffle_lst, 3)#シャッフル
  if tmp == shuffle_lst:  #高確率でシャッフルに失敗するので
    tmp = random.sample(shuffle_lst, 3)
  return tmp

def suff_2(shuffle_lst):
  tmp = [shuffle_lst[1], shuffle_lst[0]]
  return tmp

def DAE(s:str, parameter=0.5):
  stop = stopp = 100000000 # continueされない初期値
  buffer = []
  token = get_token(s)
  token_idx = [idx for idx in range(len(token))]
  
  for index in token_idx:
    if random.random() < parameter: #破損処理されるトークン率
      pass
      if random.random() < 1/4:     # 25%で除去
        pass
      elif random.random() > 1/4 and random.random() < 2/4: # 25%でシャッフル
        if index == stop+1 or index == stop+2 or index == stopp+1:
          continue
        else:
          if index !=  len(token_idx) and index != len(token_idx)-1 and index != len(token_idx)-2:
            shuffle_lst = [index, index+1, index+2]
            shuffled_lst = suff_3(shuffle_lst)
            buffer.append(token[shuffled_lst[0]])
            buffer.append(token[shuffled_lst[1]])
            buffer.append(token[shuffled_lst[2]])
            stop = index
            # print("現在のindex：", index, "shuffle_lst3：", shuffle_lst, "shuffle3結果：", shuffled_lst)
          elif index == len(token_idx)-2:
            shuffle_lst = [index, index+1]
            shuffled_lst = suff_2(shuffle_lst)
            buffer.append(token[shuffled_lst[0]])
            buffer.append(token[shuffled_lst[1]])
            stopp = index
            # print("現在のindex：", index, "shuffle_lst2：", shuffle_lst, "shuffle2結果：", shuffled_lst)
          else:
            pass
      elif random.random() > 2/4 and random.random() < 3/4: # 25%でマスク(ok)
        masked_token = "[MASK]"
        buffer.append(masked_token)
      else:                                                               # 25%でニュータイプ
        if random.random() < 1/2:                             # そのうち50%で繰り返し
          select_token = token[index]
          buffer.append(select_token)
          buffer.append(select_token)
        else:                                                            # 50%でランダム挿入
          select_index = random.choice(token_idx)
          masked_token = token[index]
          select_token = token[select_index]
          buffer.append(masked_token)
          buffer.append(select_token)
    else:
      buffer.append(token[index])
  buffer = "".join(buffer)
  return buffer


if __name__ == '__main__':
    print(mask('buffer.append(a[::1])'))
    print(DAE('buffer.append(a[::1])', parameter=0.1))