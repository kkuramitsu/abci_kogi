import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials
def connect_gspread(jsonf,key):
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(jsonf, scope)
    gc = gspread.authorize(credentials)
    SPREADSHEET_KEY = key
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1
    return worksheet

def next_available_row(sheet1):
    str_list = list(filter(None, sheet1.col_values(1)))
    return str(len(str_list)+1)


def send_gs(textlist):
  key1 = "1awtKb2uLYXgo"
  key2 = "zNWo02uOuKYjPp"
  key3 = "ChnRM4z1zmAnDn1-k"
  jsonf = "cg_eval/majestic-stage-352023-d38022826eb3.json"
  spread_sheet_key = key1+key2+key3
  
  ws = connect_gspread(jsonf,spread_sheet_key)
  ws.update_cell(int(next_available_row(ws)),1,textlist)
