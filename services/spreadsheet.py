import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

SPREADSHEET_ID = "YOUR SHEET ID"

SERVER_STATUS_RANGES = {
    1: ("서버현황!B2", "서버현황!C2", "서버현황!D2"),
    2: ("서버현황!B3", "서버현황!C3", "서버현황!D3"),
    3: ("서버현황!B4", "서버현황!C4", "서버현황!D4"),
    4: ("서버현황!B5", "서버현황!C5", "서버현황!D5"),
}

CLOUD_KEY = "GOOGLE CLOUD KEY PATH"

def get_sheets_service():
    creds = service_account.Credentials.from_service_account_file(
        CLOUD_KEY, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    service = build("sheets","v4", credentials=creds)
    return service

def update_server_status(server_number:int, name:str=None, status:bool=True, task:str=None):
    '''
    server_number = 사용중인 서버숫자
    name = 작업자 이름 (학습중일때 필수)
    status = 학습중(True), 학습완료(False)
    task = 어떤 작업을 수행하는가 (선택)
    '''
    if server_number not in SERVER_STATUS_RANGES:
        print('잘못된 서버 번호를 적어놓았습니다. 1,2,3,4 중에서 한개를 넣어주세요')
        return
    
    if status == True and name == None:
        print('학습을 시작할 땐 이름이 필수입니다. 입력해주세요')
        return
    
    name_range,status_range,task_range = SERVER_STATUS_RANGES[server_number]
    service = get_sheets_service()
    sheet = service.spreadsheets()
    
    status_value = "학습중"
    
    if status == False:
        name = "-"
        status_value = "학습 완료"
        task = "-"
        
    values = [
        {"range": name_range, 'values':[[name]]},
        {"range": status_range, 'values':[[status_value]]},
        {"range": task_range, 'values':[[task]]}
    ]
    
    body = {
        "valueInputOption": "RAW",
        "data": values
    }
    
    result = sheet.values().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body).execute()
    print(f"server :{server_number} 스프레드시트 서버 현황 업데이트 완료 ")

def append_training_log(sheet_name, data):
    '''
    sheet_name = 스프레드시트 이름 ex)이상진
    data = {"epoch":epoch, "loss":loss, "task":task}
            스프레드시트 형태와 열 이름에 맞도록 제작할것
    '''
    service = get_sheets_service()
    sheet = service.spreadsheets()
    
    values = [[data[key] for key in data]]
    
    body = {
        "values" : values
    }
    
    result = sheet.values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{sheet_name}!A:D",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()
    
    print(f"{sheet_name} 시트에 학습 로그 추가 완료")