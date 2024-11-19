import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

'''
SPREADSHEET_ID : 업데이트할 구글 스프레드시트 ID
SERVER_STATUS_RANGES : 서버 상태를 업데이트할 셀 범위 (서버 번호별 셀 위치)
CLOUD_KEY : Google Cloud 서비스 계정 키가 저장된 JSON 파일 경로
'''

SPREADSHEET_ID = "YOUR SHEET ID"

SERVER_STATUS_RANGES = {
    1: ("서버현황!B2", "서버현황!C2", "서버현황!D2"),
    2: ("서버현황!B3", "서버현황!C3", "서버현황!D3"),
    3: ("서버현황!B4", "서버현황!C4", "서버현황!D4"),
    4: ("서버현황!B5", "서버현황!C5", "서버현황!D5"),
}

CLOUD_KEY = "GOOGLE CLOUD KEY PATH"

def get_sheets_service():
    '''
    Google Sheets API 서비스 불러오는 함수
    
    Returns:
        service: Google Sheets API 서비스 객체
    '''
    creds = service_account.Credentials.from_service_account_file(
        CLOUD_KEY, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    service = build("sheets","v4", credentials=creds)
    return service

def update_server_status(server_number:int, name:str=None, status:bool=True, task:str=None):
    '''
    지정된 서버의 상태를 구글 스프레드시트에 업데이트하는 함수
    
    Parameters:
        server_number (int): 업데이트할 서버 번호 (1, 2, 3, 4 중 하나)
        name (str): 작업자 이름 (학습 중일 때 필수)
        status (bool): 학습 상태 - 학습 중(True) 또는 학습 완료(False)
        task (str): 수행 중인 작업 내용 (선택 사항)
    
    Returns:
        None
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
    학습 로그를 구글 스프레드시트에 추가하는 함수
    
    Parameters:
        sheet_name (str): 스프레드시트 시트 이름 (예: 작업자 이름)
        data (dict): 학습 로그 데이터 (예: {"epoch": epoch, "loss": loss, "task": task})
        
    Returns:
        None
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