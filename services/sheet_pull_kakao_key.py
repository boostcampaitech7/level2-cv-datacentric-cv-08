import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

'''
SPREADSHEET_ID : 업데이트할 구글 스프레드시트 ID
RANGE_NAME : 구글 스프레드시트에서 업데이트할 셀 범위 (예: 'Sheet1!A2')
KAKAO_ACCESS_KEY : 카카오톡 액세스 키가 저장된 JSON 파일 경로
CLOUD_KEY : Google Cloud 서비스 계정 키가 저장된 JSON 파일 경로
'''

SPREADSHEET_ID = "YOUR_SHEET_ID"
RANGE_NAME = "SHEET_NAME"

CLOUD_KEY = "COOGLE CLOUD KEY PATH"
KAKAO_ACCESS_KEY = "KAKAO ACCESS KEY PATH"

def get_access_token():
    '''
    구글 스프레드시트에서 카카오톡 액세스 토큰을 가져오는 함수
    Google Sheets API를 통해 지정된 셀 범위에서 저장된 액세스 토큰을 불러온다.
    
    Returns:
        str or None: 스프레드시트에 저장된 access_token 값 or
                     데이터가 없으면 None을 반환
    '''

    creds = service_account.Credentials.from_service_account_file(
        CLOUD_KEY, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    service = build('sheets', 'v4', credentials=creds)

    # 스프레드시트에서 access_token 가져오기
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, 
                                range=RANGE_NAME).execute()
    values = result.get('values', [])

    if not values:
        print("No data found.")
        return None
    return values[0][0]

def update_local_token_file(access_token):
    '''
    구글 스프레드시트에서 카카오톡 액세스 토큰을 가져와
    Local 액세스 토큰 json을 업데이트하는 함수
    
    Parameters:
        token (str): 스프레드시트에서 가져온 값으로 업데이트할
        json형태의 카카오톡 액세스 토큰 경로
        
    Returns:
        None
    '''
    if access_token:
        with open(KAKAO_ACCESS_KEY, 'r') as file:
            token_data = json.load(file)

        # access_token 갱신
        token_data['access_token'] = access_token

        # JSON 파일에 저장
        with open(KAKAO_ACCESS_KEY, 'w') as file:
            json.dump(token_data, file)
        print("Local access_token updated successfully.")
    else:
        print("Failed to update local token: access_token is None.")
        
access_token = get_access_token()

if access_token:
    update_local_token_file(access_token)