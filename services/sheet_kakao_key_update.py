import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
import json

'''
SPREADSHEET_ID : 업데이트할 구글 스프레드시트 ID
RANGE_NAME : 구글 스프레드시트에서 업데이트할 셀 범위
KAKAO_ACCESS_KEY : 카카오톡 액세스 키가 저장된 JSON 파일의 경로
CLOUD_KEY : Google Cloud 서비스 계정 키가 저장된 JSON 파일의 경로
'''

SPREADSHEET_ID = "{YOUR SHEET ID}"
RANGE_NAME = "SHEET_NAME!A2"

KAKAO_ACCESS_KEY = "ACCESS_KEY_PATH"
CLOUD_KEY = "GOOGLE_CLOUD_KEY_PATH"

with open(KAKAO_ACCESS_KEY, "r") as fp:
    kakao_tokens = json.load(fp)
    access_token = kakao_tokens["access_token"]

def update_access_token(token):
    '''
    구글 스프레드시트에 카카오톡 액세스 토큰을 업데이트하는 함수
    
    Parameters:
        token (str): 스프레드시트에 업데이트할 카카오톡 액세스 토큰
    
    Returns:
        None
    '''
    creds = service_account.Credentials.from_service_account_file(CLOUD_KEY, 
                                                                  scopes=["https://www.googleapis.com/auth/spreadsheets"])
    service = build('sheets','v4', credentials=creds)
    
    body = {
        'values':[[token]]
    }
    
    result = service.spreadsheets().values().update(
        spreadsheetId = SPREADSHEET_ID,
        range=RANGE_NAME,
        valueInputOption='RAW',
        body=body
    ).execute()
    
    print(f"Updated {result.get('updatedCells')} cells with access_token.")

update_access_token(access_token)