import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

SPREADSHEET_ID = "YOUR_SHEET_ID"
RANGE_NAME = "SHEET_NAME"

CLOUD_KEY = "COOGLE CLOUD KEY PATH"

KAKAO_ACCESS_KEY = "KAKAO ACCESS KEY PATH"

def get_access_token():
    # Google Sheets API 인증
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
    # 로컬 JSON 파일 업데이트
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