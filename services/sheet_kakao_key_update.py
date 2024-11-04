import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
import json

SPREADSHEET_ID = "{YOUR SHEET ID}"
RANGE_NAME = "SHEET_NAME!A2"

KAKAO_ACCESS_KEY = "ACCESS_KEY_PATH"
CLOUD_KEY = "GOOGLE_CLOUD_KEY_PATH"

with open(KAKAO_ACCESS_KEY, "r") as fp:
    kakao_tokens = json.load(fp)
    access_token = kakao_tokens["access_token"]

def update_access_token(token):
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