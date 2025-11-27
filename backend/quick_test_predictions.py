"""
Quick test for session predictions endpoint
"""
import sys
sys.path.insert(0, 'C:\\Users\\diont\\OneDrive\\Desktop\\Projects\\PP\\backend')

import os
os.environ["TESTING"] = "true"

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Register and login
reg_resp = client.post("/api/auth/register", json={
    "username": "quicktest",
    "password": "Test123!",
    "email": "quick@test.com"
})
print("Register:", reg_resp.status_code)

login_resp = client.post("/api/auth/login", json={
    "username": "quicktest",
    "password": "Test123!"
})
print("Login:", login_resp.status_code)

if login_resp.status_code == 200:
    token = login_resp.json()["tokens"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create session
    sess_resp = client.post("/api/sessions", headers=headers, json={"name": "Test Session"})
    print("Session created:", sess_resp.status_code, sess_resp.json().get("id") if sess_resp.status_code == 201 else sess_resp.json())
    
    if sess_resp.status_code == 201:
        session_id = sess_resp.json()["id"]
        
        # Create prediction
        pred_resp = client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=headers,
            json={"sequence": "ACDEFGHIKLMN"}
        )
        print(f"Prediction created: {pred_resp.status_code}")
        if pred_resp.status_code != 201:
            print(f"Error: {pred_resp.json()}")
        else:
            print(f"Prediction ID: {pred_resp.json().get('id')}")
        
        # List predictions
        list_resp = client.get(f"/api/sessions/{session_id}/predictions", headers=headers)
        print(f"List predictions: {list_resp.status_code}")
        if list_resp.status_code == 200:
            data = list_resp.json()
            print(f"Total predictions: {data['total']}")
            print(f"Predictions: {len(data['predictions'])}")
