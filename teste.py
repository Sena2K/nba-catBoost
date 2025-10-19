import requests

url = "https://api.balldontlie.io/v1/player_injuries"
headers = {
    "Authorization": "acb7d5a9-cadd-4327-98d0-e0aca8d25e96"
}

response = requests.get(url, headers=headers)

print("Status:", response.status_code)
print("Response text:", response.text[:500])  # Mostra só os primeiros 500 caracteres

if response.headers.get("content-type", "").startswith("application/json"):
    data = response.json()
    print(data)
else:
    print("⚠️ Resposta não é JSON, verifique o token ou headers.")
