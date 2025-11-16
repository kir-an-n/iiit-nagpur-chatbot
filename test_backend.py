import requests

# Test with PDF data
response = requests.post(
    'http://127.0.0.1:5000/ask',
    json={'question': 'What is the maternity leave policy for faculty?'}
)

print("Status Code:", response.status_code)
print("\nResponse:")
print(response.json()['answer'])