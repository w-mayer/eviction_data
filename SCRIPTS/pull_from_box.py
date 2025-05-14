from boxsdk import OAuth2, Client

# This is the piece that needs configuation.
# Acces_token comes from the authorization process--you can either use the developer token mentioned above or i can write a script
# to open the web browser to get the access token
auth = OAuth2(
    client_id='your_client_id',
    client_secret='your_client_secret',
    access_token='your_access_token'
)
client = Client(auth)

file_id = 'file_id_for_data.txt'

file = client.file(file_id).get()
with open("../DATA/cases_residential_only.txt", "wb") as f:
    file.download_to(f)
    print(f"File '{file.name}' downloaded successfully")