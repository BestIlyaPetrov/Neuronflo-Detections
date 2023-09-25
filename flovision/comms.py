import requests
import time
import json
import hashlib
import cv2



auth_token = ""
# url = 'http://192.168.0.17:6971/'



def authenticate(identifier, url):

    #INITIAL AUTHENTICATION STAGE
    # identifier = "jetson01"

    # Convert the string to bytes
    string_bytes = identifier.encode('utf-8')

    # Generate the SHA256 hash key
    sha256_key = hashlib.sha256(string_bytes).hexdigest()

    id_data = {
        'identifier':sha256_key
    }

    # Obtain a token
    # response = requests.post(url+'api-token-auth/', data=credentials)
    response = requests.post(url+'device-token-auth', data=id_data)

    if response.status_code == 200:
        token = response.json()['token']
        csrf_token = response.json()['csrf_token']
        # print('Token:', token)
        # print('CSRF Token:', csrf_token)
        return (token, csrf_token)

    else:
        print('Failed to obtain token')
        print(f"Server responded with: {response}")
        exit()
    ## END INITIAL AUTHENTICATION



def sendImageToServer(image, image_data, IP_address):
    global auth_token
    url = f'http://{IP_address}/'
    if auth_token == "":
        auth_token, csrf_token = authenticate(identifier = "jetson01", url=url) # TODO: add the identifier to dotenv
        # print("New auth token is: ", auth_token)
        print("Identifier got confirmed, so I received new auth token. Proceeding to send the image")
    else: 
        print("Successfully authorized, proceeding to send the image.")
        # print("Auth token already is: ", auth_token)



    #ALL SUBSEQUENT DATA POSTS HAPPEN HERE
    # Use the token to authenticate subsequent requests
    headers = {
        'Authorization': f'Token {auth_token}',
        }


    # Get the current time in seconds since the epoch
    timestamp = int(time.time())

    # Convert the timestamp to a string in the dd-mm-yy_hh-mm-ss format
    timestamp_str = time.strftime('%d-%m-%y_%H-%M-%S', time.localtime(timestamp))

    # convert image data to the correct format
    success, encoded_image = cv2.imencode('.jpg', image)
    if success:
        image_bytes = bytearray(encoded_image)

        print()
        print("########### DETECTION MADE #############")
        print(image_data)
        print("########### END OF DETECTION #############")
        print()

    else:
        raise ValueError("Could not encode the frame as a JPEG image")

    response = requests.post(url+'api/entrance_update', files={'image': (timestamp_str+'.jpg', image_bytes)}, data=image_data, headers=headers)


    # Check response status code
    if response.status_code == 200:
        try: 
            msg = response.json()['message']
            print(msg)
        except Exception as e:
            print(e)
    else:
        print('Failed to upload image - HTTP response status: ',response.status_code )
        try: 
            
            msg = response.json()['message']
            print(msg)
        except Exception as e:
            print(e)
        
