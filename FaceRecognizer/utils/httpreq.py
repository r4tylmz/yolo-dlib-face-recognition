import requests
from constants import constants
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_staff_credentials_by_id(id) -> str:
    response = requests.get(f'{constants.BASEURL}/Staff/{id}', verify=False).json()
    formatted_fullname = f"{response['name']}_{response['lastName']}_{response['id']}"
    return formatted_fullname

def send_staff_credentials(name, lastname, phone):
    data = {"name": name, "lastName": lastname, "phoneNumber": phone}
    response = requests.post(f"{constants.BASEURL}/Staff/", json=data, verify=False)
    index = response.json()["id"]
    print(f"[INFO] HTTP CODE: {response.status_code} | USER ID: {index} SUCCESSFULLY CREATED")
    return index

def send_staff_activity(room_id, staff_id, entry, exit):
    data = {"roomId": room_id, "staffId": staff_id,
            "entryTime": entry.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "exitTime": exit.strftime("%Y-%m-%dT%H:%M:%S.%f")}
    response = requests.post('{constants.BASEURL}/StaffActivity', json=data, verify=False)
    print(f"[INFO] HTTP CODE: {response.status_code} | USER ID: {staff_id} SUCCESSFULLY SENT TO API")