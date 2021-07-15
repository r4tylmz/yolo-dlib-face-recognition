from collections import OrderedDict
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Person():
    def __init__(self, name, entry_time, exit_time):
        self.name = name
        self.entry_time = entry_time
        self.exit_time = exit_time


class PersonTracker():
    def __init__(self):
        self.persons = OrderedDict()
        self.persons_activities = []

    def register(self, name, entry_time):
        self.persons[name] = Person(name, entry_time, 0)

    def mark_person_disappeared(self, name, exit_time):
        if name in self.persons.keys():
            self.persons_activities.append(Person(name, self.persons[name].entry_time, exit_time))

    def print_persons_activity(self):
        for person in self.persons_activities:
            print(f'Staff Name:{person.name}, Entry Time={person.entry_time.strftime("%d %b %Y %H:%M:%S")}, '
                  f'Exit Time={person.exit_time.strftime("%d %b %Y %H:%M:%S")}')

    def write_file(self):
        with open("person_activities.txt", mode="w") as f:
            sorted_by_name = sorted(self.persons_activities, key=lambda x: x.name)
            for person in sorted_by_name:
                f.write(f'Staff Name:{person.name}, Entry Time={person.entry_time.strftime("%d %b %Y %H:%M:%S")}, '
                        f'Exit Time={person.exit_time.strftime("%d %b %Y %H:%M:%S")}\n')
        f.close()

    def send_server(self, name, roomId):
        for persons_activity in reversed(self.persons_activities):
            if persons_activity.name == name:
                index = self.persons_activities.index(persons_activity)
                data = {
                    "roomId": roomId,
                    "staffId": name.split('_')[2],
                    "entryTime": self.persons_activities[index].entry_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    "exitTime": self.persons_activities[index].exit_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
                }
                x = requests.post('https://localhost:5001/api/StaffActivity', json=data, verify=False)
                print(x.text)
                print(f"http: {x.status_code} id:{name.split('_')[2]} ==> Basarili sekilde sunucuya gonderildi")
                del self.persons[name]
                break
