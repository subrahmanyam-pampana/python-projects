import sys
import clipboard
import json

SAVED_DATA = "clipboard.json"

def save_data(filepath,data):
    with open(filepath,"w") as f:
        json.dump(data,f)
def load_data(filepath):
    try:
        with open(filepath,"r") as f:
            data = json.load(f)
            return data
    except:
        return {}    

if len(sys.argv) == 2:
    cmd = sys.argv[1]
    data = load_data(SAVED_DATA)
    if cmd == "save":
        key = input("enter a key: ")
        data[key] = clipboard.paste()
        save_data(SAVED_DATA,data)
        print("successfuly saved")
    elif cmd== "load":
        key = input("enter key: ")
        if key in data:
            clipboard.copy(data[key])
            print("value: "+data[key])
            print("copeied to clipboard!")
        else: 
            print("key not found")      
    elif cmd == "list":
        print(data)  
    elif cmd == "delete":
        key = input("enter a key: ")
        del data[key]
        save_data(SAVED_DATA,data)
        print("deleted the key!")       
    else:
        print("unknown Command")    
else:print("pass only one command at a time")        