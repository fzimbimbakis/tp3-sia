import json
from Executables.adaptative import adaptative_func
from Executables.basic import basic_func
from Executables.batch import batch_func
from Executables.momentun import momentum_func

if __name__ == '__main__':
    with open("../config.json", "r") as jsonfile:
        jsonData = json.load(jsonfile)  # Reading the file
        print("Read successful")
        jsonfile.close()

    data = jsonData['Exercise1']
    adaptative_data = data['adaptative']
    basic_data = data['basic']
    batch_data = data['batch']
    momentum_data = data['momentum']

    if adaptative_data['run']:
        adaptative_func(adaptative_data['learning_rate'], adaptative_data['epochs'])
    if basic_data['run']:
        basic_func(basic_data['learning_rate'], basic_data['epochs'])
    if batch_data['run']:
        batch_func(batch_data['learning_rate'], batch_data['epochs'])
    if momentum_data['run']:
        momentum_func(momentum_data['learning_rate'], momentum_data['epochs'])
