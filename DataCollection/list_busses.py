from datetime import timedelta, date
from S3Wrapper import S3Wrapper
import numpy as np

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

s3 = S3Wrapper()

for single_date in daterange(date(2017,1,1), date(2017,1,4)):
    print(single_date.strftime("\n%Y-%m-%d"))

    files = s3.list_bucket(f"trackingHistory/{single_date.strftime('%Y-%m-%d')}/")

    vehicles = {}

    vehicles_to_save = {}

    for file in files:

        tracking_data = s3.fetchJson(file)

        # If there is no data for this day-route combination.
        if('code' in tracking_data and tracking_data['code'] == 100):
            continue

        # Loop through every stop in the tracking file and record the vehicle-route combination and start and stop times. 
        # SOme busses do multiple routes within a day so we need to do this for every file before we can be sure we are done. 
        for stop in tracking_data:
            if(stop['VehicleCode']== ""):
                continue

            vehicle_code = stop['VehicleCode']

            if(vehicle_code in vehicles):
                vehicles[vehicle_code]['lines'].append(stop['LineRef'])
                vehicles[vehicle_code]['departureTimes'].append(stop['DepartureTime'])
                vehicles[vehicle_code]['arrivalTimes'].append(stop['ArrivalTime'])
            else:
                vehicles[vehicle_code] = {'lines':[stop['LineRef']], 'departureTimes':[stop['DepartureTime']], 'arrivalTimes':[stop['ArrivalTime']]}

    
    # Now that we have all the data lets split this bus into sections where it was on the same route
    for bus_number, bus_dict in vehicles.items():

        # Convert everything to numpy as it's more powerful for what we need. 
        bus_dict['lines'] = np.array(bus_dict['lines'], dtype='str')
        bus_dict['departureTimes'] = np.array(bus_dict['departureTimes'], dtype='datetime64')
        bus_dict['arrivalTimes'] = np.array(bus_dict['arrivalTimes'], dtype='datetime64')

        # If this bus does more than one line this day
        if(np.all(bus_dict['lines'] == bus_dict['lines'][0]) == True):

            vehicles_to_save[bus_number] = [{'line':bus_dict['lines'][0], 'start':str(bus_dict['arrivalTimes'][0]), 'end':str(bus_dict['departureTimes'][-1])}]

        else:

            # Sort the arrays by the departure times. 
            correct_order = np.argsort(bus_dict['departureTimes'])
            bus_dict['lines'] = bus_dict['lines'][correct_order]
            bus_dict['departureTimes'] = bus_dict['departureTimes'][correct_order]
            bus_dict['arrivalTimes'] = bus_dict['arrivalTimes'][correct_order]

            # Make a new array like lines but shifted backwards by 1. We repeat the first element.
            lines_shifted = np.concatenate(([bus_dict['lines'][0]], bus_dict['lines'][:-1]))

            # We find all the indices where the true lines and offset lines are different. 
            # This approach won't catch the case where the last stop is different from the 
            # penultimate but that seems unlikely enough that I'm going to ignore it. 
            changes = np.where(bus_dict['lines'] != lines_shifted)[0]

            # We now have the start and end of each run.
            starts = np.concatenate(([0], changes))
            ends = np.concatenate((changes-1, [len(bus_dict['lines'])-1]))

            vehicles_to_save[bus_number] = []

            # print({bus_number})
            for run in zip(starts, ends):

                vehicles_to_save[bus_number].append({'line':bus_dict['lines'][run[0]], 'start':str(bus_dict['arrivalTimes'][run[0]]), 'end':str(bus_dict['departureTimes'][run[1]])})


    s3.uploadJsonObject(vehicles_to_save, f"bussesAndRoutes/{single_date.strftime('%Y-%m-%d')}.json")

