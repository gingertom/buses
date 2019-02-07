import requests
import time
import logging
import sys

from datetime import timedelta, date
from S3Wrapper import S3Wrapper

import http.client as http_client

payload_key = {'key':'WBrnbCzAFL'}
servicesKey = "services/services.json"

s3 = S3Wrapper() 

def enableLogging():
    http_client.HTTPConnection.debuglevel = 1

    # You must initialize logging, otherwise you'll not see debug output.
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

# enableLogging()

for single_date in daterange(date(2017,6,1), date(2017,6,3)):
    print(single_date.strftime("\n%Y-%m-%d"))

    busses = s3.fetchJson(f"bussesAndRoutes/{single_date.strftime('%Y-%m-%d')}.json")

    for bus in busses:

        # Try to do this 10 times. 
        for attempt in range(10):
            try:
                r = requests.get('https://rtl2.ods-live.co.uk/api/vehiclePositionHistory', params={'vehicle':bus, 'date':single_date.strftime("%Y-%m-%d"), **payload_key}, timeout=100)

                location_data = r.json()

                # We've tried to download too much data. Lets try splitting it in half
                if('status' in location_data and location_data['status'] == "false" and 'message' in location_data and location_data['message'].startswith("Too much data")):
                        r1 = requests.get('https://rtl2.ods-live.co.uk/api/vehiclePositionHistory', params={'vehicle':bus, 'date':single_date.strftime("%Y-%m-%d"), 'from':'00:00:00', 'to':'11:59:59', **payload_key}, timeout=100)
                        r2 = requests.get('https://rtl2.ods-live.co.uk/api/vehiclePositionHistory', params={'vehicle':bus, 'date':single_date.strftime("%Y-%m-%d"), 'from':'12:00:00', 'to':'23:59:59', **payload_key}, timeout=100)

                        s3.uploadJsonObject(r1.json(), f"vehiclePositionHistory/{single_date.strftime('%Y-%m-%d')}/vehicle-AM-{bus}.json")
                        s3.uploadJsonObject(r2.json(), f"vehiclePositionHistory/{single_date.strftime('%Y-%m-%d')}/vehicle-PM-{bus}.json")
                    

                s3.uploadJsonObject(r.json(), f"vehiclePositionHistory/{single_date.strftime('%Y-%m-%d')}/vehicle-{bus}.json")
            except Exception as e: #on exception print an x and wait a bit longer each time (in seconds)
                print(e.__doc__)
                print(str(e))
                time.sleep(attempt**2)
            else: 
                # If there is no exception break
                break
        else:
            # If we've done all 10 attempts print the failure.
            print(f"\nERROR: vehiclePositionHistory/{single_date.strftime('%Y-%m-%d')}/vehicle-{bus} FAILED 10 times")

        print(".", end="")
        sys.stdout.flush()


    # https://rtl2.ods-live.co.uk/api/scheduledJourneys?key=WBrnbCzAFL&service=1&date=2017-01-01&location=

    


