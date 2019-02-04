import requests
import time

from datetime import timedelta, date
from S3Wrapper import S3Wrapper

payload_key = {'key':'WBrnbCzAFL'}
servicesKey = "services/services.json"

s3 = S3Wrapper() 

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

if(s3.if_exists(servicesKey) == False):
    r = requests.get('https://rtl2.ods-live.co.uk//api/services', params=payload_key)
    s3.uploadJsonObject(r.json(), servicesKey)
    routes = r.json()

else:
    routes = s3.fetchJson(servicesKey) 

routes = [route['id'] for route in routes]

print(routes)

for single_date in daterange(date(2017,11,30), date(2019,2,1)):
    print(single_date.strftime("\n%Y-%m-%d"))

    for route in routes:

        # Try to do this 10 times. 
        for attempt in range(10):
            try:
                r = requests.get('https://rtl2.ods-live.co.uk/api/trackingHistory', params={'service':route, 'date':single_date.strftime("%Y-%m-%d"), **payload_key}, timeout=5)

                s3.uploadJsonObject(r.json(), f"trackingHistory/{single_date.strftime('%Y-%m-%d')}/service-{route}.json")
            except Exception as e: #on exception print an x and wait a bit longer each time (in seconds)
                print(e.__doc__)
                print(str(e))
                time.sleep(attempt**2)
            else: 
                # If there is no exception break
                break
        else:
            # If we've done all 10 attempts print the failure.
            print(f"\nERROR: trackingHistory/{single_date.strftime('%Y-%m-%d')}/service-{route} FAILED 10 times")

        for attempt in range(10):
            try:
                r = requests.get('https://rtl2.ods-live.co.uk/api/scheduledJourneys', params={'service':route, 'date':single_date.strftime("%Y-%m-%d"), **payload_key}, timeout=5)

                s3.uploadJsonObject(r.json(), f"scheduledJourneys/{single_date.strftime('%Y-%m-%d')}/service-{route}.json")
            except Exception as e: #on exception print an x and wait a bit longer each time (in seconds)
                print(e.__doc__)
                print(str(e))
                time.sleep(attempt**2)
            else: 
                break
        else:
            print(f"\nERROR: scheduledJourneys/{single_date.strftime('%Y-%m-%d')}/service-{route} FAILED 10 times")

        print(".", end="")


    # https://rtl2.ods-live.co.uk/api/scheduledJourneys?key=WBrnbCzAFL&service=1&date=2017-01-01&location=

    


