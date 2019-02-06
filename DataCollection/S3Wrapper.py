import boto3
from botocore.errorfactory import ClientError
import json



class S3Wrapper():

    def __init__(self):
        self.client = boto3.client('s3')
        self.bucket = 'reading-bus-data'

    def uploadJsonObject(self, data, key):
        self.uploadString(json.dumps(data), key)

    def uploadString(self, body, key):
        self.client.put_object(Body=body.encode('utf-8'), Bucket=self.bucket, Key=key)

    def fetchJson(self, key):
        obj = self.client.get_object(Bucket=self.bucket, Key=key)

        return json.loads(obj['Body'].read().decode('utf-8'))

    def if_exists(self, key):
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def list_bucket(self, prefix, startFrom=''):
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, StartAfter=startFrom)

        if('Contents' in response == False):
            return []

        return [ item['Key'] for item in response['Contents']]

if __name__ == "__main__":

    wrapper = S3Wrapper()

    print(wrapper.list_bucket("trackingHistory", 'trackingHistory/2017-01-11/service-50a.json'))