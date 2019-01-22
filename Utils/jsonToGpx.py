import json
import datetime

import gpxpy
import gpxpy.gpx

gpx = gpxpy.gpx.GPX()

# Create first track in our GPX:
gpx_track = gpxpy.gpx.GPXTrack()
gpx.tracks.append(gpx_track)

# Create first segment in our GPX track:
gpx_segment = gpxpy.gpx.GPXTrackSegment()
gpx_track.segments.append(gpx_segment)

with open('Reading Data/vehiclePositionHistory.json') as f:
    data = json.load(f)

    for point in data:
        # print(point)

        # Create points:
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(point['latitude'], point['longitude'], time=datetime.datetime.strptime(point['observed'], '%Y-%m-%d %H:%M:%S')))
    

with open('Temp Data/gpxFile.gpx', 'w') as f2:
    f2.write(gpx.to_xml())
    f2.close()

