import pandas as pd
import datetime as dt
import numpy as np
import geopy.distance
from tqdm.auto import tqdm

tqdm.pandas()
import dask.dataframe as dd
from dask.multiprocessing import get
from geopy.point import Point

from dask.diagnostics import ProgressBar

ProgressBar().register()


class GtfsGraphConstructor:
    def __init__(self, input_path, output_path, date, nodes_from_existing=None):
        # Initialize empty graph
        if nodes_from_existing is not None:
            self.nodes = pd.read_csv(output_path + nodes_from_existing)
        self.nodes = pd.DataFrame()
        self.edges = []
        self.date = date
        self.out_path = output_path

        # Load necessary data
        self.calendar_df = pd.read_csv(input_path + 'calendar.txt')
        self.calendar_df.columns = ['service_id',
                                    'sunday',
                                    'monday',
                                    'tuesday',
                                    'wednesday',
                                    'thursday',
                                    'friday',
                                    'saturday',
                                    'start_date',
                                    'end_date']
        self.trips_df = pd.read_csv(input_path + 'trips.txt')
        self.trips_df.columns = ['route_id', 'service_id', 'trip_id',
                                 'trip_headsign', 'direction_is', 'shape_id']
        self.stop_times_df = pd.read_csv(input_path + 'stop_times.txt')
        self.stops_df = pd.read_csv(input_path + 'stops.txt')

    def construct_nodes(self, file_name='nodes.pkl', zone_id=None):
        day_trips_df = self._get_single_day_trips()

        # Get all trips departures by getting the minimal departure time for
        # each trip
        trips_start_times_df = self.stop_times_df.groupby('trip_id').agg(
            {'departure_time': 'min'})

        # Let's join the last two tabled to get the departure times of all
        # sunday trips
        sunday_departures_df = day_trips_df.merge(trips_start_times_df,
                                                  on='trip_id', suffixes=(
                '_departures', '_trips'))

        # Add stop code and zone id to stop times
        stop_times_with_stop_codes_df = self.stop_times_df.merge(
            self.stops_df[['stop_id', 'stop_code', 'zone_id']], on='stop_id')
        stop_times_with_stop_codes_df['departure_time'] = \
            stop_times_with_stop_codes_df[
                'departure_time'].apply(self._convert_gtfs_time_to_datetime)

        # We want to (right) join this table with stop_times in order to get the
        # sunday stop times with trip departure time.
        nodes_df = self.stop_times_df.merge(sunday_departures_df, how='right',
                                            on='trip_id', suffixes=(
            '_stop', '_trip_departure'))

        # clean up

        nodes_df = nodes_df.drop(
            ['stop_desc', 'stop_name', 'zone_id', 'parent_station',
             'location_type'], axis=1)

        # Add stops data to nodes
        nodes_df = nodes_df.merge(self.stops_df, on='stop_id',
                                  suffixes=('_node', '_stop'))

        # Handle strange GTFS time (for example we might have hour 25:00 to
        # indicate 1am the following day)
        nodes_df['arrival'] = nodes_df['arrival_time'].apply(
            self._convert_gtfs_time_to_datetime)
        nodes_df['departure'] = nodes_df['departure_time_stop'].apply(
            self._convert_gtfs_time_to_datetime)

        if zone_id is not None:
            # TODO: Handle zone(s) restriction
            pass

        self.nodes = nodes_df

        nodes_df.to_pickle(self.out_path + file_name)

    def _get_single_day_trips(self):
        # TODO: we currently only handle sunday dates, extend to support more
        self.calendar_df['start_date'] = self.calendar_df['start_date'].apply(
            lambda x: dt.datetime.strptime(str(x), '%Y%m%d'))
        self.calendar_df['end_date'] = self.calendar_df['end_date'].apply(
            lambda x: dt.datetime.strptime(str(x), '%Y%m%d'))

        # Filter so we only keep services that are active on Sunday.
        sunday_services_df = self.calendar_df[self.calendar_df['sunday'] == 1][
            ['service_id', 'start_date', 'end_date']]

        # Keep only services that start during/before the given date
        sunday_services_df = sunday_services_df[
            sunday_services_df['start_date'] <= self.date]

        # Keep only services that end during/after the given date
        sunday_services_df = sunday_services_df[
            sunday_services_df['end_date'] >= self.date]

        trips_calendar_df = sunday_services_df.merge(self.trips_df,
                                                     on='service_id',
                                                     suffixes=(
                                                         '_calendar', '_trips'))
        return trips_calendar_df.drop(
            ['start_date', 'end_date', 'trip_headsign'],
            axis=1)

    def _convert_gtfs_time_to_datetime(self, gtfs_time):
        h, m, s = [int(x) for x in gtfs_time.split(':')]
        if h < 24:
            # This is a 'normal' situation, we can simply create a datetime
            # object using the date we defined before
            return self.date + dt.timedelta(hours=h, minutes=m, seconds=s)
        # Otherwise we have a 'strange' time: it's after midnight
        new_date = self.date + dt.timedelta(days=1)
        return new_date + dt.timedelta(hours=h - 24, minutes=m, seconds=s)


DATA_PATH = '../input_data/GTFS-28-Oct-19/'
OUTPUT_PATH = '../output_data/'

if __name__ == '__main__':
    gtfs_constructor = GtfsGraphConstructor(DATA_PATH, OUTPUT_PATH,
                                            dt.datetime(2019, 11, 3), None)
    gtfs_constructor.construct_nodes()
