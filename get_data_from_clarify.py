import orchest
from pyclarify import ClarifyClient
import pandas as pd
import time

clarify_client = ClarifyClient("clarify-credentials.json")

data_list = []
number_of_items_to_fetch = 566
num_per_fetch = 50

for i in range(0, number_of_items_to_fetch, num_per_fetch):
    
    print(f'fetching item {i} to {i + num_per_fetch - 1}')
    
    data_not_fetched = True
    
    while data_not_fetched:

        response = clarify_client.select_items_data(
            limit=num_per_fetch,
            skip = i,
            not_before = "2018-01-01T00:00:00Z",
            before = "2021-01-02T00:00:00Z"
        )

        if response.result == None:
            wait_time_after_fail = 10
            print(f'failed, waiting for {wait_time_after_fail} seconds and then trying again')
            time.sleep(wait_time_after_fail)
        else:
            data_list.append(response.result.data)
            data_not_fetched = False

df_list = [data.to_pandas() for data in data_list]
data_df = pd.concat(df_list, axis=1, ignore_index=False)

print(f'successfully fetched {len(data_df.columns)} items')

response = clarify_client.select_items_metadata(
    ids=list(data_df.columns),
    limit=1000
)
signal_infos_dict = response.result.items

sensor_types_to_add = ['Fastkraft', 'Fjernvarme', 'Varme', 'Elkjel', 'Kjøling']

item_id_dict = {}
for item_id, signal_info in signal_infos_dict.items():
    building_name = signal_info.labels.get('building', [None])[0]
    sensor_type = signal_info.labels.get('type', [None])[0]

    # if item does not have building name or sensor type as label continue to next item
    if building_name is None or sensor_type is None:
        continue

    # if sensor type is not in list of interseting sensor types continue to next sensor
    if sensor_type not in sensor_types_to_add:
        continue

    # if building is not in item_id_dict add empty dict
    if building_name not in item_id_dict:
        item_id_dict[building_name] = {}

    # if sensor type not in dict for build add empty list
    if sensor_type not in item_id_dict[building_name]:
        item_id_dict[building_name][sensor_type] = []

    item_id_dict[building_name][sensor_type].append(item_id)

    building_names = list(item_id_dict.keys())

energy_consumption_hourly_dfs = {}
for building_name, sensor_type_item_id_dict in item_id_dict.items():
    d = {}
    for sensor_type, item_id_list in sensor_type_item_id_dict.items():
        d[sensor_type] = data_df[item_id_list].sum(axis=1)
    energy_consumption_hourly_dfs[building_name] = pd.DataFrame(d)
    
    # add total column to dataframe
    energy_consumption_hourly_dfs[building_name]['Totalt'] = energy_consumption_hourly_dfs[building_name].sum(axis=1)


energy_consumption_daily_dfs = {}
energy_consumption_weekly_dfs = {}
for building_name, df in energy_consumption_hourly_dfs.items():
    energy_consumption_daily_dfs[building_name] = df.resample('D').sum()
    energy_consumption_weekly_dfs[building_name] = df.resample('W-MON').sum()

energy_consumption_dfs = {
    'hourly': energy_consumption_hourly_dfs,
    'daily': energy_consumption_daily_dfs,
    'weekly': energy_consumption_weekly_dfs,
}

print('outputting energy cosumption data fetched from clarify...')
orchest.output(energy_consumption_dfs, name='energy_consumption')
print('success')