import pandas as pd

df = pd.read_csv('data/processed/Vacancy.csv', index_col=0, parse_dates = [2, 3])

df = df.assign(movedTF = (df.moved > 100).astype(int))
park_df = df.drop(columns = 'reservation_time')
park_df['action'] = 'Park'
park_df.rename(columns = {'park_time':'time'}, inplace = True)
reserve_df = df.drop(columns = 'park_time')
reserve_df['action'] = 'Reservation'
reserve_df.rename(columns = {'reservation_time':'time'}, inplace = True)
df_split = pd.concat([park_df,reserve_df]).sort_values(by = 'time')

df_split[['prev_customer', 'next_customer']] = (df_split[['prev_customer', 'next_customer']] == 'Customer').values

# Define action as T/F on park/reserve
df_split['action'] = (df_split['action'] == 'Park').values

df_split.reset_index(drop = True, inplace = True)
df_split.to_csv('data/processed/VacancySplit.csv')