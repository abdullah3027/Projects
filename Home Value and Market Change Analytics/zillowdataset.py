import pandas as pd
zillow_dataset = pd.read_csv('zillow.csv')
region_names = zillow_dataset.iloc[1:, 2]
states_list = set([x for x in zillow_dataset.loc[1:,'StateName']])
# Assuming 31/01/2000 as the starting point for comparison

def region_price_increase(zillow_dataset, region_names):
    # Price increase check for 4 years
    prices_2000 = zillow_dataset.loc[1:,'31/01/2000']
    prices_2004 = zillow_dataset.loc[1:,'31/01/2004']
    price_ratio1 = prices_2004/prices_2000
    print('Price increase of cities in a span of 4 years\n')
    print('------------------------------------------------------------------------------------------\n')
    count=0
    for index, price in enumerate(price_ratio1):
        if price >= 2:
            count+=1
            print(f'City name : {region_names[index+1]} \t\t\t\t Price increase ratio: {round(price,2)}')
    if count == 0:
        print('No city had its market value doubled in 4 year span')

    # Price increase check for 8 years
    prices_2008 = zillow_dataset.loc[1:,'31/01/2008']
    price_ratio2 = prices_2008/prices_2000
    print('\n\n\n\n')
    print('Price increase of cities in a span of 8 years\n')
    print('------------------------------------------------------------------------------------------\n')
    count = 0
    for index,price in enumerate(price_ratio2):
        if price >= 2:
            count += 1
            print(f'City name : {region_names[index+1]} \t\t\t\t Price increase ratio: {round(price,2)}')
    if count == 0:
        print('No city had its market value doubled in 8 year span')

    # Price increase check for 10 years
    prices_2010 = zillow_dataset.loc[1:,'31/01/2010']
    price_ratio3 = prices_2010/prices_2000
    print('\n\n\n\n')
    print('Price increase of cities in a span of 10 years\n')
    print('------------------------------------------------------------------------------------------\n')
    count = 0
    for index,price in enumerate(price_ratio3):
        if price >= 2:
            count += 1
            print(f'City name : {region_names[index+1]} \t\t\t\t Price increase ratio: {round(price,2)}')
    if count == 0:
        print('No city had its market value doubled in 10 year span')

def country_state_comparison(zillow_dataset, states_list):
    # Price averages for every state
    state_avg_2000 = {}

    for state in states_list:
        state_set = zillow_dataset.loc[zillow_dataset['StateName'] == state]
        prices_state = state_set.loc[:,'31/01/2000']
        state_avg_2000[state] = prices_state.mean()

    state_avg_2005 = {}
    for state in states_list:
        state_set = zillow_dataset.loc[zillow_dataset['StateName'] == state]
        prices_state = state_set.loc[:,'31/01/2005']
        state_avg_2005[state] = prices_state.mean()

    state_avg_2010 = {}
    for state in states_list:
        state_set = zillow_dataset.loc[zillow_dataset['StateName'] == state]
        prices_state = state_set.loc[:,'31/01/2010']
        state_avg_2010[state] = prices_state.mean()

    country_avg_2000 = zillow_dataset.loc[0,'31/01/2000']
    country_avg_2005 = zillow_dataset.loc[0,'31/01/2005']
    country_avg_2010 = zillow_dataset.loc[0,'31/01/2010']

    print('City performance comparison with its state and country for 5 year span\n')
    print('--------------------------------------------------------------------------\n')

    for index,region in enumerate(region_names):
        st = region.split(', ')
        state_of_region = st[1]
        state_price_ratio_2005 = state_avg_2005[state_of_region]/state_avg_2000[state_of_region]
        country_price_ratio_2005 = country_avg_2005/country_avg_2000
        city_price_ratio_2005 = zillow_dataset.loc[index,'31/01/2005']/zillow_dataset.loc[index,'31/01/2000']
        print(f'City Name : {region} \t Price Increase : {round(city_price_ratio_2005,2)} \t State Price Increase : {round(state_price_ratio_2005,2)} \t Country Price Increase : {round(country_price_ratio_2005,2)}')

    print('City performance comparison with its state and country for 10 year span\n')
    print('--------------------------------------------------------------------------\n')

    for index,region in enumerate(region_names):
        st = region.split(', ')
        state_of_region = st[1]
        state_price_ratio_2010 = state_avg_2010[state_of_region]/state_avg_2000[state_of_region]
        country_price_ratio_2010 = country_avg_2010/country_avg_2000
        city_price_ratio_2010 = zillow_dataset.loc[index,'31/01/2010']/zillow_dataset.loc[index,'31/01/2000']
        print(f'City Name : {region} \t Price Increase : {round(city_price_ratio_2010,2)} \t State Price Increase : {round(state_price_ratio_2010,2)} \t Country Price Increase : {round(country_price_ratio_2010,2)}')

def state_market(zillow_dataset, states_list):
    print('State Market\n')
    print('---------------------------------------------------------------\n')
    user = input('Enter year value to compare price increase ratio (20XX) ')
    print('\n')
    date = '31/01/'+user

    max = 0
    min = 10
    for state in states_list:
        state_set = zillow_dataset.loc[zillow_dataset['StateName'] == state]
        prices_state_2000 = state_set.loc[:, '31/01/2000']
        prices_state_avg_2000 = prices_state_2000.mean()
        prices_state_input = state_set.loc[:, date]
        prices_state_avg_input = prices_state_input.mean()
        ratio = prices_state_avg_input/prices_state_avg_2000
        if ratio > max:
            max = ratio
            state_name_max = state
        if ratio < min:
            min = ratio
            state_name_min = state
        print(f'State name : {state} \t State market price increase : {round(ratio,2)}')

    print('\n')
    print(f'Best Market for appreciation -> State name : {state_name_max} \t Price Increase : {max}')
    print(f'Best Market for depreciation -> State name : {state_name_min} \t Price Increase : {min}')

print('Market Evaluation Panel\n')
print('------------------------------\n')
print('1. Region Price Increase in the years 4,8 and 10 \n2. Country & State comparsion with region \n3. State Market for given N year \n')
user = int(input('Enter your requirment : '))
print('\n')
if user == 1:
    region_price_increase(zillow_dataset, region_names)
elif user == 2:
    country_state_comparison(zillow_dataset,states_list)
elif user == 3:
    state_market(zillow_dataset,states_list)





