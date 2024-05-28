
# 2 page
# print the number of the index
print(list.index("me"))
print(list[2:3]) - [start:end]
list.append("me")

print(len(names))
print(f"length of the list: {len(list)}")

item.append(float(input(f"How much {stuff[-1]}?")))

for x in range(1,7):
    print("-"*40)

for i in range(len(numbers)):
    numbers[i] += 1
    print(numbers[i])

 actors = {}
actors = dict()
actors = {"Tom Hanks": "Forrest Gump", 
            "Leonardo DiCaprio": "Titanic"}
# add a name by the following
actors["name"] = 'Denzel Washington'

my_info = {"name": "X",
            "age": 20,
            }

print(f"I am {my_info["name"]}")

print(list(my_info["wake-up"].keys())[0])

for i, topping in enumerate(toppings, start=1):
        print(f"{i}. {topping}")

city = input("What city should be added to the list? ").title()

donors_df_renamed = donors_df.rename(columns={"donorName": "Donor",
                                                "employerName": "Employer",
                                                "zipcode": "Zip Code",
                                                "usd": "Donation Amount"
                                                })

list(range(10))

print("Population Mean: ", california_data["Population"].mean())

donors_df['Match Amount'] = donors_df['Amount'].apply(lambda x: x * 0.1 if x < 500 else x * 0.2)

joined_data_rows = pd.concat([apple_data, goog_data, meta_data], axis="rows", join="inner")

joined_data_rows = pd.concat([apple_data, goog_data, meta_data], axis="rows",
                            join="inner", keys=['Apple','Google', 'Meta'] )
products_data = pd.read_csv(products_data_path, index_col='CustomerID')

raw_data_items = {
    "customer_id": [403, 112, 543, 999, 654],
    "item": ["soda", "chips", "TV", "Laptop", "Cooler"],
    "cost": [3.00, 4.50, 600, 900, 150]
}
items_df = pd.DataFrame(raw_data_items, columns=[
                        "customer_id", "item", "cost"])
df = df.dropna(subset=['name', 'email'])

df = pd.DataFrame(df.loc[df["Year"]==2019,:])

df = df.reset_index(drop=True)

df.dtypes
grouped_ufo_duration_shape = converted_ufo_df.groupby("shape")[["duration (seconds)"]].mean()
grouped_ufo_duration_shape = converted_ufo_df.groupby("shape").agg({"duration (seconds)":'mean'})
grouped_dept_delay = flights_df.groupby("UniqueCarrier")[["ArrDelay", "DepDelay"]].agg('mean')

ufo_shape_avg_sum = converted_ufo_df.groupby("shape")[["duration (seconds)"]].agg(['mean','sum'])

df.columns = ['_'.join(column) for column in df.columns]

def custom_avg(x):
    return x.mean()

avg = df.groupby("shape").apply(lambda x: pd.Series({"Avg_Duration(seconds)": custom_avg(x["duration (seconds)"])}))

avg_sum_books = pd.pivot_table(book_sales_df, 
                                values='total_sales',
                                columns='book_name',
                                aggfunc=('sum', 'mean'))

.rename(columns={"shape": "UFO Sightings"})
.sort_values(by=["UFO Sightings"], ascending=False)

tables = pd.read_html(url)

print(json.dumps(data, indent=4))

titles.append(response[0]['show']['name'])

writers_list = writers.to_list()

by_country = happiness.drop(columns=['year']).groupby('country').mean()

wealth_happiness = by_country.plot.scatter(x="wealth",y="happiness",title="Relationship between Happiness and Wealth", figsize=(8,4),
                                            xlabel="Wealth",ylabel="Happiness",color="lightcoral",
                                            xlim=(6.5,12),ylim=(3,8),xticks=(),yticks=())

bottom = by_country.sort_values(by=['happiness']).head(15)
bottom_chart = bottom.plot.bar(y='happiness', rot=60, width=0.8, figsize=(10,3.5),ylim=(0,10),
                            title='Countries with Lowest Average Happiness, 2005-2018',
                            yticks=[0,5,10], xlabel='',color='lightsteelblue')

x_axis = np.arange(1,13,1)
x_axis = np.arange(len(users))

%matplotlib widget
plt.plot(x_axis, sin, linewidth=0, marker="o", color="blue", label="Sine")
plt.plot(x_axis, cos, linewidth=0, marker="^", color="red", label="Cosine")
plt.show()

# semi-transparent horizontal line at y = 0
plt.hlines(0, 0, 10, alpha=0.25)

plt.grid()
plt.legend(loc="lower right")
plt.legend(handles=[fahrenheit, celsius], loc="best")

fahrenheit, = plt.plot(x_axis, points_F, marker="+",color="blue", linewidth=1, label="Fahrenheit")
celsius, = plt.plot(x_axis, points_C, marker="s", color="Red", linewidth=1, label="Celsius")
plt.show()

plt.xlim(-0.75, len(x_axis)-0.25)
plt.ylim(0, max(users)+5000)
x_axis = np.arange(0, x_limit, 1)
data = [random.random() for value in x_axis]

plt.scatter(x_axis, data, marker="o", facecolors="red", edgecolors="black",
            s=x_axis, alpha=0.75)

plt.figure(figsize=(20,4))
plt.bar(x_axis, rain_df["Inches"], color='r', alpha=0.5, align="edge")
plt.xticks(tick_locations, rain_df["State"], rotation="vertical")

plt.xlim(-0.25, len(x_axis))
plt.ylim(0, max(rain_df["Inches"])+10)
plt.show()

plt.tight_layout()
plt.savefig("../Images/avg_state_rain.png")
plt.show()

# Set the index to be "State" so they will be used as labels
state_and_inches = state_and_inches.set_index("State")

patron_chart = patron_borrows.plot(kind="bar", title="Library Usage by Patron Type")
patron_chart.set_xlabel("Patron Type")
patron_chart.set_ylabel("Number of Patrons Borrowing Items")

# Remove the rows with missing values in horsepower
car_data = car_data.loc[car_data['horsepower'] != "?"]

# Set the 'car name' as our index
car_data = car_data.set_index('car name')

# Remove the 'origin' column
del car_data['origin']

# Convert the "horsepower" column to numeric so the data can be used
car_data['horsepower'] = pd.to_numeric(car_data['horsepower'])

combined_unemployed_data = combined_unemployed_data.rename(columns={"Country Code_x":"Country Code"})
# Set the 'Country Code' to be our index for easy referencing of rows
combined_unemployed_data = combined_unemployed_data.set_index("Country Code")
average_unemployment = combined_unemployed_data[[str(year) for year in range(2010, 2021)]].mean()
# Collect the years where data was collected
years = average_unemployment.keys()

average_unemployment.plot(label="World Average")
combined_unemployed_data.loc['USA', "2010":"2020"].plot(label="United States")
plt.legend()
plt.show()
# Plot the world average as a line chart
world_avg, = plt.plot(years, average_unemployment, color="blue", label="World Average" )

# Plot the unemployment values for a single country
country_one, = plt.plot(years, combined_unemployed_data.loc['USA',["2010","2011","2012","2013","2014","2015",
                                                                "2016","2017","2018","2019","2020"]], 
                        color="green",label=combined_unemployed_data.loc['USA',"Country Name"])

# Create a legend for our chart
plt.legend(handles=[world_avg, country_one], loc="best")

# Show the chart
plt.show()

combined_travel_df = pd.merge(combined_travel_df, travel_2018_df, how="outer", on="COUNTRY OF NATIONALITY")
country1_traveler_over_time = travel_reduced.loc[country1,
                                                [f"2016 {columns_to_compare}",
                                                f"2017 {columns_to_compare}", 
                                                f"2018 {columns_to_compare}"]]
plt.xticks(np.arange(min(years), max(years)+1, 1.0))

print(f"The mean MPG of all vehicles is: {round(fuel_economy['Combined_MPG'].mean(),2)}")
print(f"The standard deviation of all vehicle's MPG is: {round(fuel_economy['Combined_MPG'].std(),2)}")
plt.hist(fuel_economy['Combined_MPG'])

# Standard Error of the Mean (SEM) measures the precision of the sample mean estimate of the population mean. It indicates how much the sample mean is expected to fluctuate due to sampling variability. A lower SEM suggests that the sample mean is a more accurate estimate of the population mean.
# Plot sample means with error bars
fig, ax = plt.subplots()
ax.errorbar(x_axis, means, standard_errors, fmt="o")
ax.set_xlim(0, len(vehicle_sample_set) + 1)
ax.set_ylim(20,28)

fig, ax = plt.subplots()
ax.errorbar(np.arange(0, len(samples), 1)+1,means, yerr=sems, fmt="o", color="b",
            alpha=0.5, label="Mean of House Prices")
ax.set_xlim(0, len(means)+1)
ax.set_xlabel("Sample Number")
ax.set_ylabel("Mean of Median House Prices ($100,000)")
plt.legend(loc="best", fontsize="small", fancybox=True)
plt.show()

plt.scatter(wdi_data.iloc[:,9],wdi_data.iloc[:,7])
plt.xlabel('% Population with Access to Clean Water')
plt.ylabel('Male Life Expectancy')
plt.show()

age = wdi_data.iloc[:,3]
cell_phones = wdi_data.iloc[:,10]
correlation = st.pearsonr(age,cell_phones)
plt.scatter(age,cell_phones)
print(f"The correlation between both factors is {round(correlation[0],2)}")

flavanoids = wine_data['flavanoids']
malic_acid = wine_data['malic_acid']
plt.scatter(malic_acid,flavanoids)

# Calculate the correlation coefficient between malic_acid and flavanoids
print(f"The correlation coefficient between malic acid and flavanoids is {round(st.pearsonr(malic_acid,flavanoids)[0],2)}")

california_dataset = datasets.fetch_california_housing()
housing_data = pd.DataFrame(data=california_dataset.data,columns=california_dataset.feature_names)
housing_data['MEDV'] = california_dataset.target
# Plot out rooms versus median house price
x_values = san_diego_housing['AveRooms']
y_values = san_diego_housing['MEDV']
plt.scatter(x_values,y_values)
plt.xlabel('Rooms in House')
plt.ylabel('Median House Prices ($100,000)')
plt.show()
# Add the linear regression equation and line to plot
x_values = san_diego_housing['AveRooms']
y_values = san_diego_housing['MEDV']
(slope, intercept, rvalue, pvalue, stderr) = linregress(x_values, y_values)
regress_values = x_values * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.scatter(x_values,y_values)
plt.plot(x_values,regress_values,"r-")
plt.annotate(line_eq,(5.8,0.8),fontsize=15,color="red")
plt.xlabel('Rooms in House')
plt.ylabel('Median House Prices ($100,000)')
print(f"The r-squared is: {rvalue**2}")
plt.show()

# Generate a scatter plot of year versus number of petrol-electric cars
year = vehicle_data.loc[(vehicle_data["type"]=="Cars") & (vehicle_data["engine"]=="Petrol-Electric"),"year"]
petrol_electric_cars = vehicle_data.loc[(vehicle_data["type"]=="Cars") & (vehicle_data["engine"]=="Petrol-Electric"),"number"]
plt.scatter(year,petrol_electric_cars)
plt.xticks(year, rotation=90)
plt.xlabel('Year')
plt.ylabel('Petrol Electric Cars')
plt.show()
# Perform a linear regression on year versus petrol-electric cars
pe_slope, pe_int, pe_r, pe_p, pe_std_err = stats.linregress(year, petrol_electric_cars)
# Create equation of line to calculate predicted number of petrol-electric cars
pe_fit = pe_slope * year + pe_int
# Plot the linear model on top of scatter plot 
year = vehicle_data.loc[(vehicle_data["type"]=="Cars") & (vehicle_data["engine"]=="Petrol-Electric"),"year"]
petrol_electric_cars = vehicle_data.loc[(vehicle_data["type"]=="Cars") & (vehicle_data["engine"]=="Petrol-Electric"),"number"]
plt.scatter(year,petrol_electric_cars)
plt.plot(year,pe_fit,"--")
plt.xticks(year, rotation=90)
plt.xlabel('Year')
plt.ylabel('Petrol Electric Cars')
plt.show()
# Repeat plotting scatter and linear model for year versus petrol cars
petrol_cars = vehicle_data.loc[(vehicle_data["type"]=="Cars") & (vehicle_data["engine"]=="Petrol"), "number"]
p_slope, p_int, p_r, p_p, p_std_err = stats.linregress(year, petrol_cars)
p_fit = p_slope * year + p_int
plt.scatter(year,petrol_cars)
plt.plot(year,p_fit,"--")
plt.xticks(year, rotation=90)
plt.xlabel('Year')
plt.ylabel('Petrol Cars')
plt.show()
# Repeat plotting scatter and linear model for year versus electric cars
diesel_cars = vehicle_data.loc[(vehicle_data["type"]=="Cars") & (vehicle_data["engine"]=="Diesel"), "number"]
d_slope, d_int, d_r, d_p, d_std_err = stats.linregress(
    year, diesel_cars)
d_fit = d_slope * year + d_int
plt.scatter(year,diesel_cars)
plt.plot(year,d_fit,"--")
plt.xticks(year, rotation=90)
plt.xlabel('Year')
plt.ylabel('Diesel Cars')
plt.show()
# Generate a facet plot of all 3 figures
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
fig.suptitle("Number of Vehicles Over Time", fontsize=16, fontweight="bold")

ax1.set_xlim(min(year), max(year))
ax1.plot(year, petrol_electric_cars, linewidth=1, marker="o")
ax1.plot(year, pe_fit, "b--", linewidth=1)
ax1.set_ylabel("Petrol-Electric Cars")

ax2.plot(year, petrol_cars, linewidth=1, marker="o", color="y")
ax2.plot(year, p_fit, "y--", linewidth=1)
ax2.set_ylabel("Petrol Cars")

ax3.plot(year, diesel_cars, linewidth=1, marker="o", color="g")
ax3.plot(year, d_fit, "g--", linewidth=1)
ax3.set_ylabel("Diesel Cars")
ax3.set_xlabel("Year")
plt.show()
# Calculate the number of cars for 2024
year = 2024
print(f"The number of petrol-electic cars in 2024 will be {round(pe_slope * year + pe_int,0)}.")
print(f"The number of petrol cars in 2024 will be {round(p_slope * year + p_int,0)}.")
print(f"The number of diesel cars in 2024 will be {round(d_slope * year + d_int,0)}.")




















