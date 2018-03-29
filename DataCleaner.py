#This file will be used to clean the data before passing into our algorithm
import pandas

class DataCleaner():
    # we'll initialize the class with a dataset, and continuously update it until return is requested
    def __init__(self, dataset):
        self.dataset = pandas.read_csv(dataset)

    def to_CSV(self, name='clean_data.csv'):
        self.dataset.to_csv(name)

    def to_excel(self, name='clean_data.xlsx'):
        self.dataset.to_excel(name)

    def to_JSON(self, name='clean_data.json'):
        self.dataset.to_json(name)

    # each section of data will be cleaned in a different function, so we can limit our cleanable set
    # to data we plan on using during our runs.  Also, many will have to be cleaned using various methods

    # since land value for condominiums is listed as 0, we have to get the total value of the property and
    # populate it down.  Ideally then dividing by the number of units.
    def clean_AV_LAND(self):
        count = 0
        last = 0
        divider = 0
        column = self.dataset['AV_LAND']
        for value in column:
            # indicates we're at the start of a condo set
            if value == 0:
                tempcount = count
                # count how many zeroes we see in a row to divide the total land value
                while column[tempcount] == 0:
                    divider += 1
                    tempcount += 1
                new_value = last/divider
                tempcount = count
                # loop over again and replace those zeroes
                while column[tempcount] == 0:
                    column.update(pandas.Series([new_value], index = [tempcount]))
                    tempcount += 1
                #reset our temporary variables
                tempcount, divider = 0,0
            # now we go back to our original count location, ignoring everything with a preset value
            last = column[count]
            print(last)
            count += 1
        # once the loop is complete we delete the original series from our dataframe and add the new one back in
        del self.dataset['AV_LAND']
        column.to_frame()
        self.dataset = pandas.concat(self.dataset, column)


dc = DataCleaner('property-assessment-fy2015.csv')
dc.clean_AV_LAND()

# >>> s = pd.Series(['a', 'b', 'c'])
# >>> s.update(pd.Series(['d', 'e'], index=[0, 2]))
# >>> s
# 0    d
# 1    b
# 2    e
# dtype: object