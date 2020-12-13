import xlrd
import numpy as np

# Read xls file
workbook = xlrd.open_workbook('Raw data/Environmental conditions.xlsx')
sheet = workbook.sheet_by_name('Sheet1')
keys = np.asarray(list(filter(None, sheet.row_values(0))), dtype='str')

class Moving_average:
        
    def __init__(self, lab):
        # Get data
        if lab == 'VPD':
            T = self.get_data('T')
            RH = self.get_data('RH')
            VPD = np.zeros(len(T))
            for i in range(len(VPD)):
                VPD[i] = (100-RH[i])/100*0.61078*np.exp(17.27*T[i]/(T[i]+237.3))/101.3
            res = VPD
            self.vals = ma.masked_greater(np.asarray(res, dtype='float'), 10)
        else:
            res = self.get_data(lab)
            self.vals = ma.masked_less(np.asarray(res, dtype='float'), -100)
    
    
    def get_data(self, lab):
        
        res = np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])
        
        # Fill empth cells       
        for i in range(len(res)):
            if res[i] == '':
                res[i] = -999
        
        res = np.asarray(res, dtype='float')
        return res

    # Get daily average
    def daily_average(self):
        
        vals = self.vals
        
        valsre = np.reshape(vals, (len(vals) // 48, 48))
#        res = np.mean(valsre[:, 24:33], axis=1)
        res = np.mean(valsre[:, ], axis=1)
        
        return res

    # Get convolution (moving average) with head and rail as NaN
    def moving_average(self, window_size):
        daily_average = self.daily_average()
        window = np.ones(int(window_size)) / float(window_size)
        average = np.convolve(daily_average, window, 'same')
        average[:window_size] = np.nan
        average[-window_size:] = np.nan
        return average
