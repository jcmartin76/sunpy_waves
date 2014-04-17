from scipy.io.idl import readsav
import numpy as np
import matplotlib.pyplot as plt
import pandas
from datetime import timedelta, datetime
import scipy
from matplotlib.ticker import EngFormatter
from scipy.stats import nanmean
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
import matplotlib.dates as dates
from matplotlib.ticker import ScalarFormatter 
import matplotlib.dates as mdates

itime = datetime.strptime("2013-04-11T00:00", "%Y-%m-%dT%H:%M")

w0r1 = readsav('20130411_R1.sav')
w0r2 = readsav('20130411_R2.sav')

rad1data = w0r1['arrayb'][:,0:1439]
rad1back = w0r1['arrayb'][:,1440]
  
for i in range(0,1439):
	rad1data[:,i] = rad1data[:,i] - rad1back

rad2data = w0r2['arrayb'][:,0:1439]
rad2back = w0r2['arrayb'][:,1440]

for i in range(0,1439):
	rad2data[:,i] = rad2data[:,i] - rad2back

radtime = []
for i in range(len(rad1data[0,:])):
	radtime.append(itime + timedelta(minutes=i))

rad1freq = (4 * np.array(range(len(rad1data[:,0]))) + 20.)/1000.
rad2freq = (np.array(range(len(rad1data[:,0])))*(13.825-1.075)/255. +1.075)
nny = np.concatenate([rad1freq,rad2freq])

radtime2 = radtime
# Create a pandas DataFrame to make the spectrogram plot
index = np.array(radtime2)


yinw = scipy.r_[rad1data,rad2data]

#yinw[60*(25) + 2:60*(25) + 14,:] = np.NaN
yinw = yinw[26:-1,:]

# def pad(data):
# 	bad_indexes = np.isnan(data)
# 	good_indexes = np.logical_not(bad_indexes)
# 	good_data = data[good_indexes]
# 	interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
# 	data[bad_indexes] = interpolated
# 	return data

# yinw  = np.apply_along_axis(pad, 1, yiw)

# ffreqs = np.apply_along_axis(pad, 0, nny)

pfreqs = nny[26:-1]

spectrogramw = pandas.DataFrame(yinw.transpose(),
                               index = index,
                               columns = nny[26:-1])

fig, ax1 = plt.subplots(1,figsize=(8, 2))
#ax1 = fig.add_subplot(111)
#fig.subplots_adjust(right=0.85,bottom=0.15,left=0.1,top=0.9)
#ax1.set_title("Test: " )
ax1.tick_params(axis='both', direction='out')
ax1.set_ylim([pfreqs[1],pfreqs[-1]])
#ax1.set_xlim([datetime.strptime("2010-03-13T12:00", "%Y-%m-%dT%H:%M"),datetime.strptime("2010-03-14T12:00", "%Y-%m-%dT%H:%M")])
ax1.set_yscale("log")
ax1.set_ylabel("Frequency  MHz")
ax1.yaxis.set_visible(True)

spectrogram_nan = spectrogramw.iloc[:,[1]]*np.NaN
spectrogram_nan.plot(ax=ax1,legend=False)



# For the actual spectrogram plot, we create an image.  First, we make
# a 1000 point log spaced frequency range from the min to the max
# frequency to serve as the y axis for the image.
yinterp = np.logspace(np.log10(pfreqs[1]),np.log10(pfreqs[-1]),10*len(pfreqs))

# For each of the 1000 interpolation points on the y axis, we find the
# index of the closest element in the FFT frequency array.  At low
# frequencies, multiple 'interp' frequencies match a single FFT
# frequency.  At high frequencies, several FFT frequencies will be
# skipped over from one 'interp' frequency to the next.
yind = [np.nanargmin(np.abs(pfreqs-y)) for y in yinterp]

# Then, we fill in the image using the indices found above.
img = np.empty(shape = (len(radtime2),len(yinterp)))

for i in range(len(radtime2)):
	img[i,:] = np.interp(yinterp, pfreqs, yinw[:,i])
#	img[i,:] = spectrogram.values[i,yind]

#    img[i,:] = np.log10(spectrogram.values[i,yind])

### Background substraction 

# bkg = np.mean(img[100:200,:],axis=0)

# indx,indy = img.shape

# for i in range(indx):
# 	img[i,:] = img[i,:] - bkg
    
# Finally, the image is shown on the plot.  The array is normalized by
# default to span the entire color map.

im1 = ax1.imshow(img.transpose(), 
                 interpolation='bicubic', 
                 vmin=1,vmax=4,
                 aspect='auto',origin='lower',
                 extent = [min(ax1.get_xlim()),
                           max(ax1.get_xlim()),
                           min(ax1.get_ylim()),
                           max(ax1.get_ylim())])

formatter = DateFormatter('%H:%M')

xax = ax1.get_xaxis()
xax.set_major_locator(dates.HourLocator(interval=4))
ax1.xaxis.set_major_formatter(formatter)

yax = ax1.get_yaxis()
yax.set_major_formatter(ScalarFormatter())

ax1.text(mdates.datestr2num('2013-04-11T00:30'), 8,'Wind',fontsize=12, color='w')

fig.subplots_adjust(hspace=0.05)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
plt.xticks(rotation=0)

plt.savefig('wind_spectrogram_day.svg',format='svg')
plt.savefig('wind_spectrogram_day.eps',format='eps')

plt.show()

