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

itime = datetime.strptime("2013-04-11T00:00", "%Y-%m-%dT%H:%M")

############ Wind Data #################
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

indexw = np.array(radtime)
yinw = scipy.r_[rad1data,rad2data]

yinw = yinw[26:-1,:]

################## STEREO Data ##############################

s0 = readsav('swaves_2013_04_11T0000.sav')

s1 = s0['spectrogram']

s2a = s1['spec_a']

tina1 = np.array(s2a[0]['x'][0])
yina1 = np.array(s2a[0]['y'][0])


tina = tina1

s2 = s1['spec_b']

tin1 = np.array(s2[0]['x'][0])
yin1 = np.array(s2[0]['y'][0])
vin  = np.array(s2[0]['v'][0])

tin  = tin1

pfreqs = vin[:,0]

###
indx,indy = yin1.shape
bkg = nanmean(yin1,axis=1)
bkg = np.array([bkg,]*indy)

yin1 = yin1 - bkg.transpose()
yin = np.array(yin1) 

###
indx,indy = yina1.shape
bkga = nanmean(yina1,axis=1)
bkga = np.array([bkga,]*indy)

yina1 = yina1 - bkga.transpose()

yina = np.array(yina1) 

def pad(data):
        bad_indexes = np.isnan(data)
        good_indexes = np.logical_not(bad_indexes)
        good_data = data[good_indexes]
        interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
        data[bad_indexes] = interpolated
        return data


ffreqs = np.apply_along_axis(pad, 0, pfreqs)


tt = tin-tin[0]
times = np.empty(shape=0)

for ddt in tt:
	times = np.append(times, itime + timedelta(seconds=ddt))

# Create a pandas DataFrame to make the spectrogram plot
index = times
spectrograma = pandas.DataFrame(yina.transpose(),
                               index = index,
                               columns = vin[:,0])

spectrogram = pandas.DataFrame(yin.transpose(),
                               index = index,
                               columns = vin[:,0])

import matplotlib.dates as mdates

fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(8, 4))
#fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True, figsize=(8, 6))
ax1.tick_params(axis='both', direction='out')
ax1.set_ylim([pfreqs[1],pfreqs[-1]])

ax1.set_yscale("log")
ax1.set_ylabel("Frequency  MHz")
ax1.yaxis.set_visible(True)

spectrogram_nan = spectrograma.iloc[:,[1]]*np.NaN
spectrogram_nan.plot(ax=ax1,legend=False)

yinterp = np.logspace(np.log10(pfreqs[1]),np.log10(pfreqs[-1]),10*len(pfreqs))
yind = [np.nanargmin(np.abs(pfreqs-y)) for y in yinterp]

img = np.empty(shape = (len(tin),len(yinterp)))

for i in range(len(tin)):
	img[i,:] = np.interp(yinterp, pfreqs, yina.transpose()[i,:])

im1 = ax1.imshow(img.transpose(), 
                 interpolation='bicubic', 
                 vmin=-1,vmax=5,
                 aspect='auto',origin='lower',
                 extent = [min(ax1.get_xlim()),
                           max(ax1.get_xlim()),
                           min(ax1.get_ylim()),
                           max(ax1.get_ylim())])

ax1.text(mdates.datestr2num('2013-04-11T00:30'), 8,'STEREO A',fontsize=12, color='w')

########## Wind Plot#################################

# pfreqsw = nny[26:-1]

# spectrogramw = pandas.DataFrame(yinw.transpose(),
#                                index = indexw,
#                                columns = nny[26:-1])

# ax2.tick_params(axis='both', direction='out')
# ax2.set_ylim([pfreqsw[1],pfreqsw[-1]])

# ax2.set_yscale("log")
# ax2.set_ylabel("Frequency  MHz")
# ax2.yaxis.set_visible(True)

# spectrogram_nan = spectrogramw.iloc[:,[1]]*np.NaN
# spectrogram_nan.plot(ax=ax2,legend=False)

# yinterp = np.logspace(np.log10(pfreqsw[1]),np.log10(pfreqsw[-1]),10*len(pfreqsw))
# yind = [np.nanargmin(np.abs(pfreqsw-y)) for y in yinterp]

# img2 = np.empty(shape = (len(radtime),len(yinterp)))

# for i in range(len(radtime)):
#   img2[i,:] = np.interp(yinterp, pfreqsw, yinw[:,i])
    

# im2 = ax2.imshow(img.transpose(), 
#                  interpolation='bicubic', 
#                  vmin=1,vmax=4,
#                  aspect='auto',origin='lower',
#                  extent = [min(ax1.get_xlim()),
#                            max(ax1.get_xlim()),
#                            min(ax1.get_ylim()),
#                            max(ax1.get_ylim())])

# ax2.text(mdates.datestr2num('2013-04-11T00:30'), 8,'Wind',fontsize=12, color='w')


###########################################################################################################
#### STEREO B section

ax2.tick_params(axis='both', direction='out')
ax2.set_ylim([pfreqs[1],pfreqs[-1]])
ax2.set_yscale("log")
ax2.set_ylabel("Frequency MHz")
ax2.set_xlabel("Time (from 2013-04-11T00:00 UT)")
ax2.yaxis.set_visible(True)

spectrogram_nan = spectrogram.iloc[:,[1]]*np.NaN
spectrogram_nan.plot(ax=ax2,legend=False)

yinterp = np.logspace(np.log10(pfreqs[1]),np.log10(pfreqs[-1]),4*len(pfreqs))
yind = [np.nanargmin(np.abs(pfreqs-y)) for y in yinterp]

img = np.empty(shape = (len(tin),len(yinterp)))

for i in range(len(tin)):
	img[i,:] = np.interp(yinterp, pfreqs, yin[:,i])

im2 = ax2.imshow(img.transpose(), 
                 interpolation='bicubic', 
                 vmin=-1,vmax=5,
                 aspect='auto',origin='lower',
                 extent = [min(ax1.get_xlim()),
                           max(ax1.get_xlim()),
                           min(ax2.get_ylim()),
                           max(ax2.get_ylim())])

formatter = DateFormatter('%H:%M')

xax = ax2.get_xaxis()
xax.set_major_locator(dates.HourLocator(interval=4))
ax2.xaxis.set_major_formatter(formatter)

yax = ax2.get_yaxis()
yax.set_major_formatter(ScalarFormatter())

ax2.text(mdates.datestr2num('2013-04-11T00:30'), 8,'STEREO B',fontsize=12, color='w')

fig.subplots_adjust(hspace=0.05)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
plt.xticks(rotation=0)

plt.savefig('stereo_spectrogram_day.svg',format='svg')
plt.savefig('stereo_spectrogram_day.eps',format='eps')
plt.show()
