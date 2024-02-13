from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from collections import defaultdict
import numpy as np
import h5py
import traceback
from gcpdatautils import ROTTEN_EGGS

def parse_events_html(event_file, exclude_events_longer_than_days=7):
    soup = BeautifulSoup(open(event_file).read(), features="lxml")
    #the 'recipe' and 'statistic' columns don't always get parsed correctly due
    # to the break, but we can work around it
    rows = [[td.contents[0].rstrip() for td in row.find_all("td") ]
            for row in soup.body.table.tbody.find_all("tr")]
    #prepare the event IDs to be analyzed
    included_events = []
    for row in rows:
        if row[-1].startswith("Yes"): #only keep those events used in the GCP analysis (e.g. exclude events noted as redundant or post-hoc)
            event_num = row[0]
            event_name = row[1]
            start, end = row[2], row[3]
            start_datetime = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            end_datetime = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
            duration = 1 + (end_datetime - start_datetime).total_seconds()
            if (end_datetime < start_datetime):
                print("Skipping because end < start:", event_num, event_name, start, end)
                continue
            if (duration > exclude_events_longer_than_days*24*60*60):
                print("Skipping because event is more than",exclude_events_longer_than_days,
                      "days long!", event_num, event_name, start, end)
                continue
            if ("New Year Var" in event_name):
                print("Skipping because redundant with a corresponding 'mean' event: ",
                      event_num, event_name, start, end)
                continue
        else:
            continue
        included_events.append((event_num, event_name, start_datetime, end_datetime, duration))

    return included_events


def parse_rotten_egg_file(file):
    bad_data_list = []
    #read in the bad data durations
    for line in open(file, 'r'):
        line = line.rstrip("\n")
        if line.startswith("47")==False: #ignore lines not starting with 47
            continue
        else:
            _,start,end,deviceid = line.split(",")
            #there appear to be some cases of manual errors that need to be corrected
            if (end=="2001-06-31 23:59:59"):
                print("manually correcting",line)
                end="2001-06-30 23:59:59"
            if (end=="2008-04-55 23:59:59"):
                print("manually correcting",line)
                end="2008-04-30 23:59:59"
            #If end is 'current', set to datetime.now()
            if (end.startswith("current--*")):
                end=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            start_datetime = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            end_datetime = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
            bad_data_list.append((start_datetime, end_datetime, int(deviceid) ))
    return bad_data_list


class GCPMissingDataError(RuntimeError):
  pass

class GCPHdf5DataReader(object):

  def __init__(self, bad_data_file=ROTTEN_EGGS,
                     year_to_hdf5path=lambda x: "GCP1data_"+str(x)+".hdf5"):
    print("Parsing the bad data file:",bad_data_file)
    #reorganize the bad data list by device ID
    self.bad_data_lookup = defaultdict(list)
    for baddata_starttime, baddata_endtime, deviceid in parse_rotten_egg_file(bad_data_file):
      self.bad_data_lookup[deviceid].append((baddata_starttime, baddata_endtime))
    self.year_to_hdf5fh = {} #mapping from the year to the hdf5 file handle
    self.year_to_hdf5path = year_to_hdf5path

  def fetch_data(self, starttime, endtime, normalize=False):

    if (endtime.strftime("%Y-%m-%d") == starttime.strftime("%Y-%m-%d")):
      return self.fetch_data_within_day_normalize(starttime, endtime, normalize=normalize) #returns both data and devices
    else:
      #split into accesses over multiple days
      start_day = starttime.strftime("%Y-%m-%d")
      end_day = endtime.strftime("%Y-%m-%d")
      #for the first day:
      day_accesses = [ (
          starttime,
          datetime.strptime(start_day+" 23:59:59", '%Y-%m-%d %H:%M:%S')) ]
      #for intermediate days:
      dayoffset = 1
      inter_day = (starttime + timedelta(days=dayoffset)).strftime("%Y-%m-%d")
      while inter_day != end_day:
        day_accesses.append(
            (datetime.strptime(inter_day+" 00:00:00", '%Y-%m-%d %H:%M:%S'),
             datetime.strptime(inter_day+" 23:59:59", '%Y-%m-%d %H:%M:%S')))
        dayoffset += 1
        inter_day = (starttime + timedelta(days=dayoffset)).strftime("%Y-%m-%d")

      #for the final day
      day_accesses.append(
          (datetime.strptime(inter_day+" 00:00:00", '%Y-%m-%d %H:%M:%S'),
           endtime))
      days_data_and_devices = [self.fetch_data_within_day_normalize(t1, t2, normalize=normalize) for (t1,t2) in day_accesses]
      #if all the days have the same sets of devices...
      if (len(set([str(x[1]) for x in days_data_and_devices]))==1):
        return np.concatenate([x[0] for x in days_data_and_devices], axis=0), days_data_and_devices[0][1]
      else:
        #print("Some days don't have same devices. Rearranging")
        all_devices = set()
        [all_devices.update(x[1]) for x in days_data_and_devices]
        all_devices = sorted(all_devices)
        all_devices_newidxs = dict([(x[1],x[0]) for x in enumerate(all_devices)])
        rearranged_days_data = []
        for day_data, day_devices in days_data_and_devices:
          rearranged_day_data = np.zeros((len(day_data), len(all_devices)))
          rearranged_day_data[:,:] = np.nan
          for orig_deviceidx, deviceid in enumerate(day_devices):
            rearranged_day_data[:,all_devices_newidxs[deviceid]] = day_data[:,orig_deviceidx]
          rearranged_days_data.append(rearranged_day_data)
        return np.concatenate(rearranged_days_data, axis=0), all_devices

  def fetch_data_within_day_normalize(self, starttime, endtime, normalize):
    if (normalize==False):
      return self.fetch_data_within_day(starttime, endtime)
    else:
      day = starttime.strftime("%Y-%m-%d")
      day_start = datetime.strptime(day+" 00:00:00", '%Y-%m-%d %H:%M:%S')
      day_end = datetime.strptime(day+" 23:59:59", '%Y-%m-%d %H:%M:%S')
      full_day_data, daydevices = self.fetch_data_within_day(day_start, day_end)
      #set ddof=1 to get unbiased (more conservative) stdev estimates
      device_day_stdevs = np.nan_to_num(np.nanstd(full_day_data, axis=0, ddof=1), np.sqrt(50))
      device_day_means = np.nan_to_num(np.nanmean(full_day_data, axis=0), 100)
      #transform the data to be centered around 100, with stdev of sqrt(50)
      full_day_data_normalized = ((full_day_data - device_day_means[None,:])/
                                  (device_day_stdevs[None,:]/np.sqrt(50))) + 100
      start_offset =  int(starttime.timestamp() - day_start.timestamp())
      end_offset = int((endtime.timestamp() - day_start.timestamp()) + 1)
      return full_day_data_normalized[start_offset:end_offset], daydevices

  def get_fh_for_year(self, year):
    if (year not in self.year_to_hdf5fh):
      self.year_to_hdf5fh[year] = h5py.File(self.year_to_hdf5path(year), "r")
    return self.year_to_hdf5fh[year]

  def get_available_days_in_year(self, year):
    fh = self.get_fh_for_year(year)
    return fh.keys()

  def fetch_data_within_day(self, starttime, endtime, bail_if_missing_seconds=True, mask_bad_data=True):
    assert endtime.strftime("%Y-%m-%d") == starttime.strftime("%Y-%m-%d")
    year = starttime.year
    fh = self.get_fh_for_year(year)
    day = starttime.strftime("%Y-%m-%d")
    if (day not in fh):
      raise GCPMissingDataError("data for "+day+" not present")
    dset = fh[day]
    if dset.attrs['start_time'] > starttime.timestamp():
      raise GCPMissingDataError("query for day "
             +day+" starts at "+str(starttime.timestamp())
             +" but data for day starts at "+str(dset.attrs['start_time']))
    if dset.attrs['end_time'] < endtime.timestamp():
      raise GCPMissingDataError("query for day "
             +day+" ends at "+str(endtime.timestamp())
             +" but data for day ends at "+str(dset.attrs['end_time']))
    #now that we have verified that the day has all the data we need...
    start_offset = int(starttime.timestamp() - dset.attrs['start_time'])
    end_offset = int(endtime.timestamp()+1 - dset.attrs['start_time']) #add +1 since GCP ranges are end-inclusive
    day_data = np.array(dset[start_offset:end_offset]).astype("float")
    #replace 255 with nan; 255 was used to encode NaNs in the hdf5 file
    day_data[day_data==255] = np.nan
    
    devices_on_day = fh[day].attrs["device_ids"]
    if (mask_bad_data):
      #Radin 2023 (Anomalous entropic effects in physical systems associated
      # with collective consciousness) said "All individual samples within a matrix
      # less than 55 or greater than 145 were set to nan" so we do that here
      day_data = np.where((day_data < 55), np.nan, day_data)
      day_data = np.where((day_data > 145), np.nan, day_data)

      #Mask out columns with bad data
      masking_occurred = False
      for deviceid_idx,deviceid in enumerate(devices_on_day):
        if (deviceid in self.bad_data_lookup):
          for rottenegg_start, rottenegg_end in self.bad_data_lookup[deviceid]:
            if (rottenegg_start.timestamp() <= endtime.timestamp() and rottenegg_end.timestamp() >= starttime.timestamp()):
              if (masking_occurred == False):
                print("Before masking, fraction of nans in raw data is", np.mean(np.isnan(day_data)),"for",starttime,"to",endtime)
              masking_occurred = True
              print("Found 'rotten egg' entries for device id:",deviceid,
                    "in time range",starttime,"to",endtime,
                    "(range:",rottenegg_start,"to",rottenegg_end,")")
              mask_startidx = int(max(rottenegg_start.timestamp()-starttime.timestamp(),0))
              mask_endidx = int(min((rottenegg_end.timestamp()+1)-starttime.timestamp(), len(day_data))) #+1 because end inclusive
              day_data[mask_startidx:mask_endidx, deviceid_idx] = np.nan

      if (masking_occurred):
        print("After masking, fraction of nans in raw data is", np.mean(np.isnan(day_data)))
    
    #Check that there are no rows that completely lack data
    nonnan_devices_per_second = np.sum(np.isnan(day_data)==False, axis=1)
    if (np.min(nonnan_devices_per_second)==0 and bail_if_missing_seconds):
      raise GCPMissingDataError("Some seconds for query from "+str(starttime)+" to "+str(endtime)+" had no data")

    return day_data, devices_on_day

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def close(self):
    for fh in self.year_to_hdf5fh.values():
      fh.close()
