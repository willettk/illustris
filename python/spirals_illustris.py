from astropy.io import fits

gzdir = '/Users/willettk/Astronomy/Research/GalaxyZoo'

data = fits.getdata('%s/gz_reduction_sandbox/data/illustris_weighted_collated_01.fits' % gzdir,1)

from matplotlib import pyplot as plt

# How many classifications are we averaging?

# This is only for the "fixed_mass" sample. 10712/10832 (98.9%) of active subjects have at least one classification so far.

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(data['num_classifications'])

ax.set_xlabel('Number of classifications')
ax.set_ylabel('Count')
ax.set_title('GZ-Illustris: 11 Oct 2015')

plt.savefig('%s/illustris/plots/classifications_2015_10_11.png' % gzdir)

# Which of them are spirals?

spirals = (data['t00_smooth_or_features_a1_features_frac'] >= 0.8) & \
            (data['t00_smooth_or_features_a1_features_frac'] * data['t00_smooth_or_features_count'] >= 5) & \
            (data['t03_spiral_a0_spiral_frac'] >= 0.8) & \
            (data['t03_spiral_a0_spiral_frac'] * data['t03_spiral_count'] >= 5)

print '%i spirals' % spirals.sum()

from pymongo import MongoClient
from bson.objectid import ObjectId

client = MongoClient('localhost', 27017)
db = client['galaxy_zoo'] 
    
subjects = db['galaxy_zoo_subjects']
classifications = db['galaxy_zoo_classifications']
users = db['galaxy_zoo_users']

import requests
import shutil

wf = open('%s/illustris/spirals.csv' % gzdir,'wb')
print >> wf,"zooniverse_id,snapshot,subhalo_id,camera,background"

for ind,spiral in enumerate(data[spirals]):
    subject = subjects.find_one({'_id':ObjectId(spiral['subject_id'])})
    url = subject['location']['standard']

    # Download images
    get_images = True
    if get_images:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            imagename = '%s/illustris/spirals/spiral_%i_bg%1i_camera%1i.png' % (gzdir,subject['metadata']['subhalo_id'],subject['metadata']['background'],subject['metadata']['camera'])
            with open(imagename, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)  
        else:
            print "couldn't find image for %s, %i" % (subject['zooniverse_id'],subject['metadata']['subhalo_id'])


    # Make a quick catalog

    print >> wf,subject['zooniverse_id'],subject['metadata']['snapshot'],subject['metadata']['subhalo_id'],subject['metadata']['camera'],subject['metadata']['background']

wf.close()




# Questions to ask

# What's the distribution of morphologies (elliptical, disk, merger)?

# How do results so far compare to GZ2?
