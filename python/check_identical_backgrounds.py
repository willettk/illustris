# Check to see if any of the Illustris images have identical backgrounds; all should be different for a given camera angle.
# Melanie Beck pointed out that the images of subhalo 0 are identical for each camera angle

from astropy.io import fits
from datetime import datetime as dt

diskpath = '/Volumes/REISEPASS/illustris'

# Load path to images and a logging file for output
dirfile = '{0}/directory_catalog_135.txt'.format(diskpath)
writefile = '{0}/identical_backgrounds.txt'.format(diskpath)

wf = open(writefile,'w')
rf = open(dirfile,'r')

i = 0
for line in rf:

    subdir,subhalo,mstar = line.split()

    # Loop over all the camera angles for a given subhalo and SDSS background
    for camera in range(4):
        try:
            imgsums = []
            for bg in range(4):
                img = fits.getdata('{0}/fits/subdir_{1}/synthetic_image_{2}_band_3_camera_{4}_bg_{3}.fits'.format(diskpath,subdir,subhalo,bg,camera),0)
                # Rely on the image sums to figure out if there's accidental duplication
                imgsums.append(img.sum())

            if len(set(imgsums)) < 3:
                print >> wf,"Identical background(s) for subhalo {0}, camera {1}".format(subhalo,camera)
        except IOError:
            print >> wf,"Error reading file(s) for subhalo {0}, camera {1}".format(subhalo,camera)
            
    # Write progress to screen occasionally
    i += 1
    if not i % 100:
        print "{0} images searched {1}".format(i,dt.today().strftime("%H:%M:%S.%f"))

wf.close()

# Conclusion - it's just image 0, for some reason. - KWW
