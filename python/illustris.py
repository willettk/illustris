'''
Extensive documentation of the Illustris API and how to use it with Python here:
    http://www.illustris-project.org/data/docs/api/

'''

path = "/Users/willettk/Astronomy/Research/GalaxyZoo/illustris"

baseUrl = 'http://www.illustris-project.org/api/'
headers = {"api-key":"e14b9698476e0987cdc11d0c25e14232"}        # My (Kyle Willett's) personal API key

import numpy as np
def get(path, params=None):
    import requests

    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    if r.status_code != requests.codes.ok:
        print 'Subhalo: %s' % path.split('/')[-1]
        return None

    #r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r

def read_gz_filenames():

    filename = '%s/directory_catalog_135.txt' % path
    with open(filename,'r') as f:
        d = f.readlines()

    subdirs = [int(x.split()[0]) for x in d]
    subhalo_ids = [int(x.split()[1]) for x in d]

    return subdirs,subhalo_ids

def get_subhalo(s_id):

    url = "http://www.illustris-project.org/api/Illustris-1/snapshots/135/subhalos/%s" % str(s_id)
    sub = get(url)

    return sub

def plot_image_examples():

    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
    from StringIO import StringIO

    subdirs,subhalo_ids = read_gz_filenames()

    plt.figure(1,(12,12))

    for sub_count,s_id in enumerate(subhalo_ids[:16]):
        sub = get_subhalo(s_id)

        if 'stellar_mocks' in sub['supplementary_data']:
            png_url = sub['supplementary_data']['stellar_mocks']['image_fof']
            response = get(png_url)

            plt.subplot(4,4,sub_count)
            plt.text(0,-20,'ID=%i' % s_id,color='blue')
            plt.gca().axes.get_xaxis().set_ticks([])
            plt.gca().axes.get_yaxis().set_ticks([])

            file_object = StringIO(response.content)
            plt.imshow(mpimg.imread(file_object))

    plt.show()

    return None

def make_metadata_table():

    from astropy.table import Table,vstack
    import os,datetime
    from copy import deepcopy

    subdirs,subhalo_ids = read_gz_filenames()

    subdict = {}
    for x,y in zip(subdirs,subhalo_ids):
        subdict[y] = '%03i' % x

    N = len(subhalo_ids)

    sub0 = get_subhalo(subhalo_ids[0])
    drows = [('Illustris-1', 'illustris', 
            "http://www.galaxyzoo.org.s3.amazonaws.com/subjects/illustris/standard/subdir_%s/synthetic_image_%i_camera_%i_bg_%i.png" % (subdict[subhalo_ids[0]],sub0['id'],0,0),
            "http://www.galaxyzoo.org.s3.amazonaws.com/subjects/illustris/inverted/subdir_%s/synthetic_image_%i_camera_%i_bg_%i.png" % (subdict[subhalo_ids[0]],sub0['id'],0,0),
            "http://www.galaxyzoo.org.s3.amazonaws.com/subjects/illustris/thumbnail/subdir_%s/synthetic_image_%i_camera_%i_bg_%i.png" % (subdict[subhalo_ids[0]],sub0['id'],0,0),
            sub0['snap'], subdict[subhalo_ids[0]], sub0['id'], 0, 0, 'grz', sub0['mass_log_msun'], sub0['halfmassrad'], sub0['sfr'], 
            sub0['stellarphotometrics_b'], sub0['stellarphotometrics_g'], sub0['stellarphotometrics_i'], sub0['stellarphotometrics_k'], 
            sub0['stellarphotometrics_r'], sub0['stellarphotometrics_u'], sub0['stellarphotometrics_v'], sub0['stellarphotometrics_z'])]
    dnames = ('metadata.simulation', 'metadata.survey','location.standard','location.inverted','location.thumbnail', 'metadata.snapshot', 'metadata.subdir', 'metadata.subhalo_id', 'metadata.camera', 'metadata.background', 'metadata.bands', 'metadata.mass_log_msun', 'metadata.radius_half', 'metadata.sfr', 'metadata.mag.absmag_b', 'metadata.mag.absmag_g', 'metadata.mag.absmag_i', 'metadata.mag.absmag_k', 'metadata.mag.absmag_r', 'metadata.mag.absmag_u', 'metadata.mag.absmag_v', 'metadata.mag.absmag_z')
    dtypes = ( 'S11', 'S10', 'S120','S120','S120', 'i8', 'i3', 'i8', 'i8', 'i8', 'S3', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
    t = Table(rows=drows,names = dnames,dtype=dtypes,meta={'name':'Galaxy Zoo-Illustris'})

    sub_count = 0

    camera = 0
    bg = 0
    for s_id in subhalo_ids:
        sub = get_subhalo(s_id)
        if sub is not None:
            t.add_row(('Illustris-1', 'illustris', "http://www.galaxyzoo.org.s3.amazonaws.com/subjects/illustris/standard/subdir_%s/synthetic_image_%i_camera_%i_bg_%i.png" % (subdict[s_id],sub['id'],camera,bg), "http://www.galaxyzoo.org.s3.amazonaws.com/subjects/illustris/inverted/subdir_%s/synthetic_image_%i_camera_%i_bg_%i.png" % (subdict[s_id],sub['id'],camera,bg), "http://www.galaxyzoo.org.s3.amazonaws.com/subjects/illustris/thumbnail/subdir_%s/synthetic_image_%i_camera_%i_bg_%i.png" % (subdict[s_id],sub['id'],camera,bg), sub['snap'], subdict[s_id], sub['id'], camera, bg, 'grz', sub['mass_log_msun'], sub['halfmassrad'], sub['sfr'], sub['stellarphotometrics_b'], sub['stellarphotometrics_g'], sub['stellarphotometrics_i'], sub['stellarphotometrics_k'], sub['stellarphotometrics_r'], sub['stellarphotometrics_u'], sub['stellarphotometrics_v'], sub['stellarphotometrics_z']))

        sub_count += 1
        if not sub_count % 100:
            print '%i/%i galaxies completed; %s' % (sub_count,N,datetime.datetime.now().strftime("%H:%M:%S.%f"))

    # Remove the duplicate first row of the table
    #
    t.remove_row(1)
    
    # Now replicate the existing table to account for different backgrounds and cameras
    
    tcopy = deepcopy(t)

    for camera in range(4):
        for bg in range(4):
            if (camera > 0) or (bg > 0):
                tcopy['metadata.camera'] = np.zeros(len(tcopy),dtype=int)+camera
                tcopy['metadata.background'] = np.zeros(len(tcopy),dtype=int)+bg

                def location_fix(loc,camera,bg):

                    loclist = list(loc)
                    locnew1 = ['%scamera_%i%s' % (l.split('camera')[0],camera,l.split('camera')[-1][2:]) for l in loclist]
                    locnew2 = ['%sbg_%i%s' % (l.split('bg')[0],bg,l.split('bg')[-1][2:]) for l in locnew1]

                    return locnew2

                tcopy['location.standard']  = location_fix(tcopy['location.standard'],camera,bg)
                tcopy['location.inverted']  = location_fix(tcopy['location.inverted'],camera,bg)
                tcopy['location.thumbnail'] = location_fix(tcopy['location.thumbnail'],camera,bg)

                t = vstack((t,tcopy))

    # Group the metadata by priority
    #
    t_grouped = group_metadata(t)

    # Kluge, since no overwrite option exists in astropy

    ftypes = ('csv','fits')
    for ft in ftypes:
        fname = '%s/metadata/illustris_metadata.%s' % (path,ft)
        if os.path.isfile(fname):
            os.remove(fname)
        t_grouped.write(fname)

    return None

def group_metadata(t):

    from astropy.table import Column

    groupcol = Column(data = np.zeros(len(t)),dtype="S11",name="metadata.priority")

    # Group 1: narrow mass bins, but varying camera angle and background. 
    # Study effect of environment, orientation on morphology

    grouplabel = "fixed_mass"

    mass1,mass2 = np.log10(6e10),np.log10(6e10)+2
    ind_fixed_mass_lo = (t['metadata.mass_log_msun'] >= (mass1 - 0.25)) & (t['metadata.mass_log_msun'] <= (mass1 + 0.25))
    ind_fixed_mass_hi = (t['metadata.mass_log_msun'] >= (mass2 - 0.25)) & (t['metadata.mass_log_msun'] <= (mass2 + 0.25))

    ind_fixed_mass = ind_fixed_mass_lo | ind_fixed_mass_hi
    groupcol[ind_fixed_mass] = grouplabel

    print "Group 1a (fixed_mass),  low-mass bin: %i galaxies" % np.sum(ind_fixed_mass_lo)
    print "Group 1b (fixed_mass), high-mass bin: %i galaxies" % np.sum(ind_fixed_mass_hi)

    # Group 2: full mass range, but fixed camera angle and background. 
    # Study effect of mass on bulge-disk decomposition
    #
    # Some galaxies from Group 1 would have been included in the original criteria
    # for Group 2 (fixed mass bins with camera=0,bg=0); explicitly remove these.
    # Scientific analysis should put these back in after classifications are complete.

    grouplabel = "fixed_view"

    ind_fixed_view = (t['metadata.camera'] == 0) & (t['metadata.background'] == 0) & np.logical_not(ind_fixed_mass)
    groupcol[ind_fixed_view] = grouplabel

    print "Group 2 (fixed_view): %i galaxies" % np.sum(ind_fixed_view)

    # Group 3: all galaxies not in Group 1 or Group 2
    
    grouplabel = "full_sample"

    ind_full_sample = np.logical_not(ind_fixed_view) & np.logical_not(ind_fixed_mass)
    groupcol[ind_full_sample] = grouplabel

    print "Group 3 (full_sample): %i galaxies" % np.sum(ind_full_sample)
    print "Check: group 3 should be %i" % (len(t) - np.sum(ind_fixed_mass_lo)  - np.sum(ind_fixed_mass_hi)  - np.sum(ind_fixed_view)  )

    t.add_column(groupcol)

    return t

def check_mass_lims(N=1000):

    # Find out what the proper mass limits on Illustris are

    subdirs,subhalo_ids = read_gz_filenames()

    sub0 = get_subhalo(subhalo_ids[0])

    masses = {"mass":[],
    "mass_gas":[],
    "mass_dm":[],
    "mass_stars":[],
    "mass_bhs":[],
    "massinhalfrad":[],
    "massinhalfrad_gas":[],
    "massinhalfrad_dm":[],
    "massinhalfrad_stars":[],
    "massinhalfrad_bhs":[],
    "massinmaxrad":[],
    "massinmaxrad_gas":[],
    "massinmaxrad_dm":[],
    "massinmaxrad_stars":[],
    "massinmaxrad_bhs":[],
    "massinrad":[],
    "massinrad_gas":[],
    "massinrad_dm":[],
    "massinrad_stars":[],
    "massinrad_bhs":[],
    "mass_log_msun":[]}

    import random
    for i,s_id in enumerate(random.sample(subhalo_ids,N)):
        sub = get_subhalo(s_id)
        if sub is not None:
            for m in masses:
                masses[m].append(sub[m])

        if not i % 100:
            print i

    for m in masses:
        print "%20s: min = %.3f, max = %.3f" % (m,np.min(masses[m]),np.max(masses[m]))

    return masses

def histsubplot(ax,data,xlabel='',range=None,color="#377eb8",bins=20,histtype='bar'):

        ax.hist(data,range=range,color=color,bins=bins,histtype=histtype,normed=1)
        ax.set_xlabel(xlabel,fontsize=18)
        ax.set_ylabel('Count',fontsize=16)

        return None

def re_from_petrosian(r50,r90):

    # Convert Petrosian radii at 50% and 90% into an effective radius (r_e). 
    # Taken from Graham et al. (2005); good approximation for Sersic indices 0.1 < n < 10

    p3 = 8.0e-6
    p4 = 8.47

    r_e = r50 / (1. - p3*((r90/r50)**p4))

    return r_e

def histograms(savefig=False):

    h = 0.70

    from matplotlib import pyplot as plt
    from astropy.io import fits

    metadata = fits.getdata('%s/gz_illustris_all_metadata.fits' % path,1)

    fig,axarr = plt.subplots(2,2,figsize=(10,10))
    axravel = np.ravel(axarr)
    histsubplot(axravel[0],metadata['mass_stars']/h * 10.,'Mass '+r'$[\log$'+' '+r'$(M/M_{\odot})]$',color="#377eb8",range=(6,14),bins=30)
    histsubplot(axravel[1],metadata['halfmassrad_stars'] * h,color="#e41a1c",range=(0,20),bins=30)
    histsubplot(axravel[2],metadata['sfr'],'SFR '+r'$[M_{\odot}/yr]$',color="#4daf4a",range=(0,10),bins=30)
    histsubplot(axravel[3],metadata['stellarphotometrics_r'],color="#984ea3",range=(-26,-15),bins=30)

    # Same plots for GZ2 Legacy data?
    #
    gz2data = fits.getdata('/Users/willettk/Astronomy/Research/GalaxyZoo/fits/mpajhu_gz2.fits',1)

    from astropy.cosmology import WMAP7
    from astropy import units as u
    r50_kpc = (gz2data['PETROR50_R'] * u.arcsec / WMAP7.arcsec_per_kpc_comoving(gz2data['REDSHIFT'])).value
    r90_kpc = (gz2data['PETROR90_R'] * u.arcsec / WMAP7.arcsec_per_kpc_comoving(gz2data['REDSHIFT'])).value
    r_e = re_from_petrosian(r50_kpc,r90_kpc)

    histsubplot(axravel[0],gz2data['MEDIAN_MASS'],'Stellar mass '+r'$[\log$'+' '+r'$(M/M_{\odot})]$',color="k",histtype='step',range=(6,14),bins=30)
    histsubplot(axravel[1],r50_kpc,'Stellar half-mass radius [kpc]',color="k",range=(0,20),histtype='step',bins=30)
    histsubplot(axravel[2],10**(gz2data['MEDIAN_SFR']),'SFR '+r'$[M_{\odot}/yr]$',color="k",range=(0,10),histtype='step',bins=30)
    histsubplot(axravel[3],gz2data['PETROMAG_MR'],r'$M_r$',color="k",histtype='step',range=(-26,-15),bins=30)


    fig.tight_layout()
    if savefig:
        plt.savefig('%s/hist_compare_gz2.pdf' % path)
    else:
        plt.show()

    return None

if __name__ == "__main__":

    make_metadata_table()
