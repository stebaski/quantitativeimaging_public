#import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.widgets import LassoSelector
from matplotlib import path
import imageio
from PIL import Image
import scipy.ndimage as sci
from scipy.optimize import curve_fit
from math import *
import time
import csv

#paths to 6 tissues imaged @ SYRMEP beamline Elettra
#several mono energies were used
#email stevan.vrbaski@phd.units.it for an access
base_path = 'data/'
paths = {'path_934' : [base_path + 'tissue_934/',[24,28,36]],
         'path_9365' : [base_path + 'tissue_9365/',[24,28,38]],
         'path_1057' : [base_path + 'tissue_1057/',[28,32,35]],
         'path_9547' : [base_path + 'tissue_9547/',[26,32,35,38]],
         'path_13826' : [base_path + 'tissue_13826/',[24,28,38]],
         'path_12320' : [base_path + 'tissue_12320/',[24,28,32,35,38]]}

#choose single mastectomy sample to analyze
tissue = list(paths)[2]
#full path
path_to_data = paths[tissue][0]
#choose energy level at which delineation will be performed
Es = paths[tissue][1]

#load spectral data
image_list = []
for E in Es:
    im = str(E) + 'keV.tif'
    image_list.append(imageio.imread(path_to_data+im))
image_list = np.array(image_list)

image = image_list[0]
h,w = np.shape(image)
array = np.zeros((h,w))

mp = {}
mpind = {}
rho_dict={}
Z_dict={}
PMMA_dict={}
Al_dict={}
indarr = []

#following is related to the interactive selection of ROIs
fig = plt.figure(figsize = (20,20))
ax1 = fig.add_subplot(121)
ax1.set_title('Select region:\nSelect unlimited number of ROIs for desired material.\nPress letter N to examine new material.')
ax1.imshow(image, cmap = 'Greys_r')
ax2 = fig.add_subplot(122)
ax2.set_title('Selected single material:')
msk = ax2.imshow(array, vmin = image.min(), vmax=image.max(), interpolation='nearest') #, cmap = 'Greys_r'

#pixel coordinates
x, y = np.meshgrid(np.arange(h), np.arange(w))
pix = np.vstack((x.flatten(), y.flatten())).T


def updateArray(array,indices):
    global indarr
    lin = np.arange(array.size)
    newArray = array.flatten()
    newArray[lin[indices]] = image.flatten()[lin[indices]]
    return newArray.reshape(array.shape), indarr.extend(lin[indices])


def refreshArray(event,i=[0]):
    global array, indarr, mp, mpind
    if event.key in ['N', 'n']:
        i[0]+=1
        print('New material examination.')
        mp['Material ' + str(i[0])] = array 
        mpind['Material ' + str(i[0])] = indarr
        array = np.zeros((h,w))
        indarr = []
        return i[0]

def onselect(verts):
    global array, pix
    p = path.Path(verts)
    ind = p.contains_points(pix, radius=1)
    array = updateArray(array, ind)[0]
    msk.set_data(array)
    fig.canvas.draw_idle()


#style   
lineprops = {'color': 'red','linewidth':2,'alpha':0.3}
lasso = LassoSelector(ax1, onselect,lineprops = lineprops, button = 1)
fig.canvas.mpl_connect('key_press_event', refreshArray)
plt.show()

#Gaussian function to be fitted in histogram space
def gaussian(k, x0, y0, A, su, sv, theta):
    (x,y)=k
    u = (x - x0) * cos(theta) + (y - y0) * sin(theta)
    v = (y - y0) * cos(theta) - (x - x0) * sin(theta)
    return A * np.exp(- (u * u / (2 * su * su) + v * v / (2 * sv * sv)))

#materials & methods Vrbaski et al. 2023
rho1, rho2 = 1.19, 2.7  #densities for PMMA and Al
a,b,c,d =   rho1 / (sqrt(1 + (rho1**2/rho2**2)) * rho2),  1 / sqrt(1 + (rho1**2/rho2**2)), -1 / sqrt(1 + (rho1**2/rho2**2)),  rho1 / (sqrt(1 + (rho1**2/   rho2**2)) * rho2)
u = a*d-b*c
M = np.array([[a,b],[c,d]])

#material decomposition
mus_PMMA = [] 
mus_Al = []
with open("PMMAAl.txt") as f:
    for line in f:
        entries = line.split(',')
        if float(entries[0]) in Es:
            mus_PMMA.append(float(entries[1]) * rho1)    
            mus_Al.append(float(entries[2]) * rho2)

PMMA_params = np.polyfit(Es,mus_PMMA, deg = 2)
Al_params = np.polyfit(Es,mus_Al, deg = 2)

def mu_PMMA(E):
    return PMMA_params[0] * E * E + PMMA_params[1] * E + PMMA_params[2]
def mu_Al(E):
    return Al_params[0] * E * E + Al_params[1] * E + Al_params[2]
def mu(E, PMMA, Al):
    return PMMA * mu_PMMA(E) + Al * mu_Al(E)


#calibration curves for rho/Z (round phantom)
def Z(xZ,n,p,q):
    return ((xZ-q)/p)**(1/(n-1))

def Rho(xRho,r,s):
    return ((xRho - s)/r)

#calibration
r,t = 0.28483749, 0.06521029
n,p,q = 4.38372806, 5.78568175e-4, -2.60135013


#material decomposition of ROIs - histogram space - centers of Gaussians
values = []
pcoor = {}
for tt,mat in enumerate(mp.keys()):
    
    t0 = time.time()
    PMMA = []
    Al = []
    for i in mpind[mat]:
        fit_params, cov_mat = curve_fit(mu, Es, image_list.reshape((len(Es),h*w))[:,i], p0 = [0, 0])
        PMMA.append(fit_params[0]), Al.append(fit_params[1])
    t1 = time.time()
    print('The decomposition of the ' +str(mat)+ ' ended after ' + str(t1-t0) + ' seconds')

    #optimizing histogram range and number of bins
    pixnum = len(mpind[mat]) #number of pixels in selected region/s
    #print("Number of pixels selected for "+str(mat)+" is ", pixnum)
    mPMMA = np.median(PMMA) #median value for PMMA
    mAl = np.median(Al) #median value for Al
    sPMMA = np.std(PMMA)
    sAl = np.std(Al)
    histo_rng = [[mPMMA-(abs(sPMMA)*3), mPMMA+(abs(sPMMA)*3)], [mAl-(abs(sAl)*3), mAl+(abs(sAl)*3)]] 
    print(mPMMA,mAl)
    #print('Histogram range for '+str(mat)+" is ",histo_rng)
    histo_bins = [np.int(sqrt(pixnum)), np.int(sqrt(pixnum))]
    #print('Number of bins range for '+str(mat)+" is ",histo_bins)
    xmax, xmin = max(histo_rng[0]), min(histo_rng[0])
    ymax, ymin = max(histo_rng[1]), min(histo_rng[1])
    nx, ny = histo_bins[0],histo_bins[1]
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x, y = np.linspace(xmin + dx/2, xmax - dx/2, nx), np.linspace(ymin + dy/2, ymax - dy/2, ny)
    X, Y = np.meshgrid(x, y)
    xData = np.vstack((X.flatten(), Y.flatten()))
    hist = plt.hist2d(PMMA, Al, bins = histo_bins, range = histo_rng,cmap = 'Greys_r'); #2D histogram
    #hist = np.histogram2d(PMMA, Al, bins = histo_bins, range = histo_rng); #2D histogram
    zData = hist[0].transpose().flatten()
    histogram = Image.fromarray(hist[0])

    #Computing the center
    p0 = [mPMMA,mAl,50,0.1,0.001,-0.12] #inital vlues 
    param_bounds = ([-3,-3,0,-1,-1,-0.2],[3,3,1000,1,1,0.2]) #to speed up the process - can be skipped
    try:
        params, cov_mat = curve_fit(gaussian, xData, zData, p0 = p0, bounds = param_bounds)
        print("Retrieved parameters: " + str(params))
    except Exception as exc:
        plt.figure()
        plt.imshow(histogram)
        plt.show()
        print(str(exc) + "- please check your histogram appearance and adjust the range and the number of bins, or try selecting a      larger image region")
    #rho/Zeff decomposition              
    v_p1 = M.dot([params[0],params[1]])
    v_rho=v_p1[0]
    v_z=v_p1[1]/v_p1[0]
    Zvalue = Z(v_z,n,p,q)
    Rhovalue = Rho(v_rho,r,t)
    mat_name = input("Name the selected tissue (e.g., adipose): ")
    rho_dict[str(mat_name)]=Rhovalue
    Z_dict[str(mat_name)]=Zvalue
    PMMA_dict[str(mat_name)]=params[0]
    Al_dict[str(mat_name)]=params[1]
    print('Decomposed values in ' + mat + ' are ' + 'PMMA: ' + str(params[0]) + ' Al: ' + str(params[1]) + " and median values are " + str(mPMMA) + " and " + str(mAl))
    print('Quantitative values in ' + mat + ' are ' + 'Rho: ' + str(Rhovalue) + ' Zeff: ' + str(Zvalue))

#write results of the analysis to .csv file
with open(base_path +'mastectomies_test_Nov22/' +tissue+".csv","a") as csvfile:
    fieldnames = list(PMMA_dict)
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames,delimiter=',')
    writer.writeheader()
    writer.writerow(PMMA_dict)
    writer.writerow(Al_dict)
    writer.writerow(rho_dict)
    writer.writerow(Z_dict)

















