
"""
LG DIFFRACTION GRATING CODE

Created: June 10, 2019
Author: Isabel Sacksteder
Last Modified: July 24th 2019

Note: Based on a code created by Isabel Fernandez and Evan Villafranca
"""

#%%
#BEGIN CODE

#import libraries
import  math
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

#begin timer
t00 = time.time()

#%%
#INPUTS

Superposition = False #Generating a superposition of two LG modes?

#LG mode integer inputs
p_input =  int(input('Enter radial index of LG mode, "p" integer: '))
if Superposition == True:
    p_0_input = int(input('Enter radial index of second LG mode, "p_0" integer: '))
else:
    p_0_input = 0

l_input = int(input('Enter azimuthal index of LG mode, "l" integer: '))
if Superposition == True:
    l_0_input = int(input('Enter azimuthal index of second LG mode, "l_0" integer: '))
else:
    l_0_input = 0
    
#%%
#CALCULATING FRINGE DENSITY
    
Rb_lambda = 633*10**(-9) #m, wavelength of laser light

theta_i = np.radians(15) #rad, angle of incidence onto DMD

prop_distance = 0.66 #m, distance to viewing plane

desired_spacing = 0.005 #m, desired seperarion between diffracted orders

theta_d = -(theta_i + np.arctan(desired_spacing/prop_distance)) #rad, angle of diffraction off DMD 

#theta_d = choose your own angle of diffraction

d = Rb_lambda/(np.sin(theta_i) + np.sin(theta_d)) #m, diffraction grating equation for proper fringe density  
k0 = (2*np.pi/d)

print "fringe size =", -d, "m,", -d/(7.6*10**-6), "pixels per fringe"
#%%
#LAGUERRE GAUSSIAN FUNCTIONS

#constants
L = 633*10**(-9) #m, wavelength of light
c = 3*10**(8) #m/s, speed of light
k = (2*np.pi)/L #1/m, wave number
w = 2*np.pi*c/L #rad/s, angular frequency
w0 = 1*10**-4 #m, waist
zR = np.pi*w0**2/L #m, rayleigh range

#radius of curvature
def R(z):
    return z + (zR**2)/z
#gouey phase
def G(z):
    return np.arctan(z/zR)
#width
def w(z):
    return w0*np.sqrt(1*(z**2)/(zR**2))
#define r in caresian coordinate
def r(x,y): 
    return np.sqrt(x**2+y**2)

#required normalizing factor
def A(p,l):
    return np.sqrt((2*math.factorial(p))/(np.pi*math.factorial(p+np.abs(l))))

#argument for laguerre polynomial
def x0(x,y,z):
    return 2*(r(x,y)/w(z))**2 

#Generalized Laguerre Polynomial - from original code
def L(l,p,x,y,z): 
    total=0
    for m in range(0,p+1):
        n=((-1)**m)*(math.factorial(p+l)/(math.factorial(p-m)*math.factorial(l+m)*math.factorial(m)))*(x0(x,y,z)**m)
        total=total+n
    return total

#full form LG beam
def LG1(x,y,z,t,p,l):
    return  A(p,l) * w0/w(z) * ((np.sqrt(2)*r(x,y))/(w(z)))**(np.abs(l))* np.exp((-r(x,y)**2)/(w(z)**2)) * np.exp(-1j*G(z)) *  np.exp(1j*l*np.arctan2(x,y)) * np.exp((1j*k*z**2)/(2*R(z))) * L(np.abs(l),p,x,y,z)

#propogation distance, match image to beam size
z0 = 0.50

#phase of LG superposition
def phi_super(x,y,p,p_0,l,l_0):
    return np.angle(LG1(x,y,z0,0,p_0,l_0)+LG1(x,y,z0,0,p,l))

#amplitude of LG beam
def LGamp(x,y,p,l):
    r3 = np.abs(LG1(x,y,z0,0,p,l))
    return ((1/np.amax(r3))*r3%2*np.pi/(2*np.pi)) 

#amplitude of LG superposition
def superamp(x,y,p,p_0,l,l_0):
    r2 = np.abs((LG1(x,y,z0,0,p_0,l_0)+LG1(x,y,z0,0,p,l)))
    return ((1/np.amax(r2))*r2%2*np.pi/(2*np.pi))

#%%
#DMD GRATING
    
def DMD_Grating(x,y,l,p,l_0,p_0):
    if Superposition == False:
        pixel_phase = l*np.arctan2(x,y) + k0*x 
        amplitude = LGamp(x,y,p,l)
        
    else:
        pixel_phase = phi_super(x,y,p,p_0,l,l_0) + k0*x  #phase at pixel (x,y)
        amplitude = superamp(x,y,p,p_0,l,l_0)
        
    phase = (pixel_phase%(2*np.pi))/(2*np.pi) #phase mod 2 pi in units of 2pi
    
    Grating = np.abs(amplitude*(phase)) #final pixel color value for intensity of electric field
    return Grating**2

#%%
#SMOOSH IMAGE
    
"""
dmd_resultion_tests.py
Created on Mon Jul  8 16:30:14 2019
This script generates images that help characterize the DMD's resolution
and aspect ratio.
@author: jdm
"""

#%%

# First define the dimensions of the screen, which has 1824 x 1140 mirrors
# according to the data sheet.
mpx = 7.6*10**-6 #scaling factor in meters per pixel

Nx =  1824 #x-dimension number of pixels on DMD
Ny =  1140 #y-dimension number of pixels on DMD
Nx_input = Nx/2
Ny_input = Ny

#%%

# Then create arrays of this size in order to create a mesh grid of the 
# corresponding resolution.  Ideally each element of this array would be in
# a one-to-one correspondence with the pixel array of the DMD.
x = np.arange(0, Nx)
y = np.arange(0, Ny)
xx, yy = np.meshgrid(x,y)

#%%

# What we really want, though, is a function that takes as its input another
# arbitrary function defined over a 2d space and returns an image to be sent 
# to the DMD.  The function should be normalized so that 0 is off and 1 is
# on, i.e. no intensity and full intensity.
def generate_dmd_image(my_function, x, y):
    # First plot the desired image
    plt.imshow(my_function(mpx*x,mpx*y), cmap='gist_gray')
    plt.title("Desired function")
    plt.show()
    
    x_logical = np.arange(0, (np.max(x)+1)/2)
    y_logical = np.arange(0, (np.max(y)+1))
    
    xxl, yyl = np.meshgrid(x_logical, y_logical)
    
    # Then generate the desired image
    dmd_image_array = (np.array(my_function(2*mpx*xxl, mpx*yyl), dtype=float)*255)
    # Since the function is normalized for intensities between 0 and 1, 
    # convert to 256 bit depth

    plt.imshow(dmd_image_array, cmap='gist_gray')
    plt.title("Image sent to the DMD")
    plt.show()
    
    #print np.amax((dmd_image_array/np.amax(dmd_image_array))*255)
    img = Image.fromarray((dmd_image_array/np.amax(dmd_image_array))*255)
    img.show()
    img.save("test0.tif")
    
#Offset the DMD grating function to account for misalignment from smooshing
def MyFunc(x,y):
    xoffset = mpx*Nx/2
    yoffset = mpx*Ny/2
    return (DMD_Grating(x-xoffset,y-yoffset,l_input,p_input,l_0_input,p_0_input))

# Then generate the desired image
generate_dmd_image(MyFunc, xx, yy)

#%%

#end timer
t10 = time.time()
total0 = t10-t00
print "time to run code =", total0, "seconds"