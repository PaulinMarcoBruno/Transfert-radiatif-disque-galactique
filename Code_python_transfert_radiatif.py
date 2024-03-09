#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:15:51 2024

@author: paulin bruno
M1 physique fondamentale, Université de Paris Cité


"Transfert radiatif dans le disque galactique par la méthode de Monte-Carlo"

"""



import numpy as np
import numpy.random as rd 
import numpy.linalg as nl
import matplotlib.pyplot as plt
import time


def s(r_face,r_photon,n_unit):
    
    S = (r_face - r_photon)/n_unit
    
    return np.min(S)



def tau(rho,opac,s):
    
    return rho*opac*s



def norme(vect):
    
    return np.sqrt(np.sum(vect**2))

#fonction pour tirer de facon homogene un theta sur une sphère
def tirage_theta(cos_min,cos_max):
    
    u = rd.uniform(cos_min,cos_max)
    theta = np.arccos(u)
    
    return theta

#fonction pour passer de coordonnée sphérique (r,phi,theta) à cartésienne
def change_base(vect):
    
    r,phi,theta = vect
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    
    return np.array([x,y,z])
    
    
### fonctions pour definir le capteur

def centre_capteur(coord_centre_systeme,D,phi,theta):
    
    if type(coord_centre_systeme) != type(np.array([])):
        
        coord_centre_systeme = np.array(coord_centre_systeme)
    
    x_cap = D*np.cos(phi)*np.sin(theta)
    y_cap = D*np.sin(phi)*np.sin(theta)
    z_cap = D*np.cos(theta)
    
    coord_centre_capteur = np.array([x_cap,y_cap,z_cap]) + coord_centre_systeme
    
    return coord_centre_capteur

#definition du vecteur normal au plan, utile pour l'inclinaison du capteur
def vect_normal_plan(phi,theta):
    
    x = np.cos(phi)*np.sin(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(theta)
    
    vect = np.array([x,y,z])
    
    return vect
    

#definition de deux vecteurs formant une base du plan du capteur
#une relation est utilisé pour trouver le premier vecteur 
def base_plan_capteur(phi,theta,vect_normal):
    
    phi_prime =  phi
    x = np.cos(phi_prime)*np.sin(theta + np.pi/2) 
    y = np.sin(phi_prime)*np.sin(theta + np.pi/2)
    z = np.cos(theta + np.pi/2)
    
    n_capt_x = np.array([x,y,z])
    n_capt_x = n_capt_x/(np.sqrt(np.sum(n_capt_x**2)))
    n_capt_y = np.cross(vect_normal,n_capt_x) # produit vectoriel pour obtenir le 3 eme vecteure de base 
    n_capt_y = n_capt_y/(np.sqrt(np.sum(n_capt_y**2)))
    
    
    return n_capt_x,n_capt_y
    

#donne la matrice image du capteur
def capteur_image(D,phi,theta,coord_centre_systeme,phi_incline,theta_incline,list_coord_photon,list_n_photon,list_frequence_photon,xmax=50,xmin=-50,ymax=50,ymin=-50,resolution=1000):
    
    #definition du capteur, de son equation plan et son orientation par rapport au centre du systeme 
    centre = centre_capteur(coord_centre_systeme, D, phi, theta)
    
    n_plan = vect_normal_plan(phi_incline, theta_incline)
    a,b,c = n_plan
    
    n_capt_x,n_capt_y = base_plan_capteur(phi_incline, theta_incline, n_plan)
    
    const = n_plan[0]*centre[0] + n_plan[1]*centre[1] + n_plan[2]*centre[2]
    
    Mat_changement_base = nl.inv(np.array([[n_capt_x[0],n_capt_y[0],n_plan[0]],[n_capt_x[1],n_capt_y[1],n_plan[1]],[n_capt_x[2],n_capt_y[2],n_plan[2]]]))
    
    
    # collision des photon avec le capteur 
    list_coord_image = []
    list_nu_image = []
    
    for j in range(len(list_coord_photon)):
        
        X,Y,Z = list_coord_photon[j]
        nx,ny,nz = list_n_photon[j]
        freq = list_frequence_photon[j]
        
        if np.sum(nx + ny + nz) != 0:
            
            t = (const - a*X - b*Y - c*Z)/(a*nx + b*ny + c*nz)
            
            if (t>0):
                
                x_plan = X + nx*t
                y_plan = Y + ny*t
                z_plan = Z + nz*t
                
                x_plan -= centre[0]  
                y_plan -= centre[1]
                z_plan -= centre[2]
                
                coord_point = np.array([x_plan,y_plan,z_plan])
                
                coord_point_capt = (Mat_changement_base @ coord_point.T).T
                
                
                x_capt = coord_point_capt[0]
                y_capt = coord_point_capt[1]
                
                if round(coord_point_capt[2],8) != 0:
                    
                    print('ERREUR Z CAPT NON NUL')
                    # vu que l'on doit etre sur le plan on doit avoir cette composante = 0. or numpy garde toujours une precision donc on arrondi
                else :
                    
                    photon_coord_image = np.array([x_capt,y_capt])
                    
                    list_coord_image.append(photon_coord_image)
                    list_nu_image.append(freq)
   
    largeur_x = xmax - xmin
    largeur_y = ymax - ymin
    res_x = np.sqrt(resolution*(largeur_x/largeur_y))
    res_y = int(resolution/res_x)
    res_x = int(res_x)
    
    image = np.zeros((res_y,res_x))
    image_energie = np.zeros((res_y,res_x))
    
    # creation de la matrice image du capteur.
    for j in range(len(list_coord_image)):
        
        x_im,y_im = list_coord_image[j]
        freq_im = list_nu_image[j]
        
        if (x_im>xmin) and (x_im<xmax) and (y_im>ymin) and (y_im<ymax):
            
            x_im -= xmin
            y_im -= ymin
            
            ind_x = int(x_im*res_x/largeur_x)
            ind_y = int(y_im*res_y/largeur_y)
            
            image[ind_y,ind_x] += 1
            image_energie[ind_y,ind_x] += h*freq_im
    
    return image,image_energie    
    


### fonctions pour la trajectoire du photon

#fonction pour localiser la prochaine case de la matrice dans laquel va évoluer le photon
def localisation_case_inter(r_vect, x_grille, y_grille, z_grille,n_vect):
    
    x,y,z = r_vect
    nx,ny,nz = n_vect
    
    if (nz >= 0) :
        if((nx>=0) & (ny>=0)):
            for j in range(len(x_grille)-1):
            
                if ((x_grille[j]<=x) and (x<x_grille[j+1])):
                    
                    x_face = x_grille[j+1]
             
                if ((y_grille[j]<=y) and (y<y_grille[j+1])):
                    y_face = y_grille[j+1]
                
                if((z_grille[j]<=z) and (z<z_grille[j+1])):
                    z_face = z_grille[j+1]
            
            
            ind_x = int(x_face) -1
            ind_y = int(y_face) -1
            ind_z = int(z_face) -1
            
            
        elif (nx<0) & (ny>=0) :
            for j in range(len(x_grille)-1):
                
                if ((x_grille[j]<x) and (x<=x_grille[j+1])):
                    x_face = x_grille[j]
                
                if ((y_grille[j]<=y) and (y<y_grille[j+1])):
                    y_face = y_grille[j+1]
                    
                if((z_grille[j]<=z) and (z<z_grille[j+1])):
                    z_face = z_grille[j+1]
                    
            ind_x = int(x_face) 
            ind_y = int(y_face) -1
            ind_z = int(z_face) -1
            
            
        elif (nx>=0) & (ny<0) :
            for j in range(len(x_grille)-1):
                    
                if ((x_grille[j]<=x) and (x<x_grille[j+1])):
                    x_face = x_grille[j+1]
                    
                if ((y_grille[j]<y) and (y<=y_grille[j+1])):
                    
                    y_face = y_grille[j]
                    
                if((z_grille[j]<=z) and (z<z_grille[j+1])):
                    z_face = z_grille[j+1]
                
            ind_x = int(x_face) -1
            ind_y = int(y_face)
            ind_z = int(z_face) -1
            
            
        elif (nx<0) & (ny<0) :
            for j in range(len(x_grille)-1):
                    
                if ((x_grille[j]<x) and (x<=x_grille[j+1])):
                    x_face = x_grille[j]
                    
                if ((y_grille[j]<y) and (y<=y_grille[j+1])):
                    y_face = y_grille[j]
                    
                if((z_grille[j]<=z) and (z<z_grille[j+1])):
                    z_face = z_grille[j+1]
                
            ind_x = int(x_face)
            ind_y = int(y_face)
            ind_z = int(z_face) -1
            
    if (nz < 0) :
        if((nx>=0) & (ny>=0)):
            for j in range(len(x_grille)-1):
            
                if ((x_grille[j]<=x) and (x<x_grille[j+1])):
                    x_face = x_grille[j+1]
             
                if ((y_grille[j]<=y) and (y<y_grille[j+1])):
                    y_face = y_grille[j+1]
                
                if((z_grille[j]<z) and (z<=z_grille[j+1])):
                    z_face = z_grille[j]
            
            
            ind_x = int(x_face) -1
            ind_y = int(y_face) -1
            ind_z = int(z_face)
            
            
        elif (nx<0) & (ny>=0) :
            for j in range(len(x_grille)-1):
                
                if ((x_grille[j]<x) and (x<=x_grille[j+1])):
                    x_face = x_grille[j]
                
                if ((y_grille[j]<=y) and (y<y_grille[j+1])):
                    y_face = y_grille[j+1]
                    
                if((z_grille[j]<z) and (z<=z_grille[j+1])):
                    z_face = z_grille[j]
                    
            ind_x = int(x_face)
            ind_y = int(y_face) -1
            ind_z = int(z_face)
            
            
        elif (nx>=0) & (ny<0) :
            for j in range(len(x_grille)-1):
                    
                if ((x_grille[j]<=x) and (x<x_grille[j+1])):
                    x_face = x_grille[j+1]
                    
                if ((y_grille[j]<y) and (y<=y_grille[j+1])):
                    
                    y_face = y_grille[j]
                    
                if((z_grille[j]<z) and (z<=z_grille[j+1])):
                    z_face = z_grille[j]
                
            ind_x = int(x_face) -1
            ind_y = int(y_face)
            ind_z = int(z_face)
            
            
        elif (nx<0) & (ny<0) :
            for j in range(len(x_grille)-1):
                    
                if ((x_grille[j]<x) and (x<=x_grille[j+1])):
                    x_face = x_grille[j]
                    
                if ((y_grille[j]<y) and (y<=y_grille[j+1])):
                    y_face = y_grille[j]
                    
                if((z_grille[j]<z) and (z<=z_grille[j+1])):
                    z_face = z_grille[j]
                
            ind_x = int(x_face)
            ind_y = int(y_face)
            ind_z = int(z_face)
        
    return np.array([x_face,y_face,z_face]),np.array([ind_x,ind_y,ind_z])







            

    
#fonction qui tire une direction aléatoire avec un angle donnee par rapport à l'axe vertical

def deviation(alpha):
    
    phi_dev = rd.uniform(0,2*np.pi)
    theta_dev = alpha
    v_dev = np.array([1,phi_dev,theta_dev])
    
    return v_dev

# fonction qui donne l'angle entre deux direction
def angle_rotation(v1,v2):
    
    cos_angle_rot = np.dot(v1,v2)/(norme(v1)*norme(v2))
    angle_rot = np.arccos(cos_angle_rot) 
    
    return angle_rot
    



#fonction qui applique une rotation 
def rotation(vect,vect_rot_1,vect_rot_2,angle_rot):
    
    #definition de l'axe de rotation par un vecteur unitaire
    axe_rotation = np.cross(vect_rot_1,vect_rot_2)
    
    axe_rotation = axe_rotation/norme(axe_rotation)
    
    #definition de la matrice de rotation de vect_rot_1 vers vect_rot_2
    ux,uy,uz = axe_rotation 
    c=np.cos(angle_rot)
    s=np.sin(angle_rot)
    
    P=np.array([[ux**2,ux*uy,ux*uz],
                [ux*uy,uy**2,uy*uz],
                [ux*uz,uy*uz,uz**2]])
    
    I=np.array([[1,0,0],
                [0,1,0],
                [0,0,1]])
    
    Q=np.array([[0,-uz,uy],
                [uz,0,-ux],
                [-uy,ux,0]])
    
    R = P + c*(I - P) + s*Q
    
    vect_final = R@vect
    
    return vect_final

#fonction qui dévie un vecteur aléatoirement autour d'une direction initial avec un angle donné
def scattering_alpha(alpha,n_ini):
    
    v_dev = change_base(deviation(alpha))
    z=np.array([0,0,1])
    theta_rot = angle_rotation(z,n_ini)
    n_dev = rotation(v_dev,z,n_ini,theta_rot)
    
    return n_dev



# fonction de probabilité de Rayleigh (diffusion dans l'atmosphere) 
def P_atmo(cos_theta):
    
    return (3/8)*(1+cos_theta**2)

#tirage d'un angle suivant la probabilité de Rayleigh par monte carlo
def Monte_carlo_atmo():
    
    max_val = 0.75
    u=1
    verif=0
    
    while(u>verif):
        
        x = rd.uniform(-1,1)
        u = rd.uniform(0,max_val)
        verif = P_atmo(x)
    
    return np.arccos(x)


def Mie(x,g):
    num = 1 - g**2
    denu = 1+g**2 - 2*g*x
    return num/denu

def Monte_carlo_mie(g):
    t = np.linspace(-1,1,10000)
    max_val = np.float128(np.max(Mie(t,g)))
    
    u=1
    verif=0
    
    while(u>verif):
        
        x = rd.uniform(-1,1)
        u = rd.uniform(0,max_val)
        verif = Mie(x,g)
    
    return np.arccos(x)



    

########## fonction qui calcul la trajectoire d'un photon dans le systeme
# /!\ il faut mettre une matrice dont les élément au bord sont nul, il ne seront pas pris en compte 
def traj_inter_opti(n_ini,r_ini,taille_grille = 21,step = 1,matrice_pk = 0,probabilite_scattering=3/4,fonction_de_phase = 'isotropic',g=1):
    
    if type(matrice_pk) == type(0): #matrice info contient les info sur les cases du milieu
        
        mat = np.zeros((taille_grille,taille_grille,taille_grille))
        
    else :
        
        mat = matrice_pk
    
    n_spherique = n_ini
    n = change_base(n_ini)
    
    if (np.sum(n**2) != 1):
        
        n = n/np.sqrt(np.sum(n**2))

    r = r_ini
    
    x_grille = np.arange(0,taille_grille,step)
    y_grille = np.copy(x_grille)
    z_grille = np.copy(x_grille)
    ind = 1 
    
    xface_max = x_grille[-1]
    yface_max = xface_max
    zface_max = xface_max
    
    ind_lim = np.shape(mat)[0] -1 #indice limite du systeme
    
    while(ind != 0):
        
        tau_run = 0
        tau_react = -np.log(1-rd.uniform())
        
        while(tau_run < tau_react):
            
            r_face,ind_list = localisation_case_inter(r,x_grille,y_grille,z_grille,n)
            section = s(r_face,r,n)
        
            indx,indy,indz = ind_list
            
            pk_cell = mat[indz,indy,indx]
            tau_cell = section*pk_cell
        
            tau_run += tau_cell
            
            #condition pour les cas ou le photon va sortir de lu systeme
            condx0 = (indx==0) and (round(r[0] + section*n[0],12) == 0)
            condxmax = (indx==ind_lim) and (round(r[0] + section*n[0],12) == xface_max)
            condy0 = (indy==0) and (round(r[1] + section*n[1],12) == 0)
            condymax = (indy==ind_lim) and (round(r[1] + section*n[1],12) == yface_max)
            condz0 = (indz==0) and (round(r[2] + section*n[2],12) == 0)
            condzmax = (indz==ind_lim) and (round(r[2] + section*n[2],12) == zface_max)
        
            condx = condx0 or condxmax
            condy = condy0 or condymax
            condz = condz0 or condzmax
        
       
            if (condx or condy or condz):
                
                #bug si jamais il y a diffusion dans la case limite 
                tau_cell = 0 #pas de diffusion possible ni d'absorption
                section = s(r_face,r,n)
                r = r + section *n
                
                return r,n
            
            r = r + section*n
                
            
        prob_scat = probabilite_scattering  # si reaction => proba de scatter et pas absorbation
        
        scat_abs = rd.uniform()
        tau_run -= tau_cell
        
        if (scat_abs < prob_scat):
            
            ds = (tau_react-tau_run)/pk_cell
            r = r + ds*n
            
            if (fonction_de_phase == 'Rayleigh') :
                
                alpha = Monte_carlo_atmo()
                n= scattering_alpha(alpha,n)
            
            elif(fonction_de_phase == 'isotropic') : 
                
                alpha = tirage_theta(-1,1)
                n = scattering_alpha(alpha,n)
                
            elif(fonction_de_phase == 'Mie'):
                
                alpha = Monte_carlo_mie(g)
                n= scattering_alpha(alpha,n)
            
            else : 
                
                print('\n fonction de phase doit etre "Rayleigh", "Mie" ou "isotropic"','\n\n')
                return None
            
            
        else :
            
            ds = (tau_react-tau_run)/pk_cell
            r = r + ds*n
            n = np.array([0,0,0])
           
            return  r,n
            
            
        

            





def sphere_system(taille,pk):
    
    Mat_syst = np.zeros((taille,taille,taille))
    rayon = (taille-2)/2
    milieu = (taille)/2 
    
    for z in range(taille):
        for y in range(taille):
            for x in range(taille):
                
                r = np.sqrt((x-milieu)**2 + (y-milieu)**2 + (z-milieu)**2)
                cond = (rayon > r)
                
                if cond : 
                
                    Mat_syst[z,y,x] = pk
                
    return Mat_syst

def disc_system(taille,pk):
    
    Mat_syst = np.zeros((taille,taille,taille))
    milieu = taille/2
    rmax = (taille/2) -1
    rmin = (taille/2.5)
    
    for z in range(taille):
        for y in range(taille):
            for x in range(taille):
                
                r = np.sqrt((x-milieu)**2 + (y-milieu)**2)
                
                if (r<rmax) and (r>rmin) and (z<(taille/2 + 2)) and (z>(taille/2 - 2)):
                    
                    Mat_syst[z,y,x]=pk
                    
    return Mat_syst
    


# tirer un lambda selon une distribution de planck

h = 6.626e-34  
kb = 1.38e-23  
c = 3e8  

def planck(lam,T):
    fact = (2*h*c**2) / (lam**5)
    fac_exp = (h*c) / (lam*kb*T)
    denu = np.exp(fac_exp) - 1
    return fact / denu


def Monte_carlo_planck(T,lam_min,lam_max):
    t = np.linspace(lam_min,lam_max,10000)
    planck_val = planck(t,T)
    
    
    max_planck = np.max(planck_val)
    
    
    verif = True
    while(verif):
        x = rd.uniform(lam_min,lam_max)
        
        u = rd.uniform(0,max_planck)
        
        verif = u>planck(x,T)
        
    
    return x


def histogramme_photon(data,bins,xmax,xmin,n_list):
    dx = (xmax-xmin)/bins
    hist = np.zeros(bins)
    abcisse = np.arange(xmin,xmax,dx)
    for j in range(len(data)):
        norme_p = norme(n_list[j])
        if norme_p != 0:
            x = data[j]
            
            indice = int((x-xmin)/dx)
            
            if (indice >= 0) and (indice < (bins - 1)):
                
                hist[indice]+=1
    return abcisse,hist

########################################################################################
########################################################################################
########################################################################################
########################################################################################


# EXEMPLE UTILISATION DU CODE

#definition des coordonnee d'un astres et sa temperature (type etoile, simule par un point)
R = np.array([5,5,5])
T = 5000

g = 0.3
#nombre de photon simule
N = 10000


#on définit les bornes de longueur d'onde 
lam_min = 1e-16
lam_max = 1e-5


#definition du system
taille =10
Matrice_system_100_1000 = disc_system(taille,1)
Matrice_system_1000_2000 = disc_system(taille,1)



#creation de list de données
#ici on sépare en 2 plage de longueur d'onde

list_coord_photon_100_1000 = []
list_n_photon_100_1000 = []
list_nu_photon_100_1000 = []
list_lambda_photon_100_1000 = []

list_coord_photon_1000_2000 = []
list_n_photon_1000_2000 = []
list_nu_photon_1000_2000 = []
list_lambda_photon_1000_2000 = []

for j in range(N):
    
    if j%500 == 0:
        print(j)
    
    phi_ini = rd.uniform(0,2*np.pi)
    theta_ini = tirage_theta(-1,1)
    n_ini = np.array([1,phi_ini,theta_ini])
    lam_photon = Monte_carlo_planck(T, lam_min, lam_max)
    
    
    if lam_photon < 1000e-9 :
        r_fin,n_fin = traj_inter_opti(n_ini, R, taille_grille=taille+1,matrice_pk=Matrice_system_100_1000,probabilite_scattering = 1,fonction_de_phase='Mie',g=g)
        list_coord_photon_100_1000.append(r_fin)
        list_n_photon_100_1000.append(n_fin)
        list_nu_photon_100_1000.append(c/lam_photon)
        list_lambda_photon_100_1000.append(lam_photon)
        
    else : 
        r_fin,n_fin = traj_inter_opti(n_ini, R, taille_grille=taille+1,matrice_pk=Matrice_system_1000_2000,probabilite_scattering = 1,fonction_de_phase='Mie',g=g)
        list_coord_photon_1000_2000.append(r_fin)
        list_n_photon_1000_2000.append(n_fin)
        list_nu_photon_1000_2000.append(c/lam_photon)
        list_lambda_photon_1000_2000.append(lam_photon)
        
        
list_coord_photon = list_coord_photon_100_1000 + list_coord_photon_1000_2000
list_n_photon = list_n_photon_100_1000 + list_n_photon_1000_2000
list_nu_photon = list_nu_photon_100_1000 + list_nu_photon_1000_2000
list_lambda_photon = list_lambda_photon_100_1000 + list_lambda_photon_1000_2000

#calcul de l'image
D=100
phi = 0
theta = 0
cible = R
borne = 1000
res = 10000


image_tot,image_E_tot = capteur_image(D, phi, theta, R, phi, theta, list_coord_photon, list_n_photon, list_nu_photon,xmax=borne,xmin=-borne,ymax=borne,ymin=-borne,resolution=res)
image_100_1000,image_E_100_1000 = capteur_image(D, phi, theta, R, phi, theta, list_coord_photon_100_1000, list_n_photon_100_1000, list_nu_photon_100_1000,xmax=borne,xmin=-borne,ymax=borne,ymin=-borne,resolution=res)
image_1000_2000,image_E_1000_2000 = capteur_image(D, phi, theta, R, phi, theta, list_coord_photon_1000_2000, list_n_photon_1000_2000, list_nu_photon_1000_2000,xmax=borne,xmin=-borne,ymax=borne,ymin=-borne,resolution=res)


#affichage de l'image
graph ,(im1) = plt.subplots(1,1)
plt.title('nombre total de photon')
img = im1.imshow(image_tot,cmap = 'gray')
im1.invert_yaxis()
cbar = graph.colorbar(img,ax=im1)
cbar.set_label('Nombre de photons', rotation=270, labelpad=15)
plt.show()


graph ,(im1) = plt.subplots(1,1)
plt.title('nombre de photon pour lambda entre 100 nm et 1000 nm')
img = im1.imshow(image_100_1000,cmap = 'gray')
im1.invert_yaxis()
cbar = graph.colorbar(img,ax=im1)
cbar.set_label('Nombre de photons', rotation=270, labelpad=15)
plt.show()

graph ,(im1) = plt.subplots(1,1)
plt.title('nombre de photon pour lambda entre 1000 nm et 2000 nm')
img = im1.imshow(image_1000_2000,cmap = 'gray')
im1.invert_yaxis()
cbar = graph.colorbar(img,ax=im1)
cbar.set_label('Nombre de photons', rotation=270, labelpad=15)
plt.show()

graph ,(im1) = plt.subplots(1,1)
plt.title('Energie total')
img = im1.imshow(image_E_tot,cmap='grey')
im1.invert_yaxis()
cbar = graph.colorbar(img,ax=im1)
cbar.set_label('E (J)', rotation=270, labelpad=15)
plt.show()


x,y =   histogramme_photon(list_lambda_photon,20,lam_max,lam_min,list_n_photon)

plt.figure()

plt.plot(x,y)

plt.show()



    
    