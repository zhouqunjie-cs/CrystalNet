from pymatgen.analysis.local_env import CrystalNN,EconNN
from pymatgen.util.coord import all_distances, get_angle
from itertools import combinations
import math
import numpy as np

def get_neighbors_crystalnn(structure):
    #cnn = CrystalNN()
    cnn = EconNN()
    site_all_list,bond_l,angle_l,dihedral_l = [], [], [], []
    #nnum = sum(structure.natoms)
    for i in range(len(structure.sites)):
        crystalnn = cnn.get_nn_info(structure,i)
        one_site_list = one_site_process(crystalnn)
        for site in one_site_list:
            c1 = np.array(site.coords)
            c2 = np.array(structure[i].coords)
            bond = np.linalg.norm(c1-c2)
            bond_l.append(bond)


        if len(one_site_list) > 1:
            for pair in combinations(one_site_list,2):
                v1 = pair[0].coords - structure[i].coords
                v2 = pair[1].coords - structure[i].coords
                #angle= get_angle(v1,v2,units="degrees")
                angle= get_angle(v1,v2,units="radians")
                angle_l.append(angle)
        
        if len(one_site_list) > 2:
            for triplet in combinations(one_site_list,3):
                v1 = triplet[1].coords - triplet[2].coords
                v2 = structure[i].coords - triplet[1].coords
                v3 = triplet[0].coords - structure[i].coords
                v23 = np.cross(v2,v3)
                v12 = np.cross(v1,v2)
                #dihedral = math.degrees(math.atan2(np.linalg.norm(v2) * np.dot(v1, v23), np.dot(v12, v23)))
                dihedral = math.atan2(np.linalg.norm(v2) * np.dot(v1, v23), np.dot(v12, v23))
                dihedral_l.append(dihedral)
        
        site_all_list.append(one_site_list)
    return site_all_list,bond_l, angle_l, dihedral_l

def one_site_process(crystalnn):
    site_list, site_list_t = [],[]
    for i in range(len(crystalnn)):
        for k in crystalnn[i].items():
            if k[0] == 'site':
                site_list_t = list(k[1:])
                site_list.append(site_list_t[0])
    return site_list


