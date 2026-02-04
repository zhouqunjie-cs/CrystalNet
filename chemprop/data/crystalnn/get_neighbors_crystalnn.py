from pymatgen.analysis.local_env import CrystalNN,EconNN

def get_neighbors_crystalnn(structure):
    cnn = CrystalNN()
    enn = EconNN()
    site_all_list, site_num_list = [], []
    
    #nnum = sum(structure.natoms)
    for i in range(len(structure.sites)):
        try:
            crystalnn = cnn.get_nn_info(structure,i)
        except:
            print("No neighbors by CrystalNN Methods, trying EconNN... ")
            crystalnn = enn.get_nn_info(structure,i)

        site_num = len(crystalnn)
#        if site_num == 0 :
#            crystalnn = enn.get_nn_info(structure,i)
        site_num_list.append(site_num)
        one_site_list = one_site_process(crystalnn)
        site_all_list.append(one_site_list)
    return site_all_list, max(site_num_list)

def one_site_process(crystalnn):
    site_list, site_list_t = [],[]
    for i in range(len(crystalnn)):
        for k in crystalnn[i].items():
            if k[0] == 'site':
                site_list_t = list(k[1:])
                site_list.append(site_list_t[0])
    return site_list


