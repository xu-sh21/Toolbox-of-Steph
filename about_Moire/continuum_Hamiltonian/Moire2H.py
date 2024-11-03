'''Program for calculating the eigenvalues of the continuum Hamiltonian of twisted MoTe2 and plotting the bands!'''
# Author: xu-sh21@mails.tsinghua.edu.cn
# Date: 2024.10.29
# Description: The model of this script is from the paper of Pro. Wang. For more information, please read ther paper: http://arxiv.org/abs/2304.11864.

############################################################################################################################################################################
import ast
import yaml
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from CONSTANT import *
############################################################################################################################################################################

class Moire2H():
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.hsp = {}
        # example: self.hsp = {1: ['Gamma', 'M'], 2: ['M', 'K']}
        self.expand_k = []
        # example: self.expand_k = [(0,0), (1,0), (0,1), (1,1)]
        self.density= 30 # Default density.
        self.kpoints_all_selected = None
        self.kpoints_all_coef = None


# +-------------------+
# | Some Decorations. |
# +-------------------+
    def begin(self):
        print('''
#############################################################################################################################
#         MM       MM           OOOOOOOOOO         III     RRRRRRR        EEEEEEEEEEEE      222222         HH          HH   #
#       MM MM     MM MM       OO          OO               RR     RR      EE              22       22      HH          HH   #
#      MM  MM    MM   MM     OO            OO      III     RR       RR    EE            22          22     HH          HH   #
#     MM    MM   MM   MM    OO              OO     III     RR     RR      EEEEEEEEEE     22       22       HHHHHHHHHHHHHH   #
#    MM     MM  MM     MM   OO              OO     III     RR RR          EEEEEEEEEE            22         HHHHHHHHHHHHHH   #
#   MM      MM MM       MM   OO            OO      III     RR  RR         EE                  22           HH          HH   #
#  MM       MMMM        MM    OO          OO       III     RR    RR       EE                 22            HH          HH   #
#  MM        MM         MM      OOOOOOOOOO         III     RR       RR    EEEEEEEEEEE    22222222222       HH          HH   #
#############################################################################################################################
''')
    

    def info_print(self):
        print('''
#######################################################################################################################################################
Program for calculating the eigenvalues of the continuum Hamiltonian of twisted MoTe2 and plotting the bands!
Author: xu-sh21@mails.tsinghua.edu.cn
Date: 2024.10.29
Description: The model of this script is from the paper of Pro. Wang. For more information, please read ther paper: http://arxiv.org/abs/2304.11864.
#######################################################################################################################################################
''')


# +-----------------------------------------------------+
# | Parse the config.yaml to get necessary information. |
# +-----------------------------------------------------+
    def parse_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)

            # Read information about kpoints.
            k_points = config['kpoints']
            self.hsp = k_points['hsp']
            self.num_path = len(self.hsp)
            self.k_loop = k_points['k_loop']
            self.density = k_points['density']
            self.k_path = self.gen_kpath()
            # Read information about plot information.
            self.plot_mode = config['plot']['plot_mode']
            self.emin = config['plot']['emin']


    def gen_kpath(self):
        # This function is to generate the k-path along the high-symmetry paths provided by self.hsp.
        k_path = []
        for i in range(1, self.num_path + 1):
            if i == 1:
                k_path.append(self.hsp[i][0])
                k_path.append(self.hsp[i][1])
            else:
                k_path.append(self.hsp[i][1])
        return k_path


# +-------------------------------------------------------------------------------+
# | Choose the k points from the k-space along some high-symmetry points in fMBZ. |
# +-------------------------------------------------------------------------------+
    def set_k(self):
        self.preprocess()
        self.set_k_within_fMBZ()
        self.set_k_expand()
    

    def preprocess(self):
        # This function is to convert the loop k points to a list of combinations.
        axis_arr = np.arange(-np.abs(self.k_loop), np.abs(self.k_loop) + 1)
        X, Y = np.meshgrid(axis_arr, axis_arr)
        self.expand_k = np.vstack((X.ravel(), Y.ravel())).T
        # print(self.expand_k)
        
        mask = ~np.all(self.expand_k == [0, 0], axis=1)
        self.expand_k = self.expand_k[mask]
        # print(self.expand_k)
        self.group_num = self.expand_k.shape[0] + 1 # Number of member of each group should include (0,0)


    def set_k_within_fMBZ(self):
        # This function is to set k points along some high-symmetry paths provided by self.hsp.
        # Density = 35 is default value.
        self.k_lengths = [np.linalg.norm(self.find_coord(self.hsp[i][1]) - self.find_coord(self.hsp[i][0])) for i in range(1, self.num_path + 1)]
        self.point_num = np.int_(np.round(np.asarray(self.k_lengths) * self.density))
        # print(self.point_num)
        self.point_num_tot = sum(self.point_num)

        self.kpoints_all_selected = np.zeros((self.point_num_tot, self.group_num, 2))
        self.kpoints_all_coef = np.zeros((self.point_num_tot, self.group_num, 2), dtype=int)
        # print(self.kpoints_all_selected.shape)
        start_label = 0
        end_label = self.point_num[0]
        for i, (path, points) in enumerate(self.hsp.items()):
            t = np.linspace(0, 1, self.point_num[i])
            points = self.find_coord(self.hsp[i+1][0]) + (self.find_coord(self.hsp[i+1][1]) - self.find_coord(self.hsp[i+1][0])) * t[:, np.newaxis]
            # print(points.shape)
            self.kpoints_all_selected[start_label: end_label, :, :] = points[:, None, :] * coef_G 
            # print(start_label, end_label)
            self.kpoints_all_coef[start_label: end_label, 0, :] = np.zeros((2,), dtype=int)
            self.kpoints_all_coef[start_label: end_label, 1: , :] = self.expand_k
            # print(self.kpoints_all_coef[0,:,:])
            start_label += self.point_num[i]
            if i != len(self.point_num)-1:
                end_label += self.point_num[i+1]
            else:
                end_label = -1

    

    def find_coord(self, k):
        Hsp_dict = {
            'Gamma': Gamma_unit,
            'M'    : M_unit,
            'K'    : K_b_unit,
            'K_'   : K_t_unit
        }
        return Hsp_dict[k]


    def set_k_expand(self):
        # This function is to expand k points out of fMBZ by self.expand_k.
        G_1 = G_vec[1]
        G_2 = G_vec[2]
        for i, [m_1, m_2] in enumerate(self.expand_k):
            self.kpoints_all_selected[:, i+1, :]  += m_1 * G_1 + m_2 * G_2

    
 
# +---------------------------------------------------------+
# | Construct the Hamiltonian matrix for a certain k point. |
# +---------------------------------------------------------+
    def bulid_H(self):
        print("----------------------------------------------------------------")
        print("Begin buliding Hamiltonian matrix!")
        begin = datetime.now()
        print(f"Begin time:{begin}")

        self.Hk_dim = self.group_num * 2
        self.H_tot = np.zeros((self.point_num_tot, self.Hk_dim, self.Hk_dim), dtype=np.complex_)
        for i in tqdm(range(self.point_num_tot)):
            self.calculate_H_k(i)
        
        print("Finish buliding Hamiltonian matrix!")
        end = datetime.now()
        print(f"End time:{end}")
        print(f"Time cost:{end - begin}")
        print("----------------------------------------------------------------")


    def calculate_H_k(self, indices):
        k_group = self.kpoints_all_selected[indices, :, :]
        k_group_comb = self.kpoints_all_coef[indices, :, :]

        # Build diagonal parts of H_k.
        H_b = np.zeros((self.group_num, self.group_num), dtype=np.complex_)
        H_t = np.zeros((self.group_num, self.group_num), dtype=np.complex_)
        H_b, H_t = self.construct_Hb_Ht(H_b, H_t, k_group, k_group_comb)
        
        # Build non-diagonal parts of H_k.
        Delta_T = np.zeros((self.group_num, self.group_num), dtype=np.complex_)
        Delta_T_dagger = np.zeros((self.group_num, self.group_num), dtype=np.complex_)
        self.construct_deltaT(Delta_T, k_group_comb)
        Delta_T_dagger = Delta_T.T

        # Build H_k.
        H_k = np.block([
            [H_b,        Delta_T],
            [Delta_T_dagger, H_t]
        ])
        self.H_tot[indices, :, :] = H_k


    def construct_Hb_Ht(self, Hb, Ht, k_group, k_group_comb):
        # Construct the diagoanl part od Hb and Ht.
        self.calc_kinetic(Hb, Ht, k_group) 

        # Construct the non-diagoanl part od Hb and Ht.
        H_b_nondiag = np.zeros((self.group_num, self.group_num), dtype=np.complex_)
        self.calc_Hb_nondiag(self.group_num, k_group_comb, H_b_nondiag)
        H_t_nondiag = H_b_nondiag.T
        Hb += H_b_nondiag
        Ht += H_t_nondiag

        return Hb, Ht


    def calc_kinetic(self, Hb, Ht, k_group):
        k_diffs_b = k_group - K_b
        k_diffs_t = k_group - K_t

        kinetic_b = Kinetic_Coef * np.sum(k_diffs_b**2, axis=1)
        kinetic_t = Kinetic_Coef * np.sum(k_diffs_t**2, axis=1)

        np.fill_diagonal(Hb, kinetic_b)
        np.fill_diagonal(Ht, kinetic_t)


    def calc_Hb_nondiag(self, dim, k_group_comb, H_nondiag):
        m_1, m_2 = k_group_comb[:, 0], k_group_comb[:, 1]
        n_1, n_2 = k_group_comb[:, 0], k_group_comb[:, 1]

        index_matrix = m_1[None, :] - n_1[:, None] + m_2[None, :] * 1j - n_2[:, None] * 1j
        conditions = {
    1: 1 + 0j,  # n_1 + 1 == m_1 and n_2 == m_2
    2: -1 + 1j, # n_1 - 1 == m_1 and n_2 + 1 == m_2
    3: 0 - 1j,  # n_1 == m_1 and n_2 - 1 == m_2
    4: -1 - 0j, # n_1 - 1 == m_1 and n_2 == m_2
    5: 1 - 1j,  # n_1 + 1 == m_1 and n_2 - 1 == m_2
    6: 0 + 1j   # n_1 == m_1 and n_2 + 1 == m_2
}
        for condition, value in conditions.items():
            if condition in (1, 2, 3):
                H_nondiag[index_matrix == value] = Exp_psi_plus
            elif condition in (4, 5, 6): 
                H_nondiag[index_matrix == value] = Exp_psi_minus
    

    def construct_deltaT(self, Delta_T, k_group_comb):
        m_1, m_2 = k_group_comb[:, 0], k_group_comb[:, 1]
        n_1, n_2 = k_group_comb[:, 0], k_group_comb[:, 1]

        index_matrix = m_1[None, :] - n_1[:, None] + m_2[None, :] * 1j - n_2[:, None] * 1j
        conditions = {
    1: 0 + 0j,  # n_1 == m_1 and n_2 == m_2
    2: 0 - 1j, # n_1 == m_1 and n_2 - 1 == m_2
    3: 1 - 1j,  # n_1 + 1 == m_1 and n_2 - 1 == m_2
}
        for condition, value in conditions.items():
            Delta_T[index_matrix == value] = w


# +------------------------------------------------------------+
# | Calculate and save the eigenvalues of Halmiltonian matrix. |
# +------------------------------------------------------------+
    def is_hermitian(self):
            H_dagger = np.conjugate(np.transpose(self.H_tot, (0, 2, 1)))
            judge = np.allclose(self.H_tot, H_dagger)
            if judge == True:
                pass
            else:
                raise ValueError('The calculated Hamiltonian matrix is not hermitian!')
            

    def cal_eigen(self):
        print("Begin calculation of the eigenvalues of the Halmiltonian matrix!")
        begin = datetime.now()
        print(f"Begin time:{begin}")

        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.H_tot)

        print("End calculation of the eigenvalues of the Halmiltonian matrix!")
        end = datetime.now()
        print(f"End time:{end}")
        print(f"Time cost:{end - begin}")
        print("----------------------------------------------------------------")
        # print(self.eigenvalues)
        # print(self.eigenvectors.shape)


    def process_eigen(self):
        tolerance = 1e-5
        self.processed_eigenvalues = np.array(self.eigenvalues, dtype=np.complex128)
        for i in range(self.point_num_tot):
            if np.all(np.abs(self.processed_eigenvalues[i,:].imag) < tolerance):
                self.processed_eigenvalues[i,:] = self.processed_eigenvalues[i,:].real
# +---------------------------------------------+
# | Calculate the form factor of twisted MoTe2. |
# +---------------------------------------------+
    def calc_u(self):
        # TODO
        pass


# +---------------------------------------------------+
# | Plot the band and form factors of twisted MoTe2. |
# +---------------------------------------------------+
    def plot_band(self):
        plt.figure(figsize=(5,8))
        k_points = np.arange(self.point_num_tot)
        self.eigen_sorted = np.sort(self.processed_eigenvalues, axis=1)
        self.spec_kcoord = [0,]
        k_points, self.eigen_sorted, self.spec_kcoord = self.eigen_delete(k_points, self.eigen_sorted, self.spec_kcoord)
        for i in range(self.Hk_dim):
            energy_band = self.eigen_sorted[:, i]
            mask = energy_band > self.emin
            # plt.plot(k_points[mask], energy_band[mask], label=f'Band {i+1}')
            plt.plot(k_points[mask], energy_band[mask])
        
        plt.xlabel('k-point')
        plt.ylabel('Energy(meV)')
        plt.title('Energy Band Structure')
        plt.legend()

        
        # print(self.spec_kcoord)
        # print(self.k_path)
        special_k_points = {self.spec_kcoord[i]: self.k_path[i] for i in range(len(self.k_path))}
        # special_k_points = {
        #     0: 'G',
        #     20: 'M',
        #     40: 'K', 
        # }
        plt.xlim(k_points[0], k_points[-1])
        plt.ylim(bottom=self.emin)
        for point in special_k_points.keys():
            plt.axvline(x=point, color='k', linestyle='--', linewidth=0.5)
        plt.xticks([point for point in special_k_points.keys()], [label for label in special_k_points.values()])
        plt.savefig(f'./fig/dens={self.density}loop={self.k_loop}min={self.emin}.png', dpi=300) 


    def eigen_delete(self, kpoints, eigen, k_coord):
        lengths_list = self.point_num
        deleted_points = [sum(lengths_list[:i]) for i in range(1, len(lengths_list))]
        for i in range(1, len(lengths_list)+1):
            k_coord.append(sum(lengths_list[:i])-i)
        mask = np.ones(eigen.shape[0], dtype=bool)
        mask[deleted_points] = False
        eigen = eigen[mask]
        kpoints = kpoints[mask]

        return kpoints, eigen, k_coord


    def plot_u(self):
        # TODO

        pass


    def plot_all(self):
        if self.plot_mode == 'band':
            self.plot_band()
        elif self.plot_mode == 'form_factor':
            self.plot_u()
        elif self.plot_mode == 'both':
            self.plot_band()
            self.plot_u()
        else:
            raise TypeError("ERROR! Incorrect setting about plot mode!")


# +---------------+
# | Full process. |
# +---------------+
    def Hamiltonian(self):
        self.begin()
        self.info_print()
        self.parse_config()
        self.set_k()
        self.bulid_H()
        self.is_hermitian()
        self.cal_eigen()
        self.process_eigen()
        self.calc_u()
        self.plot_all()




test = Moire2H('config.yaml')
test.Hamiltonian()
# test.parse_config()
# test.preprocess()
# test.set_k_within_fMBZ()
# print(test.kpoints_all_selected)
# test.set_k_expand()
# print(test.kpoints_all_selected)
# print(test.kpoints_all_coef)
# print(test.H_tot.shape)
# print(test.eigenvalues.shape)
# print(test.H_tot[0])
# print(test.H_tot[0].shape)
# print(test.H_tot[0])
# print(np.linalg.eigvals(test.H_tot[0]))
# print(np.linalg.eigvals(test.H_tot[1]))
# print(test.H_tot[0]==test.H_tot[1])