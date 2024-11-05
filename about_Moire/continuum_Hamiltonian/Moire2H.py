'''Program for calculating the eigenvalues of the continuum Hamiltonian of twisted MoTe2 and plotting the bands!'''
# Author: xu-sh21@mails.tsinghua.edu.cn
# Date: 2024.10.29
# Description: The model of this script is from the paper of Pro. Wang. For more information, please read the paper: http://arxiv.org/abs/2304.11864.

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
        self.density_for_band = 30 # Default density.
        self.density_for_ff = 30 # Default density
        self.kpoints_all_band = None
        self.kpoints_all_band_coef = None


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
            self.density_for_band = k_points['density_for_band']
            self.density_for_ff = k_points['density_for_ff']
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
        self.Hk_dim = self.group_num * 2
    

    def set_k_for_band(self):
        print("----------------------------------------------------------------")
        print(f"Begin set k points for mode band!")
        self.begin_k = datetime.now()
        print(f"Begin time:{self.begin_k}")

        self.k_lengths_band = [np.linalg.norm(self.find_coord(self.hsp[i][1]) - self.find_coord(self.hsp[i][0])) for i in range(1, self.num_path + 1)]
        self.point_num_band = np.int_(np.round(np.asarray(self.k_lengths_band) * self.density_for_band))
        # print(self.point_num)
        self.point_num_band_tot = sum(self.point_num_band)

        self.kpoints_all_band = np.zeros((self.point_num_band_tot, self.group_num, 2))
        self.kpoints_all_band_coef = np.zeros((self.point_num_band_tot, self.group_num, 2), dtype=int)
        # print(self.kpoints_all_band.shape)
        start_label = 0
        end_label = self.point_num_band[0]
        for i, (path, points) in enumerate(self.hsp.items()):
            t = np.linspace(0, 1, self.point_num_band[i])
            points = self.find_coord(self.hsp[i+1][0]) + (self.find_coord(self.hsp[i+1][1]) - self.find_coord(self.hsp[i+1][0])) * t[:, np.newaxis]
            # print(points.shape)
            self.kpoints_all_band[start_label: end_label, :, :] = points[:, None, :] * coef_G 
            # print(start_label, end_label)
            self.kpoints_all_band_coef[start_label: end_label, 0, :] = np.zeros((2,), dtype=int)
            self.kpoints_all_band_coef[start_label: end_label, 1: , :] = self.expand_k
            # print(self.kpoints_all_band_coef[0,:,:])
            start_label += self.point_num_band[i]
            if i != len(self.point_num_band)-1:
                end_label += self.point_num_band[i+1]
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
    

    def set_k_for_ff(self):
        print("----------------------------------------------------------------")
        print(f"Begin set k points for mode form factor!")
        self.begin_k = datetime.now()
        print(f"Begin time:{self.begin_k}")

        G_1 = G_vec_unit[1]
        G_2 = G_vec_unit[2]
        G_arr = np.array([G_1,G_2])
        t = np.linspace(0, 1, self.density_for_ff + 1)
        X, Y = np.meshgrid(t, t)
        self.coef_within_fbz = np.vstack((X.ravel(), Y.ravel())).T
        # print(self.coef_within_fbz)
        self.coord_within_fbz = self.coef_within_fbz[:,] @ G_arr
        # print(self.coord_within_fbz)
        self.coord_within_fbz = self.transition(self.coord_within_fbz)
        # print(self.coord_within_fbz)
        self.kpoints_all_ff = self.coord_within_fbz * coef_G
        self.kpoints_all_ff = np.repeat(self.kpoints_all_ff[:, np.newaxis, :], self.group_num, axis=1)
        self.point_num_ff_tot = self.kpoints_all_ff.shape[0]
        self.kpoints_all_ff_coef = np.zeros(self.kpoints_all_ff.shape, dtype=int)
        self.kpoints_all_ff_coef[:, 1:, :] = self.expand_k


    # def transition(self, coords):
    #     # This function is to move k points in unit cell in k-space into first Brillouin zone.
    #     to_delete = []
    #     for i in range(coords.shape[0]):
    #         x = coords[i,0]
    #         y = coords[i,1]
    #         # print(y - np.sqrt(3)*(x-1))
    #         if (x <= 1/2) & (np.sqrt(3) * y + (x-1) <= -1e-7):
    #             # print(1)
    #             pass
    #         elif (x > 1/2) & (x - np.sqrt(3)*y >= 1e-7) & (np.sqrt(3) * y + (x-2) < -1e-7) & (y > 0) & (y - np.sqrt(3)*(x-1) > 0):
    #             coords[i, 0] -= 1
    #             # print(2)
    #         elif (np.sqrt(3) * y + (x-1) > -1e-7) & (x - np.sqrt(3)*y < 1e-7) & (x < 1) & (y < np.sqrt(3)/2) & (y - np.sqrt(3)*x < 0):
    #             coords[i, 0] -= 1/2
    #             coords[i, 1] -= np.sqrt(3)/2
    #             # print(3)
    #         elif (np.sqrt(3) * y + (x-2) >= -1e-7) & (x >= 1) & (x < 3/2):
    #             coords[i, 0] -= 3/2
    #             coords[i, 1] -= np.sqrt(3)/2
    #             # print(4)
    #         else:
    #             to_delete.append(i)
    #             # print(f"to_delete{coords[i]}")
        
    #     coords = np.delete(coords, to_delete, axis=0)
    #     return coords


    def transition(self, coords):
        x = coords[:, 0]
        y = coords[:, 1]

        condition1 = (x <= 1/2) & (np.sqrt(3) * y + (x-1) <= -1e-7)
        condition2 = (x > 1/2) & (x - np.sqrt(3)*y >= 1e-7) & (np.sqrt(3) * y + (x-2) < -1e-7) & (y > 0) & (y - np.sqrt(3)*(x-1) > 0)
        condition3 = (np.sqrt(3) * y + (x-1) > -1e-7) & (x - np.sqrt(3)*y < 1e-7) & (x < 1) & (y < np.sqrt(3)/2) & (y - np.sqrt(3)*x < 0)
        condition4 = (np.sqrt(3) * y + (x-2) >= -1e-7) & (x >= 1) & (x < 3/2)

        coords[condition2, 0] -= 1
        coords[condition3, 0] -= 1/2
        coords[condition3, 1] -= np.sqrt(3)/2
        coords[condition4, 0] -= 3/2
        coords[condition4, 1] -= np.sqrt(3)/2

        to_delete = np.where(~condition1 & ~condition2 & ~condition3 & ~condition4)[0]

        coords = np.delete(coords, to_delete, axis=0)
        return coords


    def set_k_expand(self, k_set):
        # This function is to expand k points out of fMBZ by self.expand_k.
        G_1 = G_vec[1]
        G_2 = G_vec[2]
        for i, [m_1, m_2] in enumerate(self.expand_k):
            k_set[:, i+1, :]  += m_1 * G_1 + m_2 * G_2

        print("Finish buliding Hamiltonian matrix!")
        end = datetime.now()
        print(f"End time:{end}")
        print(f"Time cost:{end - self.begin_k}")


# +---------------------------------------------------------+
# | Construct the Hamiltonian matrix for a certain k point. |
# +---------------------------------------------------------+
    def bulid_H(self, mode, point_num, k_set, k_coef, H):
        print("----------------------------------------------------------------")
        print(f"Begin buliding Hamiltonian matrix for mode {mode}!")
        self.begin_H = datetime.now()
        print(f"Begin time:{self.begin_H}")

        for i in tqdm(range(point_num)):
            self.calculate_H_k(k_set, k_coef, H, i)
        
        print("Finish buliding Hamiltonian matrix!")
        self.end_H = datetime.now()
        print(f"End time:{self.end_H}")
        print(f"Time cost:{self.end_H - self.begin_H}")
        print("----------------------------------------------------------------")


    def calculate_H_k(self, k_set, k_coef, H, indices):
        k_group = k_set[indices, :, :]
        k_group_comb = k_coef[indices, :, :]

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
        H[indices, :, :] = H_k


    def construct_Hb_Ht(self, Hb, Ht, k_group, k_group_comb):
        # Construct the diagoanl part od Hb and Ht.
        self.calc_kinetic(Hb, Ht, k_group) 

        # Construct the non-diagoanl part od Hb and Ht.
        H_b_nondiag = np.zeros((self.group_num, self.group_num), dtype=np.complex_)
        self.calc_Hb_nondiag(k_group_comb, H_b_nondiag)
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


    def calc_Hb_nondiag(self, k_group_comb, H_nondiag):
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
    def is_hermitian(self, H):
            H_dagger = np.conjugate(np.transpose(H, (0, 2, 1)))
            judge = np.allclose(H, H_dagger)
            if judge == True:
                pass
            else:
                raise ValueError('The calculated Hamiltonian matrix is not hermitian!')
            

    def cal_eigen(self, mode, H):
        print(f"Begin calculation of the eigenvalues of the Halmiltonian matrix for {mode}!")
        self.begin_H_cal = datetime.now()
        print(f"Begin time:{self.begin_H_cal}")

        eigenvalues, eigenvectors = np.linalg.eig(H)

        print("End calculation of the eigenvalues of the Halmiltonian matrix!")
        self.end_H_cal = datetime.now()
        print(f"End time:{self.end_H_cal}")
        print(f"Time cost:{self.end_H_cal - self.begin_H_cal}")
        print("----------------------------------------------------------------")
        
        return eigenvalues, eigenvectors
        # print(self.eigenvalues)
        # print(self.eigenvectors.shape)


    def process_eigen_for_band(self, point_num_total, point_num_list, eigenvals):
        # Remove the imaginary part of the vector, if the imaginary part is less than tolerance. 
        tolerance = 1e-5
        processed_eigenvalues = np.array(eigenvals, dtype=np.complex128)
        for i in range(point_num_total):
            if np.all(np.abs(processed_eigenvalues[i,:].imag) < tolerance):
                processed_eigenvalues[i,:] = processed_eigenvalues[i,:].real
        
        # Remove the duplicated parts in the k-point, eigenvalues, and eigenvectors.
        self.k_points = np.arange(point_num_total)
        self.eigen_sorted = np.sort(processed_eigenvalues, axis=1)
        self.spec_kcoord = [0,]
        self.k_points, self.eigen_sorted, self.spec_kcoord = self.delete(self.k_points, self.eigen_sorted, self.spec_kcoord, point_num_list)


    def delete(self, kpoints, eigen, k_coord, point_num_list):
        lengths_list = point_num_list
        deleted_points = [sum(lengths_list[:i]) for i in range(1, len(lengths_list))]
        for i in range(1, len(lengths_list)+1):
            k_coord.append(sum(lengths_list[:i])-i)
        mask = np.ones(eigen.shape[0], dtype=bool)
        mask[deleted_points] = False
        eigen = eigen[mask]
        kpoints = kpoints[mask]

        return kpoints, eigen, k_coord
    

    def process_eigen_for_ff(self, point_num_total, eigenvals, eigenvecs):
        sorted_indices = np.argsort(eigenvals, axis=1)
        self.max_eigenvalue_indices = sorted_indices[:, -1]
        self.sorted_eigenvalues = np.take_along_axis(eigenvals, sorted_indices, axis=1).squeeze()
        self.sorted_eigenvectors = np.take_along_axis(eigenvecs, sorted_indices[:, np.newaxis, :], axis=2)
        self.max_vecs = self.sorted_eigenvectors[:,:,-1]
        # print(self.sorted_eigenvalues.shape)
        # print(self.max_vecs.shape)


# +---------------------------------------------------+
# | Plot the band and form factors of twisted MoTe2. |
# +---------------------------------------------------+
    def plot_band(self):
        plt.figure(figsize=(5,8))
        for i in range(self.Hk_dim):
            energy_band = self.eigen_sorted[:, i]
            mask = energy_band > self.emin
            # plt.plot(k_points[mask], energy_band[mask], label=f'Band {i+1}')
            plt.plot(self.k_points[mask], energy_band[mask])
        
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
        plt.xlim(self.k_points[0], self.k_points[-1])
        plt.ylim(bottom=self.emin)
        for point in special_k_points.keys():
            plt.axvline(x=point, color='k', linestyle='--', linewidth=0.5)
        plt.xticks([point for point in special_k_points.keys()], [label for label in special_k_points.values()])
        plt.savefig(f'./fig/dens={self.density_for_band}loop={self.k_loop}min={self.emin}.png', dpi=300) 


# +------------------------------------------------------+
# | Calculate and plot the form factor of twisted MoTe2. |
# +------------------------------------------------------+
    def calc_u(self):
        self.form_factor_matrix = np.dot(self.max_vecs.conj(), self.max_vecs.T).T
        # print(self.form_factor_matrix.shape)


    def plot_u(self):
        # TODO
        pass


# +---------------+
# | Full process. |
# +---------------+
    def Hamiltonian(self):
        self.begin()
        self.info_print()
        self.parse_config()
        self.preprocess()
        self.operation()


    def operation(self):
        if self.plot_mode == 'band':
            self.band_job()
        elif self.plot_mode == 'form_factor':
            self.form_factor_job()
        elif self.plot_mode == 'both':
            self.band_job()
            self.form_factor_job()
        else:
            raise TypeError("ERROR! Incorrect setting about plot mode!")
        

    def band_job(self):
        self.set_k_for_band()
        self.set_k_expand(self.kpoints_all_band)
        self.H_tot_band = np.zeros((self.point_num_band_tot, self.Hk_dim, self.Hk_dim), dtype=np.complex_)
        self.bulid_H(self.plot_mode, self.point_num_band_tot, self.kpoints_all_band, self.kpoints_all_band_coef, self.H_tot_band)
        self.is_hermitian(self.H_tot_band)
        self.eigenvals_band, self.eigenvecs_band = self.cal_eigen(self.plot_mode, self.H_tot_band)
        self.process_eigen_for_band(self.point_num_band_tot, self.point_num_band, self.eigenvals_band)
        self.plot_band()


    def form_factor_job(self):
        self.set_k_for_ff()
        self.set_k_expand(self.kpoints_all_ff)
        self.H_tot_ff = np.zeros((self.point_num_ff_tot, self.Hk_dim, self.Hk_dim), dtype=np.complex_)
        self.bulid_H(self.plot_mode, self.point_num_ff_tot, self.kpoints_all_ff, self.kpoints_all_ff_coef, self.H_tot_ff)
        self.is_hermitian(self.H_tot_ff)
        self.eigenvals_ff, self.eigenvecs_ff = self.cal_eigen(self.plot_mode, self.H_tot_ff)
        self.process_eigen_for_ff(self.point_num_ff_tot, self.eigenvals_ff, self.eigenvecs_ff)
        self.calc_u()
        # self.plot_u()

test = Moire2H('config.yaml')
test.Hamiltonian()