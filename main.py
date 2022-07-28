import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename  # For files opening
from pathlib import Path  # For creating folder
from datetime import datetime  # For time stamp


# import sys  # For exceptions
# import traceback  # For exceptions

#######################################################################################################################
# CLASSES
#######################################################################################################################

class FiniteElement:

    def __init__(self, fe_id):
        self.id = fe_id

        self.fn_id = (elements_file.loc[elements_file['id'] == self.id]).iloc[0, 1]
        self.sn_id = (elements_file.loc[elements_file['id'] == self.id]).iloc[0, 2]
        self.type_name = (elements_file.loc[elements_file['id'] == self.id]).iloc[0, 3]
        self.stf_name = (elements_file.loc[elements_file['id'] == self.id]).iloc[0, 4]

        # Setting finite element's properties
        self.fn = Node(self.fn_id)
        self.sn = Node(self.sn_id)
        self.type = FEType(self.type_name)
        self.stf = Stiffness(self.stf_name)

        # Adding this finite element to it's nodes
        self.fn.add_fe(self)
        self.sn.add_fe(self)

        # Forming vector of FE
        self.vector = [self.sn.x - self.fn.x, self.sn.y - self.fn.y, self.sn.z - self.fn.z]

        # Calculating finite element's length
        self.L = np.linalg.norm(self.vector, ord=2)

        # Creating matrices templates
        #
        # Stiffness Matrix in local coordinates
        self.sm_local = np.zeros((len(self.type.matrix), len(self.type.matrix)))
        # Stiffness Matrix in global coordinates
        self.sm_global = np.zeros((len(self.type.matrix), len(self.type.matrix)))
        # Rotation Matrix
        self.rm = np.zeros((len(self.type.matrix), len(self.type.matrix)))

        # LN vector -- gives global dof for element's dof
        self.LM = []

    def set_sm_numerical_format(self):
        for i in range(len(self.type.matrix)):
            for j in range(len(self.type.matrix)):
                self.sm_local[i][j] = (
                    eval(str(self.type.matrix.iloc[i, j]), {},
                         {"E": self.stf.E, "A": self.stf.A, "I_z": self.stf.I_z, "L": self.L}))
                # str - на случай, если на вход eval попадется integer
                # с минусом получается


class Node:

    def __init__(self, node_id):
        self.id = node_id

        self.x = (nodes_file.loc[nodes_file['id'] == self.id]).iloc[0, 1]
        self.y = (nodes_file.loc[nodes_file['id'] == self.id]).iloc[0, 2]
        self.z = (nodes_file.loc[nodes_file['id'] == self.id]).iloc[0, 3]

        self.fe_array = []

    def add_fe(self, adding_fe):
        # Adding finite element to this node
        self.fe_array.append(adding_fe)


class FEType:  # Stiffness Matrix in symbol format

    def __init__(self, fe_type_name):
        self.name = fe_type_name

        self.file_name = self.name + '.csv'
        self.file_path = 'my_catalogues/input/fe_types/' + self.file_name
        self.matrix = pd.read_csv(self.file_path, header=None)


class Stiffness:

    def __init__(self, stf_name):
        self.name = stf_name

        self.file_name = self.name + '.csv'
        self.file_path = 'my_catalogues/input/stiffness/' + self.file_name

        self.table = pd.read_csv(self.file_path, header=0)
        self.E = self.table.iloc[0, 0]
        self.A = self.table.iloc[0, 1]
        self.I_z = self.table.iloc[0, 2]


class GlobalCoordinateSystem:

    def __init__(self):
        print('GCS is created')


class LocalCoordinateSystem:

    def __init__(self, fe_id):
        print('LCS is created')


#######################################################################################################################
# MAIN
#######################################################################################################################


if __name__ == '__main__':

    # Dealing with exceptions
    '''
    # try:
    d = datetime.now()
    log = open("my_catalogues/output/errors.txt", "w")
    log.write("----------------------------" + "\n")
    log.write("----------------------------" + "\n")
    log.write("Log: " + str(d) + "\n")
    log.write("\n")
    # Start process...
    start_time = datetime.now()
    log.write("Begin process:\n")
    log.write("     Process started at "
              + str(start_time) + "\n")
    log.write("\n")
    '''
    ###############################################################################################################
    # IMPORTING DATA
    ###############################################################################################################

    # Initial data is imported from .csv files into object DataFrame of Pandas package
    #
    # Choosing .csv files
    toggle = True  # For fast testing
    if toggle:
        print("\n Choose nodes file: ")
        nodes_file_name = askopenfilename()
        print("\n Choose elements file: ")
        elements_file_name = askopenfilename()
        # print("\n Choose file of node load: ")
        # node_load_file_name = askopenfilename()
        print("\n Choose file of node load: ")
        node_load_file_name = askopenfilename()
        print("\n Choose file of boundary conditions: ")
        bc_file_name = askopenfilename()
    else:
        nodes_file_name = 'my_catalogues/input/examples/Пример1_задача_из_методички/Nodes.csv'
        elements_file_name = 'my_catalogues/input/examples/Пример1_задача_из_методички/Elements.csv'
        # node_load_file_name = 'my_catalogues/input/examples/Пример1_задача_из_методички/P.csv'
        node_load_file_name = 'my_catalogues/input/examples/Пример1_задача_из_методички/P_for_K.csv'
        bc_file_name = 'my_catalogues/input/examples/Пример1_задача_из_методички/BoundaryConditions.csv'

    # Reading .csv files
    nodes_file = pd.read_csv(nodes_file_name, header=0)
    elements_file = pd.read_csv(elements_file_name, header=0)
    # node_load_file = pd.read_csv(node_load_file_name, header=None)
    node_load_file = pd.read_csv(node_load_file_name, header=None)

    print('Nodes: ')
    print(nodes_file)
    print('Elements: ')
    print(elements_file)
    print('Node load: ')
    print(node_load_file)
    print()

    #
    # INITIAL DATA
    #
    NODES_IN_FE = 2  # number of nodes in single FE
    NODES_TOTAL = len(nodes_file)  # number of nodes

    GC_i = [1, 0, 0]
    GC_j = [0, 1, 0]
    GC_k = [0, 0, 1]

    ###############################################################################################################
    # MAIN PART
    ###############################################################################################################

    #
    # PROCESSING FINITE ELEMENTS
    #

    # Creating array of FE by ids from file
    fe_array = []
    for i in range(len(elements_file)):
        fe = FiniteElement(elements_file.iloc[i, 0])  # [i,0] -- ids of FE in file "elements_file"
        fe_array.append(fe)

    # Назначение КЭ их МЖ в численном виде
    for fe in fe_array:
        fe.set_sm_numerical_format()
        print("FE length:")
        print(fe.L)
        print()

    print("\n МЖ первого КЭ в численном виде: ")
    for line in fe_array[0].sm_local:
        print(*line)

    # Привести МЖ всех КЭ к общей размерности?

    ################################################################################################################
    # TRANSFORMATION MATRICES
    ################################################################################################################

    # 4.1.2 Direction Cosines
    #for fe in fe_array:


    ################################################################################################################
    # Boundary Conditions
    ################################################################################################################

    # 5.2.1 Boundary Conditions, [ID] Matrix
    # Reading the .csv file of Boundary Conditions
    ID = pd.read_csv(bc_file_name, header=0)
    print("ID^T:")
    print(ID)

    # Reshaping BC matrix for further use
    ID = ID.T
    new_header = ID.iloc[0]  # grab the first row for the header
    ID = ID[1:]  # take the data less the header row
    ID.columns = new_header  # set the header row as the df header
    print("ID:")
    print(ID)

    # Number of Equation Augmented = total number of dof = number of nodes * dof in one node
    NEQA = 0
    NEQ = 0
    # Incrementally indexing dof of nodes to global dof of structure
    for j in range(ID.shape[1]):  # iterate over columns
        for i in range(ID.shape[0]):  # iterate over rows
            if ID.iloc[i, j] == 0:
                ID.iloc[i, j] = NEQ
                NEQ = NEQ + 1
                NEQA = NEQA + 1
    # Restrained dof are negative to distinguish them from unrestrained later
    for j in range(ID.shape[1]):  # iterate over columns
        for i in range(ID.shape[0]):  # iterate over rows
            if ID.iloc[i, j] == -1:
                ID.iloc[i, j] = -NEQA
                NEQA = NEQA + 1

    print("ID:")
    print(ID)
    print()

    # 5.2.2 LM Vector -- shows the global dof for each element
    for fe in fe_array:
        j_fn = fe.fn_id
        j_sn = fe.sn_id
        for i in range(ID.shape[0]):  # iterate over rows
            fe.LM.append(ID.iloc[i, j_fn])
        for i in range(ID.shape[0]):  # iterate over rows
            fe.LM.append(ID.iloc[i, j_sn])
    print("fe[0] LM:")
    print(fe_array[0].LM)
    print("fe[1] LM:")
    print(fe_array[1].LM)
    print()

    #
    # 5.2.3 Assembly of Global Stiffness Matrix
    #
    K = np.zeros((NEQA, NEQA))
    print("K:")
    print(K)
    print()

    for fe in fe_array:
        for i in range(fe.sm_local.shape[0]):  # iterate over rows
            for j in range(fe.sm_local.shape[1]):  # iterate over columns
                i_K = abs(fe.LM[i])
                j_K = abs(fe.LM[j])
                K[i_K, j_K] = K[i_K, j_K] + fe.sm_local[i, j]

    print("K:")
    print(K)
    for line in K:
        print(*line)
    print()

    # Extracting structure's stiffness matrix K_tt from the augmented stiffness matrix K and other parts
    #
    #     | K_tt | K_tu |
    # K = | ——————————— |
    #     | K_ut | K_uu |
    K_tt = K[0:NEQ, 0:NEQ]
    print("K_tt:")
    print(K_tt)
    for line in K_tt:
        print(*line)

    K_uu = K[NEQ:NEQA + 1, NEQ:NEQA + 1]
    print("K_uu:")
    print(K_uu)

    K_tu = K[0:NEQ, NEQ:NEQA + 1]
    print("K_tu:")
    print(K_tu)

    K_ut = K[NEQ:NEQA + 1, 0:NEQ]
    print("K_ut:")
    print(K_ut)

    print()

    # Determine the vector ∆_u which stores the initial displacements
    D_u = np.zeros((NEQA - NEQ, 1))
    print("D_u:")
    print(D_u)
    print()

    # Reading node load file
    P_t = np.zeros((len(node_load_file), 1))
    for i in range(len(node_load_file)):
        P_t[i][0] = (eval(node_load_file.iloc[i, 0]))
    print("P_t:")
    print(P_t)
    print()

    # Solving equations
    D_t = np.linalg.inv(K_tt).dot((P_t - K_tu.dot(D_u)))
    print("D_t:")
    print(D_t)
    print()

    R_t = K_ut.dot(D_t) + K_uu.dot(D_u)
    print("R_t:")
    print(R_t)
    print()

    ################################################################################################################
    # Вывод результатов
    ################################################################################################################

    # Нарисую график U(x)
    '''
    nodesXArray = []
    for i in range(len(nodes_file)):
        node = Node(nodes_file.iloc[i, 0])  # [i,0] -- id элементов в файле elementsFile
        nodesXArray.append(node.x)

    plt.plot(nodesXArray, U, '-ok')  # график перемещений
    
    # нарисуем КЭ схему
    # plt.axhline(y=U[0], color='b', linestyle='-')
    #  Сохраню результат в новую папку
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y__%H-%M-%S")
    Path("my_catalogues/output/results/", date_time).mkdir(parents=True, exist_ok=True)
    plt.savefig("my_catalogues/output/results/" + date_time + "/U")
    '''
    ################################################################################################################
    # Dealing with exceptions
    ################################################################################################################
    '''    
    endtime = datetime.now()
    # Process Completed...
    log.write("     Completed successfully in "
              + str(endtime - start_time) + "\n")
    log.write("\n")
    log.close()

    # Dealing with exceptions
    
    except:
        # Get the traceback object
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        # Concatenate information together concerning
        # the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback info:\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1])
        # Print Python error messages for use in
        # Python / Python Window
        log.write("" + pymsg + "\n")
        log.close()
    '''
