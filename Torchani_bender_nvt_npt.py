from tempfile import TemporaryDirectory
from openbabel import pybel
from ssl import ALERT_DESCRIPTION_DECOMPRESSION_FAILURE
import sys, os
from openbabel import openbabel as ob
import subprocess
from types import CellType
import torchani
import ase
from ase.io import read, write
from ase.optimize import FIRE, BFGS
from ase import units
import subprocess
import pkg_resources
from ase import units
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.langevin import Langevin
from ase.calculators.checkpoint import Checkpoint
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt



Number_A = 6.02214076 * 10**23

elements_dic = {'H':1,'HE':2,'LI':3,'BE': 4,'B':5,'C':6, 'N':7, 'O':8, 'F':9,'NE':10,
                 'NA': 10,'MG':12,'AL':13,'SI':14,'P':15,'S':16, 'CL':17, 'AR':18,'K':19,'CA':20}

weight_dic ={'water': 18.01528 ,'acetonitrile': 41.05, 'methanol':32.04, 'chloroform':119.38 } ##g/mol

density_dic = {'water' : 997,'acetonitrile': 786, 'methanol':792, 'chloroform': 1490} ##kg/m3

closeness_dic = {'water': 0.5, 'acetonitrile':1.80, 'methanol':0.6, 'nma': 0.58 ,'chloroform':0.58}

def calculateSolvnumber(solvPrefix,volume):  ###Volume should be m3
    denisty = density_dic[solvPrefix]
    weight = weight_dic[solvPrefix]
    mass = volume * denisty  ## kg
    Mol = mass*1000/weight
    NumberOfSolv = Mol * Number_A
    return int(NumberOfSolv) 

class MLFF(object):
    def __init__(self, xyz, solvent, closeness, slu_netcharge=0, cube_size=20, tempi=0, temp0=300, dt=0.2, nstlim_heating=10000, nstlim_npt = 100000):
        self.xyz = xyz
        self.solvent = solvent
        self.closeness = closeness_dic[closeness]
        self.cube_size = cube_size
        self.slu_pos = self.cube_size/2.0
        self.pbcbox_size = self.cube_size+2
        self.solvPrefix = self.solvent.split('.pdb')[0]
        self.volumn = (self.cube_size * 10**(-10))**3 ### m
        self.slv_count = calculateSolvnumber(self.solvPrefix, self.volumn)
        self.tempi = tempi
        self.temp0 = temp0
        self.dt = dt
        self.nstlim_heating = nstlim_heating
        self.nstlim_npt = nstlim_npt


    def packSLUSLV(self):

        solute_xyzfile = read(self.xyz)
       # pdbfilename = solute_xyzfile.split('.xyz')[0]+'.pdb'
        write('solute.pdb',solute_xyzfile)

        solvent_pdb = self.solvent
        output_pdb = self.solvPrefix + "_solvated.packmol.pdb"
    
    #   solvent_pdb_origin = pkg_resources.resource_filename('autosolvate', 
    #           os.path.join('data/', solvPrefix, solvent_pdb))
    #   subprocess.call(['cp',solvent_pdb_origin,solvent_pdb])

        packmol_inp = open('packmol.inp','w')
        packmol_inp.write("# All atoms from diferent molecules will be at least %s Angstroms apart\n" % self.closeness)
        packmol_inp.write("tolerance %s\n" % self.closeness)
        packmol_inp.write("\n")
        packmol_inp.write("filetype pdb\n")
        packmol_inp.write("\n")
        packmol_inp.write("output " + output_pdb + "\n")
        packmol_inp.write("\n")
        packmol_inp.write("# add the solute\n")
        packmol_inp.write("structure solute.pdb\n")
        packmol_inp.write("   number 1\n")
        packmol_inp.write("   fixed " + " " + str(self.slu_pos) + " "+ str(self.slu_pos) + " " + str(self.slu_pos) + " 0. 0. 0.\n")
        packmol_inp.write("   centerofmass\n")
        packmol_inp.write("   resnumbers 2 \n")
        packmol_inp.write("end structure\n")
        packmol_inp.write("\n")
        packmol_inp.write("# add first type of solvent molecules\n")
        packmol_inp.write("structure "+ solvent_pdb + "\n")
        packmol_inp.write("  number " + str(self.slv_count) + " \n")
        packmol_inp.write("  inside cube 0. 0. 0. " + str(self.cube_size) + " \n")
        packmol_inp.write("  resnumbers 2 \n")
        packmol_inp.write("end structure\n")
        packmol_inp.close()

        cmd ="packmol < packmol.inp > packmol.log"
        subprocess.call(cmd, shell=True)
    
    def torchaniFFPre(self):
        input_pdb = self.solvPrefix + "_solvated.packmol.pdb"
        f_input_pdb = open(input_pdb,'r').readlines()
        line = "%-6s"%"CRYST1" + "%9.3f"%self.cube_size + "%9.3f"%self.cube_size + "%9.3f"%self.cube_size + "%7.2f"%90.0 + "%7.2f"%90.0 + "%7.2f"%90.0 + '%11s'%'P 1'
        print(line)
        f_pdb_pbc = open(self.solvPrefix + "_solvated.packmol.pbc.pdb",'w')
        f_pdb_pbc.write(line+'\n')
        for line in f_input_pdb:
            if 'ATOM' == line.split()[0]:
                f_pdb_pbc.write(line)
        f_pdb_pbc.write('END')
        f_pdb_pbc.close()
        input_pdb_pbc = self.solvPrefix + "_solvated.packmol.pbc.pdb"
       
    def torchaniFFMin(self):
        input_pdb_pbc = self.solvPrefix + "_solvated.packmol.pbc.pdb"
        atoms = read(input_pdb_pbc)
        print(atoms)
        calculator = torchani.models.ANI1ccx().ase()
        atoms.set_calculator(calculator)
        print("Begin minimizing...")
        opt = BFGS(atoms,trajectory='mmmin.traj')
        opt.run(fmax=0.001)
        print()
        

    
    def torchaniFFHeating(self):
        calculator = torchani.models.ANI1ccx().ase()
        trajmin = Trajectory('mmmin.traj')
        atoms = trajmin[-1]
        atoms.set_calculator(calculator)
        temp_init = self.tempi
        temp_end = self.temp0
        temp = temp_init
        temp_all = []
        
        while temp < temp_end :
            temp_all.append(temp)
            temp = temp + 30
        temp_all.append(temp_end)
        print(temp_all)
        Nstep= int(self.nstlim_heating/len(temp_all))
        calculator = torchani.models.ANI1ccx().ase()
       # print(temp_all)
        for temp in temp_all[1:]:
            print('npv_heating_'+str(temp)+'.traj')
            dyn = NVTBerendsen(atoms, timestep=self.dt * units.fs, temperature_K = temp , trajectory='nvt_heating_'+str(temp)+'.traj', logfile='nvt_heating_'+str(temp)+'.log', taut=0.5*1000*units.fs)
            dyn.run(Nstep)
            traj = Trajectory('nvt_heating_'+str(temp)+'.traj')
            atoms = traj[-1]
            atoms.set_calculator(calculator)
            traj.close()
        
        

    def torchaniFFNPT(self):
        cmd = 'cp nvt_heating_'+str(self.temp0)+'.traj mmheating.traj'
        os.system(cmd)
        calculator = torchani.models.ANI1ccx().ase()
        trajheat = Trajectory('mmheating.traj')
        atoms = trajheat[-1]
        atoms.set_calculator(calculator)
        trajheat.close()
        dyn = NPTBerendsen(atoms, timestep=self.dt * units.fs, temperature_K = self.temp0, trajectory = 'mmnpt.traj', logfile='mmnpt.log',pressure_au=4050 * units.bar,
                           taup=1000 * units.fs, compressibility=8.17e-5 / units.bar)
        dyn.run(self.nstlim_npt)
                      

A = MLFF('./try.xyz','acetonitrile.pdb', 'acetonitrile')
#A.packSLUSLV()
#A.torchaniFFPre()
#A.torchaniFFMin()
#A.torchaniFFHeating()
A.torchaniFFNPT()



