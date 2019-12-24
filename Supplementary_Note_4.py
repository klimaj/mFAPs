import os
import pyrosetta
import pyrosetta.distributed.dask
import pyrosetta.distributed.io as io
import pyrosetta.distributed.packed_pose as packed_pose
import re

def circularly_permute(pose, new_Nterm_pdb_res):
    
    lines_list = io.to_pdbstring(pose).split("\n")
    
    new_lines_list = []
    for line in lines_list:
        if line.startswith("ATOM"):
            pdb_res = int(line[22:26].split()[-1])
            if pdb_res >= new_Nterm_pdb_res:
                 new_lines_list.append(line)
    for line in lines_list:
        if line.startswith("ATOM"):
            pdb_res = int(line[22:26].split()[-1])
            if pdb_res < new_Nterm_pdb_res:
                 new_lines_list.append(line)
    for line in lines_list:
        if line.startswith("HETATM"):
            new_lines_list.append(line)
    
    out_pose = packed_pose.to_pose(io.pose_from_pdbstring("\n".join(new_lines_list)))
    
    # Renumber pdb
    for i, s in enumerate(out_pose.sequence(), start=1):
        if s == "Z":
            out_pose.pdb_info().set_resinfo(res=i, chain_id="B", pdb_res=i)
            out_pose.residue(i).chain(2)
        else:
            out_pose.pdb_info().set_resinfo(res=i, chain_id="A", pdb_res=i)
            out_pose.residue(i).chain(1)
    
    return out_pose

notebook_flags = """
-out:level 300 
-ignore_unrecognized_res 1
-ex2 1
-ex1aro 1
-holes:dalphaball DAlphaBall.gcc
-extra_res_fa HBI.fa.params
-extra_res_cen HBI.cen.params
"""
pyrosetta.distributed.dask.init_notebook(notebook_flags)

new_Nterm_pdb_res = [35, 63, 89, 106]

for pdb in ["mFAP2a", "mFAP2b"]:

    pose = pyrosetta.io.pose_from_file(f"{pdb}.pdb")

    for new_Nterm in new_Nterm_pdb_res:
        
        tmp_pose = pose.clone()
        out_pose = circularly_permute(tmp_pose, new_Nterm)
        
        scorefxn = pyrosetta.create_score_function("ref2015")
        
        Nterminus_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector("1A-2A")
        Cterminus_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector("114A-115A")
        termini_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(Nterminus_selector, Cterminus_selector)
        not_termini_selector = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(termini_selector)
        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
            pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), not_termini_selector))
        mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
        mmf.all_bb(setting=False)
        mmf.all_bondangles(setting=False)
        mmf.all_bondlengths(setting=False)
        mmf.all_chi(setting=False)
        mmf.all_jumps(setting=False)
        mmf.set_cartesian(setting=False)
        enable = pyrosetta.rosetta.core.select.movemap.move_map_action.mm_enable
        mmf.add_chi_action(action=enable, selector=termini_selector)
        mmf.add_bb_action(action=enable, selector=termini_selector)
        fast_relax = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn_in=scorefxn, standard_repeats=1)
        fast_relax.cartesian(False)
        fast_relax.set_task_factory(tf)
        fast_relax.set_movemap_factory(mmf)
        fast_relax.minimize_bond_angles(False)
        fast_relax.minimize_bond_lengths(False)
        fast_relax.min_type("dfpmin_armijo_nonmonotone")
        fast_relax.apply(out_pose)
        
        outdir = "circularly_permuted_relaxed_pdbs"
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        new_name = "cp" + str(new_Nterm) + "-" + str(new_Nterm - 1) + f"_{pdb}.pdb"
        
        # Output pose with a TER line at cutpoint
        cutpoint = int(re.search("SRAAQLLPGTWQ", out_pose.sequence()).start() + 1)
        lines_list = io.to_pdbstring(out_pose).split("\n")
        cont = True
        new_lines_list = []
        for line in lines_list:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                pdb_res = int(line[22:26].split()[-1])
                if (pdb_res == cutpoint) and cont:
                    new_lines_list.append("TER")
                    cont = False
                new_lines_list.append(line)
            elif line.startswith("TER"):
                new_lines_list.append(line)
        with open(os.path.join(outdir, new_name), "w") as f:
            f.write("\n".join(new_lines_list))
