"""
Custom DFTTK Workflows
"""

import numpy as np
from uuid import uuid4
from fireworks import Workflow, Firework
from atomate.vasp.config import VASP_CMD, DB_FILE
from dfttk.fworks import OptimizeFW
from dfttk.input_sets import RelaxSet
from dfttk.EVcheck_QHA import EVcheck_QHA
from dfttk.utils import check_relax_path, add_modify_incar_by_FWname, add_modify_kpoints_by_FWname


def get_wf_gibbs(structure, num_deformations=7, deformation_fraction=(-0.1, 0.1),
                 phonon=False, phonon_supercell_matrix=None, Pos_Shape_relax=False,
                 t_min=5, t_max=2000, t_step=5, tolerance = 0.01, volume_spacing_min = 0.03,
                 vasp_cmd=None, db_file=None, metadata=None, name='EV_QHA', symmetry_tolerance = 0.05,
                 passinitrun=False, relax_path='', modify_incar_params={}, 
                 modify_kpoints_params={}, verbose=False):
    """
    E - V
    curve

    workflow
    Parameters
    ------
    structure: pymatgen.Structure
    num_deformations: int
    deformation_fraction: float
        Can be a float (a single value) or a 2-type of a min,max deformation fraction.
        Default is (-0.05, 0.1) leading to volumes of 0.95-1.10. A single value gives plus/minus
        by default.
    phonon : bool
        Whether to do a phonon calculation. Defaults to False, meaning the Debye model.
    phonon_supercell_matrix : list
        3x3 array of the supercell matrix, e.g. [[2,0,0],[0,2,0],[0,0,2]]. Must be specified if phonon is specified.
    Pos_Shape_relax : bool
        Whether to relax for shaple before every static calculation.
    t_min : float
        Minimum temperature
    t_step : float
        Temperature step size
    t_max : float
        Maximum temperature (inclusive)
    tolerance: float
        Acceptable value for average RMS, recommend >= 0.005.
    volume_spacing_min: float
        Minimum ratio of Volumes spacing
    vasp_cmd : str
        Command to run VASP. If None (the default) is passed, the command will be looked up in the FWorker.
    db_file : str
        Points to the database JSON file. If None (the default) is passed, the path will be looked up in the FWorker.
    name : str
        Name of the workflow
    metadata : dict
        Metadata to include
    passinitrun : bool
        Set True to pass initial VASP running if the results exist in DB, use carefully to keep data consistent.
    relax_path : str
        Set the path already exists for new static calculations; if set as '', will try to get the path from db_file.
    modify_incar_params : dict
        User can use these params to modify the INCAR set. It is a dict of class ModifyIncar with keywords in Workflow name.
    modify_kpoints_params : dict
        User can use these params to modify the KPOINTS set. It is a dict of class ModifyKpoints with keywords in Workflow name.
        Only 'kpts' supported now.
    """
    vasp_cmd = vasp_cmd or VASP_CMD
    db_file = db_file or DB_FILE

    metadata = metadata or {}
    tag = metadata.get('tag', '{}'.format(str(uuid4())))

    if isinstance(deformation_fraction, (list, tuple)):
        deformations = np.linspace(1+deformation_fraction[0], 1+deformation_fraction[1], num_deformations)
        vol_spacing = max((deformation_fraction[1] - deformation_fraction[0]) / (num_deformations - 0.999999) + 0.001, 
                          volume_spacing_min)
    else:
        deformations = np.linspace(1-deformation_fraction, 1+deformation_fraction, num_deformations)
        vol_spacing = max(deformation_fraction / (num_deformations - 0.999999) * 2 + 0.001, 
                          volume_spacing_min)
            
    fws = []
    if 'tag' not in metadata.keys():
        metadata['tag'] = tag
    relax_path, Pos_Shape_relax = check_relax_path(relax_path, db_file, tag, Pos_Shape_relax)
    
    if (relax_path == ''):
        # follow a scheme of
        # 1. Full relax + symmetry check
        # 2. If symmetry check fails, detour to 1. Volume relax, 2. inflection detection
        # 3. Inflection detection
        # 4. Static EV
        # 5. Phonon EV
        # for each FW, we set the structure to the original structure to verify to ourselves that the
        # volume deformed structure is set by input set.
    
        vis_relax = RelaxSet(structure)
        print('Full relax will be running ...')
        full_relax_fw = OptimizeFW(structure, symmetry_tolerance=symmetry_tolerance, job_type='normal', name='Full relax', 
                                   prev_calc_loc=False, vasp_input_set=vis_relax, vasp_cmd=vasp_cmd, db_file=db_file, 
                                   metadata=metadata, record_path = True, Pos_Shape_relax = Pos_Shape_relax,
                                   modify_incar_params=modify_incar_params, modify_kpoints_params = modify_kpoints_params,
                                   spec={'_preserve_fworker': True})
        fws.append(full_relax_fw)
    else:
        full_relax_fw = None
    check_result = Firework(EVcheck_QHA(db_file = db_file, tag = tag, relax_path = relax_path, deformations =deformations,
                                        tolerance = tolerance, threshold = 14, vol_spacing = vol_spacing, vasp_cmd = vasp_cmd, 
                                        metadata = metadata, t_min=t_min, t_max=t_max, t_step=t_step, phonon = phonon, symmetry_tolerance = symmetry_tolerance,
                                        phonon_supercell_matrix = phonon_supercell_matrix, verbose = verbose, Pos_Shape_relax = Pos_Shape_relax,
                                        modify_incar_params=modify_incar_params, modify_kpoints_params = modify_kpoints_params), 
                            parents=full_relax_fw, name='%s-EVcheck_QHA' %structure.composition.reduced_formula)
    fws.append(check_result)

    wfname = "{}:{}".format(structure.composition.reduced_formula, name)
    wf = Workflow(fws, name=wfname, metadata=metadata)
    add_modify_incar_by_FWname(wf, modify_incar_params = modify_incar_params)
    add_modify_kpoints_by_FWname(wf, modify_kpoints_params = modify_kpoints_params)

    return wf
