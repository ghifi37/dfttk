# coding: utf-8

import math
import numpy as np
from atomate.vasp.database import VaspCalcDb
from dfttk.utils import sort_x_by_y, mark_adopted
from itertools import combinations
from pymatgen.analysis.eos import EOS
from fireworks import FiretaskBase, LaunchPad, Workflow, Firework
from fireworks.utilities.fw_utilities import explicit_serialize
from atomate.utils.utils import env_chk
from dfttk.input_sets import RelaxSet, StaticSet, ForceConstantsSet
from dfttk.fworks import StaticFW, OptimizeFW, PhononFW
from dfttk.ftasks import QHAAnalysis

@explicit_serialize
class EVcheck_QHA(FiretaskBase):
    ''' Properties:
    correct: result satisfies the tolerance
    points: the selected data index
    error: actual fitting error
    eos_fit: eos fitting
    
    '''
    _fw_name = 'EVcheck_QHA'
    required_params = ['db_file', 'tag', 'structure', 'vasp_cmd', 'metadata']
    optional_params = ['run_num', 'tolerance', 'threshold', 'del_limited', 'vol_spacing', 't_min',
                       't_max', 't_step', 'phonon', 'phonon_supercell_matrix', 'verbose']
    
    def  run_task(self, fw_spec):
        ''' Args:
        run_num: maximum number of appending VASP run
        tolerance: acceptable value for average RMS, recommend >= 0.001
        threshold: total point number above the value should be reduced, recommend <= 16
        del_limited: maximum deletion ration for large results
        vol_spacing: the minimium ratio step between two volumes
        '''
        db_file = env_chk(self.get('db_file'), fw_spec)
        tag = env_chk(self.get('tag'), fw_spec)
        structure = env_chk(self.get('structure'), fw_spec)
        run_num = env_chk(self.get('run_num'), fw_spec) or 10
        tolerance = env_chk(self.get('tolerance'), fw_spec) or 0.005
        threshold = env_chk(self.get('threshold'), fw_spec) or 14
        del_limited = env_chk(self.get('del_limited'), fw_spec) or 0.8
        vol_spacing = env_chk(self.get('vol_spacing'), fw_spec) or 0.02
        vasp_cmd = env_chk(self.get('vasp_cmd'), fw_spec)
        metadata = env_chk(self.get('metadata'), fw_spec)
        t_min = env_chk(self.get('t_min'), fw_spec) or 5 
        t_max = env_chk(self.get('t_max'), fw_spec) or 2000
        t_step = env_chk(self.get('t_step'), fw_spec) or 5
        phonon = env_chk(self.get('phonon'), fw_spec) or False
        verbose = env_chk(self.get('verbose'), fw_spec) or False
        run_num -= 1
        
        volumes, energies = self.get_org_EV(db_file, tag)
        self.check_points(db_file, metadata, tolerance, threshold, del_limited, volumes, energies, verbose)
        
        EVcheck_result = {}
        EVcheck_result['append_run_num'] = 10 - run_num
        EVcheck_result['correct'] = self.correct
        EVcheck_result['volumes'] = volumes
        EVcheck_result['energies'] = energies
        EVcheck_result['tolerance'] = tolerance
        EVcheck_result['threshold'] = threshold
        EVcheck_result['vol_spacing'] = vol_spacing
        EVcheck_result['error'] = self.error
        EVcheck_result['metadata'] = metadata

        if self.correct:
            vol_ave = (volumes[0] + volumes[-1]) / 2
            volume, energy = self.gen_volenerg(self.points, volumes, energies)
            vol_adds = self.check_vol_spacing(volume, vol_spacing, vol_ave)   # Normalized to 1
            EVcheck_result['sellected'] = volume
            EVcheck_result['Append'] = vol_adds
            lpad = LaunchPad.auto_load()
            fws = []
            if vol_adds != []:
                if run_num > 0:
                    # Do VASP and check again
                    print('Append the volumes of : %s to calculate in VASP(%.3f set to 1)!' %(vol_adds, vol_ave))
                    calcs = []
                    # Full relax
                    vis = RelaxSet(structure)
                    full_relax_fw = OptimizeFW(structure, symmetry_tolerance=0.05, job_type='normal', name='Full relax', prev_calc_loc=False, vasp_input_set=vis, vasp_cmd=vasp_cmd, db_file=db_file, metadata=metadata, spec={'_preserve_fworker': True})
                    fws.append(full_relax_fw)
                    calcs.append(full_relax_fw)
                    vis = StaticSet(structure)
                    visphonon = ForceConstantsSet(structure)
                    for vol_add in vol_adds:
                        static = StaticFW(structure, scale_lattice = vol_add, name = 'structure_%.3f-static' %vol_add, 
                                          vasp_input_set = vis, vasp_cmd = vasp_cmd, db_file = db_file, metadata = metadata, parents = full_relax_fw)
                        fws.append(static)
                        calcs.append(static)
                        if phonon:
                            phonon_supercell_matrix = env_chk(self.get('phonon_supercell_matrix'), fw_spec)
                            phonon_fw = PhononFW(structure, phonon_supercell_matrix, t_min=t_min, t_max=t_max, t_step=t_step,
                                     name='structure_{}-phonon'.format(vol_add), vasp_input_set=visphonon,
                                     vasp_cmd=vasp_cmd, db_file=db_file, metadata=metadata,
                                     prev_calc_loc=True, parents=static)
                            fws.append(phonon_fw)
                            calcs.append(phonon_fw)
                    check_result = Firework(EVcheck_QHA(db_file = db_file, tag = tag, structure = structure, tolerance = tolerance, 
                                                        threshold = threshold, vol_spacing = vol_spacing, vasp_cmd = vasp_cmd, 
                                                        metadata = metadata, run_num = run_num, phonon = phonon, verbose = verbose), 
                                            parents = calcs, name='%s_EVcheck_QHA' %structure.composition.reduced_formula)
                    fws.append(check_result)
                else:
                    print('''

#######################################################################
                 Too many appended VASP running, abort!
                        Please check setting!
#######################################################################

                         ''')
            else:  # No need to do more VASP calculation, can cun QHA
                # Marked as adopted in db
                mark_adopted(tag, db_file, volume)
                # Debye
                debye_fw = Firework(QHAAnalysis(phonon=False, t_min=t_min, t_max=t_max, t_step=t_step, db_file=db_file, tag=tag, metadata=metadata), 
                                    name="{}-qha_analysis-Debye".format(structure.composition.reduced_formula))
                fws.append(debye_fw)
                if phonon:
                    phonon_supercell_matrix = env_chk(self.get('phonon_supercell_matrix'), fw_spec)
                    # do a Debye run before the phonon, so they can be done in stages.
                    phonon_fw = Firework(QHAAnalysis(phonon=True, t_min=t_min, t_max=t_max, t_step=t_step, db_file=db_file, tag=tag, 
                                                     metadata=metadata), parents=debye_fw, name="{}-qha_analysis-phonon".format(structure.composition.reduced_formula))
                    fws.append(phonon_fw)
            strname = "{}:{}".format(structure.composition.reduced_formula, 'EV_QHA_Append')
            wfs = Workflow(fws, name = strname, metadata=metadata)
            lpad.add_wf(wfs)
        else:   # failure to meet the tolerance
            print('''

#######################################################################
           Can not achieve the tolerance requirement, abort!
#######################################################################

                  ''')
        import json
        with open('E-V check_summary.json', 'w') as fp:
            json.dump(EVcheck_result, fp)
  
    
    def get_org_EV(self, db_file, tag):
        vasp_db = VaspCalcDb.from_db_file(db_file, admin = True)
        energies = []
        volumes = []
        static_calculations = vasp_db.collection.find({'$and':[ {'metadata.tag': tag}, {'adopted': True} ]})
        m = 0
        for calc in static_calculations:
            m += 1
        if m <= 3:
            static_calculations = vasp_db.collection.find({'metadata.tag': tag})
        vol_last = 0
        for calc in static_calculations:
            vol = calc['output']['structure']['lattice']['volume']
            if abs((vol - vol_last) / vol) > 0.00001:
                energies.append(calc['output']['energy'])
                volumes.append(vol)
            vol_last = vol
        energies = sort_x_by_y(energies, volumes)
        volumes = sorted(volumes)
        return(volumes, energies)
        
        
    def gen_volenerg(self, num, volumes, energies):
        volume = []
        energy = []
        for m in range(len(num)):
            volume.append(volumes[num[m]])
            energy.append(energies[num[m]])
        return volume, energy
      
        
    def check_fit(self, volumes, energies):
        eos = EOS('vinet')
        self.eos_fit = eos.fit(volumes, energies)
        
        
    def check_points(self, db_file, metadata, tolerance, threshold, del_limited, volumes, energies, verbose = False):
        self.correct = False
        error = 1e10
        num = np.arange(len(volumes))
        comb = num
        limit = len(volumes) * del_limited
        
        # Decrease the quantity of large results
        while (error > tolerance) and (len(num) > limit) and (len(num) > threshold):
            volume, energy = self.gen_volenerg(num, volumes, energies)
            try:
                self.check_fit(volume, energy)
            except:
                if verbose:
                    print('Fetal error in Fitting : ', num)         # Seldom 
            fit_value = self.eos_fit.func(volume)
            errors = abs(fit_value - energy)
            num = sort_x_by_y(num, errors)
            errors = sorted(errors)
            for m in range(min(len(errors) - threshold, 1)):
                errors.pop(-1)
                num.pop(-1)
            temperror = 0
            for m in range(len(num)):
                temperror += math.pow(errors[m], 2) 
            temperror = math.sqrt(temperror) / len(num)
            if verbose:
                print('Absolutely average offest is: %.4f in %s numbers combination.' %(temperror, len(num)))
            if temperror < error:
                error = temperror
                comb = num

        # combinations

        len_comb = len(comb)
        if (error > tolerance) and (len_comb <= threshold):
            comb_source = comb
            while (error > tolerance) and (len_comb >= 4):
                print('Combinations in "%s"...' %len_comb)
                combination = combinations(comb_source, len_comb)
                for combs in combination:
                    volume, energy = self.gen_volenerg(combs, volumes, energies)
                    try:
                        self.check_fit(volume, energy)
                    except:
                        if verbose:
                            print('Fitting error in: ', combs)
                        continue
                    fit_value = self.eos_fit.func(volume)
                    temperror = 0
                    for m in range(len(volume)):
                        temperror += math.pow((fit_value[m] - energy[m]), 2) 
                    temperror = math.sqrt(temperror) / len(volume)
                    if verbose:
                        print('error = %.4f in [ ' %(temperror), end = '')
                        for vol in volume:
                            print('%.3f, ' %vol, end = '') 
                        print(']')
                    if temperror < error:
                        error = temperror
                        comb = combs
                len_comb -= 1

        if verbose:
            print('Minimum error = %s' %error, comb)
        if error <= tolerance:
            self.correct = True
        self.points = comb
        self.error = error
        
    
    def check_vol_spacing(self, volume, vol_spacing, vol_ave):
        result = []
        volumer = volume.copy()
        for m in range(len(volumer)):
            volumer[m] = volumer[m] / vol_ave
        for m in range(len(volumer) - 1):
            if (volumer[m + 1] - volumer[m]) > vol_spacing:
                step = (volumer[m + 1] - volumer[m]) / (int((volumer[m + 1] - volumer[m]) / vol_spacing) + 0.9999)
                vol = volumer[m] + step
                while vol < volumer[m + 1]:
                    result.append(vol)
                    vol += step
        return(result)
