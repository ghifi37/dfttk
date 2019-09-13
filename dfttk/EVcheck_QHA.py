# coding: utf-8

import math
import numpy as np
from atomate.vasp.database import VaspCalcDb
from dfttk.utils import sort_x_by_y, mark_adopted
from itertools import combinations
from pymatgen.analysis.eos import EOS
from fireworks import FiretaskBase, LaunchPad, Workflow, Firework
from fireworks.utilities.fw_utilities import explicit_serialize
from dfttk.input_sets import RelaxSet, StaticSet, ForceConstantsSet
from dfttk.fworks import StaticFW, OptimizeFW, PhononFW
from dfttk.ftasks import QHAAnalysis
from dfttk.analysis.quasiharmonic import Quasiharmonic

@explicit_serialize
class EVcheck_QHA(FiretaskBase):
    ''' 
    If EVcheck(Energies versus Volumes) meets the tolerance, it will launch QHA;
        otherwise it will append more volumes to VASP calculation and take EVcheck again.
    The maximum appending VASP running times set by run_num;
    
    Important Properties:
    correct: whether result satisfies the tolerance
    points: the selected data index
    error: actual fitting error
    eos_fit: eos fitting
    
    '''
    _fw_name = 'EVcheck_QHA'
    required_params = ['db_file', 'tag', 'structure', 'vasp_cmd', 'metadata']
    optional_params = ['run_num', 'tolerance', 'threshold', 'del_limited', 'vol_spacing', 't_min',
                       't_max', 't_step', 'phonon', 'phonon_supercell_matrix', 'verbose']
    
    def run_task(self, fw_spec):
        ''' 
        run_num: maximum number of appending VASP running; this limitation is to avoid always running due to bad settings;
            only for internal usage;

        Important args:
        tolerance: acceptable value for average RMS, recommend >= 0.001;
        threshold: total point number above the value should be reduced, recommend < 16;
        del_limited: maximum deletion ration for large results;
        vol_spacing: the maximum ratio step between two volumes, larger step will be inserted points to calculate;
        '''
        max_run = 10
        db_file = self['db_file']
        tag = self['tag']
        structure = self['structure']
        vasp_cmd = self['vasp_cmd']
        metadata = self['metadata']
        run_num = self.get('run_num') or 0
        tolerance = self.get('tolerance') or 0.005
        threshold = self.get('threshold') or 14
        del_limited = self.get('del_limited') or 0.3
        vol_spacing = self.get('vol_spacing') or 0.02
        t_min = self.get('t_min') or 5 
        t_max = self.get('t_max') or 2000
        t_step = self.get('t_step') or 5
        phonon = self.get('phonon') or False
        phonon_supercell_matrix = self.get('phonon_supercell_matrix') or None
        verbose = self.get('verbose') or False
        run_num += 1
        
        volumes, energies, dos_objs = self.get_orig_EV(db_file, tag)
        self.check_points(db_file, metadata, tolerance, threshold, del_limited, volumes, energies, verbose)
        
        EVcheck_result = {}
        EVcheck_result['append_run_num'] = run_num
        EVcheck_result['correct'] = self.correct
        EVcheck_result['volumes'] = volumes
        EVcheck_result['energies'] = energies
        EVcheck_result['tolerance'] = tolerance
        EVcheck_result['threshold'] = threshold
        EVcheck_result['vol_spacing'] = vol_spacing
        EVcheck_result['error'] = self.error
        EVcheck_result['metadata'] = metadata

        if self.correct:
            vol_orig = structure.volume
            volume, energy, dos_obj = self.gen_volenergdos(self.points, volumes, energies, dos_objs)
            vol_adds = self.check_vol_coverage(volume, vol_spacing, vol_orig, run_num, 
                                               energy, structure, dos_obj, t_min, t_max)   # Normalized to 1
            EVcheck_result['sellected'] = volume
            EVcheck_result['append'] = str(vol_adds * vol_orig)
            # Marked as adopted in db
            mark_adopted(tag, db_file, volume)
            lpad = LaunchPad.auto_load()
            fws = []
            if vol_adds != []:
                if run_num < max_run:
                    # Do VASP and check again
                    print('Append the volumes of : %s to calculate in VASP!' %(vol_adds * vol_orig))
                    calcs = []
                    # Full relax
                    vis = RelaxSet(structure)
                    full_relax_fw = OptimizeFW(structure, symmetry_tolerance=0.05, job_type='normal', name='Full relax', prev_calc_loc=False, vasp_input_set=vis, vasp_cmd=vasp_cmd, db_file=db_file, metadata=metadata, spec={'_preserve_fworker': True})
                    fws.append(full_relax_fw)
                    calcs.append(full_relax_fw)
                    vis = StaticSet(structure)
                    for vol_add in vol_adds:
                        static = StaticFW(structure, scale_lattice = vol_add, name = 'structure_%.3f-static' %(vol_add * vol_orig), 
                                          vasp_input_set = vis, vasp_cmd = vasp_cmd, db_file = db_file, metadata = metadata, parents = full_relax_fw)
                        fws.append(static)
                        calcs.append(static)
                        if phonon:
                            visphonon = ForceConstantsSet(structure)
                            phonon_fw = PhononFW(structure, phonon_supercell_matrix, t_min=t_min, t_max=t_max, t_step=t_step,
                                     name='structure_%.3f-phonon' %(vol_add * vol_orig), vasp_input_set=visphonon,
                                     vasp_cmd=vasp_cmd, db_file=db_file, metadata=metadata,
                                     prev_calc_loc=True, parents=static)
                            fws.append(phonon_fw)
                            calcs.append(phonon_fw)
                    check_result = Firework(EVcheck_QHA(db_file = db_file, tag = tag, structure = structure, tolerance = tolerance, 
                                                        threshold = threshold, vol_spacing = vol_spacing, vasp_cmd = vasp_cmd, 
                                                        metadata = metadata, t_min=t_min, t_max=t_max, t_step=t_step, phonon = phonon,
                                                        phonon_supercell_matrix = phonon_supercell_matrix, verbose = verbose), 
                                            parents = calcs, name='%s-EVcheck_QHA' %structure.composition.reduced_formula)
                    fws.append(check_result)
                    strname = "{}:{}".format(structure.composition.reduced_formula, 'EV_QHA_Append')
                    wfs = Workflow(fws, name = strname, metadata=metadata)
                    lpad.add_wf(wfs)
                else:
                    print('''

#######################################################################
             Too many times appended VASP running, abort!
                      Please check VASP setting!
#######################################################################

                         ''')
            else:  # No need to do more VASP calculation, QHA could be running 
                print('Success in Volumes-Energies checking, enter QHA ...')
                # Debye
                debye_fw = Firework(QHAAnalysis(phonon=False, t_min=t_min, t_max=t_max, t_step=t_step, db_file=db_file, tag=tag, metadata=metadata), 
                                    name="{}-qha_analysis-Debye".format(structure.composition.reduced_formula))
                fws.append(debye_fw)
                if phonon:
                    phonon_supercell_matrix = self.get('phonon_supercell_matrix')
                    # do a Debye run before the phonon, so they can be done in stages.
                    phonon_fw = Firework(QHAAnalysis(phonon=True, t_min=t_min, t_max=t_max, t_step=t_step, db_file=db_file, tag=tag, 
                                                     metadata=metadata), parents=debye_fw, name="{}-qha_analysis-phonon".format(structure.composition.reduced_formula))
                    fws.append(phonon_fw)
                strname = "{}:{}".format(structure.composition.reduced_formula, 'QHA')
                wfs = Workflow(fws, name = strname, metadata=metadata)
                lpad.add_wf(wfs)
        else:   # failure to meet the tolerance
            if self.error == 1e10:   # Bad initial running set
                print('''

#######################################################################
#                                                                     #
#  "passinitrun = True" could not set while initial results absent.   #
#                                                                     #
#######################################################################

                      
                      ''')
            else:                      # fitting fails
                print('''

#######################################################################
#                                                                     #
#           Can not achieve the tolerance requirement, abort!         #
#                                                                     #
#######################################################################

                      ''')
        import json
        with open('E-V check_summary.json', 'w') as fp:
            json.dump(EVcheck_result, fp)
  
    
    def get_orig_EV(self, db_file, tag):
        vasp_db = VaspCalcDb.from_db_file(db_file, admin = True)
        energies = []
        volumes = []
        dos_objs = []  # pymatgen.electronic_structure.dos.Dos objects
        static_calculations = vasp_db.collection.find({'$and':[ {'metadata.tag': tag}, {'adopted': True},
                                                               {'output.structure.lattice.volume': {'$exists': True} }]})
        if static_calculations.count() <= 4:
            static_calculations = vasp_db.collection.find({'$and':[ {'metadata.tag': tag},
                                                               {'output.structure.lattice.volume': {'$exists': True} }]})
        vol_last = 0
        for calc in static_calculations:
            vol = calc['output']['structure']['lattice']['volume']
            if abs((vol - vol_last) / vol) > 1e-8:
                energies.append(calc['output']['energy'])
                volumes.append(vol)
                dos_objs.append(vasp_db.get_dos(calc['task_id']))
            else:
                energies[-1] = calc['output']['energy']
                volumes[-1] = vol
                dos_objs[-1] = vasp_db.get_dos(calc['task_id'])
            vol_last = vol
        energies = sort_x_by_y(energies, volumes)
        dos_objs = sort_x_by_y(dos_objs, volumes)
        volumes = sorted(volumes)
        n = len(volumes) - 1           # Delete duplicated
        while n > 0:
            if abs((volumes[n] - volumes[n - 1]) / volumes[n]) < 1e-8:
                volumes.pop(n)
                energies.pop(n)
                dos_objs.pop(n)
            n -= 1
        print('%s Volumes  = %s' %(len(volumes), volumes))
        print('%s Energies = %s' %(len(energies), energies))
        return(volumes, energies, dos_objs)
        
        
    def gen_volenerg(self, num, volumes, energies):
        volume = []
        energy = []
        for m in range(len(num)):
            volume.append(volumes[num[m]])
            energy.append(energies[num[m]])
        return volume, energy


    def gen_volenergdos(self, num, volumes, energies, dos_objs):
        volume = []
        energy = []
        dos_obj = []
        for m in range(len(num)):
            volume.append(volumes[num[m]])
            energy.append(energies[num[m]])
            dos_obj.append(dos_objs[num[m]])
        return volume, energy, dos_obj

        
    def check_fit(self, volumes, energies):
        eos = EOS('vinet')
        self.eos_fit = eos.fit(volumes, energies)
        
        
    def check_points(self, db_file, metadata, tolerance, threshold, del_limited, volumes, energies, verbose = False):
        self.correct = False
        self.error = 1e11
        error = 1e10
        num = np.arange(len(volumes))
        comb = num
        limit = len(volumes) * del_limited
        
        # For len(num) > threshold case, do a whole number fitting to pass numbers delete if met tolerance
        for i in range(1):     # To avoid the codes after except running
            if (len(num) > threshold):
                try:
                    self.check_fit(volumes, energies)
                except:
                    if verbose:
                        print('Fitting error in: ', comb)
                    break
                fit_value = self.eos_fit.func(volumes)
                temperror = 0
                for m in range(len(volumes)):
                    temperror += math.pow((fit_value[m] - energies[m]), 2) 
                temperror = math.sqrt(temperror) / len(volumes)
                if verbose:
                    print('error = %.4f in %s ' %(temperror, comb))
                if temperror < error:
                    error = temperror
        
        # Decrease the quantity of large results
        while (error > tolerance) and (len(num) > limit) and (len(num) > threshold):
            volume, energy = self.gen_volenerg(num, volumes, energies)
            try:
                self.check_fit(volume, energy)
            except:
                if verbose:
                    print('Fetal error in Fitting : ', num)         # Seldom
                break
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
                        print('error = %.4f in %s ' %(temperror, combs))
                    if temperror < error:
                        error = temperror
                        comb = combs
                len_comb -= 1

        print('Minimum error = %s' %error, comb)
        if error <= tolerance:
            self.correct = True
        comb = list(comb)
        comb.sort()
        self.points = comb
        self.error = error
        
    
    def check_vol_coverage(self, volume, vol_spacing, vol_orig, run_num, energy, structure, 
                          dos_objects, t_min, t_max):
        result = []
        volumer = volume.copy()
        for m in range(len(volumer)):
            volumer[m] = volumer[m] / vol_orig
        for m in range(len(volumer) - 1):
            if (volumer[m + 1] - volumer[m]) > vol_spacing:
                step = (volumer[m + 1] - volumer[m]) / (int((volumer[m + 1] - volumer[m]) / vol_spacing) + 1 - 0.0002 * run_num)
                vol = volumer[m] + step
                while vol < volumer[m + 1]:
                    result.append(vol)
                    vol += step
        # To check and extend deformation rations
        vol_spacing = vol_spacing * 0.98   # To make sure that coverage extension smaller than interpolation spacing
        qha = Quasiharmonic(energy, volume, structure, dos_objects=dos_objects, F_vib=None,
                            t_min=t_min, t_max=t_max, t_step=(t_max-t_min)/100, poisson=0.363615, bp2gru=1)
        vol_max = np.max(qha.optimum_volumes) / vol_orig
        vol_min = np.min(qha.optimum_volumes) / vol_orig
        if volumer[-1] * 1.01 < vol_max:
            result.append(volumer[-1] + vol_spacing)
            while result[-1] < vol_max:
                result.append(result[-1] + vol_spacing)
        if volumer[0] * 0.99 > vol_min:
            result.append(volumer[0] - vol_spacing)
            while result[-1] > vol_min:
                result.append(result[-1] - vol_spacing)
        return(np.array(result))

