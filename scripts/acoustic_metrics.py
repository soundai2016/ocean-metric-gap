"""Sensitivity-weighted error metric of the normal-mode receiver field."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from acoustic_example import (reference_speed, solve_modes, pressure, array_mismatch,
                               depth_rms, write_csv)
from error_metrics import ranking_bound
from quadratic_order import reversal_witness

ROOT = Path(__file__).resolve().parents[1]


def cosine_basis(z: np.ndarray, dimension: int) -> np.ndarray:
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)) or dimension < 2:
        raise ValueError('at least two real perturbation directions required')
    if z.ndim != 1 or len(z) < dimension + 1 or z[0] != 0 or np.any(np.diff(z) <= 0):
        raise ValueError('an increasing depth grid from zero is required')
    b = np.cos(np.pi * z[:, None] * np.arange(dimension)[None, :] / z[-1])
    b[:, 1:] *= np.sqrt(2)
    return b


def response_matrix(z, c, b, ranges, receivers, frequency, source_depth,
                    basis_size: int, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(step) or step <= 0:
        raise ValueError('positive finite derivative step required')
    reference = solve_modes(z, c, frequency, basis_size)
    p0 = pressure(reference, ranges, source_depth, receivers)
    derivative = np.empty((len(ranges), len(receivers), b.shape[1]), dtype=complex)
    for j in range(b.shape[1]):
        fields = {}
        for multiple in (-2, -1, 1, 2):
            modes = solve_modes(z, c + multiple*step*b[:,j], frequency, basis_size)
            if len(modes.k) != len(reference.k):
                raise ValueError('derivative interval crosses a propagating-mode cutoff')
            fields[multiple] = pressure(modes, ranges, source_depth, receivers)
        derivative[:,:,j] = (8*(fields[1]-fields[-1])-(fields[2]-fields[-2])) / (12*step)
    norm = np.sum(abs(p0)**2,axis=1)
    projected = derivative - p0[:,:,None] * (
        np.einsum('ri,rij->rj',p0.conj(),derivative)/norm[:,None])[:,None,:]
    q=np.einsum('rik,ril->rkl',projected.conj(),projected).real / norm[:,None,None]
    return p0, derivative, (q+q.transpose(0,2,1))/2


def run() -> dict:
    spec=json.loads((ROOT/'data/acoustic_metric_spec.json').read_text())
    base=json.loads((ROOT/'data/acoustic_spec.json').read_text())
    z=np.linspace(0,base['depth_m'],round(base['depth_m']/base['depth_grid_m'])+1)
    c=reference_speed(z)
    receivers=np.arange(base['receiver_min_depth_m'],base['receiver_max_depth_m']+.1,base['receiver_spacing_m'])
    ranges=np.asarray(spec['ranges_m'])
    d=spec['dimension']; b=cosine_basis(z,d)
    w=np.trapezoid(b[:,:,None]*b[:,None,:],z,axis=0)/z[-1]
    args=(z,c,b,ranges,receivers,spec['frequency_hz'],spec['source_depth_m'])
    p0,g,matrices=response_matrix(*args,spec['report_basis_size'],spec['derivative_step_mps'])
    _,_,qstep=response_matrix(*args,spec['report_basis_size'],spec['derivative_check_step_mps'])
    _,_,qbasis=response_matrix(*args,spec['check_basis_size'],spec['derivative_step_mps'])
    spectrum, checks, witnesses, profiles, bounds = [], [], [], [], []
    eigenvectors=[]
    for i,distance in enumerate(ranges):
        q=matrices[i]
        values, vectors=eigh(q,w); eigenvectors.append(vectors)
        values_step=eigh(qstep[i],w,eigvals_only=True)
        values_basis=eigh(qbasis[i],w,eigvals_only=True)
        for j,value in enumerate(values):
            spectrum.append({'range_m':float(distance),'index':j+1,'lambda_per_mps_squared':float(value),
                'lambda_step_check':float(values_step[j]),'lambda_basis_check':float(values_basis[j])})
        checks.append({'range_m':float(distance),
            'relative_Q_step_difference':float(np.linalg.norm(q-qstep[i],2)/np.linalg.norm(q,2)),
            'relative_Q_basis_difference':float(np.linalg.norm(q-qbasis[i],2)/np.linalg.norm(q,2)),
            'relative_lambda_min_step_difference':float(abs(values[0]-values_step[0])/values[0]),
            'relative_lambda_min_basis_difference':float(abs(values[0]-values_basis[0])/values[0]),
            'lambda_min':float(values[0]),'lambda_max':float(values[-1]),
            'norm_ratio_threshold':float(np.sqrt(values[0]/values[-1]))})
        for amplitude in spec['witness_amplitudes_mps']:
            pair=reversal_witness(q,w,amplitude=amplitude)
            row={'range_m':float(distance),'amplitude_mps':amplitude}
            for label in ('a','b'):
                coef=np.asarray(pair[label]);dc=b@coef
                estimated=pressure(solve_modes(z,c+dc,spec['frequency_hz'],spec['report_basis_size']),
                                  np.array([distance]),spec['source_depth_m'],receivers)
                full=float(array_mismatch(p0[i:i+1],estimated)[0]);quad=float(coef@q@coef)
                row.update({f'field_rms_{label}':depth_rms(dc,z),f'full_mismatch_{label}':full,
                            f'quadratic_mismatch_{label}':quad,
                            f'quadratic_relative_error_{label}':abs(full-quad)/full})
            row['full_reversal']=bool(row['field_rms_a']<row['field_rms_b'] and row['full_mismatch_a']>row['full_mismatch_b'])
            witnesses.append(row)
        for direction,column in [('minimum',0),('maximum',-1)]:
            shape=b@vectors[:,column]
            for depth,value in zip(z,shape):
                profiles.append({'range_m':float(distance),'direction':direction,'depth_m':float(depth),'unit_rms_shape':float(value)})
        sb=spec['bound_test_amplitude_mps'];threshold=np.sqrt(values[0]/values[-1])
        for factor in spec['bound_threshold_factors']:
            aa=sb*factor*threshold*vectors[:,-1];bb=sb*vectors[:,0]
            full=[]
            for coef in (aa,bb):
                pp=pressure(solve_modes(z,c+b@coef,spec['frequency_hz'],spec['report_basis_size']),
                            np.array([distance]),spec['source_depth_m'],receivers)
                full.append(float(array_mismatch(p0[i:i+1],pp)[0]))
            bound=ranking_bound(q,w,sb*factor*threshold,sb)
            bounds.append({'range_m':float(distance),'threshold_factor':factor,
                'norm_a_mps':sb*factor*threshold,'norm_b_mps':sb,
                'quadratic_ratio_a_over_b':float((aa@q@aa)/(bb@q@bb)),
                'full_ratio_a_over_b':full[0]/full[1],
                'certified_quadratic_order':bound['strictly_ordered_for_all_directions']})
    dimensions=[]
    for dimension in spec['dimension_checks']:
        if dimension==d:qd=matrices;wd=w
        else:
            bd=cosine_basis(z,dimension)
            wd=np.trapezoid(bd[:,:,None]*bd[:,None,:],z,axis=0)/z[-1]
            _,_,qd=response_matrix(z,c,bd,ranges,receivers,spec['frequency_hz'],spec['source_depth_m'],
                                  spec['report_basis_size'],spec['derivative_step_mps'])
        for i,distance in enumerate(ranges):
            vals=eigh(qd[i],wd,eigvals_only=True)
            dimensions.append({'dimension':dimension,'range_m':float(distance),
                'lambda_min':float(vals[0]),'lambda_max':float(vals[-1]),
                'norm_ratio_threshold':float(np.sqrt(max(0.,vals[0])/vals[-1]))})
    out=ROOT/'results';out.mkdir(exist_ok=True)
    write_csv(out/'acoustic_metric_spectrum.csv',spectrum)
    write_csv(out/'acoustic_metric_refinement.csv',checks)
    write_csv(out/'acoustic_metric_witnesses.csv',witnesses)
    write_csv(out/'acoustic_metric_directions.csv',profiles)
    write_csv(out/'acoustic_ranking_bound.csv',bounds)
    write_csv(out/'acoustic_metric_dimensions.csv',dimensions)
    np.savez_compressed(out/'acoustic_metric_matrices.npz',depth_m=z,basis=b,W=w,Q=matrices,
                        G=g,p0=p0,range_m=ranges,eigenvectors=np.asarray(eigenvectors))
    result={'spec':spec,'checks':checks,'dimensions':dimensions,'witness_rows':len(witnesses),
        'full_reversals':sum(r['full_reversal'] for r in witnesses),
        'small_amplitude_max_quadratic_relative_error':max(r[f'quadratic_relative_error_{k}']
            for r in witnesses if r['amplitude_mps']<=.01 for k in ('a','b')),
        'origin':'new finite-difference receiver sensitivities and recomputed modal fields'}
    (out/'acoustic_metric_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
