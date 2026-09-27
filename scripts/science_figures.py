"""Plot the transport and acoustic experiments from their numerical tables."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
COLORS = ('#17658c', '#c36a24', '#3a8579')


def rows(name: str) -> list[dict]:
    with (RESULTS / name).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def values(records: list[dict], name: str) -> np.ndarray:
    return np.array([float(row[name]) for row in records])


def panel(ax, label: str, title: str) -> None:
    ax.set_title(title, loc='left', fontsize=10, pad=10)
    ax.text(-.17, 1.045, label, transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', length=3, width=.7)


def save(fig, name: str, formats: tuple[str, ...]) -> None:
    FIGURES.mkdir(exist_ok=True)
    for suffix in formats:
        metadata = {'Creator':'ocean-evaluation 2.0.0'} if suffix == 'pdf' else ({'Date':None} if suffix == 'svg' else None)
        if suffix == 'pdf':
            metadata.update({'CreationDate':None,'ModDate':None})
        fig.savefig(FIGURES / f'{name}.{suffix}', dpi=320, bbox_inches='tight', metadata=metadata)
    plt.close(fig)


def tracer_figure(formats: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(2,2, figsize=(7.15,5.95), layout='constrained')
    profiles=rows('tracer_profiles.csv'); time=rows('tracer_time_series.csv')
    for case,color in zip(('A','B'),COLORS):
        p=[x for x in profiles if x['case']==case]
        t=[x for x in time if x['case']==case]
        axes[0,0].plot(values(p,'x_m')/1000,values(p,'initial_error'),color=color,
                       label=f'Error {case}',lw=1.7,ls='-' if case=='A' else '--')
        axes[0,1].plot(values(t,'lead_days'),np.abs(values(t,'output_error')),color=color,
                       label=f'Error {case}',lw=1.7,ls='-' if case=='A' else '--')
    ax=axes[0,0]; panel(ax,'a','Zero-inventory initial errors')
    ax.axhline(0,color='.65',lw=.6); ax.set(xlabel='Position (km)',ylabel='Concentration error',xlim=(0,100))
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    ax.text(.025,.97,r'$E_A=0.04,\quad E_B=0.05$',transform=ax.transAxes,va='top',fontsize=9)
    ax=axes[0,1]; panel(ax,'b','Error in the regional mean')
    ax.axvline(5,color='.45',lw=.8,ls=':'); ax.set(xlabel='Lead time (days)',ylabel='Absolute output error',xlim=(0,10))
    ratio=json.loads((RESULTS/'tracer_results.json').read_text())['output_error_ratio_A_over_B']
    ax.text(.51,.9,f'5 days: ratio {ratio:.3f}',transform=ax.transAxes,fontsize=8)
    summary=rows('tracer_summary.csv')
    before=next(x for x in summary if x['case']=='before_projection')
    after=next(x for x in summary if x['case']=='after_projection')
    ratios=[float(after['initial_field_rms'])/float(before['initial_field_rms']),
            float(after['squared_output_error'])/float(before['squared_output_error'])]
    ax=axes[1,0]; panel(ax,'c','Projection to the correct inventory')
    xp=np.arange(2)
    ax.bar(xp-.17,[1,1],width=.30,color='.82',label='Before projection')
    ax.bar(xp+.17,ratios,width=.30,color=COLORS[2],label='After projection')
    ax.set(xticks=xp,xticklabels=['Field RMS','Squared output error'],ylabel='Relative to the unprojected error',ylim=(0,4.95))
    for x,y in zip(xp+.17,ratios): ax.text(x,y+.10,f'{y:.3f}',ha='center',fontsize=9)
    ax.legend(frameon=False,fontsize=8,loc='upper left')
    con=rows('tracer_convergence.csv'); ax=axes[1,1];panel(ax,'d','Spatial convergence at five days')
    dx=values(con,'grid_spacing_m')
    err=values(con,'max_output_error')
    ax.loglog(dx,err,'o-',color=COLORS[0],lw=1.5,ms=4,label='Finite-volume solution')
    ax.loglog(dx,err[-1]*(dx/dx[-1])**2*1.8,':',color='.35',lw=1.3,label=r'$O(\Delta x^2)$')
    ax.set(xlabel='Cell width (m)',ylabel='Maximum absolute output error')
    ax.legend(frameon=False,fontsize=8)
    save(fig,'tracer_conservation',formats)


def metric_figure(formats: tuple[str, ...]) -> None:
    fig,axes=plt.subplots(2,2,figsize=(7.15,6.0),layout='constrained')
    spec=rows('acoustic_metric_spectrum.csv')
    for r,c in zip((10000,20000,30000),COLORS):
        data=[x for x in spec if float(x['range_m'])==r]
        axes[0,0].semilogy(values(data,'index'),values(data,'lambda_per_mps_squared'),
                           'o-',lw=1.4,ms=3.5,color=c,label=f'{r/1000:g} km')
    ax=axes[0,0];panel(ax,'a','Generalized eigenvalue spectrum')
    ax.set(xlabel='Ordered eigenvalue index',ylabel=r'$\lambda_i$  [$(\mathrm{m\,s^{-1}})^{-2}$]',xticks=[1,2,3,4,5,6,7,8])
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    dirs=rows('acoustic_metric_directions.csv')
    ax=axes[0,1]
    for direction,c,ls in zip(('maximum','minimum'),COLORS,('-','--')):
        data=[x for x in dirs if float(x['range_m'])==10000 and x['direction']==direction]
        ax.plot(values(data,'unit_rms_shape'),values(data,'depth_m'),color=c,ls=ls,lw=1.6,
                label=r'$\lambda_{\max}$' if direction=='maximum' else r'$\lambda_{\min}$')
    panel(ax,'b','Extreme directions at 10 km')
    ax.set(xlabel='Unit depth-RMS perturbation shape',ylabel='Depth (m)',ylim=(200,0))
    ax.axvline(0,color='.65',lw=.6);ax.legend(frameon=False,fontsize=8,loc='upper right')
    wit=[x for x in rows('acoustic_metric_witnesses.csv') if float(x['range_m'])==10000]
    ax=axes[1,0];panel(ax,'c','Full and quadratic mismatches')
    for case,c in zip(('a','b'),COLORS):
        e=values(wit,f'field_rms_{case}');d=values(wit,f'full_mismatch_{case}')
        ax.loglog(e,d,'o',color=c,ms=4,label=f'Full, error {case.upper()}')
        ax.loglog(e,values(wit,f'quadratic_mismatch_{case}'),'-',lw=1.2,color=c,
                  label=f'Quadratic, error {case.upper()}')
    ax.set(xlabel=r'Depth-RMS error (m s$^{-1}$)',ylabel='Array mismatch $D$')
    ax.legend(frameon=False,fontsize=7.6,loc='lower right')
    bound=rows('acoustic_ranking_bound.csv');ax=axes[1,1];panel(ax,'d','Sharp ordering threshold')
    for r,c,marker in zip((10000,20000,30000),COLORS,('o','s','^')):
        data=[x for x in bound if float(x['range_m'])==r]
        ax.loglog(values(data,'threshold_factor'),values(data,'full_ratio_a_over_b'),marker,
                   ms=5,color=c,fillstyle='none',label=f'Full, {r/1000:g} km')
    x=np.array([.45,2.2]);ax.loglog(x,x*x,color='.3',lw=1.1,label=r'Quadratic: $x^2$')
    ax.axhline(1,color='.6',ls=':',lw=.8);ax.axvline(1,color='.6',ls=':',lw=.8)
    ax.set(xlabel=r'Norm ratio divided by $\tau$',ylabel=r'Output-error ratio $D_A/D_B$',xticks=[.5,1,2],xticklabels=['0.5','1','2'])
    ax.legend(frameon=False,fontsize=7.5,loc='upper left')
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    save(fig,'acoustic_metric_analysis',formats)


def operator_figure(formats: tuple[str, ...]) -> None:
    fig,axes=plt.subplots(2,2,figsize=(7.15,6.15),layout='constrained')
    profile=rows('acoustic_profiles.csv')
    z=values([x for x in profile if x['perturbation']=='uniform'],'depth_m')
    ax=axes[0,0]
    ax.plot(np.full_like(z,.4),z,color=COLORS[0],lw=1.7,label='Uniform: RMS 0.4')
    local=np.exp(-.5*((z-45)/15)**2);local*=.5/np.sqrt(np.trapezoid(local*local,z)/200)
    ax.plot(local,z,color=COLORS[1],lw=1.7,ls='--',label='Localized: RMS 0.5')
    panel(ax,'a','Finite-amplitude sound-speed errors')
    ax.set(xlabel=r'Sound-speed error (m s$^{-1}$)',ylabel='Depth (m)',ylim=(200,0))
    ax.legend(frameon=False,fontsize=7.7,loc='lower right')
    sweep=rows('acoustic_amplitude_sweep.csv'); ax=axes[0,1]
    for shape,c,ls in zip(('uniform','localized'),COLORS,('-','--')):
        data=[x for x in sweep if x['perturbation']==shape and float(x['range_m'])==10000]
        ax.plot(values(data,'rms_mps'),values(data,'mismatch'),color=c,ls=ls,lw=1.6,label=shape.capitalize())
    ax.plot(.4,.748,'o',color=COLORS[0],ms=5); ax.plot(.5,.516,'s',color=COLORS[1],ms=5)
    ax.annotate('(0.4, 0.748)',(.4,.748),(.16,.91),fontsize=8,arrowprops={'arrowstyle':'-','lw':.6})
    ax.annotate('(0.5, 0.516)',(.5,.516),(.25,.27),fontsize=8,arrowprops={'arrowstyle':'-','lw':.6})
    panel(ax,'b','Amplitude dependence at 100 Hz, 10 km')
    ax.set(xlabel=r'Depth-RMS error (m s$^{-1}$)',ylabel='Array mismatch $D$',xlim=(0,.6),ylim=(0,1))
    curves=rows('acoustic_contrast_curves.csv');ax=axes[1,0]
    for shape,c,ls in zip(('uniform','localized'),COLORS,('-','--')):
        ax.plot(values(curves,'range_m')/1000,values(curves,shape),color=c,ls=ls,lw=.5,label=shape.capitalize())
    panel(ax,'c','Interference changes the error ordering')
    ax.set(xlabel='Source–receiver range (km)',ylabel='Array mismatch $D$',ylim=(0,1),xlim=(.5,30))
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    sweep=rows('acoustic_operator_sensitivity.csv')
    combos=[(f,zs) for f in (50,100,200) for zs in (25,50,100)]
    array=np.zeros((9,3))
    for i,(f,zs) in enumerate(combos):
        for j,r in enumerate((10000,20000,30000)):
            pair={x['perturbation']:float(x['mismatch']) for x in sweep if float(x['frequency_hz'])==f and float(x['source_depth_m'])==zs and float(x['range_m'])==r}
            array[i,j]=pair['uniform']-pair['localized']
    ax=axes[1,1];panel(ax,'d','All frequency–depth comparisons')
    image=ax.imshow(array,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vcenter=0,vmin=-1,vmax=1))
    ax.set(xticks=[0,1,2],xticklabels=['10','20','30'],xlabel='Source–receiver range (km)',
           yticks=np.arange(9),yticklabels=[f'{f} Hz / {zs} m' for f,zs in combos])
    ax.tick_params(axis='y',labelsize=7.5)
    for i in range(9):
        for j in range(3): ax.text(j,i,f'{array[i,j]:+.2f}',ha='center',va='center',fontsize=7.5,color='white' if abs(array[i,j])>.58 else '#202020')
    cb=fig.colorbar(image,ax=ax,pad=.025,fraction=.05)
    cb.set_label(r'$D_{\rm uniform}-D_{\rm localized}$',fontsize=8);cb.ax.tick_params(labelsize=7)
    save(fig,'acoustic_operator_tests',formats)


def main(formats: tuple[str, ...] = ('pdf','png','svg')) -> None:
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                         'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.7,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'svg.hashsalt':'ocean-evaluation-v2'})
    for make in (tracer_figure,metric_figure,operator_figure): make(formats)


if __name__ == '__main__':
    main()
