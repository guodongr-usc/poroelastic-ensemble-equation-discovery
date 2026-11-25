# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

from pathlib import Path
import json
import os
import tempfile
import numpy as np

NAMES = ('P', 'Ux', 'Uy')
LABELS = (r'$P$', r'$U_x$', r'$U_y$')
UNITS = ('MPa', 'mm', 'mm')
FACTORS = np.array([1e-6, 1e3, 1e3])
DISPLAY = {'kernel': 'Kernel-based method', 'fpde': 'Fractional PDE'}
COLORS = {'kernel': '#2070af', 'fpde': '#df541e'}
POINTS = ((200., 215.), (110., 290.), (400., 405.))


def score(y, p):
    error = np.asarray(p, float) - np.asarray(y, float)
    sse = float(np.sum(error ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    norm = float(np.linalg.norm(y))
    return dict(R2=1 - sse / sst if sst > 0 else None,
                absolute_L2=float(np.sqrt(sse)),
                relative_L2=float(np.sqrt(sse) / norm) if norm > 0 else None)


def load_inputs(root, data_path=None):
    path = data_path if data_path is not None else root / 'data' / '443EA_CLEAN_ICSTRAIN0.1_P_U1_U2.npz'
    with np.load(path, allow_pickle=False) as z:
        truth = np.stack([z[k][0] for k in ('P_EA', 'U1_EA', 'U2_EA')], axis=1)
        xy = z['node_xy_m'].copy()
        times = z['times_s'].copy()
    with np.load(root / 'assets/fpde/evaluation_nodes.npz', allow_pickle=False) as z:
        free = [z['free_' + h].copy() for h in NAMES]
    split = json.loads((root / 'data' / 'splits.json').read_text())
    return truth, xy, times, free, split


def load_prediction(path, times, xy):
    with np.load(path, allow_pickle=False) as z:
        fields = tuple(map(str, z['field_order'].tolist()))
        if len(fields) != 3 or set(fields) != set(NAMES):
            raise ValueError(f'Unexpected prediction field order: {fields}')
        prediction = z['prediction_si'][:, [fields.index(h) for h in NAMES]].copy()
        np.testing.assert_allclose(z['times_s'], times, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(z['node_xy_m'], xy)
    if prediction.shape != (len(times), 3, len(xy)) or not np.isfinite(prediction).all():
        raise ValueError('Incomplete/nonfinite prediction; cannot score as successful.')
    return prediction


def evaluate(root, output, models, *, prediction_paths=None, data_path=None):
    truth_si, xy, times, free, split = load_inputs(root, data_path)
    truth = truth_si * FACTORS[None, :, None]
    report = dict(field_order=NAMES, time_days=(times / 86400).tolist(), split=split,
                  metrics_definition={
                      'absolute_L2': 'sqrt(sum((prediction-truth)^2)); spatial discrete Euclidean norm, no mesh weighting or N division',
                      'relative_L2': 'absolute_L2 / sqrt(sum(truth^2))',
                      'R2_per_frame': '1-SSE/SST; SST uses spatial mean of this field at this frame on its free nodes',
                      'R2_split': '1-SSE/SST pooled over listed frames and this field free nodes',
                      'initial_frame': 'Known initial condition; included in curves, excluded from training aggregate'},
                  models={})
    for model in models:
        path = (root / 'assets/fpde' / (model + '_prediction.npz')
                if prediction_paths is None else prediction_paths[model])
        raw = load_prediction(path, times, xy)
        pred = raw * FACTORS[None, :, None]
        result = dict(initial_max_abs_error_si={}, fields={})
        for j, h in enumerate(NAMES):
            y, p = truth[:, j][:, free[j]], pred[:, j][:, free[j]]
            result['initial_max_abs_error_si'][h] = float(np.max(np.abs(raw[0, j] - truth_si[0, j])))
            frame = [score(y[i], p[i]) for i in range(len(times))]
            f = dict(unit=UNITS[j], free_node_count=len(free[j]), per_frame=frame)
            for label, ids in [('train', split['train_score_frames']),
                               ('temporal_heldout', split['temporal_heldout_frames']),
                               ('last', [len(times) - 1])]:
                f[label] = score(y[ids], p[ids])
            result['fields'][h] = f
        report['models'][model] = result
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for model, result in report['models'].items():
        print(DISPLAY[model])
        for h in NAMES:
            print(h, 'train', result['fields'][h]['train'],
                  'temporal heldout', result['fields'][h]['temporal_heldout'])
    return report


def plot(root, output, models, *, prediction_paths=None, data_path=None):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'clean443_frozen_forward_matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    truth, xy, times, free, split = load_inputs(root, data_path)
    predictions = {m: load_prediction(root / 'assets/fpde' / (m + '_prediction.npz')
                                      if prediction_paths is None else prediction_paths[m], times, xy)
                   * FACTORS[None, :, None] for m in models}
    truth = truth * FACTORS[None, :, None]
    days = times / 86400.
    cutoff = days[split['training_last_frame']]
    metrics = json.loads((output / 'metrics.json').read_text())
    out = output
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 19, 'axes.labelsize': 16,
                         'legend.fontsize': 12, 'pdf.fonttype': 42})

    def finish(fig, name):
        fig.savefig(out / (name + '.png'), dpi=180)
        plt.close(fig)

    def interval(ax):
        ax.axvspan(cutoff, days[-1], color='#f5dfcf', alpha=.6, zorder=0)
        ax.axvline(cutoff, color='.5', ls=':', lw=1.5)
        ax.set(xlim=(days[0], days[-1]), xlabel='Time (days)')
        ax.grid(alpha=.2)

    for metric, filename in [('absolute_L2', 'absolute_L2_R2'), ('relative_L2', 'relative_L2_R2')]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 9), layout='constrained')
        for j, h in enumerate(NAMES):
            for model in models:
                frame = metrics['models'][model]['fields'][h]['per_frame']
                for row, key in enumerate((metric, 'R2')):
                    axes[row, j].plot(days, [v[key] for v in frame],
                        'o-' if model == 'kernel' else 's--', ms=3,
                        color=COLORS[model], label=DISPLAY[model])
            axes[0, j].set(title=LABELS[j], ylabel=(f'$L_2$ norm error ({UNITS[j]})'
                          if metric == 'absolute_L2' else 'Relative $L_2$ error'))
            axes[1, j].set(ylabel='$R^2$')
            for row in range(2): interval(axes[row, j])
        axes[0, 0].legend()
        axes[0, 0].text(.895, .22, 'Held-out\ninterval', transform=axes[0, 0].transAxes,
                        ha='center', weight='bold', color='#765947')
        fig.suptitle(('Kernel-based method and fractional PDE' if len(models) == 2 else DISPLAY[models[0]])
                     + ' vs. ensemble-average truth', fontsize=21)
        finish(fig, filename)

    nx, ny = len(np.unique(xy[:, 0])), len(np.unique(xy[:, 1]))
    np.testing.assert_array_equal(xy, np.stack(np.meshgrid(np.unique(xy[:, 0]), np.unique(xy[:, 1])), axis=-1).reshape(-1, 2))
    columns = 1 + 2*len(models)
    fig, axes = plt.subplots(3, columns, figsize=(4.2*columns, 11), layout='constrained', squeeze=False)
    quantiles = (5., 95.)
    for j, h in enumerate(NAMES):
        low, high = np.percentile(truth[-1, j], quantiles)
        error_limit = max(float(np.percentile(np.abs(predictions[m][-1,j]-truth[-1,j]), 95)) for m in models)
        error_limit = max(error_limit, 1e-15)
        fields = [truth[-1,j]] + [predictions[m][-1,j] for m in models] + [predictions[m][-1,j]-truth[-1,j] for m in models]
        titles = ['Ensemble-average truth'] + [DISPLAY[m] for m in models] + [f'Error: {"kernel" if m=="kernel" else "fractional PDE"} vs. truth' for m in models]
        for c, values in enumerate(fields):
            is_error = c > len(models)
            im = axes[j, c].imshow(values.reshape(ny,nx), origin='lower', extent=[0,500,0,500],
                 cmap='RdBu_r' if is_error else 'viridis',
                 vmin=-error_limit if is_error else low, vmax=error_limit if is_error else high)
            axes[j,c].set(xlabel='x (m)', xticks=[0,250,500], yticks=[0,250,500])
            if c == 0: axes[j,c].set_ylabel(LABELS[j]+'\ny (m)')
            else: axes[j,c].tick_params(labelleft=False)
            if j == 0: axes[j,c].set_title(titles[c], fontsize=16)
            if c in (len(models), columns-1): fig.colorbar(im, ax=axes[j,c], label=UNITS[j], extend='neither', shrink=.86)
    fig.suptitle('Final-frame comparison with ensemble-average truth', fontsize=21)
    finish(fig, 'last_frame_comparison')

    fig, axes = plt.subplots(3,3,figsize=(17,11),layout='constrained')
    for c, (point, label) in enumerate(zip(POINTS, ('Near the well','Intermediate distance','Far from the well'))):
        matches = np.flatnonzero(np.all(xy == point,axis=1))
        if len(matches)!=1: raise ValueError('Selected point is not a unique mesh node.')
        i=matches.item()
        for j in range(3):
            ax=axes[j,c]
            ax.plot(days,truth[:,j,i],color='#222222',label='Ensemble-average truth',lw=2)
            for model in models:
                ax.plot(days,predictions[model][:,j,i], '--' if model=='kernel' else '-.',
                        color=COLORS[model],label=DISPLAY[model],lw=1.8)
            ax.set_ylabel(LABELS[j]+f' ({UNITS[j]})')
            interval(ax)
            if j==0:ax.set_title(f'{label}\n({point[0]:g}, {point[1]:g}) m')
    axes[0,0].legend()
    fig.suptitle('Time histories at selected locations comparison',fontsize=21)
    finish(fig,'selected_locations')
    print('Figures:',out)
