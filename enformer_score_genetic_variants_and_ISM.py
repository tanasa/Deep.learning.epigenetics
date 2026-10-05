# -*- coding: utf-8 -*-
"""
Enformer variant-effect scoring on ALL variants of a VCF  --  Google Colab script
=================================================================================

Adapted from the DeepMind tutorial
  https://github.com/google-deepmind/deepmind-research/blob/master/enformer/enformer-usage.ipynb
and from the notebook  Enformer.score.genetic.variants.NHCFV.v5.on.5.variants.pics.ipynb

WHAT THIS SCRIPT DOES
---------------------
For every variant in the input VCF (HCM GWAS hits):
  1. Cut a 393,216 bp window centred on the variant from the reference genome and build
     two sequences: REF (as in the genome) and ALT (with the variant substituted).
  2. Run Enformer on both. Enformer predicts 896 bins (128 bp each, ~115 kb in the middle of
     the window) for each of 5,313 human tracks (DNase/ATAC, ChIP-seq, CAGE).
  3. Variant effect per track ("SAD" score) = mean over the 896 bins of (ALT - REF).
     A second, more local score uses only the 3 bins around the variant.
  4. Standardise the SAD scores with the fitted scaler shipped with Enformer (z-scores, so
     tracks are comparable), and also compute the first 20 principal components.
  5. RANK tissues / cell types for every variant, separately for each assay group:
        CAGE | DNase | ATAC | histone ChIP | TF/other ChIP
     For a cell type with several tracks (e.g. many histone marks, replicates), the effect of
     the variant on that cell type is the track with the largest |z|.

The model is run ONCE per variant (2 forward passes); every table below is derived from those
predictions. Progress is checkpointed to Drive so a Colab disconnect does not lose the run:
just re-run the script and it resumes.

EXTRAS IN THIS VERSION (cells 11-12; resumable, chunkable via RANGE_START/RANGE_END, run AFTER all the core outputs are saved, and guarded so that
a failure there cannot destroy the scoring results):
  * gradient x input attribution (saliency) for the N strongest variants: which bases around the
    variant drive the prediction of each assay group's strongest track, for REF and ALT
  * in-silico mutagenesis (ISM) around the strongest few variants: effect of every possible
    single-base change on the same tracks
  * clustered heatmap of all variants over the most variable tracks
NOT included: embeddings / attention. The TF Hub model as loaded only exposes predict_on_batch,
so internal activations are not reachable from it.

OUTPUTS  (written to OUT_DIR on Google Drive)
---------------------------------------------
  track_annotations.tsv                       track index -> assay, group, mark, cell type
  variants_scored.tsv                         one row per scored variant (+ REF-check flag)
  matrix_raw_SAD_variants_x_tracks.tsv.gz     variants x 5313, alt-ref mean over all bins
  matrix_raw_local3bins_variants_x_tracks.tsv.gz   variants x 5313, alt-ref over 3 central bins
  matrix_zscore_variants_x_tracks.tsv.gz      variants x 5313, z-scored SAD  (main matrix)
  matrix_zscore_variants_x_tracks.npy         same, numpy
  pca20_variants.tsv                          first 20 PCs of the z-scored SAD
  ranking_celltypes_per_variant_<group>.tsv   TOP_N cell types per variant, per assay group
  ranking_celltypes_per_variant_ALL_groups.tsv  the same, all groups in one long table
  matrix_celltype_effect_<group>.tsv.gz       variants x cell types (signed z at the best track)
  summary_top_celltypes_per_variant.tsv       one row per variant: top 3 cell types per group and
                                              'largest_change_overall' (the single strongest cell type)
  ranking_variants_per_celltype_<group>.tsv   REVERSE ranking: for each cell type / tissue, the top
                                              TOP_VARIANTS variants by |z| (+ ..._ALL_groups.tsv)
  matrix_peak_distance_bp_variants_x_tracks.tsv.gz  where (bp from the variant) each track changes most
  matrix_peak_delta_variants_x_tracks.tsv.gz  ALT-REF value at that peak bin
  (rankings also carry peak_pos / peak_dist_bp / peak_delta for each cell type's strongest track)
  figures/per_variant/<variant>.png + figures/all_variants_top_celltypes.pdf
        one figure per variant: top cell types per assay group + the locus (ALT-REF along the
        ~115 kb window for each group's strongest track, peak marked)
  figures/fig1..fig4  overview figures (top cell types, heatmaps, PCA, genome tracks of top 5)

  RAW MODEL UNITS (no z-scores; units differ between assays, compare within an assay only):
  matrix_raw_REF_mean / ALT_mean / mean_abs_delta / max_abs_delta _variants_x_tracks.tsv.gz
        REF mean, ALT mean, mean |ALT-REF| and max |ALT-REF| over the 896 bins
        (the signed mean change ALT-REF is matrix_raw_SAD_...; all five also in matrices_raw_5_measures.npz)
  ranking_celltypes_per_variant_RAW_mean_abs_delta.tsv
        cell types per variant and assay group ranked by the MEDIAN mean |ALT-REF| of their tracks
  preflight_errors.tsv                        every variant that failed the pre-run check (empty = all OK)
  spatial_heart_fibroblast/<variant>.npz      REF and ALT at all 896 bins for every heart/fibroblast
                                              track (+ tracks.tsv listing those tracks)
  gene_effects_heart_fibroblast_TSS.tsv       cell 13: every gene with its TSS in the output window,
                                              log2 fold change ALT/REF at the TSS per assay group,
                                              heart and fibroblast separately (from the saved
                                              predictions, no extra model run)
"""

# %% ----------------------------------------------------------------------------------
# CELL 1 - configuration.  Edit this block only.
# ---------------------------------------------------------------------------------------
import os, re, gzip, json, sys, time, subprocess

# Data that already lives on your Drive (same paths as the original notebook).
BASE_DIR = '/content/drive/MyDrive/Enformer_tutorial_Google_DeepMind'
TRANSFORM_PATH = f'{BASE_DIR}/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
MODEL_PATH = f'{BASE_DIR}/model/1'
FASTA_FILE = f'{BASE_DIR}/genome.fa'
TARGETS_TXT = f'{BASE_DIR}/targets_human.txt'
VCF_PATH = (f'{BASE_DIR}/gwas.amgen/'
            'gwas.41467_2025_64070_MOESM4_ESM.2b_HCM.snp.txt.POS.ensembl_snv2.dbsnp138.broad.vcf.gz')

# Drive folder:  results / Claude_model_embeddings_ext2
OUT_DIR = '/content/drive/MyDrive/Enformer_use_GWAS_hits_Amgen/results/Claude_model_embeddings_ext2'

SEQUENCE_LENGTH = 393_216      # Enformer input length (bp). Do not change.
NUM_BINS = 896                 # Enformer output bins (128 bp each). Do not change.
CENTER_BINS = 3                # bins around the variant used for the "local" score
NUM_PCS = 20                   # principal components to keep
TOP_N = 10                     # cell types kept per variant and assay group
TOP_VARIANTS = 50              # variants kept per cell type / tissue in the reverse ranking
RANK_BY = 'max_abs_z'          # 'max_abs_z' (strongest single track) or 'mean_abs_z'
CHECKPOINT_EVERY = 25          # save progress to Drive every N variants
# ---- gradient x input / ISM selection (cell 11) -------------------------------------
# Variants are ranked by their strongest |z| (strongest first); RANGE_START/RANGE_END slice that
# ranked list so a long run can be split into chunks. Files that already exist are skipped, so
# re-running (or resuming after a disconnect) continues where it stopped.
ATTR_TOP_VARIANTS = None       # gradient x input: None = ALL variants, N = the N strongest, 0 = skip
ISM_TOP_VARIANTS = None        # ISM: None = ALL variants, N = the N strongest, 0 = skip
ISM_HEART_FIBRO_TOP = 0        # (only needed when ISM_TOP_VARIANTS is a number) ISM also for the N variants with the strongest effect on heart/cardiac/
                               # fibroblast tracks (ranked by per-track percentile of |z|, 0 = skip)
ISM_EXTRA_IDS = []             # ISM also for these rsIDs, e.g. ['rs17037388', 'rs13306561']
RANGE_START = 0                # first / last position (in the strongest-first order) to process
RANGE_END = None               #   e.g. RANGE_START=0, RANGE_END=150, then 150-300, ... (None = to the end)
ATTR_WINDOW = 300              # bp on each side of the variant shown in attribution plots
ATTR_HALF_BINS = 8             # output bins (each side of the centre) whose predictions are attributed
ISM_HALF_WIDTH = 40            # bp on each side of the variant mutated in ISM (3 mutations per base)
ISM_BATCH_SIZE = 8             # sequences per forward pass in ISM (halved automatically if GPU memory runs out)
MAX_VARIANTS = None            # None = all variants; set e.g. 5 for a quick test run
CHR_PREFIX = 'chr'             # VCF has '1', the genome FASTA has 'chr1'
# ---- preflight, raw measures, heart/fibroblast spatial predictions, gene effects (cells 6, 7, 13)
PREFLIGHT_STRICT = True        # check EVERY variant before any model run and stop if one fails
                               # (False = old behaviour: only the first 100, 90% must match)
HEART_RE = r'\b(?:heart|cardiac|cardio\w*|ventric\w*|atri(?:al|um|a)|aort\w*|coronary)\b'
FIBRO_RE = r'\bfibroblast\w*'  # both matched on the cell-type text, never on the ChIP target
SAVE_HF_SPATIAL = True         # save REF and ALT at all 896 bins for every heart/fibroblast track
GTF_URL = ('https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/'
           'gencode.v46.basic.annotation.gtf.gz')
GTF_PATH = f'{BASE_DIR}/gencode.v46.basic.annotation.gtf.gz'   # downloaded once (cell 13)
GENE_TYPES = ('protein_coding', 'lncRNA')   # genes whose TSS effect is computed in cell 13
TSS_HALF_BINS = 1              # signal summed over the TSS bin +-1 (3 bins, as in the Enformer paper)
PSEUDOCOUNT = 1.0              # log2 fold change = log2((ALT + pc) / (REF + pc))

# %% ----------------------------------------------------------------------------------
# CELL 2 - install libraries, check GPU, mount Drive
# ---------------------------------------------------------------------------------------
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kipoiseq==0.5.2', 'pyfaidx'],
               check=False)          # a pyYAML warning here can be ignored

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import tensorflow_hub as hub
import pyfaidx
import kipoiseq
from kipoiseq import Interval

assert tf.config.list_physical_devices('GPU'), \
    'No GPU. In Colab: Runtime -> Change runtime type -> GPU'

from google.colab import drive
drive.mount('/content/drive')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/checkpoint', exist_ok=True)
print('Outputs will be written to:', OUT_DIR)

# %% ----------------------------------------------------------------------------------
# CELL 3 - helpers: annotate tracks and rank cell types.
# Pure numpy/pandas (no TensorFlow), so they can be tested without a GPU.
# ==== HELPERS BEGIN
# ---------------------------------------------------------------------------------------
HISTONE_RE = re.compile(r'^H\d')     # H3K27ac, H3K4me1, H2AFZ, H4K20me1 ... = histone marks


def clean_cell_type(text):
    """Normalise a track's cell-type text so replicates/donors collapse onto one label.

    Removes parenthesised details (ages, donors), sex/age words, pool/donor/replicate tags and
    treatment/origin suffixes. Developmental stage words 'embryo' and 'fetal' are KEPT, so adult
    and fetal heart stay separate. It is a heuristic: the untouched description is always kept
    in the track annotation table.
    """
    s = text.lower()
    s = re.sub(r'\([^)]*\)', ' ', s)                                        # (27 years), (GFH2)
    s = re.sub(r'\b(originated from|genetically modified|treated with)\b.*$', '', s)
    s = re.sub(r',?\s*\b(pool|donor|rep|biol_rep|tech_rep)\S*', '', s)      # ', pool1'
    s = re.sub(r'\b(male|female|adult|child|newborn)\b', ' ', s)
    for _ in range(3):
        s = re.sub(r'\s+', ' ', s).strip(' ,;-')
        s = re.sub(r'\s+and(\s+embryo)?$', '', s)                           # '... and embryo'
    return s or text.lower()


def annotate_tracks(df_targets):
    """One row per Enformer track: index, assay, group, histone mark / TF, cell type.

    Groups: CAGE | DNase | ATAC | ChIP_histone | ChIP_TF.
    Description formats in targets_human.txt:
        'CAGE:heart, adult, pool1'   'DNASE:heart male adult (27 years)'
        'CHIP:H3K27ac:heart left ventricle ...'   'CHIP:CTCF:IMR-90'
    """
    rows = []
    for idx, desc in zip(df_targets['index'], df_targets['description']):
        assay, rest = desc.split(':', 1)
        assay = assay.upper()
        mark = ''
        if assay == 'CHIP':
            mark, rest = rest.split(':', 1) if ':' in rest else (rest, '')
            group = 'ChIP_histone' if HISTONE_RE.match(mark) else 'ChIP_TF'
        elif assay == 'DNASE':
            group = 'DNase'
        else:
            group = assay                      # 'CAGE' or 'ATAC'
        rows.append({'track_index': int(idx), 'assay': assay, 'group': group,
                     'mark_or_factor': mark, 'cell_type': clean_cell_type(rest),
                     'description': desc,
                     'track_name': f'{idx}_{desc}'})
    return pd.DataFrame(rows)


def rank_cell_types(Z, variants, annot, top_n=10, rank_by='max_abs_z',
                    peak_bin=None, peak_delta=None, window_starts=None):
    """Rank cell types / tissues by how strongly each variant affects them.

    Z         : (n_variants, n_tracks) z-scored effects, column j = track_index j
    variants  : DataFrame with one row per variant (chrom, pos, id, ref, alt, variant_key)
    annot     : output of annotate_tracks()
    peak_bin, peak_delta, window_starts : optional. (n_variants, n_tracks) peak bin and ALT-REF at
              the peak, and the genomic start of each variant's 896-bin window. When given, the ranking
              gets peak_pos / peak_dist_bp / peak_delta columns for each cell type's strongest track:
              WHERE in the locus that track changes most (bin b spans window_start + 128 b ...).
    Returns   : (long_df, matrices)
      long_df  - top_n cell types per variant and per assay group
      matrices - {group: DataFrame variants x cell types}, signed z at the strongest track

    For each (variant, group, cell type) the strongest track = largest |z| among that cell
    type's tracks in the group. 'z_at_best_track' keeps its sign (+ = variant increases the
    predicted signal, - = decreases it). n_tracks says how many tracks back the cell type, since
    cell types with many tracks (e.g. IMR-90 histone marks) have more chances to score high.
    """
    assert rank_by in ('max_abs_z', 'mean_abs_z')
    n = Z.shape[0]
    rows, matrices = [], {}
    for group, gdf in annot.groupby('group'):
        cells = list(gdf.groupby('cell_type')['track_index'].apply(list).items())
        names = [c for c, _ in cells]
        max_abs = np.zeros((n, len(cells)), dtype=np.float32)
        z_best = np.zeros_like(max_abs)
        mean_abs = np.zeros_like(max_abs)
        mean_z = np.zeros_like(max_abs)
        best_idx = np.zeros((n, len(cells)), dtype=np.int32)
        n_tracks = np.array([len(ix) for _, ix in cells])
        for k, (_, ix) in enumerate(cells):
            ix = np.asarray(ix)
            sub = Z[:, ix]
            j = np.abs(sub).argmax(axis=1)
            z_best[:, k] = sub[np.arange(n), j]
            max_abs[:, k] = np.abs(z_best[:, k])
            best_idx[:, k] = ix[j]
            mean_abs[:, k] = np.abs(sub).mean(axis=1)
            mean_z[:, k] = sub.mean(axis=1)
        score = max_abs if rank_by == 'max_abs_z' else mean_abs
        matrices[group] = pd.DataFrame(z_best, index=variants['variant_key'].values, columns=names)
        track_name = annot.set_index('track_index')['track_name']
        for v in range(n):
            order = np.argsort(-score[v])[:top_n]
            meta = variants.iloc[v]
            for rank, k in enumerate(order, 1):
                rows.append({'variant_id': meta['id'], 'chrom': meta['chrom'], 'pos': meta['pos'],
                             'ref': meta['ref'], 'alt': meta['alt'], 'group': group, 'rank': rank,
                             'cell_type': names[k], 'n_tracks': int(n_tracks[k]),
                             'z_at_best_track': float(z_best[v, k]),
                             'max_abs_z': float(max_abs[v, k]),
                             'mean_abs_z': float(mean_abs[v, k]),
                             'mean_z': float(mean_z[v, k]),
                             'direction': 'INCREASE' if z_best[v, k] > 0 else 'DECREASE',
                             'best_track_index': int(best_idx[v, k]),
                             'best_track': track_name[int(best_idx[v, k])]})
                if peak_bin is not None:
                    b = int(peak_bin[v, best_idx[v, k]])
                    ppos = int(window_starts[v]) + b * 128 + 64          # bin centre
                    rows[-1].update({'peak_pos': ppos, 'peak_dist_bp': ppos - int(meta['pos']),
                                     'peak_delta': float(peak_delta[v, best_idx[v, k]])})
    return pd.DataFrame(rows), matrices


def rank_variants_per_celltype(Z, variants, annot, top_v=50,
                               peak_bin=None, peak_delta=None, window_starts=None):
    """REVERSE ranking: for every cell type / tissue, the variants with the strongest effect.

    For each (assay group, cell type) the variants are ordered by |z| at the cell type's
    strongest track (largest |z| among its tracks). Keeps the top_v variants per cell type.
    Columns: group, cell_type, rank, variant, z_at_best_track (signed), direction, n_tracks
    (tracks backing the cell type), best_track, and - if given - where that track peaks
    (peak_dist_bp from the variant, peak_delta).
    """
    n = Z.shape[0]
    track_name = annot.set_index('track_index')['track_name']
    rows = []
    for group, gdf in annot.groupby('group'):
        for cell, ix in gdf.groupby('cell_type')['track_index'].apply(list).items():
            ix = np.asarray(ix)
            sub = Z[:, ix]
            j = np.abs(sub).argmax(axis=1)
            zb = sub[np.arange(n), j]
            bi = ix[j]
            for rank, v in enumerate(np.argsort(-np.abs(zb))[:top_v], 1):
                meta = variants.iloc[v]
                r = {'group': group, 'cell_type': cell, 'rank': rank, 'variant_id': meta['id'],
                     'chrom': meta['chrom'], 'pos': meta['pos'], 'ref': meta['ref'], 'alt': meta['alt'],
                     'z_at_best_track': float(zb[v]), 'abs_z': float(abs(zb[v])),
                     'direction': 'INCREASE' if zb[v] > 0 else 'DECREASE',
                     'n_tracks': len(ix), 'best_track_index': int(bi[v]),
                     'best_track': track_name[int(bi[v])]}
                if peak_bin is not None:
                    ppos = int(window_starts[v]) + int(peak_bin[v, bi[v]]) * 128 + 64
                    r.update({'peak_dist_bp': ppos - int(meta['pos']),
                              'peak_delta': float(peak_delta[v, bi[v]])})
                rows.append(r)
    return pd.DataFrame(rows)


def summarise_top(long_df, variants, top=3):
    """One row per variant; per assay group the top cell types as 'name (z)' joined by ' | '."""
    out = variants[['variant_key', 'id', 'chrom', 'pos', 'ref', 'alt']].copy().set_index('variant_key')
    for group, g in long_df.groupby('group'):
        g = g[g['rank'] <= top].copy()
        g['txt'] = g['cell_type'] + ' (' + g['z_at_best_track'].map('{:+.1f}'.format) + ')'
        s = g.groupby(['chrom', 'pos', 'ref', 'alt', 'variant_id'])['txt'].apply(' | '.join)
        out[f'top_{group}'] = [s.get((c, p, r, a, i), '') for c, p, r, a, i in
                               zip(variants['chrom'], variants['pos'], variants['ref'],
                                   variants['alt'], variants['id'])]
    # single overall answer: the cell type (any assay group) with the largest |z| for this variant
    top1 = long_df[long_df['rank'] == 1]
    best = top1.loc[top1.groupby(['variant_id', 'chrom', 'pos', 'ref', 'alt'])['max_abs_z'].idxmax()] \
        .set_index(['variant_id', 'chrom', 'pos', 'ref', 'alt'])
    def _overall(v):
        b = best.loc[(v['id'], v['chrom'], v['pos'], v['ref'], v['alt'])]
        return f"{b['cell_type']} [{b['group']}] ({b['z_at_best_track']:+.1f})"
    out['largest_change_overall'] = [_overall(v) for _, v in variants.iterrows()]
    return out.reset_index()


def rank_cell_types_raw(mean_abs, delta, variants, annot, top_n=10):
    """Ranking in RAW model units: for each assay group and cell type, the MEDIAN over its tracks
    of mean |ALT-REF| (effect size) and of the signed mean ALT-REF (net direction).
    mean_abs, delta: (n_variants, n_tracks). Raw units differ between assays, so ranks are only
    meaningful within a group. Returns the top_n cell types per variant and group."""
    rows = []
    for group, gdf in annot.groupby('group'):
        cells = list(gdf.groupby('cell_type')['track_index'].apply(list).items())
        med_abs = np.stack([np.median(mean_abs[:, ix], axis=1) for _, ix in cells], axis=1)
        med_d = np.stack([np.median(delta[:, ix], axis=1) for _, ix in cells], axis=1)
        for v in range(mean_abs.shape[0]):
            meta = variants.iloc[v]
            for rank, k in enumerate(np.argsort(-med_abs[v])[:top_n], 1):
                rows.append({'variant_id': meta['id'], 'chrom': meta['chrom'], 'pos': meta['pos'],
                             'ref': meta['ref'], 'alt': meta['alt'], 'group': group, 'rank': rank,
                             'cell_type': cells[k][0], 'n_tracks': len(cells[k][1]),
                             'median_mean_abs_delta': float(med_abs[v, k]),
                             'median_delta_mean': float(med_d[v, k]),
                             'net_direction': 'INCREASE' if med_d[v, k] > 0 else
                                              'DECREASE' if med_d[v, k] < 0 else 'NONE'})
    return pd.DataFrame(rows)


def load_gene_tss(gtf_path, gene_types):
    """Genes of the given types from a GENCODE GTF: chrom, gene_id (no version), gene_name,
    gene_type, strand, tss (1-based; Ensembl_canonical transcript start when tagged, else gene
    start/end by strand). chrY copies of pseudo-autosomal genes are dropped."""
    genes, canon = [], {}
    opener = gzip.open if gtf_path.endswith('.gz') else open
    with opener(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.rstrip('\n').split('\t')
            if len(p) < 9 or p[2] not in ('gene', 'transcript'):
                continue
            raw = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
            if raw.endswith('_PAR_Y'):
                continue
            gid, start, end, strand = raw.split('.')[0], int(p[3]), int(p[4]), p[6]
            if p[2] == 'gene':
                gtype = re.search(r'gene_type "([^"]+)"', p[8])
                name = re.search(r'gene_name "([^"]+)"', p[8])
                if gtype and gtype.group(1) in gene_types:
                    genes.append({'chrom': p[0], 'gene_id': gid, 'gene_name': name.group(1) if name else gid,
                                  'gene_type': gtype.group(1), 'strand': strand, 'start': start, 'end': end})
            elif 'Ensembl_canonical' in p[8]:
                canon[gid] = start if strand == '+' else end
    g = pd.DataFrame(genes)
    g['tss'] = [canon.get(i, s if st == '+' else e)
                for i, s, e, st in zip(g['gene_id'], g['start'], g['end'], g['strand'])]
    return g.drop(columns=['start', 'end'])


def tss_gene_effects(p_ref, p_alt, genes_chrom, pos, half_bins=1, pc=1.0):
    """Genes whose TSS lies in the variant's 896-bin output window, and their log2 fold change
    ALT/REF at the TSS. p_ref, p_alt: (896, n_tracks) predictions for the saved tracks.
    Bin b covers the 0-based genome interval [pos - 57344 + 128 b, +128) (same centring as the
    rest of the script; the variant itself is in bin 447). Signal = sum over TSS bin +-half_bins.
    Returns (genes in window with tss_bin / tss_dist_bp, lfc (n_genes, n_tracks),
    ref_signal (n_genes, n_tracks))."""
    o = int(pos) - NUM_BINS * 64
    b = (genes_chrom['tss'].values - 1 - o) // 128
    keep = (b >= half_bins) & (b < NUM_BINS - half_bins)
    g = genes_chrom.loc[keep].copy()
    g['tss_bin'], g['tss_dist_bp'] = b[keep], g['tss'].values - int(pos)
    lfc = np.zeros((len(g), p_ref.shape[1]), np.float32)
    ref_sig = np.zeros_like(lfc)
    for i, bb in enumerate(g['tss_bin'].values):
        r = p_ref[bb - half_bins:bb + half_bins + 1].sum(axis=0)
        a = p_alt[bb - half_bins:bb + half_bins + 1].sum(axis=0)
        lfc[i], ref_sig[i] = np.log2((a + pc) / (r + pc)), r
    return g.reset_index(drop=True), lfc, ref_sig
# ==== HELPERS END

# %% ----------------------------------------------------------------------------------
# CELL 4 - load track table and build the annotation
# ---------------------------------------------------------------------------------------
df_targets = pd.read_csv(TARGETS_TXT, sep='\t')
annot = annotate_tracks(df_targets)
assert len(annot) == 5313 and (annot['track_index'].values == np.arange(5313)).all()
annot['heart'] = annot['cell_type'].str.contains(HEART_RE, regex=True)
annot['fibroblast'] = annot['cell_type'].str.contains(FIBRO_RE, regex=True)
annot['heart_or_fibroblast'] = annot['heart'] | annot['fibroblast']
print(f"heart tracks: {annot['heart'].sum()}, fibroblast tracks: {annot['fibroblast'].sum()}, "
      f"either: {annot['heart_or_fibroblast'].sum()}")
annot.to_csv(f'{OUT_DIR}/track_annotations.tsv', sep='\t', index=False)
print(annot.groupby('group').agg(n_tracks=('track_index', 'size'),
                                 n_cell_types=('cell_type', 'nunique')))

# %% ----------------------------------------------------------------------------------
# CELL 5 - model wrapper, sequence extraction, VCF reader
# ---------------------------------------------------------------------------------------
class Enformer:
    """Thin wrapper around the saved Enformer model (loaded once)."""

    def __init__(self, model_path):
        self._model = hub.load(model_path).model

    def predict_on_batch(self, inputs):
        """inputs: (batch, 393216, 4) one-hot -> {'human': (batch, 896, 5313), 'mouse': ...}"""
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}


class FastaStringExtractor:
    """Reads a genomic interval from the FASTA; pads with N when it runs off a chromosome."""

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval, **kwargs):
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed = Interval(interval.chrom, max(interval.start, 0),
                           min(interval.end, chromosome_length))
        sequence = str(self.fasta.get_seq(trimmed.chrom, trimmed.start + 1,   # pyfaidx is 1-based
                                          trimmed.stop).seq).upper()
        pad_up = 'N' * max(-interval.start, 0)
        pad_down = 'N' * max(interval.end - chromosome_length, 0)
        return pad_up + sequence + pad_down

    def base_at(self, chrom, pos1):
        """Reference base at a 1-based position (used to check that REF matches the genome)."""
        return str(self.fasta[chrom][pos1 - 1:pos1]).upper()


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def iter_vcf(vcf_path):
    """Yield (chrom, pos, id, ref, alt) for every ALT allele of every VCF record."""
    opener = gzip.open if vcf_path.endswith('.gz') else open
    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            chrom, pos, vid, ref, alts = line.rstrip('\n').split('\t')[:5]
            for alt in alts.split(','):
                yield chrom, int(pos), vid, ref, alt


def variant_key(chrom, pos, ref, alt, vid):
    return f'{CHR_PREFIX}{chrom}:{pos}:{ref}:{alt}:{vid}'


fasta_extractor = FastaStringExtractor(FASTA_FILE)
seq_extractor = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=fasta_extractor)


def make_ref_alt(chrom, pos, vid, ref, alt):
    """REF and ALT one-hot sequences (393216 x 4) centred on the variant."""
    variant = kipoiseq.dataclasses.Variant(chrom=CHR_PREFIX + chrom, pos=pos, ref=ref, alt=alt, id=vid)
    # Same centring as the DeepMind tutorial (interval built on the 1-based position).
    interval = Interval(CHR_PREFIX + chrom, pos, pos).resize(SEQUENCE_LENGTH)
    center = interval.center() - interval.start
    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)
    return one_hot_encode(reference), one_hot_encode(alternate)

# %% ----------------------------------------------------------------------------------
# CELL 6 - PRE-FLIGHT: EVERY variant is checked before any model run.
#   chromosome present in the FASTA, position inside it, A/C/G/T single-base REF and ALT,
#   REF != ALT, REF equal to the genome base. Every failure is written to preflight_errors.tsv.
# A mismatch on most variants means the VCF and genome.fa are different assemblies
# (the VCF header says GATK human_g1k_v37 = GRCh37, Enformer/hg38 needs GRCh38 coordinates).
# ---------------------------------------------------------------------------------------
pf_errors, n_chk = [], 0
for chrom, pos, vid, ref, alt in iter_vcf(VCF_PATH):
    n_chk += 1
    c, err = CHR_PREFIX + chrom, ''
    if c not in fasta_extractor._chromosome_sizes:
        err = f'chromosome {c} not in the FASTA'
    elif not 1 <= pos <= fasta_extractor._chromosome_sizes[c]:
        err = 'position outside the chromosome'
    elif len(ref) != 1 or len(alt) != 1 or ref.upper() not in 'ACGT' or alt.upper() not in 'ACGT':
        err = 'not an A/C/G/T single-base substitution'
    elif ref.upper() == alt.upper():
        err = 'REF equals ALT'
    else:
        base = fasta_extractor.base_at(c, pos)
        if base != ref.upper():
            err = f'REF mismatch: VCF={ref} FASTA={base}'
    if err:
        pf_errors.append({'chrom': c, 'pos': pos, 'id': vid, 'ref': ref, 'alt': alt, 'error': err})
pf_errors = pd.DataFrame(pf_errors, columns=['chrom', 'pos', 'id', 'ref', 'alt', 'error'])
pf_errors.to_csv(f'{OUT_DIR}/preflight_errors.tsv', sep='\t', index=False)
print(f'Pre-flight: {n_chk - len(pf_errors)}/{n_chk} variants OK')
if len(pf_errors):
    print(pf_errors['error'].str.split(':').str[0].value_counts().to_string())
    print(pf_errors.head(20).to_string())
if PREFLIGHT_STRICT:
    assert pf_errors.empty, (
        f'{len(pf_errors)} variants failed the pre-flight check (see {OUT_DIR}/preflight_errors.tsv); '
        'no model was run. If most are REF mismatches, the VCF is on another assembly (GRCh37 vs '
        'GRCh38): lift it over. To score only the valid ones, remove the failing records or set '
        'PREFLIGHT_STRICT = False.')
else:
    assert 1 - len(pf_errors) / max(n_chk, 1) > 0.9, (
        'more than 10% of the variants fail the pre-flight check; see preflight_errors.tsv')

# %% ----------------------------------------------------------------------------------
# CELL 7 - score every variant (resumable)
#   raw SAD   = mean over 896 bins of (ALT - REF)             -> (5313,)
#   local SAD = mean over the CENTER_BINS central bins         -> (5313,)
#   peak      = for every track, the bin with the largest |ALT - REF| and its value
#   curves    = ALT - REF along the window for the strongest track (largest |z|) of each
#               assay group, kept so every per-variant figure can show the locus
#   raw       = REF mean, ALT mean, mean |ALT - REF|, max |ALT - REF| over the 896 bins
#   spatial   = REF and ALT at all 896 bins for every heart/fibroblast track, one file per variant
# Variants in a checkpoint from an older version (no raw measures / no spatial file) are
# predicted once more to fill those in; their other scores are kept.
# ---------------------------------------------------------------------------------------
with tf.io.gfile.GFile(TRANSFORM_PATH, 'rb') as f:
    pipeline = joblib.load(f)
scaler = pipeline.steps[0][1]              # first step of the pickled sklearn pipeline (z-scores)

GROUPS = sorted(annot['group'].unique())
group_tracks = [annot.loc[annot['group'] == g, 'track_index'].values for g in GROUPS]
model = Enformer(MODEL_PATH)
CKPT = f'{OUT_DIR}/checkpoint/scores_checkpoint.npz'
CKPT_META = f'{OUT_DIR}/checkpoint/scores_checkpoint_meta.tsv'

RAW_KEYS = ('ref_mean', 'alt_mean', 'mean_abs_delta', 'max_abs_delta')
sad_rows, local_rows, peakbin_rows, peakdelta_rows, curve_rows, curvetrack_rows, meta_rows = \
    [], [], [], [], [], [], []
raw_rows = {k: [] for k in RAW_KEYS}
if os.path.exists(CKPT) and os.path.exists(CKPT_META):        # resume
    _c = np.load(CKPT)
    if {'peak_bin', 'peak_delta', 'curves', 'curve_track'} <= set(_c.files):
        sad_rows, local_rows = list(_c['sad']), list(_c['local'])
        peakbin_rows, peakdelta_rows = list(_c['peak_bin']), list(_c['peak_delta'])
        curve_rows, curvetrack_rows = list(_c['curves']), list(_c['curve_track'])
        meta_rows = pd.read_csv(CKPT_META, sep='\t', dtype=str, keep_default_na=False).to_dict('records')
        if set(RAW_KEYS) <= set(_c.files):
            raw_rows = {k: list(_c[k]) for k in RAW_KEYS}
        else:                                                   # older checkpoint: fill in below
            raw_rows = {k: [np.full(5313, np.nan, np.float32) for _ in meta_rows] for k in RAW_KEYS}
            print('Checkpoint has no raw REF/ALT measures -> those variants are predicted once more to add them')
        print(f'Resuming: {len(meta_rows)} variants already scored')
    else:
        print('Checkpoint from an older script version (no peak/curve data) -> starting fresh')
key_index = {m['variant_key']: j for j, m in enumerate(meta_rows)}

HF_DIR = f'{OUT_DIR}/spatial_heart_fibroblast'
hf_idx = annot.index[annot['heart_or_fibroblast']].values.astype(np.int32)
if SAVE_HF_SPATIAL:
    os.makedirs(HF_DIR, exist_ok=True)
    annot.loc[hf_idx].to_csv(f'{HF_DIR}/tracks.tsv', sep='\t', index=False)


def hf_file(chrom_with_prefix, pos, vid, ref, alt):
    return f"{HF_DIR}/{re.sub(r'[^A-Za-z0-9._-]+', '_', f'{vid}_{chrom_with_prefix}_{pos}_{ref}_{alt}')}.npz"


def save_checkpoint():
    np.savez(CKPT, sad=np.array(sad_rows, dtype=np.float32), local=np.array(local_rows, dtype=np.float32),
             peak_bin=np.array(peakbin_rows, dtype=np.int16),
             peak_delta=np.array(peakdelta_rows, dtype=np.float32),
             curves=np.array(curve_rows, dtype=np.float32),
             curve_track=np.array(curvetrack_rows, dtype=np.int32),
             **{k: np.array(raw_rows[k], dtype=np.float32) for k in RAW_KEYS})
    pd.DataFrame(meta_rows).to_csv(CKPT_META, sep='\t', index=False)


mid = NUM_BINS // 2
central = slice(mid - CENTER_BINS // 2, mid - CENTER_BINS // 2 + CENTER_BINS)
all_tracks = np.arange(5313)
t0, n_new, n_fill = time.time(), 0, 0
for i, (chrom, pos, vid, ref, alt) in enumerate(iter_vcf(VCF_PATH)):
    if MAX_VARIANTS is not None and i >= MAX_VARIANTS:
        break
    key = variant_key(chrom, pos, ref, alt, vid)
    f_hf = hf_file(CHR_PREFIX + chrom, pos, vid, ref, alt)
    j = key_index.get(key)                                             # None = not scored yet
    need_raw = j is not None and not np.isfinite(raw_rows['ref_mean'][j]).all()
    need_hf = SAVE_HF_SPATIAL and not os.path.exists(f_hf)
    if j is not None and not (need_raw or need_hf):
        continue
    ref_ok = fasta_extractor.base_at(CHR_PREFIX + chrom, pos) == ref.upper()
    x_ref, x_alt = make_ref_alt(chrom, pos, vid, ref, alt)
    p_ref = model.predict_on_batch(x_ref[np.newaxis])['human'][0]      # (896, 5313)
    p_alt = model.predict_on_batch(x_alt[np.newaxis])['human'][0]
    d = p_alt - p_ref                                                  # (896, 5313)
    if SAVE_HF_SPATIAL:
        np.savez_compressed(f_hf, ref=p_ref[:, hf_idx].astype(np.float32),
                            alt=p_alt[:, hf_idx].astype(np.float32), track_index=hf_idx,
                            variant_key=np.array(key))
    raw_v = {'ref_mean': p_ref.mean(axis=0), 'alt_mean': p_alt.mean(axis=0),
             'mean_abs_delta': np.abs(d).mean(axis=0), 'max_abs_delta': np.abs(d).max(axis=0)}
    if j is not None:                              # scored by an older version: only fill in
        for k in RAW_KEYS:
            raw_rows[k][j] = raw_v[k].astype(np.float32)
        n_fill += 1
    else:
        sad_v = d.mean(axis=0)
        pk = np.abs(d).argmax(axis=0)                                  # peak bin of each track
        z_v = scaler.transform(sad_v[np.newaxis])[0]
        strongest = np.array([gt[np.abs(z_v[gt]).argmax()] for gt in group_tracks])  # one per group
        sad_rows.append(sad_v.astype(np.float32))
        local_rows.append(d[central].mean(axis=0).astype(np.float32))
        peakbin_rows.append(pk.astype(np.int16))
        peakdelta_rows.append(d[pk, all_tracks].astype(np.float32))
        curve_rows.append(d[:, strongest].T.astype(np.float32))        # (n_groups, 896)
        curvetrack_rows.append(strongest.astype(np.int32))
        for k in RAW_KEYS:
            raw_rows[k].append(raw_v[k].astype(np.float32))
        meta_rows.append({'variant_key': key, 'chrom': CHR_PREFIX + chrom, 'pos': str(pos), 'id': vid,
                          'ref': ref, 'alt': alt, 'ref_matches_genome': str(bool(ref_ok))})
        key_index[key] = len(meta_rows) - 1
    n_new += 1
    if n_new % CHECKPOINT_EVERY == 0:
        save_checkpoint()
        rate = (time.time() - t0) / n_new
        print(f'{len(meta_rows)} scored, {n_fill} filled in | {rate:.1f} s/variant | '
              f'~{rate * max(0, n_chk - len(meta_rows)) / 60:.0f} min left for new variants', flush=True)
save_checkpoint()
n_missing_raw = sum(not np.isfinite(r).all() for r in raw_rows['ref_mean'])
print(f'Done: {len(meta_rows)} variants scored ({n_new - n_fill} new, {n_fill} filled in with raw '
      f'measures / spatial predictions this session; {n_missing_raw} still without raw measures)')

# %% ----------------------------------------------------------------------------------
# CELL 8 - normalise (z-scores, PCA) and save the variant x track matrices
# ---------------------------------------------------------------------------------------
variants = pd.DataFrame(meta_rows)
variants['pos'] = variants['pos'].astype(int)
sad = np.array(sad_rows, dtype=np.float32)
local = np.array(local_rows, dtype=np.float32)
peak_bin = np.array(peakbin_rows, dtype=np.int32)          # (n_variants, 5313)
peak_delta = np.array(peakdelta_rows, dtype=np.float32)    # ALT - REF at the peak bin
curves = np.array(curve_rows, dtype=np.float32)            # (n_variants, n_groups, 896)
curve_track = np.array(curvetrack_rows, dtype=np.int32)    # (n_variants, n_groups)

Z = scaler.transform(sad).astype(np.float32)        # (n_variants, 5313) z-scores
pcs = pipeline.transform(sad)[:, :NUM_PCS]          # (n_variants, NUM_PCS)
assert Z.shape == (len(variants), 5313)

# Genomic coordinates of the output window: bin b covers [window_start + 128 b, +128)
window_starts = np.array([Interval(c, p, p).resize(NUM_BINS * 128).start
                          for c, p in zip(variants['chrom'], variants['pos'])])
peak_dist = (window_starts[:, None] + peak_bin * 128 + 64) - variants['pos'].values[:, None]

cols = annot['track_name'].values                   # '<index>_<full description>'
idx = variants['variant_key'].values
variants.to_csv(f'{OUT_DIR}/variants_scored.tsv', sep='\t', index=False)
pd.DataFrame(sad, index=idx, columns=cols).to_csv(f'{OUT_DIR}/matrix_raw_SAD_variants_x_tracks.tsv.gz', sep='\t')
pd.DataFrame(local, index=idx, columns=cols).to_csv(f'{OUT_DIR}/matrix_raw_local3bins_variants_x_tracks.tsv.gz', sep='\t')
pd.DataFrame(Z, index=idx, columns=cols).to_csv(f'{OUT_DIR}/matrix_zscore_variants_x_tracks.tsv.gz', sep='\t')
pd.DataFrame(peak_dist, index=idx, columns=cols).to_csv(f'{OUT_DIR}/matrix_peak_distance_bp_variants_x_tracks.tsv.gz', sep='\t')
pd.DataFrame(peak_delta, index=idx, columns=cols).to_csv(f'{OUT_DIR}/matrix_peak_delta_variants_x_tracks.tsv.gz', sep='\t')
np.save(f'{OUT_DIR}/matrix_zscore_variants_x_tracks.npy', Z)
# RAW model units, no z-scores: REF mean, ALT mean, (signed mean change = sad), mean |change|, max |change|
raw = {k: np.array(raw_rows[k], dtype=np.float32) for k in RAW_KEYS}
for k, name in (('ref_mean', 'REF_mean'), ('alt_mean', 'ALT_mean'),
                ('mean_abs_delta', 'mean_abs_delta'), ('max_abs_delta', 'max_abs_delta')):
    pd.DataFrame(raw[k], index=idx, columns=cols) \
        .to_csv(f'{OUT_DIR}/matrix_raw_{name}_variants_x_tracks.tsv.gz', sep='\t')
np.savez_compressed(f'{OUT_DIR}/matrices_raw_5_measures.npz', ref_mean=raw['ref_mean'],
                    alt_mean=raw['alt_mean'], delta_mean=sad, mean_abs_delta=raw['mean_abs_delta'],
                    max_abs_delta=raw['max_abs_delta'], variant_keys=idx.astype(str),
                    track_names=cols.astype(str))
pd.concat([variants[['variant_key', 'chrom', 'pos', 'id', 'ref', 'alt']],
           pd.DataFrame(pcs, columns=[f'PC{i}' for i in range(pcs.shape[1])])], axis=1) \
    .to_csv(f'{OUT_DIR}/pca{NUM_PCS}_variants.tsv', sep='\t', index=False)
print('Saved matrices:', Z.shape)

# %% ----------------------------------------------------------------------------------
# CELL 9 - RANKING of tissues / cell types per variant, separately per assay group
# ---------------------------------------------------------------------------------------
ranking, celltype_matrices = rank_cell_types(Z, variants, annot, top_n=TOP_N, rank_by=RANK_BY,
                                                  peak_bin=peak_bin, peak_delta=peak_delta,
                                                  window_starts=window_starts)
ranking.to_csv(f'{OUT_DIR}/ranking_celltypes_per_variant_ALL_groups.tsv', sep='\t', index=False)
for group, g in ranking.groupby('group'):
    g.to_csv(f'{OUT_DIR}/ranking_celltypes_per_variant_{group}.tsv', sep='\t', index=False)
    celltype_matrices[group].to_csv(f'{OUT_DIR}/matrix_celltype_effect_{group}.tsv.gz', sep='\t')
# REVERSE ranking: for every cell type / tissue, the variants with the strongest effect
var_rank = rank_variants_per_celltype(Z, variants, annot, top_v=TOP_VARIANTS, peak_bin=peak_bin,
                                      peak_delta=peak_delta, window_starts=window_starts)
var_rank.to_csv(f'{OUT_DIR}/ranking_variants_per_celltype_ALL_groups.tsv', sep='\t', index=False)
for group, g in var_rank.groupby('group'):
    g.to_csv(f'{OUT_DIR}/ranking_variants_per_celltype_{group}.tsv', sep='\t', index=False)
summary = summarise_top(ranking, variants, top=3)
summary.to_csv(f'{OUT_DIR}/summary_top_celltypes_per_variant.tsv', sep='\t', index=False)
# the same question in RAW units: median mean |ALT-REF| per cell type (compare within a group only)
if np.isfinite(raw['mean_abs_delta']).all():
    rank_cell_types_raw(raw['mean_abs_delta'], sad, variants, annot, top_n=TOP_N) \
        .to_csv(f'{OUT_DIR}/ranking_celltypes_per_variant_RAW_mean_abs_delta.tsv', sep='\t', index=False)
else:
    print('raw ranking skipped: some variants still lack raw measures (re-run cell 7)')

print(f'\nRanking saved for {ranking.variant_id.nunique()} variants, groups: '
      f'{sorted(ranking.group.unique())}')
print('\nTop-1 cell type per variant and group (first 10 variants):')
print(ranking[ranking['rank'] == 1]
      .pivot_table(index='variant_id', columns='group', values='cell_type', aggfunc='first')
      .head(10).to_string())

# %% ----------------------------------------------------------------------------------
# CELL 10 - FIGURES (saved as PNG in OUT_DIR/figures). Four kinds, chosen to answer:
#   (a) Which tissues/cells are hit most often?      -> fig1_top_celltypes_<...>.png
#   (b) Which variants hit which cell types, and in
#       which direction?                              -> fig2_heatmap_<group>.png (one per group)
#   (c) Do variants cluster by overall effect?        -> fig3_pca_scatter.png
#   (d) What does one strong variant look like along
#       the genome (REF, ALT, ALT-REF)?               -> fig4_tracks_<variant>.png  (top 5 only)
# ==== PLOTS BEGIN
# ---------------------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt

FIG_DIR = f'{OUT_DIR}/figures'
N_TRACK_PLOTS = 5              # number of individual variants drawn as genome tracks
# (set N_TRACK_PLOTS = len(variants) to draw genome tracks for ALL variants: 2 extra forward
#  passes and one large PNG per variant, roughly doubling the run time)


def fig_top_celltypes_bar(ranking, path, top_k=15):
    """(a) For each assay group: the cell types that most often rank #1 across all variants."""
    groups = sorted(ranking['group'].unique())
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    for ax, group in zip(axes.ravel(), groups):
        counts = ranking[(ranking['group'] == group) & (ranking['rank'] == 1)]['cell_type'] \
            .value_counts().head(top_k)[::-1]
        ax.barh([c[:40] for c in counts.index], counts.values, color='steelblue')
        ax.set_title(f'{group}: cell types ranked #1 most often', fontsize=11)
        ax.set_xlabel('number of variants')
    for ax in axes.ravel()[len(groups):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_group_heatmap(matrix, ranking_group, path, title, n_var=40, n_ct=30):
    """(b) Variants x cell types heatmap of signed z (red = increase, blue = decrease).

    Columns: the cell types that appear most often in the per-variant top-N ranking.
    Rows: the variants with the largest |z| over those columns.
    """
    cols = ranking_group['cell_type'].value_counts().head(n_ct).index
    sub = matrix[cols]
    rows = sub.abs().max(axis=1).sort_values(ascending=False).head(n_var).index
    sub = sub.loc[rows]
    vmax = max(float(np.percentile(np.abs(sub.values), 99)), 1e-6)
    fig, ax = plt.subplots(figsize=(0.35 * len(cols) + 4, 0.25 * len(rows) + 3))
    im = ax.imshow(sub.values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[:30] for c in cols], rotation=90, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.split(':')[-1] for r in rows], fontsize=7)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.6, label='z-score of ALT - REF (strongest track)')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_pca_scatter(pcs, variants, path, label_n=10):
    """(c) PC0 vs PC1 of all variants; the label_n most extreme variants are named."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pcs[:, 0], pcs[:, 1], s=12, alpha=0.6)
    for i in np.argsort(-np.hypot(pcs[:, 0], pcs[:, 1]))[:label_n]:
        ax.annotate(variants['id'].iloc[i], (pcs[i, 0], pcs[i, 1]), fontsize=7)
    ax.set_xlabel('PC0 (overall effect size)')
    ax.set_ylabel('PC1')
    ax.set_title(f'{len(variants)} variants in Enformer PCA space')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_variant_summary(rank_v, title, save_png=None, pdf=None, loci=None):
    """One figure for ONE variant: (top) its top-N cell types in every assay group,
    (bottom) WHERE in the locus each group's strongest track changes.

    rank_v : rows of the ranking table belonging to this variant (all groups).
    loci   : optional {group: dict(x_kb, y, label, peak_kb)} -> one ALT-REF curve per group along
             the ~115 kb output window (x = distance from the variant in kb; dashed line = the
             variant; red dot = the bin with the largest |ALT-REF|).
    Bars = z-score of ALT-REF at the cell type's strongest track (red = increases the predicted
    signal, blue = decreases). The number in brackets is how many tracks back that cell type.
    """
    groups = sorted(rank_v['group'].unique())
    xmax = max(float(rank_v['max_abs_z'].max()), 1e-6) * 1.15      # same x scale in all panels
    n_loc = len(loci) if loci else 0
    fig = plt.figure(figsize=(15, 11 + 1.6 * n_loc))
    gs = fig.add_gridspec(3 + n_loc, 2, height_ratios=[3, 3, 3] + [1.2] * n_loc, hspace=0.55,
                          top=0.965, bottom=0.03)
    for k, group in enumerate(groups):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        g = rank_v[rank_v['group'] == group].sort_values('rank')[::-1]
        ax.barh([f'{c[:38]} ({n})' for c, n in zip(g['cell_type'], g['n_tracks'])],
                g['z_at_best_track'], color=['firebrick' if z > 0 else 'steelblue'
                                              for z in g['z_at_best_track']])
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlim(-xmax, xmax)
        ax.set_title(group, fontsize=11)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel('z-score ALT - REF')
    if loci:
        shared = None
        for r, (group, L) in enumerate(loci.items()):
            ax = fig.add_subplot(gs[3 + r, :], sharex=shared)
            shared = shared or ax
            ax.fill_between(L['x_kb'], L['y'], color='firebrick', alpha=0.8)
            ax.axhline(0, color='black', lw=0.6)
            ax.axvline(0, color='black', ls='--', lw=0.9)
            ax.plot([L['peak_kb']], [L['y'][int(np.abs(L['y']).argmax())]], 'o', color='black', ms=5)
            ax.set_title(f"{group}: {L['label'][:70]} | peak {L['peak_kb'] * 1000:+,.0f} bp from variant",
                         fontsize=8)
            ax.tick_params(labelsize=7)
        ax.set_xlabel('distance from variant (kb)   [dashed = variant, dot = largest change]')
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.985)
    if save_png:
        fig.savefig(save_png, dpi=110, bbox_inches='tight')
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
# ==== PLOTS END


def fig_variant_tracks(model, row, tracks, path):
    """(d) REF / ALT / ALT-REF along the ~115 kb output window for chosen tracks of one variant.

    row: variant record (chrom, pos, id, ref, alt); tracks: list of (track_index, label).
    """
    chrom = row['chrom'].replace(CHR_PREFIX, '', 1)
    x_ref, x_alt = make_ref_alt(chrom, int(row['pos']), row['id'], row['ref'], row['alt'])
    p_ref = model.predict_on_batch(x_ref[np.newaxis])['human'][0]
    p_alt = model.predict_on_batch(x_alt[np.newaxis])['human'][0]
    iv = Interval(row['chrom'], int(row['pos']), int(row['pos'])).resize(NUM_BINS * 128)
    x = np.linspace(iv.start, iv.end, NUM_BINS)
    fig, axes = plt.subplots(3 * len(tracks), 1, figsize=(16, 1.6 * 3 * len(tracks)), sharex=True)
    for t, (ti, label) in enumerate(tracks):
        for k, (y, name, col) in enumerate([(p_ref[:, ti], 'REF', 'steelblue'),
                                            (p_alt[:, ti], 'ALT', 'darkorange'),
                                            (p_alt[:, ti] - p_ref[:, ti], 'ALT - REF', 'firebrick')]):
            ax = axes[3 * t + k]
            ax.fill_between(x, y, color=col, alpha=0.85)
            ax.axvline(row['pos'], color='black', ls='--', lw=0.8)
            ax.set_title(f'{row["id"]} {row["chrom"]}:{row["pos"]} {row["ref"]}>{row["alt"]} | '
                         f'{label[:60]} | {name}', fontsize=8)
    axes[-1].set_xlabel(f'{row["chrom"]} (bp)')
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


os.makedirs(FIG_DIR, exist_ok=True)
fig_top_celltypes_bar(ranking, f'{FIG_DIR}/fig1_top_celltypes_per_group.png')
for group in sorted(ranking['group'].unique()):
    fig_group_heatmap(celltype_matrices[group], ranking[ranking['group'] == group],
                      f'{FIG_DIR}/fig2_heatmap_{group}.png',
                      f'{group}: variants x cell types (top-{TOP_N} ranking)')
fig_pca_scatter(pcs, variants, f'{FIG_DIR}/fig3_pca_scatter.png')

# One figure PER VARIANT (all variants): top cell types per assay group + locus (where it changes).
# -> figures/per_variant/<variant>.png  and one multi-page PDF (one page per variant).
from matplotlib.backends.backend_pdf import PdfPages
os.makedirs(f'{FIG_DIR}/per_variant', exist_ok=True)
x_rel_bp = (np.arange(NUM_BINS) + 0.5) * 128 - NUM_BINS * 64      # bin centres relative to the window centre
with PdfPages(f'{FIG_DIR}/all_variants_top_celltypes.pdf') as pdf:
    for vi, v in variants.iterrows():
        rv = ranking[(ranking['variant_id'] == v['id']) & (ranking['chrom'] == v['chrom'])
                     & (ranking['pos'] == v['pos']) & (ranking['alt'] == v['alt'])]
        # locus panels: ALT-REF of each group's strongest track, x = distance from the variant (kb)
        off_kb = (window_starts[vi] + NUM_BINS * 64 - v['pos'] + x_rel_bp) / 1000.0
        loci = {}
        for gi, group in enumerate(GROUPS):
            y = curves[vi, gi]
            tname = annot.loc[curve_track[vi, gi], 'track_name']
            loci[group] = {'x_kb': off_kb, 'y': y, 'label': f'strongest track {tname}',
                           'peak_kb': float(off_kb[int(np.abs(y).argmax())])}
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', f"{v['id']}_{v['chrom']}_{v['pos']}_{v['ref']}_{v['alt']}")
        fig_variant_summary(rv, f"{v['id']}   {v['chrom']}:{v['pos']}  {v['ref']}>{v['alt']}",
                            save_png=f'{FIG_DIR}/per_variant/{safe}.png', pdf=pdf, loci=loci)
print(f'Per-variant figures: {len(variants)} PNGs + all_variants_top_celltypes.pdf')

# Genome-track plots for the N_TRACK_PLOTS variants with the strongest single effect in
# CAGE and histone ChIP (re-runs the model, 2 forward passes per variant).
best = ranking[(ranking['rank'] == 1) & ranking['group'].isin(['CAGE', 'ChIP_histone'])]
for vid in best.sort_values('max_abs_z', ascending=False)['variant_id'].drop_duplicates().head(N_TRACK_PLOTS):
    r1 = best[best['variant_id'] == vid].set_index('group')
    row = variants[variants['id'] == vid].iloc[0]
    tracks = [(int(r1.loc[g, 'best_track_index']), r1.loc[g, 'best_track'])
              for g in ('CAGE', 'ChIP_histone') if g in r1.index]
    fig_variant_tracks(model, row, tracks, f'{FIG_DIR}/fig4_tracks_{re.sub(r"[^A-Za-z0-9._-]+", "_", vid)}.png')
print('Figures saved to', FIG_DIR, ':', sorted(os.listdir(FIG_DIR)))

print('\nAll files in', OUT_DIR, ':')
print(sorted(os.listdir(OUT_DIR)))


# %% ----------------------------------------------------------------------------------
# CELL 11 - GRADIENT x INPUT and IN-SILICO MUTAGENESIS for the strongest variants
#   gradient x input : d(prediction of a track, averaged over the central output bins) / d(one-hot
#                      input) multiplied by the input, summed over the 4 bases -> one number per
#                      position. Large |value| = that base pushes the prediction up/down.
#   ISM              : replace each base near the variant by each of the 3 alternatives and record
#                      the change in the same prediction. Direct, no gradients needed.
# Tracks used: the strongest track (largest |z|) of each assay group for that variant.
# ==== ATTR BEGIN
# ---------------------------------------------------------------------------------------
LETTERS = np.array(list('ACGT'))


def compute_ism(x_ref, var_idx, half_width, predict_batch, tracks, central, batch_size=4):
    """In-silico mutagenesis around a variant.

    x_ref: (L, 4) one-hot REF sequence; var_idx: index of the variant base; predict_batch(x) maps
    (b, L, 4) -> (b, 896, n_tracks); tracks: track indices to report; central: slice of output bins.
    Returns (effect, ref_bases): effect[t, i, b] = change (mutant - REF) of track t's prediction,
    averaged over the central bins, when base i (position var_idx - half_width + i) becomes base b
    (0 for the reference base itself).
    """
    lo = var_idx - half_width
    ref_bases = x_ref[lo:var_idx + half_width].argmax(axis=1)
    base = predict_batch(x_ref[np.newaxis])[0][central][:, tracks].mean(axis=0)
    jobs = [(i, b) for i in range(2 * half_width) for b in range(4) if b != ref_bases[i]]
    effect = np.zeros((len(tracks), 2 * half_width, 4), dtype=np.float32)
    for s in range(0, len(jobs), batch_size):
        chunk = jobs[s:s + batch_size]
        xb = np.repeat(x_ref[np.newaxis], len(chunk), axis=0)
        for k, (i, b) in enumerate(chunk):
            xb[k, lo + i] = 0
            xb[k, lo + i, b] = 1
        pr = predict_batch(xb)
        for k, (i, b) in enumerate(chunk):
            effect[:, i, b] = pr[k][central][:, tracks].mean(axis=0) - base
    return effect, ref_bases


def fig_attribution(attr, letters_ref, letters_alt, groups, labels, title, path, zoom=20):
    """Gradient x input around one variant. attr: (n_groups, 2 [REF, ALT], 2W). One row per assay
    group: left = full window (REF blue, ALT orange), right = +-zoom bp with the bases as tick labels."""
    n, _, w2 = attr.shape
    W = w2 // 2
    x = np.arange(-W, W)
    fig, axes = plt.subplots(n, 2, figsize=(17, 2.3 * n), gridspec_kw={'width_ratios': [3, 2]})
    axes = np.atleast_2d(axes)
    for r in range(n):
        ax = axes[r, 0]
        ax.plot(x, attr[r, 0], color='steelblue', lw=0.8, label='REF')
        ax.plot(x, attr[r, 1], color='darkorange', lw=0.8, label='ALT')
        ax.axvline(0, color='black', ls='--', lw=0.8)
        ax.set_title(f'{groups[r]}: {labels[r][:60]}', fontsize=8)
        if r == 0:
            ax.legend(fontsize=7, loc='upper right')
        ax = axes[r, 1]
        sel = (x >= -zoom) & (x <= zoom)
        ax.bar(x[sel] - 0.2, attr[r, 0][sel], width=0.4, color='steelblue')
        ax.bar(x[sel] + 0.2, attr[r, 1][sel], width=0.4, color='darkorange')
        ax.axvline(0, color='black', ls='--', lw=0.8)
        ax.set_xticks(x[sel])
        ax.set_xticklabels([f'{a}' if a == b else f'{a}>{b}' for a, b in
                            zip(letters_ref[sel], letters_alt[sel])], fontsize=5, rotation=90)
    axes[-1, 0].set_xlabel('position relative to the variant (bp)')
    fig.suptitle(title + '   (gradient x input; blue = REF, orange = ALT)', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def fig_ism(effect, ref_bases, groups, labels, title, path):
    """ISM heatmaps. effect: (n_groups, 2H, 4); rows of each panel = the 4 bases, columns = positions."""
    n, h2, _ = effect.shape
    fig, axes = plt.subplots(n, 1, figsize=(16, 1.9 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for r in range(n):
        vmax = max(float(np.abs(effect[r]).max()), 1e-9)
        ax = axes[r]
        im = ax.imshow(effect[r].T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.plot(np.arange(h2), ref_bases, 'k.', ms=2)          # dots mark the reference base
        ax.set_yticks(range(4))
        ax.set_yticklabels(list('ACGT'), fontsize=7)
        ax.axvline(h2 / 2 - 0.5, color='black', ls='--', lw=0.6)
        ax.set_title(f'{groups[r]}: {labels[r][:60]}', fontsize=8)
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    axes[-1].set_xlabel(f'position (variant at the centre, +-{h2 // 2} bp)   dots = reference base')
    fig.suptitle(title + '   (in-silico mutagenesis: change of predicted signal)', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
# ==== ATTR END


@tf.function
def input_grad(seq, mask):
    """Gradient x input of the masked mean prediction w.r.t. the one-hot input (as in the DeepMind notebook)."""
    x = seq[tf.newaxis]
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model._model.predict_on_batch(x)['human']            # (1, 896, 5313)
        target = tf.reduce_sum(mask[tf.newaxis] * pred) / tf.reduce_sum(mask)
    g = tape.gradient(target, x) * x
    return tf.reduce_sum(tf.squeeze(g, axis=0), axis=-1)            # (L,)


def predict_batch_safe(xb):
    """Forward pass for a batch; on GPU out-of-memory the batch is split in halves and retried."""
    try:
        return model.predict_on_batch(xb)['human']
    except tf.errors.ResourceExhaustedError:
        if len(xb) == 1:
            raise
        h = len(xb) // 2
        return np.concatenate([predict_batch_safe(xb[:h]), predict_batch_safe(xb[h:])])


try:
    attr_dir = f'{OUT_DIR}/attribution'
    os.makedirs(attr_dir, exist_ok=True)
    n_var = len(variants)
    strength = np.abs(Z).max(axis=1)
    order = np.argsort(-strength)                              # strongest first
    id_to_idx = {v: i for i, v in enumerate(variants['id'])}

    grad_set = set(range(n_var)) if ATTR_TOP_VARIANTS is None else set(order[:ATTR_TOP_VARIANTS].tolist())
    ism_set = set(range(n_var)) if ISM_TOP_VARIANTS is None else set(order[:ISM_TOP_VARIANTS].tolist())
    if ISM_HEART_FIBRO_TOP > 0:                                # variants with the strongest heart/fibroblast effect
        hf = annot.index[annot['cell_type'].str.contains(
            r'\b(?:heart|cardiac|ventric\w*|atri\w*|aort\w*|coronary|fibroblast\w*)\b', regex=True)].values
        absz = np.abs(Z[:, hf])
        pct = (absz.argsort(axis=0).argsort(axis=0) + 1) / n_var     # per-track percentile of |z| among variants
        ism_set |= set(np.argsort(-pct.max(axis=1))[:ISM_HEART_FIBRO_TOP].tolist())
        print(f'ISM heart/fibroblast subset built from {len(hf)} tracks')
    ism_set |= {id_to_idx[i] for i in ISM_EXTRA_IDS if i in id_to_idx}

    todo = [int(vi) for vi in order if (vi in grad_set or vi in ism_set)][RANGE_START:RANGE_END]
    print(f'Attribution: {len(todo)} variants to process (gradient for {len(grad_set)}, ISM for {len(ism_set)} variants overall)')
    grad_ok = True
    t_start, n_done = time.time(), 0
    for k, vi in enumerate(todo, 1):
        v = variants.iloc[vi]
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', f"{v['id']}_{v['chrom']}_{v['pos']}")
        f_grad, f_ism = f'{attr_dir}/{safe}_gradxinput.npz', f'{attr_dir}/{safe}_ism.npz'
        do_grad = vi in grad_set and grad_ok and not os.path.exists(f_grad)
        do_ism = vi in ism_set and not os.path.exists(f_ism)
        if not (do_grad or do_ism):
            continue                                               # already done in an earlier run
        x_ref, x_alt = make_ref_alt(v['chrom'].replace(CHR_PREFIX, '', 1), int(v['pos']),
                                    v['id'], v['ref'], v['alt'])
        diff = np.where((x_ref != x_alt).any(axis=1))[0]
        if len(diff) == 0:
            print(f'{v["id"]}: REF and ALT sequences identical, skipped')
            continue
        var_idx = int(diff[0])                                     # index of the variant base in the input
        tracks = curve_track[vi]                                   # strongest track of each assay group
        labels = [annot.loc[t, 'track_name'] for t in tracks]
        title = f"{v['id']}  {v['chrom']}:{v['pos']} {v['ref']}>{v['alt']}"
        if do_grad:
            try:
                attr = np.zeros((len(GROUPS), 2, 2 * ATTR_WINDOW), dtype=np.float32)
                for gi, t in enumerate(tracks):
                    mask = np.zeros((NUM_BINS, 5313), dtype=np.float32)
                    mask[mid - ATTR_HALF_BINS:mid + ATTR_HALF_BINS, t] = 1.0
                    for ai, xs in enumerate((x_ref, x_alt)):
                        g = input_grad(tf.constant(xs), tf.constant(mask)).numpy()
                        attr[gi, ai] = g[var_idx - ATTR_WINDOW:var_idx + ATTR_WINDOW]
                lr = LETTERS[x_ref[var_idx - ATTR_WINDOW:var_idx + ATTR_WINDOW].argmax(axis=1)]
                la = LETTERS[x_alt[var_idx - ATTR_WINDOW:var_idx + ATTR_WINDOW].argmax(axis=1)]
                fig_attribution(attr, lr, la, GROUPS, labels, title, f'{attr_dir}/{safe}_gradxinput.png')
                np.savez_compressed(f_grad, attr=attr, letters_ref=lr, letters_alt=la,
                                    tracks=tracks, groups=np.array(GROUPS))   # written last = "done" marker
            except Exception as e:                                 # e.g. GPU out of memory in the backward pass
                grad_ok = False
                print(f'gradient x input failed ({type(e).__name__}: {str(e)[:200]}); skipping it, ISM continues')
        if do_ism:
            eff, rb = compute_ism(x_ref, var_idx, ISM_HALF_WIDTH, predict_batch_safe, tracks,
                                  central, batch_size=ISM_BATCH_SIZE)
            fig_ism(eff, rb, GROUPS, labels, title, f'{attr_dir}/{safe}_ism.png')
            np.savez_compressed(f_ism, effect=eff, ref_bases=rb, tracks=tracks, groups=np.array(GROUPS))
        n_done += 1
        rate = (time.time() - t_start) / n_done
        print(f'[{k}/{len(todo)}] done {v["id"]} (grad={do_grad}, ism={do_ism}) | {rate:.0f} s/variant | '
              f'~{rate * (len(todo) - k) / 60:.0f} min left', flush=True)
    print('Attribution files in', attr_dir)
except Exception as e:                                # never lose the core results because of this step
    import traceback
    print('CELL 11 (attribution / ISM) FAILED - core results are already saved.\n', traceback.format_exc())

# %% ----------------------------------------------------------------------------------
# CELL 12 - clustered heatmap of all variants over the most variable tracks
# ---------------------------------------------------------------------------------------
try:
    import seaborn as sns
    n_t = min(300, Z.shape[1])
    top_t = np.argsort(-Z.std(axis=0))[:n_t]
    palette = dict(zip(GROUPS, plt.cm.tab10(np.arange(len(GROUPS)))))
    col_colors = [palette[annot.loc[t, 'group']] for t in top_t]
    sub = pd.DataFrame(Z[:, top_t], index=variants['id'].values,
                       columns=[annot.loc[t, 'track_name'] for t in top_t])
    keep_c = (sub.std(axis=0) > 0).values                       # constant rows/columns have no correlation
    sub, col_colors = sub.loc[:, keep_c], [c for c, k in zip(col_colors, keep_c) if k]
    sub = sub.loc[sub.std(axis=1) > 0]
    assert sub.shape[0] > 2 and sub.shape[1] > 2, 'too few variable variants/tracks to cluster'
    for metric in ('correlation', 'euclidean'):                # fall back if correlation is undefined
        try:
            cg = sns.clustermap(sub, method='average', metric=metric, cmap='RdBu_r', center=0, robust=True,
                                figsize=(16, 14), col_colors=col_colors, xticklabels=False,
                                yticklabels=len(sub) <= 80, cbar_kws={'label': 'z-score ALT - REF'})
            break
        except ValueError as err:
            print(f'clustermap with metric={metric} failed ({err}); trying the next one')
    cg.fig.suptitle(f'{len(sub)} variants x {n_t} most variable tracks (column colours = assay group: '
                    + ', '.join(GROUPS) + ')', fontsize=11)
    cg.savefig(f'{FIG_DIR}/fig5_variant_clustermap.png', dpi=130, bbox_inches='tight')
    plt.close(cg.fig)
    print('Saved fig5_variant_clustermap.png')
except Exception as e:
    import traceback
    print('CELL 12 (clustermap) FAILED - core results are already saved.\n', traceback.format_exc())

# %% ----------------------------------------------------------------------------------
# CELL 13 - GENE EFFECTS AT TSSs from the saved heart/fibroblast predictions (no model run)
#   For every gene (GENE_TYPES) whose TSS lies in a variant's 896-bin output window:
#   log2 fold change ALT/REF of the signal summed over the TSS bin +-TSS_HALF_BINS, for every
#   heart/fibroblast track. CAGE at the TSS = predicted expression change (a predicted eQTL);
#   DNase / ATAC / histone at the TSS = promoter activity change.
#   Summaries per assay group, heart and fibroblast separately: mean over tracks and the signed
#   value with the largest |log2 FC| (+ that track). REF signal says whether the gene is active.
# ---------------------------------------------------------------------------------------
try:
    assert SAVE_HF_SPATIAL, 'SAVE_HF_SPATIAL is False: no saved predictions to read'
    if not os.path.exists(GTF_PATH):
        import urllib.request, shutil
        print('downloading', GTF_URL)
        with urllib.request.urlopen(GTF_URL) as r, open(GTF_PATH + '.part', 'wb') as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(GTF_PATH + '.part', GTF_PATH)
    genes_tss = load_gene_tss(GTF_PATH, GENE_TYPES)
    genes_by_chrom = {c: g.reset_index(drop=True) for c, g in genes_tss.groupby('chrom')}
    hf = annot.loc[hf_idx].reset_index(drop=True)
    hf_names = hf['track_name'].values
    summary_sets = [(grp, tag, (hf['group'].values == grp) & hf[col].values)
                    for grp in GROUPS for tag, col in (('heart', 'heart'), ('fibro', 'fibroblast'))]
    summary_sets = [s for s in summary_sets if s[2].any()]
    gene_rows, gene_lfc, n_missing = [], [], 0
    for vi, v in variants.iterrows():
        f = hf_file(v['chrom'], v['pos'], v['id'], v['ref'], v['alt'])
        g = genes_by_chrom.get(v['chrom'])
        if not os.path.exists(f):
            n_missing += 1
            continue
        if g is None:
            continue
        z = np.load(f)
        assert np.array_equal(z['track_index'], hf_idx), f'{f}: saved with a different track set'
        gw, lfc, ref_sig = tss_gene_effects(z['ref'], z['alt'], g, int(v['pos']), TSS_HALF_BINS, PSEUDOCOUNT)
        for k, gene in gw.iterrows():
            row = {'variant_key': v['variant_key'], 'id': v['id'], 'chrom': v['chrom'], 'pos': int(v['pos']),
                   'ref': v['ref'], 'alt': v['alt'], 'gene_id': gene['gene_id'], 'gene_name': gene['gene_name'],
                   'gene_type': gene['gene_type'], 'strand': gene['strand'], 'tss': int(gene['tss']),
                   'tss_dist_bp': int(gene['tss_dist_bp'])}
            for grp, tag, m in summary_sets:
                sub = lfc[k, m]
                j = int(np.abs(sub).argmax())
                row[f'{grp}_{tag}_lfc_mean'] = float(sub.mean())
                row[f'{grp}_{tag}_lfc_maxabs'] = float(sub[j])
                row[f'{grp}_{tag}_best_track'] = hf_names[m][j]
                row[f'{grp}_{tag}_ref_signal_mean'] = float(ref_sig[k, m].mean())
            gene_rows.append(row)
            gene_lfc.append(lfc[k])
    gene_eff = pd.DataFrame(gene_rows)
    gene_eff.to_csv(f'{OUT_DIR}/gene_effects_heart_fibroblast_TSS.tsv', sep='\t', index=False)
    pd.DataFrame(np.array(gene_lfc, np.float32).reshape(-1, len(hf_idx)),
                 index=(gene_eff['variant_key'] + '|' + gene_eff['gene_name']) if len(gene_eff) else None,
                 columns=hf_names).to_csv(f'{OUT_DIR}/gene_effects_heart_fibroblast_TSS_lfc_all_tracks.tsv.gz', sep='\t')
    print(f'Gene effects: {len(gene_eff)} variant-gene pairs for {gene_eff["id"].nunique() if len(gene_eff) else 0} '
          f'variants ({n_missing} variants have no saved spatial file: re-run cell 7)')
    col = 'CAGE_heart_lfc_mean'
    if len(gene_eff) and col in gene_eff:
        top = gene_eff.reindex(gene_eff[col].abs().sort_values(ascending=False).index).head(40)[::-1]
        fig, ax = plt.subplots(figsize=(9, 0.28 * len(top) + 1.5))
        ax.barh([f"{i} -> {g} ({d / 1000:+.0f} kb)" for i, g, d in zip(top['id'], top['gene_name'], top['tss_dist_bp'])],
                top[col], color=['firebrick' if x > 0 else 'steelblue' for x in top[col]])
        ax.axvline(0, color='black', lw=0.8)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_xlabel('predicted log2 fold change ALT/REF at the TSS (mean over heart CAGE tracks)')
        ax.set_title('Strongest predicted gene effects, heart CAGE (single pass)')
        fig.savefig(f'{FIG_DIR}/fig6_top_gene_effects_heart_CAGE.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(top[::-1][['id', 'gene_name', 'tss_dist_bp', col]].head(15).to_string(index=False))
except Exception:
    import traceback
    print('CELL 13 (gene effects) FAILED - core results are already saved.\n', traceback.format_exc())
