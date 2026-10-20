"""Build the accompanying review from recorded results; no experiments run here."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).parent
OUT=ROOT.parent/'deliverables';OUT.mkdir(exist_ok=True)
FONT='/usr/share/fonts/truetype/dejavu/'
for name,file in [('Sans','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,FONT+file))
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='Bold',italic='Sans',boldItalic='Bold')
navy=HexColor('#17324D'); teal=HexColor('#16795D'); gray=HexColor('#5D6673')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title7',fontName='Bold',fontSize=22,leading=27,textColor=navy,spaceAfter=10))
styles.add(ParagraphStyle(name='H7',fontName='Bold',fontSize=13,leading=17,textColor=navy,spaceBefore=9,spaceAfter=6))
styles.add(ParagraphStyle(name='P7',fontName='Sans',fontSize=9.2,leading=13,spaceAfter=7))
styles.add(ParagraphStyle(name='Small7',fontName='Sans',fontSize=7.5,leading=10,textColor=gray,spaceAfter=5))
styles.add(ParagraphStyle(name='Cell7',fontName='Sans',fontSize=7.5,leading=10))
styles.add(ParagraphStyle(name='Code7',fontName='Mono',fontSize=7.2,leading=10,spaceAfter=7))
story=[]
def p(text,style='P7'):story.append(Paragraph(text,styles[style]))
def h(text):p(text,'H7')
def table(rows,widths):
    data=[[Paragraph(escape(str(x)).replace('\n','<br/>'),styles['Cell7']) for x in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#E7EEF4')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
        ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,navy),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#F7F9FB')])]))
    story.extend([t,Spacer(1,8)])
def read(name):return json.loads((ROOT/name).read_text())
base=read('audit_baselines.json')
def b(recipe,k):return next(x for x in base if x['population']==recipe and x['K']==k)
# Normalize distinct experiment runners' recorded formats.
def ordinary(path,label):
    z=read(path)
    return dict(label=label,cost=z['total_fe'],net=z['net_fe'],rows=z['rows'],m=z['metrics'])
def aggregate(k):
    z=read(f'aggregates_mf4pulse_unif_k{k}.json')['results']['groups']
    m={a:{'rms':v['rms_points'],'mean':v['mean_points'],'ratio':v['mse_ratio']} for a,v in z['summary'].items()}
    return dict(label=f'MF4 + pulses + H4, K={k} (full kernel)',cost=z['total_fe'],net=z['net_fe'],rows=z['anchor_rows'],m=m)
def baseline(k):
    # Audit schemas use recipe/k; read robustly from headline consolidation.
    z=next(x for x in read('headline_summary.json')['records'] if x['family']=='g' and x['population']=='unif' and x['k']==k)
    m={'crossfit':dict(rms=z['crossfit']['rms_points'],mean=z['crossfit']['mean_points'],ratio=z['crossfit']['mse_ratio'])}
    m.update({g:z['external'][g] for g in ('b128','b64')});m['mystery']={'mean':z['external']['mystery_error']}
    return dict(label=f'Original G(1,{k})',cost=z['total_fe'],net=z['cost_parts']['network_fe'],rows=1440*k,m=m)
chosen=[baseline(1),ordinary('combo_results/unif_k1_ends_mfboth_p1_sup128.json','Compact + pulses, K=1'),
 ordinary('cost_results/unif_k2_all_mf4_sup1024.json','MF4 only, K=2 (full kernel)'),
 ordinary('combo_results/unif_k2_all_mf4_p1_sup128.json','MF4 + pulses, K=2 (128)'),
 ordinary('extended_results/unif_k2_mf4_pulse_h1_sup128.json','MF4 + pulses + H4, K=2 (128)'),
 aggregate(2),baseline(2),aggregate(4),baseline(4)]

p('A larger low-budget advantage','Title7')
p('Round 7 | AlgZoo 432-parameter RNN | 20 October 2026','Small7')
p('<b>The population-supervised estimator survives review, and several inexpensive changes improve it.</b> '
  'A compact deterministic probe design reaches 0.951 percentage-point RMS at 176 forward equivalents (FE). '
  'MF4-only probes, signed pulses and temporal summaries reach 0.621 points at 598 FE, or 0.274 times sampling\'s MSE. '
  'With a full kernel, the two- and four-slice versions reach 0.606 points at 651 FE and 0.515 points at 1,216 FE.')
p('The experiments widen the margin in the useful low-budget range. They do not establish a new win beyond approximately 4,000 FE. '
  'All quoted new errors are cross-fitted across independently trained networks; the extra groups are reported separately.')
story.append(Image(str(ROOT/'round7_frontier.png'),width=530,height=214))
h('Primary population: representative results')
table([['Method','Total FE','RMS, pp','Mean, pp','MSE / sampling']]+[
 [z['label'],f"{z['cost']:.1f}",f"{z['m']['crossfit']['rms']:.3f}",f"{z['m']['crossfit']['mean']:+.3f}",f"{z['m']['crossfit']['ratio']:.3f}"] for z in chosen],
 [250,64,66,66,84])
p('Sources: audit_baselines.json; cost_results, combo_results and extended_results; aggregates_mf4pulse_unif_k2/k4.json. '
  'The plotted lines connect tested K settings; they are not a claim of global optimality. Errors throughout are percentage points.','Small7')
story.append(PageBreak())

h('1. Review: validation, scoring and actual deployment cost')
p('<b>No direct label leakage was found in G.</b> Feature standardization, response centering and inner hyperparameter selection use only the '
  'applicable training fold. A network\'s own margin normalization and weight statistics are allowed target computations. The sampled cell-target '
  'fields in the AlgZoo caches are not used. The population trainers keep the networks\' inputs, parameters and updates separate.')
p('<b>The shared label bank is a qualification, not a fatal flaw.</b> It induces correlated integration error across networks, and a fitted estimator '
  'can partly learn that bank\'s particular error. One must not subtract independent label variances as if the training and test label errors were '
  'independent. The mean individual label standard errors are only about 0.0142 points for the primary population and 0.0257 for AlgZoo. '
  'Those are far below current residuals. A second independent label bank is the appropriate final verification.')
p('<b>The fixed-cost comparator is correct:</b> ratio = mean((prediction - p)<super>2</super>) C / mean(p(1-p)). '
  'If cost varies by target or fold, its denominator must be mean(p<sub>i</sub>(1-p<sub>i</sub>)/C<sub>i</sub>), not a formula using mean cost. '
  'We conservatively charge 1,024 supports for full kernels although an outer fit has 896 supports. A selected linear predictor is cheaper still.')
p('<b>Two missing implementation optimizations matter.</b> First, gfeat.summary charges input products once per distinct scalar but actually '
  'recomputes them at every trie node. That omits 33.378, 62.928 and 122.028 FE at K=1,2,4. Precompute a value-by-hidden-unit table and gather from it. '
  'Second, fit_predict calls a helper that evaluates all 63 candidate models on the target. Its six target kernel-by-eigenvector matrix products alone '
  'cost 2,184.53 FE. After offline model choice, contract the eigensystem into one coefficient vector. Deployment then needs one kernel query. '
  'The supplied standalone implementation performs these two optimizations.')
p('The recurrent trie count checks out. For prefix counts n<sub>1</sub>,...,n<sub>10</sub> and v distinct scalar values, its numerator is '
  '256 sum(n<sub>1</sub>,...,n<sub>9</sub>) + 16v + 160n<sub>10</sub> + 16 sum(n<sub>2</sub>,...,n<sub>10</sub>) '
  '+ 16 sum(n<sub>1</sub>,...,n<sub>10</sub>) + 20n<sub>10</sub>. Divide by 2,880. '
  'Exponentials are included in the readout\'s 34-times-support-count allowance. The small-operation allowances are arithmetic conventions, '
  'not hardware latency measurements.')
p('<b>The historical numbers reproduce, but the inner folds depend unnecessarily on array layout.</b> The expression '
  'tr*7919 % len(y) changes inner assignments when unused rows are appended. This uses metadata, not their labels, but makes results unstable. '
  'For primary K=4, the historical z55 layout gives 0.557059 points, the 1,153-row cache layout gives 0.549883, and stable 1,024-population IDs give '
  '0.583058. We use the stable IDs and the same outer permutation for paired new comparisons. The old numbers were not transcription errors.')
table([['Recipe','G K=1 RMS / ratio','G K=2 RMS / ratio','G K=4 RMS / ratio'],
 ['Primary','0.96355 / 0.68237','0.67802 / 0.61706','0.58306 / 0.86924'],
 ['AlgZoo','0.59673 / 0.31873','0.51503 / 0.43361','0.43451 / 0.58792']], [106,142,142,140])
p('Nested tuning protects the stated search, not the entire history of research choices. Several combinations were motivated by earlier '
  'outer-CV results, and the extra groups have been examined repeatedly. Treat these as exploratory results with strong supporting checks, '
  'then freeze a method and test a fresh seed batch. Batch 64 is a recipe-shift diagnostic, not an exact batch-128 holdout.','Small7')
story.append(PageBreak())

h('2. Cost decomposition and external networks')
p('The following costs include weight statistics, summaries, standardization and the selected readout. Anchor rows are listed separately '
  'from the two signed pulse trajectories: the latter decode 20 age outputs, and their entire computation is charged at 6 FE.')
table([['Method','Anchor rows','Anchor FE','Other FE','Total FE']]+[
 [z['label'],str(z['rows']),f"{z['net']:.1f}",f"{z['cost']-z['net']:.1f}",f"{z['cost']:.1f}"] for z in chosen], [244,70,72,72,72])
p('The MF4 K=2 design uses the same 1,440 anchor rows as G K=1, but allocates them to two S values instead of two anchor types. '
  'Its prefix cost is almost identical. This is the largest straightforward cost reduction. The compact design keeps all 90 ordered (M,S) '
  'position pairs and puts R at the first and last unused positions, retaining 180 of 720 arrangements. Those are fixed temporal probes, '
  'not random or quasi-random input samples.')
table([['Method','B128 RMS / mean','B64 RMS / mean','Mystery error']]+[
 [z['label'],f"{z['m']['b128']['rms']:.3f} / {z['m']['b128']['mean']:+.3f}",
  f"{z['m']['b64']['rms']:.3f} / {z['m']['b64']['mean']:+.3f}",f"{z['m']['mystery']['mean']:+.3f}"] for z in chosen], [242,105,105,78])
p('For the full-kernel MF4 + pulse + H4 versions, external MSE ratios are 0.2271 (B128) and 0.5395 (B64) at K=2, '
  'and 0.3360 / 0.5002 at K=4 (exact JSON values are authoritative). The near-zero mystery error at K=4 is much smaller than its '
  'reference-label uncertainty and should be viewed as a fortunate close match. It does not show deterministic sub-noise precision.')
p('External hyperparameters in the audit and block-tuned runs use index-mod-8 folds. The compact/Nystrom runs reuse permutation-0 training '
  'CV scores. Both procedures are training-only, but individual external differences cannot all be attributed to the feature change alone. '
  'The main population comparisons share the same nested outer/inner folds. Tables retain the original conservative runner allowances; '
  'the consolidated shared extractor saves a few small operations without changing the mathematical predictor.','Small7')
story.append(PageBreak())

h('3. Features: what carries additional information?')
table([['Priority and feature','Mechanistic interpretation','Cost and observed result'],
 ['1. Signed input pulses','Start at ReLU(+Wih) and ReLU(-Wih), then propagate through nine zero-input steps. Summarize age-aligned margins and output-amplitude growth. These are reachable transient-memory tests.',
  '10 features; complete extraction 6 FE, plus 7.12 FE for 10 full-kernel coordinates. G K=1 primary: 0.964 to 0.841 RMS.'],
 ['2. Temporal group summaries','Per S slice: M last; S last; M immediately after S; M after S; R after S; M-S gap at least five steps. Distinguishes handoffs, direct writes, late distractors and retention.',
  'H4 adds 12K features, about 0.821K FE arithmetic with reused summaries. MF4 + pulse at K=2: 0.663 to 0.606 RMS with full kernel.'],
 ['3. Anchor margin gradients','Differentiate target-minus-largest-rival margin through actual gates, then summarize variation from the six lower inputs and adjacent directions.',
  '68 rows at K=4; +150.89 FE in the standalone code, or +77.51 FE with retained forward gates. Primary G K=4: 0.583 to 0.516; AlgZoo: 0.435 to 0.406.'],
 ['4. Margin tails and MF disagreement','CDF counts around zero; MF4/MF3 disagreement; size of the margin change. Looks for near-tie fragility hidden by average margins.',
  '15K features; about 8K FE extraction plus 10.68K FE full-kernel query. At K=4, primary 0.583 to 0.529; no useful gain at K=1.'],
 ['5. Designed adjacent lower-input perturbations','Test gate and rival switches that a local derivative cannot predict, especially during late handoffs near extreme S slices.',
  'Not tested this round. A fixed 32-64 row panel is roughly 30-70 FE before sharing. Select directions offline and keep selection inside training.'],
 ['6. Additive kernel or residual correction','Preserve the base predictor while adding a mechanism-specific correction, so an unusual temporal signature cannot erase all original neighbors.',
  'Not tested. About one extra small kernel query, depending on support count. Motivated by the rare-network failure below.'],
 ['7. Small hidden-unit encoder','Pool per-unit readout, input, recurrence and finite-horizon descriptors; a small model could replace many support queries.',
  'Not tested. A small 16-unit encoder can fit within roughly 5-20 FE, but 1,024 networks may limit training. Include relational information, not only unordered scalar summaries.'],
 ['Lower priority: bare spectrum / hidden basis impulses','Raw Whh eigenvalues omit changing ReLU gates; arbitrary hidden-state injections may not be reachable.',
  'The tested basis-impulse summaries did not help, even before their extra cost. Finite signed-input trajectories are preferable.']], [123,220,187])
p('The pulse age label is the location of the pulse in a zero-padded sequence. Its alignment is not second-argmax task correctness. '
  'The normalized pulse features are invariant to hidden-unit gauge and positive output scaling where numerical floors are inactive. '
  'Gradient fragility is likewise a feature, not a probability or certificate: both gates and the winning rival can change off the anchor.')
story.append(PageBreak())

h('4. Recipe differences, failures and the high-budget limit')
p('<b>AlgZoo benefits from pulses, but not from every temporal kernel.</b> G plus pulse summaries gives RMS 0.579 / 0.489 / 0.405 at '
  '632 / 1,143 / 2,165 FE for K=1/2/4; MSE ratios are 0.306 / 0.396 / 0.514. MF4 plus pulses with 128 supports gives '
  '0.548 at 591 FE and 0.500 at 1,132 FE. The corresponding zoo seed errors are shown below.')
table([['AlgZoo method','Seed 0','Seed 1','Seed 2','Seed 3','Seed 4'],
 ['MF4 + pulse, K=2, 128','-0.182','+0.436','+0.027','-0.262','+0.176'],
 ['MF4 + pulse, K=4, 128','-0.218','+0.407','-0.213','-0.165','-0.479'],
 ['G + pulse, K=4, full','-0.264','+0.610','-0.435','-0.331','-0.208']], [225,61,61,61,61,61])
p('At AlgZoo K=1, temporal groups increase full-population RMS from 0.597 to 0.718. Network 994, whose reference accuracy is '
  '78.605%, explains essentially all of the loss: its error rises from +4.210 points under G to +13.694 under the selected aggregate model. '
  'It remains in every score. Appending distances multiplies RBF similarities; an unusual poor network can lose its neighbors and be pulled '
  'toward the population mean. This is a reason to try additive kernels or carefully cross-fitted residual correction.')
p('The primary K=1 gap is not explained simply by anchor values exceeding the Unif training interval. With positive homogeneity, every '
  'direction can be scaled into that cube. The recipes differ in directional weighting, optimization and learned mechanisms. More S slices '
  'expose different relative gaps; the stronger benefit of near-zero summaries and gradients at K=4 supports that explanation, without proving it.')
p('<b>More S slices did not push the crossover higher.</b> At K=16, MF4 plus pulses with 128 supports costs 4,377.72 FE and gives '
  '0.5731 primary RMS (1.709 times sampling MSE) and 0.4257 AlgZoo RMS (1.148 times). Adding fixed H4 improves the primary result to '
  '0.5214 at 4,432.05 FE, still 1.432 times sampling. These are negative tests of particular designs and readouts, not a lower bound on '
  'all deterministic estimators. Low-rank readout capacity, feature quality and estimation floors remain distinguishable possibilities.')
h('Would 2,048 or 4,096 training networks help?')
p('A learning curve for the fixed MF4 + pulse K=2, 128-support estimator keeps the query cost at 591.448 FE. Every row scores all '
  '1,024 population networks out of fold, with smaller nested training subsets.')
table([['Outer training networks','Final-refit count','CF RMS','CF mean','MSE ratio'],
 ['224','256','0.9019','-0.0568','0.5717'],['448','512','0.7721','-0.0049','0.4189'],
 ['672','768','0.6512','-0.0055','0.2980'],['896','1,024','0.6506','+0.0013','0.2975']], [160,108,90,90,82])
p('The last interval is flat in this subset sequence. That does not prove more data is useless, nor justify a fitted power-law forecast. '
  'Repeat the curve for a richer feature model and several subset orders before committing to thousands more networks. Add extra batch-128 '
  'networks after freezing validation; batch-64 networks require recipe-aware modeling rather than automatic pooling.','Small7')
story.append(PageBreak())

h('5. Recommended next steps')
p('<b>1. Freeze and confirm the efficient methods.</b> Test compact + pulses at about 176 FE, MF4 + pulses + H4 at approximately 600-650 FE, '
  'and the four-slice full-kernel version at 1,216 FE on a fresh batch-128 seed cohort. Use a second independent reference input bank. '
  'This would distinguish a reproducible practical improvement from accumulated research selection.')
p('<b>2. Refine the probe budget.</b> The current compact design preserves every ordered (M,S) pair. Learn additional R/Q placements or '
  'S slices within training folds, optimizing expected squared-error reduction per added trie operation. The cost is the change to the union '
  'of prefixes, not the number of rows alone. Learned feature-cell selection must also be repeated inside the folds.')
p('<b>3. Target the residual with a few nonlinear perturbations.</b> Use the validated gradient block to pick a small offline family of '
  'joint, adjacent lower-input perturbations in late-handoff and extreme-S cells. Charge every new prefix, backward operation and summary. '
  'This directly tests the gate-switching behavior that local derivatives and original anchors miss.')
p('<b>4. Preserve robust extrapolation while improving the readout.</b> Test an additive base-plus-mechanism kernel or an out-of-fold residual '
  'model. Include the poorest networks as a diagnostic. Select 64/128/256 supports inside a fixed-budget comparison. The 128-support '
  'approximation is effective on the primary population but degrades some full-G AlgZoo results; compression is not uniformly free.')
p('<b>5. Revisit data size only after the representation is fixed.</b> Use several nested subset learning curves and report both bulk and '
  'poor-network errors. If the curve still falls, train 2,048 networks before 4,096. Otherwise prioritize new information rather than a larger '
  'readout training set.')
h('Other inexpensive optimizations')
p('Compile a selected linear readout to a single coefficient vector. Compress exact duplicate summaries while preserving the induced '
  'kernel metric; simply deleting them changes the estimator. At K=1 the duplicated ten summary coordinates can be merged with a square-root-of-two '
  'weight while retaining the original gamma denominator, saving about 7.1 FE. For target transforms, clipping to [0,1] is always safe in squared '
  'error; logit or error-rate transforms must be assessed after inverse transformation in probability-space MSE. These last transforms were not '
  'tested here. More flexible models should be compared by their compiled inference cost, not just their fitting score.')
h('Reproduction and delivered code')
p('round7_estimator.py is a self-contained NumPy/SciPy program. It reads the supplied caches, can generate deterministic MF4 designs at '
  'larger K, reproduces the main nested fits, saves a compiled model, and evaluates a new weight file without rerunning model selection. '
  'The archive retains the experiment scripts, exact JSON metrics, per-network predictions, derivative checks and plot data.')
p('Validation checks: the optimized prefix implementation exactly matched direct float64 logits and argmaxes in nine design/model cases. '
  'Compiled exact/inducing predictors matched their candidate predictions within 2.8e-12 in probability. The gradient checks satisfied Euler '
  'homogeneity within 4e-13 and 192 deterministic central differences within 1.5e-7. These are implementation checks, separate from '
  'statistical generalization.')
p('No new RNNs were trained, no GPU was needed, and no random/quasi-random target inputs or weight-noise copies were used. Offline work '
  'included cached nested fits (typically 20-35 seconds per full-kernel variant, 2-4 seconds for 128 supports), 78-237 seconds per block-tuned '
  'search, deterministic new-cache generation, and the learning curve. Wall times overlap across processes and are not target FE timings.','Small7')
p('Primary evidence: supplied QUESTION7_FOR_CHATGPT_PRO.md; gfeat.py, z49.py, z55.py, z59.py and their logs. New evidence is preserved in '
  'the named JSON/NPZ result files and verification.json. The fixed challenge rules allow a population-learned deterministic weight/probe '
  'estimator. This result is a learned predictor with mechanistic features; it is not an accuracy theorem derived completely from the circuit.','Small7')

def footer(canvas,doc):
    canvas.setStrokeColor(HexColor('#D6DEE6'));canvas.line(40,38,572,38)
    canvas.setFont('Sans',7);canvas.setFillColor(gray)
    canvas.drawString(40,26,'AlgZoo accuracy estimation | Round 7 | Reproducible experimental review')
    canvas.drawRightString(572,26,str(doc.page))

SimpleDocTemplate(str(OUT/'ROUND7_REPORT.pdf'),pagesize=(612,792),leftMargin=40,rightMargin=40,topMargin=34,bottomMargin=49,
                  title='AlgZoo Round 7: accuracy estimation review',author='OpenAI').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT/'ROUND7_REPORT.pdf')
